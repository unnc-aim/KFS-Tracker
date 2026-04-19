from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import random
import re
import uuid

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, load_settings
from .database import Base, create_engine_and_session
from .models import AugmentJob, ExportJob, GeneratedSample, SourceSample
from .schemas import AugmentJobCreate, BatchBBoxUpdate, ExportCreate, ReviewUpdate
from .services.exporter import build_boxed_images_export_zip, build_yolo_export_zip
from .services.transform import (
    CameraParams,
    quad_to_yolo_line,
    transform_image_and_bbox,
)


def _ensure_dirs(settings: Settings):
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.generated_image_dir.mkdir(parents=True, exist_ok=True)
    settings.generated_label_dir.mkdir(parents=True, exist_ok=True)
    settings.export_dir.mkdir(parents=True, exist_ok=True)


def _abs_path(settings: Settings, relative_path: str) -> Path:
    return Path(settings.data_dir) / relative_path


def _make_filename(prefix: str, suffix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"


def _parse_text_from_filename(filename: str) -> str:
    return Path(filename).stem.strip() or "text"


def _filename_text_token(text: str, max_len: int = 40) -> str:
    t = (text or "text").strip()
    # keep CJK/alnum/_/- ; replace others with underscore
    t = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "text"
    return t[:max_len]


def _parse_quad_norm_from_yolo_line(
    line: str, fallback_bbox: tuple[float, float, float, float] | None = None
) -> list[list[float]]:
    parts = line.strip().split()
    if len(parts) == 9:
        vals = [float(v) for v in parts[1:]]
        pts = [[vals[i], vals[i + 1]] for i in range(0, 8, 2)]
        return [[max(0.0, min(1.0, p[0])), max(0.0, min(1.0, p[1]))] for p in pts]

    if len(parts) == 5 and fallback_bbox is not None:
        x, y, w, h = fallback_bbox
        # fallback assumes bbox is pixel-like in an unknown frame, return empty to avoid false precision
        return []

    return []


def create_app(settings_override: Settings | None = None) -> FastAPI:
    settings = settings_override or load_settings()
    _ensure_dirs(settings)

    engine, session_local = create_engine_and_session(settings.database_url)
    Base.metadata.create_all(bind=engine)

    app = FastAPI(title="Wulin OCR 3D Augmentation", version="1.0.0")
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_local = session_local

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent.parent / "static"
    app.mount("/media", StaticFiles(directory=settings.data_dir), name="media")
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

    def get_db():
        db = session_local()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @app.get("/", response_class=HTMLResponse)
    def home():
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)

    @app.post("/api/v1/samples/upload")
    async def upload_sample(
        file: UploadFile = File(...),
        text: str = Form(""),
        bbox_x: float = Form(...),
        bbox_y: float = Form(...),
        bbox_w: float = Form(...),
        bbox_h: float = Form(...),
        class_id: int = Form(0),
        use_filename_as_text: bool = Form(False),
        db: Session = Depends(get_db),
    ):
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        filename = _make_filename("raw", suffix)
        rel_path = str(Path("raw_images") / filename)
        abs_path = _abs_path(settings, rel_path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        abs_path.write_bytes(content)

        final_text = (
            _parse_text_from_filename(file.filename or "")
            if use_filename_as_text or not text.strip()
            else text.strip()
        )

        sample = SourceSample(
            image_path=rel_path,
            text=final_text,
            class_id=class_id,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
        )
        db.add(sample)
        db.flush()
        return {
            "sample_id": sample.id,
            "image_url": f"/media/{sample.image_path}",
            "text": sample.text,
            "bbox": [sample.bbox_x, sample.bbox_y, sample.bbox_w, sample.bbox_h],
        }

    @app.post("/api/v1/samples/batch-upload")
    async def batch_upload_samples(
        files: list[UploadFile] = File(...),
        class_id: int = Form(0),
        use_filename_as_text: bool = Form(True),
        default_text: str = Form("text"),
        db: Session = Depends(get_db),
    ):
        if not files:
            raise HTTPException(status_code=400, detail="files 不能为空")

        created = []
        for file in files:
            suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
            filename = _make_filename("raw", suffix)
            rel_path = str(Path("raw_images") / filename)
            abs_path = _abs_path(settings, rel_path)
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            content = await file.read()
            abs_path.write_bytes(content)

            decoded = cv2.imread(str(abs_path))
            if decoded is None:
                continue
            h, w = decoded.shape[:2]
            text = (
                _parse_text_from_filename(file.filename or "")
                if use_filename_as_text
                else (default_text.strip() or "text")
            )

            sample = SourceSample(
                image_path=rel_path,
                text=text,
                class_id=class_id,
                bbox_x=0.0,
                bbox_y=0.0,
                bbox_w=float(max(1, w - 1)),
                bbox_h=float(max(1, h - 1)),
            )
            db.add(sample)
            db.flush()
            created.append(
                {
                    "sample_id": sample.id,
                    "image_url": f"/media/{sample.image_path}",
                    "text": sample.text,
                    "bbox": [
                        sample.bbox_x,
                        sample.bbox_y,
                        sample.bbox_w,
                        sample.bbox_h,
                    ],
                }
            )

        return {"count": len(created), "items": created}

    @app.get("/api/v1/samples")
    def list_samples(offset: int = 0, limit: int = 100, db: Session = Depends(get_db)):
        items = (
            db.execute(
                select(SourceSample)
                .order_by(SourceSample.id.asc())
                .offset(offset)
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [
            {
                "sample_id": s.id,
                "image_url": f"/media/{s.image_path}",
                "text": s.text,
                "class_id": s.class_id,
                "bbox_x": s.bbox_x,
                "bbox_y": s.bbox_y,
                "bbox_w": s.bbox_w,
                "bbox_h": s.bbox_h,
                "created_at": s.created_at.isoformat(),
            }
            for s in items
        ]

    @app.patch("/api/v1/samples/batch-bbox")
    def batch_update_bbox(req: BatchBBoxUpdate, db: Session = Depends(get_db)):
        updated = 0
        for item in req.items:
            s = db.get(SourceSample, item.sample_id)
            if s is None:
                continue
            s.bbox_x = item.bbox_x
            s.bbox_y = item.bbox_y
            s.bbox_w = max(1.0, item.bbox_w)
            s.bbox_h = max(1.0, item.bbox_h)
            if item.text is not None and item.text.strip():
                s.text = item.text.strip()
            updated += 1
        db.flush()
        return {"updated": updated}

    @app.post("/api/v1/augment/jobs")
    def create_augment_job(req: AugmentJobCreate, db: Session = Depends(get_db)):
        if not req.sample_ids:
            raise HTTPException(status_code=400, detail="sample_ids 不能为空")

        rng = random.Random(req.seed)
        config_json = req.model_dump_json()
        job = AugmentJob(
            status="running",
            config_json=config_json,
            total_count=len(req.sample_ids) * req.count_per_sample,
            success_count=0,
            error_count=0,
        )
        db.add(job)
        db.flush()

        samples = (
            db.execute(select(SourceSample).where(SourceSample.id.in_(req.sample_ids)))
            .scalars()
            .all()
        )
        sample_map = {s.id: s for s in samples}

        for sid in req.sample_ids:
            source = sample_map.get(sid)
            if source is None:
                job.error_count += req.count_per_sample
                continue

            src_img = cv2.imread(str(_abs_path(settings, source.image_path)))
            if src_img is None:
                job.error_count += req.count_per_sample
                continue

            bbox = (source.bbox_x, source.bbox_y, source.bbox_w, source.bbox_h)
            src_h, src_w = src_img.shape[:2]
            src_ratio = (source.bbox_w * source.bbox_h) / max(1.0, float(src_w * src_h))

            for _ in range(req.count_per_sample):
                try:
                    params = CameraParams(
                        focal=rng.uniform(*req.focal_range),
                        pitch=rng.uniform(*req.pitch_range),
                        yaw=rng.uniform(*req.yaw_range),
                        roll=rng.uniform(*req.roll_range),
                        tx=rng.uniform(*req.tx_range),
                        ty=rng.uniform(*req.ty_range),
                        tz=rng.uniform(*req.tz_range),
                        scale=rng.uniform(*req.scale_range),
                    )
                    out_img, out_bbox, quad_pts = transform_image_and_bbox(
                        src_img, bbox, params
                    )
                    oh, ow = out_img.shape[:2]
                    out_ratio = (out_bbox[2] * out_bbox[3]) / max(1.0, float(ow * oh))

                    # quality guard: drop obviously degenerate boxes caused by extreme projections
                    lower = max(0.0015, src_ratio * 0.22)
                    upper = min(0.85, max(src_ratio * 3.2, src_ratio + 0.08))
                    if out_ratio < lower or out_ratio > upper:
                        job.error_count += 1
                        continue

                    yolo = quad_to_yolo_line(source.class_id, quad_pts, ow, oh)

                    label_token = _filename_text_token(source.text)
                    img_name = _make_filename(
                        f"gen_{label_token}_sid{source.id}", ".jpg"
                    )
                    lbl_name = Path(img_name).with_suffix(".txt").name
                    rel_img = str(Path("generated") / "images" / img_name)
                    rel_lbl = str(Path("generated") / "data" / lbl_name)
                    abs_img = _abs_path(settings, rel_img)
                    abs_lbl = _abs_path(settings, rel_lbl)
                    abs_img.parent.mkdir(parents=True, exist_ok=True)
                    abs_lbl.parent.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(str(abs_img), out_img)
                    abs_lbl.write_text(yolo + "\n", encoding="utf-8")

                    generated = GeneratedSample(
                        source_id=source.id,
                        job_id=job.id,
                        image_path=rel_img,
                        label_path=rel_lbl,
                        yolo_line=yolo,
                        bbox_x=out_bbox[0],
                        bbox_y=out_bbox[1],
                        bbox_w=out_bbox[2],
                        bbox_h=out_bbox[3],
                        pitch=params.pitch,
                        yaw=params.yaw,
                        roll=params.roll,
                        tx=params.tx,
                        ty=params.ty,
                        tz=params.tz,
                        scale=params.scale,
                        focal=params.focal,
                    )
                    db.add(generated)
                    job.success_count += 1
                except Exception:
                    job.error_count += 1

        job.status = "completed"
        job.finished_at = datetime.now(UTC)
        db.flush()

        return {
            "job_id": job.id,
            "status": job.status,
            "total_count": job.total_count,
            "success_count": job.success_count,
            "error_count": job.error_count,
        }

    @app.get("/api/v1/augment/jobs/{job_id}")
    def get_job(job_id: int, db: Session = Depends(get_db)):
        job = db.get(AugmentJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job 不存在")
        return {
            "job_id": job.id,
            "status": job.status,
            "total_count": job.total_count,
            "success_count": job.success_count,
            "error_count": job.error_count,
        }

    @app.get("/api/v1/generated")
    def list_generated(
        offset: int = 0, limit: int = 100, db: Session = Depends(get_db)
    ):
        items = (
            db.execute(
                select(GeneratedSample)
                .order_by(GeneratedSample.id.desc())
                .offset(offset)
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [
            {
                "id": g.id,
                "source_id": g.source_id,
                "job_id": g.job_id,
                "image_url": f"/media/{g.image_path}",
                "label_url": f"/api/v1/generated/{g.id}/label",
                "yolo_line": g.yolo_line,
                "quad_norm": _parse_quad_norm_from_yolo_line(
                    g.yolo_line, (g.bbox_x, g.bbox_y, g.bbox_w, g.bbox_h)
                ),
                "bbox_x": g.bbox_x,
                "bbox_y": g.bbox_y,
                "bbox_w": g.bbox_w,
                "bbox_h": g.bbox_h,
                "pitch": g.pitch,
                "yaw": g.yaw,
                "roll": g.roll,
                "tx": g.tx,
                "ty": g.ty,
                "tz": g.tz,
                "scale": g.scale,
                "focal": g.focal,
                "review_status": g.review_status,
                "created_at": g.created_at.isoformat(),
            }
            for g in items
        ]

    @app.get("/api/v1/generated/{gen_id}/image")
    def get_generated_image(gen_id: int, db: Session = Depends(get_db)):
        g = db.get(GeneratedSample, gen_id)
        if not g:
            raise HTTPException(status_code=404, detail="generated sample 不存在")
        return FileResponse(_abs_path(settings, g.image_path))

    @app.get("/api/v1/generated/{gen_id}/label")
    def get_generated_label(gen_id: int, db: Session = Depends(get_db)):
        g = db.get(GeneratedSample, gen_id)
        if not g:
            raise HTTPException(status_code=404, detail="generated sample 不存在")
        return PlainTextResponse(
            _abs_path(settings, g.label_path).read_text(encoding="utf-8")
        )

    @app.patch("/api/v1/generated/{gen_id}/review")
    def review_generated(gen_id: int, req: ReviewUpdate, db: Session = Depends(get_db)):
        g = db.get(GeneratedSample, gen_id)
        if not g:
            raise HTTPException(status_code=404, detail="generated sample 不存在")
        if req.status not in {"approved", "rejected", "pending"}:
            raise HTTPException(status_code=400, detail="status 非法")
        g.review_status = req.status
        db.flush()
        return {"id": g.id, "review_status": g.review_status}

    @app.post("/api/v1/export/yolo")
    def export_yolo(req: ExportCreate, db: Session = Depends(get_db)):
        export = ExportJob(
            status="running",
            filter_json=json.dumps(req.model_dump(), ensure_ascii=False),
        )
        db.add(export)
        db.flush()

        query = select(GeneratedSample)
        if req.approved_only:
            query = query.where(GeneratedSample.review_status == "approved")
        if req.sample_ids:
            query = query.where(GeneratedSample.source_id.in_(req.sample_ids))

        generated = db.execute(query).scalars().all()
        pairs: list[tuple[Path, Path]] = []
        for g in generated:
            img = _abs_path(settings, g.image_path)
            lbl = _abs_path(settings, g.label_path)
            if img.exists() and lbl.exists():
                pairs.append((img, lbl))

        zip_name = _make_filename("yolo_export", ".zip")
        zip_path = settings.export_dir / zip_name
        build_yolo_export_zip(settings.export_dir, pairs, zip_path)

        export.status = "completed"
        export.zip_path = str(zip_path.relative_to(Path(settings.data_dir)))
        export.finished_at = datetime.now(UTC)
        db.flush()

        return {
            "export_job_id": export.id,
            "status": export.status,
            "download_url": f"/api/v1/export/yolo/{export.id}/download",
        }

    @app.post("/api/v1/export/boxed-images")
    def export_boxed_images(req: ExportCreate, db: Session = Depends(get_db)):
        export = ExportJob(
            status="running",
            filter_json=json.dumps(req.model_dump(), ensure_ascii=False),
        )
        db.add(export)
        db.flush()

        query = select(GeneratedSample)
        if req.approved_only:
            query = query.where(GeneratedSample.review_status == "approved")
        if req.sample_ids:
            query = query.where(GeneratedSample.source_id.in_(req.sample_ids))

        generated = db.execute(query).scalars().all()
        image_label_pairs: list[tuple[Path, Path, str]] = []
        for g in generated:
            img = _abs_path(settings, g.image_path)
            lbl = _abs_path(settings, g.label_path)
            if img.exists() and lbl.exists():
                text_label = g.source.text if g.source is not None else "text"
                image_label_pairs.append((img, lbl, text_label))

        zip_name = _make_filename("boxed_export", ".zip")
        zip_path = settings.export_dir / zip_name
        build_boxed_images_export_zip(settings.export_dir, image_label_pairs, zip_path)

        export.status = "completed"
        export.zip_path = str(zip_path.relative_to(Path(settings.data_dir)))
        export.finished_at = datetime.now(UTC)
        db.flush()

        return {
            "export_job_id": export.id,
            "status": export.status,
            "download_url": f"/api/v1/export/yolo/{export.id}/download",
        }

    @app.get("/api/v1/export/yolo/{export_job_id}/download")
    def download_export(export_job_id: int, db: Session = Depends(get_db)):
        export = db.get(ExportJob, export_job_id)
        if not export:
            raise HTTPException(status_code=404, detail="export job 不存在")
        if export.status != "completed" or not export.zip_path:
            raise HTTPException(status_code=400, detail="export 尚未完成")
        return FileResponse(
            _abs_path(settings, export.zip_path), filename=Path(export.zip_path).name
        )

    @app.get("/api/v1/health")
    def health():
        return JSONResponse({"status": "ok"})

    return app


app = create_app()

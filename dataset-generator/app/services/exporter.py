from __future__ import annotations

from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np


def _filename_text_token(text: str, max_len: int = 40) -> str:
    t = (text or "text").strip()
    t = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "text"
    return t[:max_len]


def build_yolo_export_zip(export_root: Path, image_label_pairs: list[tuple[Path, Path]], zip_path: Path) -> Path:
    yolo_dir = export_root / "yolo"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, lbl_path) in enumerate(image_label_pairs, start=1):
        stem = f"sample_{idx:06d}"
        target_img = images_dir / f"{stem}{img_path.suffix.lower() or '.jpg'}"
        target_lbl = labels_dir / f"{stem}.txt"
        shutil.copy2(img_path, target_img)
        shutil.copy2(lbl_path, target_lbl)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in yolo_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(export_root))
    return zip_path


def build_boxed_images_export_zip(
    export_root: Path,
    image_label_pairs: list[tuple[Path, Path, str]],
    zip_path: Path,
) -> Path:
    boxed_dir = export_root / "boxed_images"
    if boxed_dir.exists():
        shutil.rmtree(boxed_dir)
    boxed_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, lbl_path, text_label) in enumerate(image_label_pairs, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        line = lbl_path.read_text(encoding="utf-8").strip()
        parts = line.split()

        # new format: class + 8 values (quad points normalized)
        if len(parts) == 9:
            vals = [float(v) for v in parts[1:]]
            pts = np.array(vals, dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 127), thickness=2)
        # backward compatibility: class + xc yc bw bh
        elif len(parts) == 5:
            xc, yc, bw, bh = [float(v) for v in parts[1:]]
            x1 = int(round((xc - bw / 2.0) * w))
            y1 = int(round((yc - bh / 2.0) * h))
            x2 = int(round((xc + bw / 2.0) * w))
            y2 = int(round((yc + bh / 2.0) * h))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 127), 2)

        token = _filename_text_token(text_label)
        target_path = boxed_dir / f"boxed_{idx:06d}_{token}.jpg"
        cv2.imwrite(str(target_path), image)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in boxed_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(export_root))
    return zip_path

from pathlib import Path
import io
import zipfile

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import Settings
from app.main import create_app


def _make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (200, 80), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 20, 160, 60], outline=(20, 20, 20), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_end_to_end_pipeline(tmp_path: Path):
    data_dir = tmp_path / "data"
    db_path = tmp_path / "test.db"
    settings = Settings(database_url=f"sqlite:///{db_path}", data_dir=str(data_dir))
    app = create_app(settings)
    client = TestClient(app)

    img_bytes = _make_test_image_bytes()
    files = [
        ("files", ("武林第一式.jpg", img_bytes, "image/jpeg")),
        ("files", ("武林第二式.jpg", img_bytes, "image/jpeg")),
    ]
    data = {
        "class_id": "0",
        "use_filename_as_text": "true",
        "default_text": "fallback",
    }
    r = client.post("/api/v1/samples/batch-upload", files=files, data=data)
    assert r.status_code == 200
    uploaded = r.json()
    assert uploaded["count"] == 2
    sample_ids = [item["sample_id"] for item in uploaded["items"]]
    assert sample_ids == sorted(sample_ids)

    r = client.get("/api/v1/samples?limit=10")
    assert r.status_code == 200
    samples = r.json()
    assert len(samples) == 2
    assert samples[0]["text"] == "武林第一式"
    assert samples[1]["text"] == "武林第二式"

    r = client.patch(
        "/api/v1/samples/batch-bbox",
        json={
            "items": [
                {
                    "sample_id": sample_ids[0],
                    "bbox_x": 40,
                    "bbox_y": 20,
                    "bbox_w": 120,
                    "bbox_h": 40,
                },
                {
                    "sample_id": sample_ids[1],
                    "bbox_x": 42,
                    "bbox_y": 22,
                    "bbox_w": 118,
                    "bbox_h": 38,
                },
            ]
        },
    )
    assert r.status_code == 200
    assert r.json()["updated"] == 2

    r = client.post(
        "/api/v1/augment/jobs",
        json={
            "sample_ids": sample_ids,
            "count_per_sample": 2,
            "seed": 42,
            "pitch_range": [-10, 10],
            "yaw_range": [-10, 10],
            "roll_range": [-5, 5],
            "tx_range": [-5, 5],
            "ty_range": [-5, 5],
            "tz_range": [-5, 5],
            "scale_range": [0.95, 1.05],
            "focal_range": [700, 900],
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["success_count"] == 4
    assert payload["error_count"] == 0

    r = client.get("/api/v1/generated?limit=10")
    assert r.status_code == 200
    generated = r.json()
    assert len(generated) >= 4
    assert any("/generated/images/gen_" in item["image_url"] for item in generated)

    data_dir = data_dir / "generated" / "data"
    assert data_dir.exists()
    txt_files = list(data_dir.glob("*.txt"))
    assert len(txt_files) >= 4
    assert any("武林" in p.name for p in txt_files)

    gen_id = generated[0]["id"]
    assert "quad_norm" in generated[0]
    assert isinstance(generated[0]["quad_norm"], list)
    assert len(generated[0]["quad_norm"]) == 4

    r = client.get(f"/api/v1/generated/{gen_id}/label")
    assert r.status_code == 200
    parts = r.text.strip().split()
    assert parts[0] == "0"
    assert len(parts) == 9

    r = client.post("/api/v1/export/yolo", json={"approved_only": False})
    assert r.status_code == 200
    download_url = r.json()["download_url"]

    r = client.get(download_url)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert any(name.startswith("yolo/images/") for name in names)
    assert any(name.startswith("yolo/labels/") for name in names)

    r = client.post("/api/v1/export/boxed-images", json={"approved_only": False})
    assert r.status_code == 200
    download_url = r.json()["download_url"]

    r = client.get(download_url)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert any(name.startswith("boxed_images/") for name in names)

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class SourceSampleOut(BaseModel):
    id: int
    image_path: str
    text: str
    class_id: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    created_at: datetime


class AugmentJobCreate(BaseModel):
    sample_ids: list[int]
    count_per_sample: int = Field(default=5, ge=1, le=200)
    pitch_range: tuple[float, float] = (-25.0, 25.0)
    yaw_range: tuple[float, float] = (-25.0, 25.0)
    roll_range: tuple[float, float] = (-15.0, 15.0)
    tx_range: tuple[float, float] = (-20.0, 20.0)
    ty_range: tuple[float, float] = (-20.0, 20.0)
    tz_range: tuple[float, float] = (-30.0, 30.0)
    scale_range: tuple[float, float] = (0.9, 1.1)
    focal_range: tuple[float, float] = (600.0, 1200.0)
    seed: int | None = None


class AugmentJobOut(BaseModel):
    id: int
    status: str
    total_count: int
    success_count: int
    error_count: int


class GeneratedSampleOut(BaseModel):
    id: int
    source_id: int
    job_id: int
    image_url: str
    label_url: str
    yolo_line: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    pitch: float
    yaw: float
    roll: float
    tx: float
    ty: float
    tz: float
    scale: float
    focal: float
    review_status: str
    created_at: datetime


class ReviewUpdate(BaseModel):
    status: str
    comment: str | None = None


class ExportCreate(BaseModel):
    approved_only: bool = False
    sample_ids: list[int] | None = None


class ExportOut(BaseModel):
    export_job_id: int
    status: str
    download_url: str | None = None


class BatchBBoxItem(BaseModel):
    sample_id: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    text: str | None = None


class BatchBBoxUpdate(BaseModel):
    items: list[BatchBBoxItem]

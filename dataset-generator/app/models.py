from __future__ import annotations

from datetime import UTC, datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def now_utc() -> datetime:
    return datetime.now(UTC)


class SourceSample(Base):
    __tablename__ = "source_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    text: Mapped[str] = mapped_column(String(255), nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, default=0)
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_w: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_h: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    generated_samples: Mapped[list["GeneratedSample"]] = relationship(back_populates="source")


class AugmentJob(Base):
    __tablename__ = "augment_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    generated_samples: Mapped[list["GeneratedSample"]] = relationship(back_populates="job")


class GeneratedSample(Base):
    __tablename__ = "generated_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("source_samples.id"), nullable=False)
    job_id: Mapped[int] = mapped_column(ForeignKey("augment_jobs.id"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    label_path: Mapped[str] = mapped_column(String(512), nullable=False)
    yolo_line: Mapped[str] = mapped_column(String(256), nullable=False)
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_w: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_h: Mapped[float] = mapped_column(Float, nullable=False)
    pitch: Mapped[float] = mapped_column(Float, nullable=False)
    yaw: Mapped[float] = mapped_column(Float, nullable=False)
    roll: Mapped[float] = mapped_column(Float, nullable=False)
    tx: Mapped[float] = mapped_column(Float, nullable=False)
    ty: Mapped[float] = mapped_column(Float, nullable=False)
    tz: Mapped[float] = mapped_column(Float, nullable=False)
    scale: Mapped[float] = mapped_column(Float, nullable=False)
    focal: Mapped[float] = mapped_column(Float, nullable=False)
    review_status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    source: Mapped[SourceSample] = relationship(back_populates="generated_samples")
    job: Mapped[AugmentJob] = relationship(back_populates="generated_samples")


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    filter_json: Mapped[str] = mapped_column(Text, nullable=False)
    zip_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

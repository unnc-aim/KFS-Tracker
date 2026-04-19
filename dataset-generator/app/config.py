from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    app_name: str = "wulin-ocr-3d"
    database_url: str = "sqlite:///./data/app.db"
    data_dir: str = "./data"

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw_images"

    @property
    def generated_image_dir(self) -> Path:
        return Path(self.data_dir) / "generated" / "images"

    @property
    def generated_label_dir(self) -> Path:
        return Path(self.data_dir) / "generated" / "data"

    @property
    def export_dir(self) -> Path:
        return Path(self.data_dir) / "exports"


def load_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./data/app.db"),
        data_dir=os.getenv("DATA_DIR", "./data"),
    )

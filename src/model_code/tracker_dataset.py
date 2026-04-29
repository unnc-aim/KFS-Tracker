from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset


VALID_LABEL_NAMES: tuple[str, ...] = (
    "R_R1",
    "B_R1",
    "T_03",
    "T_04",
    "T_05",
    "T_06",
    "T_07",
    "T_08",
    "T_09",
    "T_10",
    "T_11",
    "T_12",
    "T_13",
    "T_14",
    "T_15",
    "T_16",
    "T_17",
    "F_18",
    "F_19",
    "F_20",
    "F_21",
    "F_22",
    "F_23",
    "F_24",
    "F_25",
    "F_26",
    "F_27",
    "F_28",
    "F_29",
    "F_30",
    "F_31",
    "F_32",
)


@dataclass(frozen=True)
class TrackerSample:
    image_path: Path
    label_path: Path
    label_name: str
    school_name: str
    class_id: int
    bbox_xyxy_norm: tuple[float, float, float, float]
    corners_xy_norm: tuple[float, float, float, float, float, float, float, float]


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class KFSTrackerLocalizationDataset(Dataset[dict[str, object]]):
    """
    Dataset for KFS localization training.

    Directory format expected:
      labeled/<label_name>/<school_name>/<image>.jpg
      labeled/<label_name>/<school_name>/<image>.txt

    Supported label formats:
      1) class_id cx cy w h
      2) class_id x1 y1 x2 y2 x3 y3 x4 y4
      3) x1 y1 x2 y2 x3 y3 x4 y4

    If multiple lines exist in one txt, only the first valid line is used.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Image.Image], object] | None = None,
        target_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
        valid_label_names: Iterable[str] = VALID_LABEL_NAMES,
        prefer_first_valid_line: bool = True,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.valid_label_names = set(valid_label_names)
        self.prefer_first_valid_line = prefer_first_valid_line

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root}")

        self.samples: list[TrackerSample] = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No valid labeled samples found under: {self.root}")

    def _collect_samples(self) -> list[TrackerSample]:
        samples: list[TrackerSample] = []

        for label_dir in sorted(self.root.iterdir()):
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            if label_name not in self.valid_label_names:
                continue

            for school_dir in sorted(label_dir.iterdir()):
                if not school_dir.is_dir():
                    continue
                school_name = school_dir.name

                for image_path in sorted(school_dir.iterdir()):
                    if not image_path.is_file() or not _is_image_file(image_path):
                        continue

                    label_path = image_path.with_suffix(".txt")
                    if not label_path.exists():
                        continue

                    parsed = self._parse_label_file(label_path)
                    if parsed is None:
                        continue

                    class_id, xyxy_norm, corners_xy = parsed
                    samples.append(
                        TrackerSample(
                            image_path=image_path,
                            label_path=label_path,
                            label_name=label_name,
                            school_name=school_name,
                            class_id=VALID_LABEL_NAMES.index(label_name),
                            bbox_xyxy_norm=xyxy_norm,
                            corners_xy_norm=corners_xy,
                        )
                    )

        return samples

    def _bbox_to_corners_xyxy(
        self, bbox_xyxy: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float, float, float, float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        return (x1, y1, x2, y1, x2, y2, x1, y2)

    def _corners_to_bbox_xyxy(
        self, corners_xy: tuple[float, float, float, float, float, float, float, float]
    ) -> tuple[float, float, float, float]:
        xs = [corners_xy[0], corners_xy[2], corners_xy[4], corners_xy[6]]
        ys = [corners_xy[1], corners_xy[3], corners_xy[5], corners_xy[7]]
        return (_clamp01(min(xs)), _clamp01(min(ys)), _clamp01(max(xs)), _clamp01(max(ys)))

    def _parse_label_file(
        self, label_path: Path
    ) -> tuple[int, tuple[float, float, float, float], tuple[float, float, float, float, float, float, float, float]] | None:
        raw = label_path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return None

        candidates = lines if self.prefer_first_valid_line else reversed(lines)
        for line in candidates:
            parts = line.split()
            values: list[float] = []
            for token in parts:
                try:
                    values.append(float(token))
                except ValueError:
                    values = []
                    break
            if not values:
                continue

            # Format B: class x1 y1 x2 y2 x3 y3 x4 y4
            if len(values) >= 9:
                class_id = int(values[0])
                corners_xy = tuple(_clamp01(v) for v in values[1:9])
                corners_xy = (
                    corners_xy[0],
                    corners_xy[1],
                    corners_xy[2],
                    corners_xy[3],
                    corners_xy[4],
                    corners_xy[5],
                    corners_xy[6],
                    corners_xy[7],
                )
                bbox_xyxy = self._corners_to_bbox_xyxy(corners_xy)
                if bbox_xyxy[2] > bbox_xyxy[0] and bbox_xyxy[3] > bbox_xyxy[1]:
                    return class_id, bbox_xyxy, corners_xy

            # Format A: class cx cy w h
            if len(values) >= 5:
                class_id = int(values[0])
                cx, cy, w, h = values[1], values[2], values[3], values[4]
                x1 = _clamp01(cx - w * 0.5)
                y1 = _clamp01(cy - h * 0.5)
                x2 = _clamp01(cx + w * 0.5)
                y2 = _clamp01(cy + h * 0.5)
                if x2 > x1 and y2 > y1:
                    bbox_xyxy = (x1, y1, x2, y2)
                    corners_xy = self._bbox_to_corners_xyxy(bbox_xyxy)
                    return class_id, bbox_xyxy, corners_xy

            # Format C: x1 y1 x2 y2 x3 y3 x4 y4
            if len(values) >= 8:
                corners_xy = tuple(_clamp01(v) for v in values[:8])
                corners_xy = (
                    corners_xy[0],
                    corners_xy[1],
                    corners_xy[2],
                    corners_xy[3],
                    corners_xy[4],
                    corners_xy[5],
                    corners_xy[6],
                    corners_xy[7],
                )
                bbox_xyxy = self._corners_to_bbox_xyxy(corners_xy)
                if bbox_xyxy[2] > bbox_xyxy[0] and bbox_xyxy[3] > bbox_xyxy[1]:
                    return -1, bbox_xyxy, corners_xy

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")

        if self.transform is not None:
            image_out = self.transform(image)
        else:
            width, height = image.size
            image_tensor = (
                torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
                .view(height, width, 3)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            image_out = image_tensor

        target: dict[str, object] = {
            "bbox_xyxy_norm": torch.tensor(sample.bbox_xyxy_norm, dtype=torch.float32),
            "corners_xy_norm": torch.tensor(sample.corners_xy_norm, dtype=torch.float32),
            "class_id": torch.tensor(sample.class_id, dtype=torch.long),
            "label_name": sample.label_name,
            "school_name": sample.school_name,
            "image_path": str(sample.image_path),
            "label_path": str(sample.label_path),
        }

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "image": image_out,
            "target": target,
        }


def build_dataset(
    root: str | Path = "data/coworkers_for_KFS/labeled",
    transform: Callable[[Image.Image], object] | None = None,
    target_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> KFSTrackerLocalizationDataset:
    return KFSTrackerLocalizationDataset(
        root=root,
        transform=transform,
        target_transform=target_transform,
    )

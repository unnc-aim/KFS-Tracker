from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from model_code.tracker_dataset import VALID_LABEL_NAMES


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _letterbox_to_square(image_bgr: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0:
        return image_bgr.copy()

    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return image_bgr.copy()

    scale = min(float(image_size) / float(width), float(image_size) / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    pad_x = (image_size - new_width) // 2
    pad_y = (image_size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas


def _normalize_color_name(label_name: str) -> str | None:
    upper = label_name.upper()
    if upper.startswith("R_"):
        return "red"
    if upper.startswith("B_"):
        return "blue"
    return None


def _order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]

    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _is_rectangle_like(corners: np.ndarray) -> bool:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    widths = [
        float(np.linalg.norm(pts[0] - pts[1])),
        float(np.linalg.norm(pts[2] - pts[3])),
    ]
    heights = [
        float(np.linalg.norm(pts[0] - pts[3])),
        float(np.linalg.norm(pts[1] - pts[2])),
    ]
    width = sum(widths) * 0.5
    height = sum(heights) * 0.5
    if width < 1e-6 or height < 1e-6:
        return False

    ratio = min(width, height) / max(width, height)
    if ratio < 0.45:
        return False

    for i in range(4):
        prev_pt = pts[(i - 1) % 4]
        cur_pt = pts[i]
        next_pt = pts[(i + 1) % 4]
        v1 = prev_pt - cur_pt
        v2 = next_pt - cur_pt
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return False
        cos_val = abs(float(np.dot(v1, v2) / denom))
        if cos_val > 0.35:
            return False
    return True


def detect_colored_quad_corners(image_bgr: np.ndarray, color: str=None) -> tuple[bool, list[float] | None]:
    if image_bgr is None or image_bgr.size == 0:
        return False, None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if color == "red":
        lower_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        upper_mask = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
        mask = cv2.bitwise_or(lower_mask, upper_mask)
    elif color == "blue":
        mask = cv2.inRange(hsv, (90, 70, 50), (130, 255, 255))
    else:
        red_result=detect_colored_quad_corners(image_bgr,color="red")
        blue_result=detect_colored_quad_corners(image_bgr,color="blue")
        if red_result[0] and blue_result[0]:
            return False,None
        elif red_result[0]:
            return red_result
        elif blue_result[0]:
            return blue_result
        else:
            return False,None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0.0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area = 0.0
    best_corners: np.ndarray | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 500.0:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        ordered = _order_corners(approx.reshape(-1, 2))
        if not _is_rectangle_like(ordered):
            continue

        if area > best_area:
            best_area = area
            best_corners = ordered

    if best_corners is None:
        return False, None

    height, width = image_bgr.shape[:2]
    if width <= 0 or height <= 0:
        return False, None

    normalized = []
    for x, y in best_corners:
        normalized.append(_clamp01(float(x) / float(width)))
        normalized.append(_clamp01(float(y) / float(height)))
    return True, normalized


@dataclass(frozen=True)
class CornerPointSample:
    image_path: Path
    label_name: str
    school_name: str
    class_id: int
    corners_xy_norm: tuple[float, float, float, float, float, float, float, float]


class CornerPointDataset(Dataset[dict[str, object]]):
    """
    Dataset for corner-point regression + classification.

    Directory format expected:
      labeled/<label_name>/<school_name>/<image>.jpg

    During initialization, every image is passed through the built-in 2D tracker.
    Only images with successfully detected red/blue quadrilateral corners are kept.

    Cached files:
      data/<cache_prefix>_images.npy
      data/<cache_prefix>_labels.npy

    labels.npy stores shape [N, 9]:
      [x1, y1, x2, y2, x3, y3, x4, y4, class_id]
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Image.Image], object] | None = None,
        target_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
        valid_label_names: Iterable[str] = VALID_LABEL_NAMES,
        image_size: int = 300,
        cache_dir: str | Path = "data",
        cache_prefix: str = "corner_point_dataset",
    ) -> None:
        super().__init__()
        self.data_root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.valid_label_names = set(valid_label_names)
        self.image_size = image_size
        self.cache_dir = Path(cache_dir)
        self.cache_prefix = cache_prefix
        self.images_cache_path = self.cache_dir / f"{self.cache_prefix}_images.npy"
        self.labels_cache_path = self.cache_dir / f"{self.cache_prefix}_labels.npy"
        self.meta_cache_path = self.cache_dir / f"{self.cache_prefix}_meta.npy"

        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.data_root}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples: list[CornerPointSample] = []
        self.data: np.ndarray
        self.labels: np.ndarray

        if self.images_cache_path.exists() and self.labels_cache_path.exists():
            self.data = np.load(self.images_cache_path)
            self.labels = np.load(self.labels_cache_path)
            if self.data.shape[0] != self.labels.shape[0] or self.labels.ndim != 2 or self.labels.shape[1] != 9:
                self.samples = self._rebuild_cache()
            elif self.meta_cache_path.exists():
                self.samples = self._load_meta_cache()
        else:
            self.samples = self._rebuild_cache()

        if len(self.labels) == 0:
            raise RuntimeError(f"No valid corner-point samples found under: {self.data_root}")

    def _rebuild_cache(self) -> list[CornerPointSample]:
        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        samples: list[CornerPointSample] = []

        for label_dir in sorted(self.data_root.iterdir()):
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            if label_name not in self.valid_label_names:
                continue

            color = _normalize_color_name(label_name)

            class_id = VALID_LABEL_NAMES.index(label_name)

            for school_dir in sorted(label_dir.iterdir()):
                if not school_dir.is_dir():
                    continue
                school_name = school_dir.name

                for image_path in sorted(school_dir.iterdir()):
                    if not image_path.is_file() or not _is_image_file(image_path):
                        continue

                    raw_image_bgr = cv2.imread(str(image_path))
                    if raw_image_bgr is None or raw_image_bgr.size == 0:
                        continue
                    image_bgr = _letterbox_to_square(raw_image_bgr, self.image_size)
                    success, corners_xy = detect_colored_quad_corners(image_bgr, color)
                    if not success or corners_xy is None:
                        continue

                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)
                    image_np = np.asarray(image_pil, dtype=np.uint8)

                    corner_tuple = tuple(float(v) for v in corners_xy)
                    label = np.asarray([*corner_tuple, float(class_id)], dtype=np.float32)

                    samples.append(
                        CornerPointSample(
                            image_path=image_path,
                            label_name=label_name,
                            school_name=school_name,
                            class_id=class_id,
                            corners_xy_norm=corner_tuple,
                        )
                    )
                    images.append(image_np)
                    labels.append(label)

        if images:
            self.data = np.stack(images, axis=0).astype(np.uint8)
            self.labels = np.stack(labels, axis=0).astype(np.float32)
        else:
            self.data = np.empty((0, self.image_size, self.image_size, 3), dtype=np.uint8)
            self.labels = np.empty((0, 9), dtype=np.float32)

        np.save(self.images_cache_path, self.data)
        np.save(self.labels_cache_path, self.labels)
        meta_array = np.asarray(
            [
                [
                    str(sample.image_path),
                    sample.label_name,
                    sample.school_name,
                    str(sample.class_id),
                ]
                for sample in samples
            ],
            dtype=object,
        )
        np.save(self.meta_cache_path, meta_array, allow_pickle=True)
        return samples

    def _load_meta_cache(self) -> list[CornerPointSample]:
        raw_meta = np.load(self.meta_cache_path, allow_pickle=True)
        samples: list[CornerPointSample] = []
        for row, label_row in zip(raw_meta, self.labels):
            image_path = Path(str(row[0]))
            label_name = str(row[1])
            school_name = str(row[2])
            class_id = int(row[3])
            corners_xy_norm = tuple(float(v) for v in label_row[:8])
            samples.append(
                CornerPointSample(
                    image_path=image_path,
                    label_name=label_name,
                    school_name=school_name,
                    class_id=class_id,
                    corners_xy_norm=corners_xy_norm,
                )
            )
        return samples

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        image_np = self.data[index]
        label_np = self.labels[index]

        image = Image.fromarray(image_np)
        if self.transform is not None:
            image_out = self.transform(image)
        else:
            image_out = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        corners_xy_norm = torch.tensor(label_np[:8], dtype=torch.float32)
        class_id = torch.tensor(int(label_np[8]), dtype=torch.long)

        sample_meta = self.samples[index] if index < len(self.samples) else None
        target: dict[str, object] = {
            "corners_xy_norm": corners_xy_norm,
            "class_id": class_id,
        }
        if sample_meta is not None:
            target["label_name"] = sample_meta.label_name
            target["school_name"] = sample_meta.school_name
            target["image_path"] = str(sample_meta.image_path)

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
    image_size: int = 300,
    cache_dir: str | Path = "data",
    cache_prefix: str = "corner_point_dataset",
) -> CornerPointDataset:
    return CornerPointDataset(
        root=root,
        transform=transform,
        target_transform=target_transform,
        image_size=image_size,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
    )
corner_point_dataset = CornerPointDataset

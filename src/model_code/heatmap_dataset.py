from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from model_code.corner_point_dataset import detect_colored_quad_corners
from model_code.tracker_dataset import VALID_LABEL_NAMES


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_color_name(label_name: str) -> str | None:
    upper = label_name.upper()
    if upper.startswith("R_"):
        return "red"
    if upper.startswith("B_"):
        return "blue"
    return None


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


def _map_corners_to_letterboxed_norm(
    corners_xy_norm: tuple[float, float, float, float, float, float, float, float],
    orig_width: int,
    orig_height: int,
    image_size: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    if image_size <= 0 or orig_width <= 0 or orig_height <= 0:
        return corners_xy_norm

    scale = min(float(image_size) / float(orig_width), float(image_size) / float(orig_height))
    new_width = max(1, int(round(orig_width * scale)))
    new_height = max(1, int(round(orig_height * scale)))
    pad_x = (image_size - new_width) // 2
    pad_y = (image_size - new_height) // 2

    mapped: list[float] = []
    for i in range(4):
        x_norm = _clamp01(float(corners_xy_norm[2 * i]))
        y_norm = _clamp01(float(corners_xy_norm[2 * i + 1]))

        x_px = x_norm * float(orig_width)
        y_px = y_norm * float(orig_height)

        x_new = x_px * scale + float(pad_x)
        y_new = y_px * scale + float(pad_y)

        mapped.append(_clamp01(x_new / float(image_size)))
        mapped.append(_clamp01(y_new / float(image_size)))

    return (
        mapped[0],
        mapped[1],
        mapped[2],
        mapped[3],
        mapped[4],
        mapped[5],
        mapped[6],
        mapped[7],
    )


def _order_corners_tl_bl_tr_br(
    corners_xy_norm: tuple[float, float, float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float, float, float]:
    pts = np.asarray(corners_xy_norm, dtype=np.float32).reshape(4, 2)

    # Split into left/right by x, then sort each side by y.
    x_sorted = pts[np.argsort(pts[:, 0])]
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    tl = left[0]
    bl = left[1]
    tr = right[0]
    br = right[1]
    ordered = np.asarray([tl, bl, br, tr], dtype=np.float32)
    return (
        float(ordered[0, 0]),
        float(ordered[0, 1]),
        float(ordered[1, 0]),
        float(ordered[1, 1]),
        float(ordered[2, 0]),
        float(ordered[2, 1]),
        float(ordered[3, 0]),
        float(ordered[3, 1]),
    )


def _compute_reprojection_error_px(
    corners_xy_norm: tuple[float, float, float, float, float, float, float, float],
    image_size: int,
) -> float:
    if image_size <= 0:
        return float("inf")

    # Input order: tl, bl, tr, br -> convert to tl, tr, br, bl for square PnP.
    pts_tl_bl_tr_br = np.asarray(corners_xy_norm, dtype=np.float32).reshape(4, 2)
    image_points = np.asarray(
        [
            pts_tl_bl_tr_br[0],  # tl
            pts_tl_bl_tr_br[2],  # tr
            pts_tl_bl_tr_br[3],  # br
            pts_tl_bl_tr_br[1],  # bl
        ],
        dtype=np.float32,
    )
    image_points[:, 0] *= float(image_size - 1)
    image_points[:, 1] *= float(image_size - 1)

    object_points = np.asarray(
        [
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ],
        dtype=np.float32,
    )

    f = float(image_size)
    cx = float(image_size) * 0.5
    cy = float(image_size) * 0.5
    camera_matrix = np.asarray([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return float("inf")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    err = np.linalg.norm(projected - image_points, axis=1).mean()
    return float(err)


def _build_corner_heatmaps(
    corners_xy_norm: tuple[float, float, float, float, float, float, float, float],
    heatmap_size: int,
    sigma: float,
) -> np.ndarray:
    if heatmap_size <= 0:
        raise ValueError("heatmap_size must be positive.")
    sigma = max(float(sigma), 1e-3)

    heatmaps = np.zeros((4, heatmap_size, heatmap_size), dtype=np.float32)
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    denom = 2.0 * sigma * sigma

    for i in range(4):
        x_norm = _clamp01(float(corners_xy_norm[2 * i]))
        y_norm = _clamp01(float(corners_xy_norm[2 * i + 1]))
        cx = x_norm * float(heatmap_size - 1)
        cy = y_norm * float(heatmap_size - 1)
        heatmaps[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / denom)

    return heatmaps


def _read_corner_label_from_txt(label_path: Path) -> tuple[float, float, float, float, float, float, float, float] | None:
    if not label_path.exists():
        return None

    raw = label_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    for line in lines:
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

        # Corner label format with class id:
        # class_id x1 y1 x2 y2 x3 y3 x4 y4
        if len(values) >= 9:
            corners = tuple(_clamp01(v) for v in values[1:9])
            return (
                corners[0],
                corners[1],
                corners[2],
                corners[3],
                corners[4],
                corners[5],
                corners[6],
                corners[7],
            )

        # Corner label format without class id:
        # x1 y1 x2 y2 x3 y3 x4 y4
        if len(values) >= 8:
            corners = tuple(_clamp01(v) for v in values[:8])
            return (
                corners[0],
                corners[1],
                corners[2],
                corners[3],
                corners[4],
                corners[5],
                corners[6],
                corners[7],
            )

    return None


@dataclass(frozen=True)
class HeatmapSample:
    image_path: Path
    label_name: str
    school_name: str
    class_id: int
    corners_xy_norm: tuple[float, float, float, float, float, float, float, float]


class HeatmapCornerDataset(Dataset[dict[str, object]]):
    """
    Dataset for corner heatmap prediction + classification.

    Data source is traditional CV corner detection:
      detect_colored_quad_corners(...)
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Image.Image], object] | None = None,
        target_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
        valid_label_names: Iterable[str] = VALID_LABEL_NAMES,
        image_size: int = 300,
        heatmap_size: int = 300,
        heatmap_sigma: float = 3.0,
        reprojection_max_error_px: float = 8.0,
        cache_dir: str | Path = "data",
        cache_prefix: str = "heatmap_corner_dataset",
    ) -> None:
        super().__init__()
        self.data_root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.valid_label_names = set(valid_label_names)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.reprojection_max_error_px = reprojection_max_error_px
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

        self.samples: list[HeatmapSample] = []
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
        else:
            self.samples = self._rebuild_cache()

        if len(self.labels) == 0:
            raise RuntimeError(f"No valid heatmap samples found under: {self.data_root}")

    def _rebuild_cache(self) -> list[HeatmapSample]:
        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        samples: list[HeatmapSample] = []

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

                    # Priority 1: use explicit corner labels from txt when available.
                    # Fallback: use traditional CV corner tracking.
                    label_path = image_path.with_suffix(".txt")
                    corners_from_label = _read_corner_label_from_txt(label_path)

                    raw_image_bgr = cv2.imread(str(image_path))
                    if raw_image_bgr is None or raw_image_bgr.size == 0:
                        continue
                    orig_h, orig_w = raw_image_bgr.shape[:2]
                    image_bgr = _letterbox_to_square(raw_image_bgr, self.image_size)

                    if corners_from_label is not None:
                        corners_xy = _map_corners_to_letterboxed_norm(
                            corners_from_label,
                            orig_width=orig_w,
                            orig_height=orig_h,
                            image_size=self.image_size,
                        )
                    else:
                        success, corners_xy = detect_colored_quad_corners(image_bgr, color)
                        if not success or corners_xy is None:
                            continue

                        # Geometric constraint for CV-generated quadrilateral:
                        # keep only samples with reasonable reprojection error.
                        corners_xy = _order_corners_tl_bl_tr_br(tuple(float(v) for v in corners_xy))
                        reproj_err = _compute_reprojection_error_px(corners_xy, self.image_size)
                        if not np.isfinite(reproj_err) or reproj_err > self.reprojection_max_error_px:
                            continue

                    corners_tuple = _order_corners_tl_bl_tr_br(tuple(float(v) for v in corners_xy))
                    label = np.asarray([*corners_tuple, float(class_id)], dtype=np.float32)

                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    image_np = np.asarray(Image.fromarray(image_rgb), dtype=np.uint8)

                    samples.append(
                        HeatmapSample(
                            image_path=image_path,
                            label_name=label_name,
                            school_name=school_name,
                            class_id=class_id,
                            corners_xy_norm=corners_tuple,
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
            [[str(sample.image_path), sample.label_name, sample.school_name, str(sample.class_id)] for sample in samples],
            dtype=object,
        )
        np.save(self.meta_cache_path, meta_array, allow_pickle=True)
        return samples

    def _load_meta_cache(self) -> list[HeatmapSample]:
        raw_meta = np.load(self.meta_cache_path, allow_pickle=True)
        samples: list[HeatmapSample] = []
        for row, label_row in zip(raw_meta, self.labels):
            image_path = Path(str(row[0]))
            label_name = str(row[1])
            school_name = str(row[2])
            class_id = int(row[3])
            corners_xy_norm = tuple(float(v) for v in label_row[:8])
            samples.append(
                HeatmapSample(
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

        corners_xy_norm = tuple(float(v) for v in label_np[:8])
        class_id = int(label_np[8])
        heatmaps = _build_corner_heatmaps(corners_xy_norm, self.heatmap_size, self.heatmap_sigma)

        sample_meta = self.samples[index] if index < len(self.samples) else None
        target: dict[str, object] = {
            "heatmaps": torch.from_numpy(heatmaps).float(),
            "corners_xy_norm": torch.tensor(corners_xy_norm, dtype=torch.float32),
            "class_id": torch.tensor(class_id, dtype=torch.long),
        }
        if sample_meta is not None:
            target["label_name"] = sample_meta.label_name
            target["school_name"] = sample_meta.school_name
            target["image_path"] = str(sample_meta.image_path)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"image": image_out, "target": target}


def build_heatmap_dataset(
    root: str | Path = "data/coworkers_for_KFS/labeled",
    transform: Callable[[Image.Image], object] | None = None,
    target_transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
    image_size: int = 300,
    heatmap_size: int = 300,
    heatmap_sigma: float = 3.0,
    reprojection_max_error_px: float = 8.0,
    cache_dir: str | Path = "data",
    cache_prefix: str = "heatmap_corner_dataset",
) -> HeatmapCornerDataset:
    return HeatmapCornerDataset(
        root=root,
        transform=transform,
        target_transform=target_transform,
        image_size=image_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        reprojection_max_error_px=reprojection_max_error_px,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
    )

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class CameraParams:
    focal: float = 800.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    scale: float = 1.0


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([pitch, yaw, roll])
    rxm = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    rym = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    rzm = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    return rzm @ rym @ rxm


def _project_points(points_3d: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts_cam = (r @ points_3d.T).T + t.reshape(1, 3)
    z = np.clip(pts_cam[:, 2], 1e-6, None)
    x = pts_cam[:, 0] / z
    y = pts_cam[:, 1] / z
    return np.stack([k[0, 0] * x + k[0, 2], k[1, 1] * y + k[1, 2]], axis=1).astype(np.float32)


def build_homography(w: int, h: int, p: CameraParams):
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    src_3d = np.array([[-cx, -cy, 0], [cx, -cy, 0], [cx, cy, 0], [-cx, cy, 0]], dtype=np.float32) * p.scale
    r = _rotation_matrix(p.pitch, p.yaw, p.roll)
    k = np.array([[p.focal, 0, cx], [0, p.focal, cy], [0, 0, 1]], dtype=np.float32)
    t = np.array([p.tx, p.ty, p.focal + p.tz], dtype=np.float32)
    dst = _project_points(src_3d, k, r, t)
    h_mat = cv2.getPerspectiveTransform(src, dst)
    return h_mat


def _text_aware_seed_mask(image: np.ndarray, bbox_xywh: tuple[float, float, float, float]) -> np.ndarray:
    """Build a mask for the text area inside bbox.

    Fallback to rectangle bbox mask when foreground extraction is unreliable.
    """
    h, w = image.shape[:2]
    x, y, bw, bh = bbox_xywh
    x0 = max(0, min(w - 1, int(np.floor(x))))
    y0 = max(0, min(h - 1, int(np.floor(y))))
    x1 = max(0, min(w, int(np.ceil(x + bw))))
    y1 = max(0, min(h, int(np.ceil(y + bh))))

    mask = np.zeros((h, w), dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return mask

    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return mask

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thr_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def _score(binary: np.ndarray) -> tuple[float, float]:
        ratio = float(np.count_nonzero(binary)) / max(1.0, float(binary.size))
        # prefer sparse but non-trivial foreground ratio for characters
        quality = 1.0 - abs(ratio - 0.20)
        return quality, ratio

    q1, r1 = _score(thr)
    q2, r2 = _score(thr_inv)
    candidate = thr if q1 >= q2 else thr_inv
    ratio = r1 if q1 >= q2 else r2

    # Morphology to stabilize connected strokes
    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Keep only meaningful components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    filtered = np.zeros_like(candidate)
    min_area = max(6, int(candidate.size * 0.002))
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == i] = 255

    valid_ratio = float(np.count_nonzero(filtered)) / max(1.0, float(filtered.size))
    reliable = (0.01 <= ratio <= 0.90) and (0.005 <= valid_ratio <= 0.85) and np.count_nonzero(filtered) > 0

    if reliable:
        mask[y0:y1, x0:x1] = filtered
    else:
        # fallback to bbox rectangle, guaranteed usable
        mask[y0:y1, x0:x1] = 255

    return mask


def transform_image_and_bbox(image: np.ndarray, bbox_xywh: tuple[float, float, float, float], p: CameraParams):
    h, w = image.shape[:2]
    h_mat = build_homography(w, h, p)
    warped = cv2.warpPerspective(
        image,
        h_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(127, 127, 127),
    )

    x, y, bw, bh = bbox_xywh
    pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32).reshape(-1, 1, 2)
    npts = cv2.perspectiveTransform(pts, h_mat).reshape(-1, 2)

    # Robust + text-aware remapping:
    # 1) Build text seed mask from bbox ROI (fallback to rectangle).
    # 2) Warp mask with same homography.
    # 3) Get visible bbox from non-zero pixels.
    mask = _text_aware_seed_mask(image, bbox_xywh)
    if not np.any(mask):
        raise ValueError("invalid input bbox")

    warped_mask = cv2.warpPerspective(
        mask,
        h_mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # close tiny holes caused by projection aliasing
    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    nz = cv2.findNonZero(warped_mask)
    if nz is None:
        raise ValueError("bbox projected out of frame")

    bx, by, bw2, bh2 = cv2.boundingRect(nz)
    return warped, (float(bx), float(by), float(max(1, bw2)), float(max(1, bh2))), npts


def _order_quad_points_clockwise(pts: np.ndarray) -> np.ndarray:
    if pts.shape != (4, 2):
        raise ValueError("quad must be (4,2)")
    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)
    q = pts[order]
    # rotate sequence so first point is nearest top-left
    sums = q[:, 0] + q[:, 1]
    start = int(np.argmin(sums))
    q = np.roll(q, -start, axis=0)
    return q


def bbox_to_yolo_line(class_id: int, bbox_xywh: tuple[float, float, float, float], img_w: int, img_h: int) -> str:
    x, y, w, h = bbox_xywh
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    xc, yc, wn, hn = [float(np.clip(v, 0.0, 1.0)) for v in [xc, yc, wn, hn]]
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def quad_to_yolo_line(class_id: int, quad_points: np.ndarray, img_w: int, img_h: int) -> str:
    """YOLO quadrilateral format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
    q = np.asarray(quad_points, dtype=np.float32).reshape(4, 2)
    q = _order_quad_points_clockwise(q)
    q[:, 0] = np.clip(q[:, 0], 0, img_w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, img_h - 1)
    qn = np.zeros_like(q, dtype=np.float32)
    qn[:, 0] = q[:, 0] / float(img_w)
    qn[:, 1] = q[:, 1] / float(img_h)
    qn = np.clip(qn, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in qn.reshape(-1))
    return f"{class_id} {coords}"

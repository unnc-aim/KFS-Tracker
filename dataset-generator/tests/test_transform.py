import numpy as np
import cv2

from app.services.transform import CameraParams, transform_image_and_bbox


def test_identity_transform_keeps_bbox_close():
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    bbox = (40.0, 30.0, 60.0, 40.0)

    params = CameraParams(
        focal=900.0,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        tx=0.0,
        ty=0.0,
        tz=0.0,
        scale=1.0,
    )
    _, new_bbox, _ = transform_image_and_bbox(img, bbox, params)
    x, y, w, h = new_bbox

    assert abs(x - bbox[0]) < 1.5
    assert abs(y - bbox[1]) < 1.5
    assert abs(w - bbox[2]) < 2.0
    assert abs(h - bbox[3]) < 2.0


def test_perspective_transform_bbox_not_full_image():
    img = np.zeros((160, 220, 3), dtype=np.uint8)
    bbox = (60.0, 40.0, 80.0, 70.0)

    params = CameraParams(
        focal=850.0,
        pitch=12.0,
        yaw=16.0,
        roll=-8.0,
        tx=6.0,
        ty=-4.0,
        tz=8.0,
        scale=1.02,
    )
    _, new_bbox, _ = transform_image_and_bbox(img, bbox, params)
    x, y, w, h = new_bbox

    assert x >= 0
    assert y >= 0
    assert w > 1
    assert h > 1
    assert (w * h) < (img.shape[0] * img.shape[1] * 0.75)


def test_text_aware_mask_projection_is_reasonable():
    img = np.full((180, 260, 3), 230, dtype=np.uint8)
    # draw pseudo character strokes in ROI
    cv2.putText(img, "W", (90, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (20, 20, 20), 7, cv2.LINE_AA)
    bbox = (70.0, 40.0, 120.0, 100.0)

    params = CameraParams(
        focal=900.0,
        pitch=10.0,
        yaw=-14.0,
        roll=6.0,
        tx=-6.0,
        ty=4.0,
        tz=10.0,
        scale=1.0,
    )
    _, new_bbox, _ = transform_image_and_bbox(img, bbox, params)
    _, _, w, h = new_bbox
    area_ratio = (w * h) / float(img.shape[0] * img.shape[1])

    assert 0.01 < area_ratio < 0.6

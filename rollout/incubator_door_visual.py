"""Rigid parent-feature servo for a recessed incubator door.

The tiny red maker label is attached to the same rigid door as the recessed
handle and remains visible when the handle is partly hidden by the gripper.
HSV thresholds and component-size limits are configuration, while all pixel
coordinates are normalized by the live image shape.
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_feature(image_bgr: np.ndarray, settings: dict) -> tuple[np.ndarray, dict]:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("door feature image must be BGR")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width), dtype=np.uint8)
    for low, high in settings["hsv_ranges"]:
        mask |= cv2.inRange(
            hsv, np.asarray(low, dtype=np.uint8), np.asarray(high, dtype=np.uint8)
        )
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    x_range = settings["normalized_x_range"]
    y_range = settings["normalized_y_range"]
    candidates = []
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        if (
            area >= int(settings["minimum_area_px"])
            and box_width >= int(settings["minimum_width_px"])
            and box_height >= int(settings["minimum_height_px"])
            and float(x_range[0]) * width < x < float(x_range[1]) * width
            and float(y_range[0]) * height < y < float(y_range[1]) * height
        ):
            candidates.append((int(area), index))
    if not candidates:
        raise RuntimeError("rigid red door feature was not detected")
    area, index = max(candidates)
    x, y, box_width, box_height, _ = stats[index]
    feature = np.asarray(
        [
            centers[index, 0] / width,
            centers[index, 1] / height,
            np.log(area / float(width * height)),
        ],
        dtype=float,
    )
    return feature, {
        "box_xywh": [int(x), int(y), int(box_width), int(box_height)],
        "area_px": int(area),
        "feature_uv_log_area": feature.tolist(),
    }


def fit_ridge(features, targets, *, ridge: float) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if features.ndim != 2 or targets.shape != (len(features), 3):
        raise ValueError("feature-model arrays have incompatible shapes")
    return np.linalg.solve(
        features.T @ features + float(ridge) * np.eye(features.shape[1]),
        features.T @ targets,
    )


def predict_local_delta(model: dict, feature) -> np.ndarray:
    goal = np.asarray(model["goal_feature_mean"], dtype=float)
    feature = np.asarray(feature, dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    value = np.r_[goal - feature, 1.0] @ coefficients
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise RuntimeError("door visual model returned an invalid correction")
    return value

"""Conservative visual checks for a wrist camera observing the other arm.

The observer must not infer a target from dark silhouettes: on Pasteur those
are commonly arm shadows on the incubator.  Target acquisition is therefore
positive-evidence only (configured colour components), and motion progress is
checked against the commanded joint direction so an encoder/wrap mismatch is
stopped before a long trajectory is streamed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColourComponent:
    area_px: int
    bbox_xywh: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]


def blue_components(
    bgr: np.ndarray,
    *,
    hue_range: tuple[int, int] = (78, 115),
    minimum_saturation: int = 70,
    minimum_value: int = 35,
    minimum_area_px: int = 80,
) -> tuple[ColourComponent, ...]:
    """Return blue/teal components; dark pixels can never become targets."""

    image = np.asarray(bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("bgr must be a uint8 HxWx3 image")
    low_hue, high_hue = map(int, hue_range)
    if not 0 <= low_hue <= high_hue <= 179:
        raise ValueError("invalid OpenCV hue range")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([low_hue, minimum_saturation, minimum_value], np.uint8),
        np.array([high_hue, 255, 255], np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    result = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if area < int(minimum_area_px):
            continue
        result.append(
            ColourComponent(
                area_px=area,
                bbox_xywh=(x, y, width, height),
                centroid_xy=tuple(map(float, centroids[index])),
            )
        )
    return tuple(sorted(result, key=lambda value: value.area_px, reverse=True))


def component_intersection_fraction(
    component: ColourComponent, mask: np.ndarray
) -> float:
    """Fraction of a component bounding box intersecting a reference mask."""

    reference = np.asarray(mask, dtype=bool)
    if reference.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    x, y, width, height = component.bbox_xywh
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(reference.shape[1], x + width), min(reference.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.count_nonzero(reference[y0:y1, x0:x1])) / float(width * height)


def target_blue_components(
    bgr: np.ndarray,
    *,
    self_mask: np.ndarray,
    maximum_self_intersection: float = 0.10,
    **component_kwargs,
) -> tuple[ColourComponent, ...]:
    """Keep positive blue evidence that is separate from the observer tool."""

    if not 0.0 <= maximum_self_intersection <= 1.0:
        raise ValueError("maximum_self_intersection must be in [0, 1]")
    return tuple(
        component
        for component in blue_components(bgr, **component_kwargs)
        if component_intersection_fraction(component, self_mask)
        <= maximum_self_intersection
    )


@dataclass(frozen=True)
class DirectionCheck:
    accepted: bool
    progress_fraction: float
    reverse_joint_indices: tuple[int, ...]
    maximum_tracking_error_rad: float


def assess_command_direction(
    start_q,
    target_q,
    measured_q,
    *,
    minimum_command_delta_rad: float = 0.04,
    reverse_tolerance_rad: float = 0.015,
    maximum_tracking_error_rad: float = 0.30,
) -> DirectionCheck:
    """Reject measured motion opposite to a material commanded joint delta."""

    start = np.asarray(start_q, dtype=float)
    target = np.asarray(target_q, dtype=float)
    measured = np.asarray(measured_q, dtype=float)
    if any(value.shape != (6,) for value in (start, target, measured)):
        raise ValueError("joint vectors must all have shape (6,)")
    if not all(np.all(np.isfinite(value)) for value in (start, target, measured)):
        raise ValueError("joint vectors must be finite")
    commanded = target - start
    observed = measured - start
    material = np.abs(commanded) >= float(minimum_command_delta_rad)
    reverse = material & (observed * np.sign(commanded) < -float(reverse_tolerance_rad))
    denominator = float(commanded @ commanded)
    progress = float(observed @ commanded / denominator) if denominator > 1e-12 else 1.0
    tracking_error = float(np.max(np.abs(target - measured)))
    accepted = not np.any(reverse) and tracking_error <= float(maximum_tracking_error_rad)
    return DirectionCheck(
        accepted=bool(accepted),
        progress_fraction=progress,
        reverse_joint_indices=tuple(np.flatnonzero(reverse).astype(int).tolist()),
        maximum_tracking_error_rad=tracking_error,
    )


def require_joint_limits(target_q, lower_q, upper_q, *, margin_rad: float = 0.0) -> None:
    """Fail before streaming a target outside the calibrated joint interval."""

    target = np.asarray(target_q, dtype=float)
    lower = np.asarray(lower_q, dtype=float) + float(margin_rad)
    upper = np.asarray(upper_q, dtype=float) - float(margin_rad)
    if any(value.shape != (6,) for value in (target, lower, upper)):
        raise ValueError("joint vectors must all have shape (6,)")
    bad = np.flatnonzero((target < lower) | (target > upper))
    if bad.size:
        raise ValueError(f"joint target outside limits at indices {bad.tolist()}")

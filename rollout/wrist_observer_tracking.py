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


@dataclass(frozen=True)
class LocalImageJacobian:
    """Locally measured pixel motion for each enabled Cartesian control axis."""

    matrix_px_per_unit: np.ndarray
    motion_axes: tuple[str, ...]
    residual_rms_px: float
    condition_number: float


@dataclass(frozen=True)
class ImageServoStep:
    motion_delta: np.ndarray
    predicted_pixel_delta: np.ndarray
    residual_pixel_error: np.ndarray


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


def fit_local_image_jacobian(
    motion_deltas,
    observed_pixel_deltas,
    *,
    motion_axes: tuple[str, ...],
) -> LocalImageJacobian:
    """Fit camera-specific motion signs from small, measured probe moves.

    Rows in ``motion_deltas`` are robot-frame commands and matching rows in
    ``observed_pixel_deltas`` are feature displacement ``(du, dv)``.  No
    camera orientation or left/right sign is assumed, which prevents a camera
    remount or mirrored stream from silently reversing visual pursuit.
    """

    motion = np.asarray(motion_deltas, dtype=float)
    pixels = np.asarray(observed_pixel_deltas, dtype=float)
    if motion.ndim != 2 or pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("probe deltas must be two-dimensional arrays")
    if motion.shape[0] != pixels.shape[0] or motion.shape[1] != len(motion_axes):
        raise ValueError("probe rows and motion-axis count must agree")
    if (
        motion.shape[0] < motion.shape[1]
        or np.linalg.matrix_rank(motion) < motion.shape[1]
    ):
        raise ValueError("probe motions do not independently excite every axis")
    if not np.all(np.isfinite(motion)) or not np.all(np.isfinite(pixels)):
        raise ValueError("probe deltas must be finite")
    if len(set(motion_axes)) != len(motion_axes):
        raise ValueError("motion_axes must be unique")

    coefficients, _, _, singular_values = np.linalg.lstsq(
        motion, pixels, rcond=None
    )
    prediction = motion @ coefficients
    residual = pixels - prediction
    rms = float(np.sqrt(np.mean(np.square(residual))))
    condition = float(singular_values[0] / singular_values[-1])
    return LocalImageJacobian(
        matrix_px_per_unit=coefficients.T.copy(),
        motion_axes=tuple(motion_axes),
        residual_rms_px=rms,
        condition_number=condition,
    )


def image_servo_step(
    pixel_error,
    calibration: LocalImageJacobian,
    *,
    maximum_abs_motion,
    damping: float = 1e-6,
) -> ImageServoStep:
    """Return a bounded motion that reduces a measured two-pixel error.

    All axes are scaled together when one reaches its bound.  This preserves
    the learned image-space direction instead of clipping axes independently.
    """

    error = np.asarray(pixel_error, dtype=float)
    jacobian = np.asarray(calibration.matrix_px_per_unit, dtype=float)
    bounds = np.asarray(maximum_abs_motion, dtype=float)
    axis_count = len(calibration.motion_axes)
    if error.shape != (2,) or jacobian.shape != (2, axis_count):
        raise ValueError("pixel error or calibrated Jacobian has the wrong shape")
    if bounds.shape != (axis_count,) or np.any(bounds <= 0.0):
        raise ValueError("maximum_abs_motion must contain positive per-axis bounds")
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("servo inputs must be finite")
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")

    normal = jacobian.T @ jacobian + float(damping) * np.eye(axis_count)
    motion = np.linalg.solve(normal, jacobian.T @ error)
    scale = float(np.max(np.abs(motion) / bounds))
    if scale > 1.0:
        motion = motion / scale
    predicted = jacobian @ motion
    residual = error - predicted
    if float(predicted @ error) <= 0.0 and np.linalg.norm(error) > 0.0:
        raise RuntimeError("calibrated step does not reduce the requested pixel error")
    return ImageServoStep(
        motion_delta=motion,
        predicted_pixel_delta=predicted,
        residual_pixel_error=residual,
    )


@dataclass(frozen=True)
class DirectionCheck:
    accepted: bool
    progress_fraction: float
    reverse_joint_indices: tuple[int, ...]
    maximum_tracking_error_rad: float


@dataclass(frozen=True)
class DirectionMonitorState:
    accepted: bool
    reverse_joint_indices: tuple[int, ...]
    reverse_counts: tuple[int, ...]


class JointDirectionMonitor:
    """Debounce genuinely reversed motion during a streamed joint path.

    Each observation is compared with the command sent at that sample, not
    with the eventual endpoint. This keeps initial compliance or static
    friction from looking like a controller/encoder sign inversion.
    """

    def __init__(
        self,
        start_q,
        *,
        minimum_command_excursion_rad: float = 0.07,
        reverse_excursion_rad: float = 0.02,
        consecutive_reverse_samples: int = 4,
    ):
        self.start_q = np.asarray(start_q, dtype=float)
        if self.start_q.shape != (6,) or not np.all(np.isfinite(self.start_q)):
            raise ValueError("start_q must contain six finite values")
        self.minimum_command_excursion_rad = float(
            minimum_command_excursion_rad
        )
        self.reverse_excursion_rad = float(reverse_excursion_rad)
        self.consecutive_reverse_samples = int(consecutive_reverse_samples)
        if (
            self.minimum_command_excursion_rad <= 0.0
            or self.reverse_excursion_rad <= 0.0
            or self.consecutive_reverse_samples <= 0
        ):
            raise ValueError("direction-monitor thresholds must be positive")
        self._reverse_counts = np.zeros(6, dtype=int)

    def observe(self, commanded_q, measured_q) -> DirectionMonitorState:
        commanded = np.asarray(commanded_q, dtype=float)
        measured = np.asarray(measured_q, dtype=float)
        if any(value.shape != (6,) for value in (commanded, measured)):
            raise ValueError("joint vectors must all have shape (6,)")
        if not np.all(np.isfinite(commanded)) or not np.all(np.isfinite(measured)):
            raise ValueError("joint vectors must be finite")
        command_excursion = commanded - self.start_q
        measured_excursion = measured - self.start_q
        material = (
            np.abs(command_excursion) >= self.minimum_command_excursion_rad
        )
        reverse = material & (
            measured_excursion * np.sign(command_excursion)
            <= -self.reverse_excursion_rad
        )
        self._reverse_counts = np.where(reverse, self._reverse_counts + 1, 0)
        tripped = self._reverse_counts >= self.consecutive_reverse_samples
        return DirectionMonitorState(
            accepted=not bool(np.any(tripped)),
            reverse_joint_indices=tuple(
                np.flatnonzero(tripped).astype(int).tolist()
            ),
            reverse_counts=tuple(self._reverse_counts.astype(int).tolist()),
        )


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

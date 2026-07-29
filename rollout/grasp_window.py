"""Resolution-independent, tool-relative grasp-window geometry.

The wrist camera is rigidly mounted to the tool.  A light-coloured finger pad
and the adjacent cyan gripper body define a local tool frame in every image.
All target measurements are expressed in that frame, so no live decision
depends on an absolute pixel coordinate or image resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from rollout.realtime_sam_servo import (
    GRIPPER_CYAN_HSV_LOWER,
    GRIPPER_CYAN_HSV_UPPER,
)


GraspWindowMethod = Literal["WHITE_WINDOW", "MASK_GEOMETRY", "HYBRID"]


@dataclass(frozen=True)
class ToolImageFrame:
    origin_px: tuple[float, float]
    forward_xy: tuple[float, float]
    lateral_xy: tuple[float, float]
    scale_px: float
    cyan_pixels: int
    light_pad_pixels: int

    def image_to_tool(self, points_xy) -> np.ndarray:
        points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        origin = np.asarray(self.origin_px, dtype=float)
        basis = np.column_stack(
            (
                np.asarray(self.forward_xy, dtype=float),
                np.asarray(self.lateral_xy, dtype=float),
            )
        )
        return (points - origin) @ basis / self.scale_px

    def tool_to_image(self, points_uv) -> np.ndarray:
        points = np.asarray(points_uv, dtype=float).reshape(-1, 2)
        basis = np.column_stack(
            (
                np.asarray(self.forward_xy, dtype=float),
                np.asarray(self.lateral_xy, dtype=float),
            )
        )
        return np.asarray(self.origin_px, dtype=float) + (
            points * self.scale_px
        ) @ basis.T


@dataclass(frozen=True)
class GraspWindowTemplate:
    method: GraspWindowMethod
    square_center_uv: tuple[float, float]
    square_side_u: float
    reference_center_uv: tuple[float, float]
    reference_quantiles_uv: tuple[float, float, float, float]
    minimum_target_inside_fraction: float = 0.90
    maximum_center_error_scale: float = 0.10
    maximum_quantile_error_scale: float = 0.10

    def __post_init__(self):
        if self.method not in ("WHITE_WINDOW", "MASK_GEOMETRY", "HYBRID"):
            raise ValueError(f"unknown grasp-window method: {self.method}")
        if not np.isfinite(self.square_side_u) or self.square_side_u <= 0:
            raise ValueError("grasp-window side must be positive")
        if not 0 < self.minimum_target_inside_fraction <= 1:
            raise ValueError("inside fraction must be in (0, 1]")
        if (
            self.maximum_center_error_scale < 0
            or self.maximum_quantile_error_scale < 0
        ):
            raise ValueError("normalized error limits must be nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "GraspWindowTemplate":
        return cls(**value)


@dataclass(frozen=True)
class GraspWindowAssessment:
    selected_method: GraspWindowMethod
    allowed_to_close: bool
    white_window_ready: bool
    mask_geometry_ready: bool
    target_inside_fraction: float
    normalized_center_error: float
    normalized_quantile_error: float
    target_center_uv: tuple[float, float]
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _largest_component(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8
    )
    if count <= 1:
        raise ValueError("mask has no connected foreground")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[index, cv2.CC_STAT_AREA]) < int(minimum_pixels):
        raise ValueError("largest mask component is too small")
    return labels == index


def _principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise ValueError("tool mask is too small")
    points = np.column_stack((xs, ys)).astype(float)
    center = points.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(
        points - center, full_matrices=False
    )
    if singular_values[1] <= 0 or singular_values[0] / singular_values[1] < 1.2:
        raise ValueError("tool mask has no stable longitudinal axis")
    axis = axes[0]
    if axis[0] < 0:
        axis = -axis
    lateral = np.array([-axis[1], axis[0]], dtype=float)
    longitudinal = (points - center) @ axis
    transverse = (points - center) @ lateral
    length = float(np.percentile(longitudinal, 99) - np.percentile(longitudinal, 1))
    width = float(np.percentile(transverse, 95) - np.percentile(transverse, 5))
    return center, axis, length, width


def detect_light_pad_tool_frame(
    image_bgr: np.ndarray,
    *,
    gripper_mask: np.ndarray | None = None,
    cyan_hsv_lower=GRIPPER_CYAN_HSV_LOWER,
    cyan_hsv_upper=GRIPPER_CYAN_HSV_UPPER,
    maximum_light_saturation: int = 105,
    minimum_light_value: int = 55,
    maximum_light_value: int = 250,
) -> ToolImageFrame:
    """Detect the light finger pad adjacent to the cyan gripper body.

    Colour thresholds select materials, not image locations.  The only spatial
    search region is derived from the observed cyan component dimensions.
    """

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 BGR image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(
        hsv,
        np.asarray(cyan_hsv_lower, dtype=np.uint8),
        np.asarray(cyan_hsv_upper, dtype=np.uint8),
    ) > 0
    if gripper_mask is not None:
        semantic = np.asarray(gripper_mask, dtype=bool)
        if semantic.shape != cyan.shape:
            raise ValueError("gripper_mask shape does not match image")
        cyan &= semantic
    cyan = _largest_component(cyan, minimum_pixels=100)
    center, axis, length, width = _principal_axis(cyan)
    lateral = np.array([-axis[1], axis[0]], dtype=float)

    yy, xx = np.indices(cyan.shape)
    pixels = np.stack((xx, yy), axis=-1).astype(float)
    longitudinal = (pixels - center) @ axis
    transverse = (pixels - center) @ lateral
    cyan_points = np.column_stack(np.nonzero(cyan)[::-1]).astype(float)
    cyan_longitudinal = (cyan_points - center) @ axis
    cyan_transverse = (cyan_points - center) @ lateral
    lower = float(np.percentile(cyan_longitudinal, 1))
    upper = float(np.percentile(cyan_longitudinal, 99))
    transverse_center = float(np.median(cyan_transverse))

    # The pad touches the distal half of the cyan jaw.  Dilation is scaled by
    # the observed jaw width, avoiding resolution-specific pixel constants.
    dilation = max(3, int(round(0.15 * width)))
    if dilation % 2 == 0:
        dilation += 1
    adjacent = (
        cv2.dilate(
            cyan.astype(np.uint8),
            np.ones((dilation, dilation), dtype=np.uint8),
        )
        > 0
    )
    light = (
        (hsv[:, :, 1] <= int(maximum_light_saturation))
        & (hsv[:, :, 2] >= int(minimum_light_value))
        & (hsv[:, :, 2] <= int(maximum_light_value))
        & (longitudinal >= lower + 0.25 * length)
        & (longitudinal <= upper + 0.15 * length)
        & (np.abs(transverse - transverse_center) <= 0.85 * width)
        & adjacent
        & ~cyan
    )
    if gripper_mask is not None:
        light &= semantic
    light_pixels = int(np.count_nonzero(light))
    if light_pixels < max(50, int(0.01 * np.count_nonzero(cyan))):
        raise ValueError("light gripper pad is not visible")

    pad_points = np.column_stack(np.nonzero(light)[::-1]).astype(float)
    # The pad is at the contact end; its robust centroid is the origin.  The
    # grasp window extends away from the cyan body, opposite its long axis.
    origin = np.median(pad_points, axis=0)
    forward = -axis
    forward_from_pad = (pad_points - origin) @ forward
    lateral_from_pad = (pad_points - origin) @ lateral
    pad_span = max(
        float(np.percentile(forward_from_pad, 95) - np.percentile(forward_from_pad, 5)),
        float(np.percentile(lateral_from_pad, 95) - np.percentile(lateral_from_pad, 5)),
    )
    scale = float(np.clip(pad_span, 0.50 * width, 2.0 * width))
    if not np.isfinite(scale) or scale <= 1:
        raise ValueError("light pad scale is invalid")
    return ToolImageFrame(
        origin_px=(float(origin[0]), float(origin[1])),
        forward_xy=(float(forward[0]), float(forward[1])),
        lateral_xy=(float(lateral[0]), float(lateral[1])),
        scale_px=scale,
        cyan_pixels=int(np.count_nonzero(cyan)),
        light_pad_pixels=light_pixels,
    )


def normalized_pad_target_gap(
    image_bgr: np.ndarray,
    target_mask: np.ndarray,
    *,
    gripper_mask: np.ndarray | None = None,
) -> tuple[float, ToolImageFrame]:
    """Return the shortest pad-to-target image gap in tool-scale units."""

    frame = detect_light_pad_tool_frame(
        image_bgr, gripper_mask=gripper_mask
    )
    image = np.asarray(image_bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(
        hsv,
        np.asarray(GRIPPER_CYAN_HSV_LOWER, dtype=np.uint8),
        np.asarray(GRIPPER_CYAN_HSV_UPPER, dtype=np.uint8),
    ) > 0
    if gripper_mask is not None:
        cyan &= np.asarray(gripper_mask, dtype=bool)
    cyan = _largest_component(cyan, minimum_pixels=100)
    radius = max(3, int(round(0.20 * frame.scale_px)))
    if radius % 2 == 0:
        radius += 1
    adjacent = (
        cv2.dilate(
            cyan.astype(np.uint8),
            np.ones((radius, radius), dtype=np.uint8),
        )
        > 0
    )
    light = (
        (hsv[:, :, 1] <= 105)
        & (hsv[:, :, 2] >= 55)
        & (hsv[:, :, 2] <= 250)
        & adjacent
        & ~cyan
    )
    target = np.asarray(target_mask, dtype=bool)
    if target.shape != light.shape or np.count_nonzero(target) < 100:
        raise ValueError("target mask is invalid")
    if not np.any(light):
        raise ValueError("light pad support is unavailable")
    distance = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
    gap_px = float(np.min(distance[light]))
    return gap_px / frame.scale_px, frame


def target_points_in_tool_frame(
    target_mask: np.ndarray, frame: ToolImageFrame
) -> np.ndarray:
    ys, xs = np.nonzero(np.asarray(target_mask, dtype=bool))
    if len(xs) < 100:
        raise ValueError("target mask is too small")
    return frame.image_to_tool(np.column_stack((xs, ys)))


def calibrate_grasp_window(
    reference_image_bgr: np.ndarray,
    reference_target_mask: np.ndarray,
    *,
    method: GraspWindowMethod = "HYBRID",
    reference_gripper_mask: np.ndarray | None = None,
) -> tuple[GraspWindowTemplate, ToolImageFrame]:
    frame = detect_light_pad_tool_frame(
        reference_image_bgr, gripper_mask=reference_gripper_mask
    )
    points = target_points_in_tool_frame(reference_target_mask, frame)
    q_low = np.percentile(points, 5, axis=0)
    q_high = np.percentile(points, 95, axis=0)
    side = float(np.max(q_high - q_low) * 1.10)
    if not np.isfinite(side) or side <= 0:
        raise ValueError("reference target has invalid tool-frame extent")
    center = 0.5 * (q_low + q_high)
    template = GraspWindowTemplate(
        method=method,
        square_center_uv=(float(center[0]), float(center[1])),
        square_side_u=side,
        reference_center_uv=(
            float(np.mean(points[:, 0])),
            float(np.mean(points[:, 1])),
        ),
        reference_quantiles_uv=(
            float(q_low[0]),
            float(q_high[0]),
            float(q_low[1]),
            float(q_high[1]),
        ),
    )
    return template, frame


def assess_grasp_window(
    image_bgr: np.ndarray,
    target_mask: np.ndarray,
    template: GraspWindowTemplate,
    *,
    gripper_mask: np.ndarray | None = None,
    method: GraspWindowMethod | None = None,
) -> tuple[GraspWindowAssessment, ToolImageFrame]:
    selected = template.method if method is None else method
    frame = detect_light_pad_tool_frame(image_bgr, gripper_mask=gripper_mask)
    points = target_points_in_tool_frame(target_mask, frame)
    center = np.mean(points, axis=0)
    q_low = np.percentile(points, 5, axis=0)
    q_high = np.percentile(points, 95, axis=0)
    half = 0.5 * template.square_side_u
    square_center = np.asarray(template.square_center_uv)
    inside = np.all(np.abs(points - square_center) <= half, axis=1)
    inside_fraction = float(np.mean(inside))
    center_error = float(
        np.linalg.norm(center - np.asarray(template.reference_center_uv))
        / template.square_side_u
    )
    reference_q = np.asarray(template.reference_quantiles_uv)
    live_q = np.asarray((q_low[0], q_high[0], q_low[1], q_high[1]))
    quantile_error = float(
        np.max(np.abs(live_q - reference_q)) / template.square_side_u
    )
    white_ready = (
        inside_fraction >= template.minimum_target_inside_fraction
        and center_error <= template.maximum_center_error_scale
    )
    geometry_ready = (
        center_error <= template.maximum_center_error_scale
        and quantile_error <= template.maximum_quantile_error_scale
    )
    allowed = {
        "WHITE_WINDOW": white_ready,
        "MASK_GEOMETRY": geometry_ready,
        "HYBRID": white_ready and geometry_ready,
    }[selected]
    reasons = []
    if not white_ready:
        reasons.append("target_outside_white_window")
    if not geometry_ready:
        reasons.append("target_geometry_mismatch")
    return (
        GraspWindowAssessment(
            selected_method=selected,
            allowed_to_close=bool(allowed),
            white_window_ready=bool(white_ready),
            mask_geometry_ready=bool(geometry_ready),
            target_inside_fraction=inside_fraction,
            normalized_center_error=center_error,
            normalized_quantile_error=quantile_error,
            target_center_uv=(float(center[0]), float(center[1])),
            failure_reasons=tuple(reasons),
        ),
        frame,
    )


def grasp_window_polygon_px(
    template: GraspWindowTemplate, frame: ToolImageFrame
) -> np.ndarray:
    half = 0.5 * template.square_side_u
    center = np.asarray(template.square_center_uv)
    corners = np.asarray(
        [
            center + (-half, -half),
            center + (+half, -half),
            center + (+half, +half),
            center + (-half, +half),
        ]
    )
    return frame.tool_to_image(corners)


def render_grasp_window(
    image_bgr: np.ndarray,
    target_mask: np.ndarray,
    template: GraspWindowTemplate,
    assessment: GraspWindowAssessment,
    frame: ToolImageFrame,
) -> np.ndarray:
    out = np.asarray(image_bgr).copy()
    mask = np.asarray(target_mask, dtype=bool)
    tint = np.zeros_like(out)
    tint[:] = (0, 180, 255)
    out[mask] = cv2.addWeighted(out[mask], 0.45, tint[mask], 0.55, 0)
    polygon = np.rint(grasp_window_polygon_px(template, frame)).astype(np.int32)
    color = (0, 220, 0) if assessment.allowed_to_close else (0, 0, 255)
    cv2.polylines(out, [polygon], True, (255, 255, 255), 5)
    cv2.polylines(out, [polygon], True, color, 2)
    origin = tuple(np.rint(frame.origin_px).astype(int))
    cv2.drawMarker(out, origin, (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
    label = (
        f"{assessment.selected_method} close={assessment.allowed_to_close} "
        f"inside={assessment.target_inside_fraction:.2f} "
        f"center={assessment.normalized_center_error:.2f} "
        f"shape={assessment.normalized_quantile_error:.2f}"
    )
    cv2.putText(
        out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3
    )
    cv2.putText(
        out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1
    )
    return out

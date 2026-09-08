"""SAM-free right-wrist checks for a blue-marked transparent lid."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from rollout.grasp_window import ToolImageFrame, detect_light_pad_tool_frame


@dataclass(frozen=True)
class ToolRelativeMarkerTemplate:
    marker_center_uv: tuple[float, float]
    maximum_error_scale: float = 0.35


@dataclass(frozen=True)
class MarkerAssessment:
    visible: bool
    aligned: bool
    marker_center_uv: tuple[float, float] | None
    error_uv: tuple[float, float] | None
    error_scale: float | None
    reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _blue_components(image_bgr, hsv_low=(100, 70, 35), hsv_high=(125, 255, 255)):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(hsv_low, np.uint8), np.asarray(hsv_high, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    result = []
    image_area = float(mask.size)
    for label in range(1, count):
        fraction = float(stats[label, cv2.CC_STAT_AREA]) / image_area
        if 2e-5 <= fraction <= 0.03:
            result.append((np.asarray(centers[label]), fraction, labels == label))
    return result


def learn_marker_template(
    reference_bgr: np.ndarray,
    *,
    tool_frame: ToolImageFrame | None = None,
    marker_hint_px=None,
) -> ToolRelativeMarkerTemplate:
    tool = tool_frame or detect_light_pad_tool_frame(reference_bgr)
    candidates = _blue_components(reference_bgr)
    if not candidates:
        raise ValueError("reference contains no usable blue marker")
    if marker_hint_px is None:
        # A template is built offline from an operator-approved success image;
        # area can rank components here, but never identifies the live target.
        center = max(candidates, key=lambda item: item[1])[0]
    else:
        hint = np.asarray(marker_hint_px, dtype=float)
        center = min(candidates, key=lambda item: np.linalg.norm(item[0] - hint))[0]
    local = tool.image_to_tool(center)[0]
    return ToolRelativeMarkerTemplate((float(local[0]), float(local[1])))


def assess_marker_alignment(
    image_bgr: np.ndarray,
    template: ToolRelativeMarkerTemplate,
    *,
    previous_marker_uv=None,
    tool_frame: ToolImageFrame | None = None,
) -> MarkerAssessment:
    """Track the marker nearest its approved tool-relative identity."""

    try:
        tool = tool_frame or detect_light_pad_tool_frame(image_bgr)
    except ValueError as error:
        return MarkerAssessment(False, False, None, None, None, str(error))
    candidates = _blue_components(image_bgr)
    if not candidates:
        return MarkerAssessment(False, False, None, None, None, "blue marker not visible")
    local = np.vstack([tool.image_to_tool(item[0])[0] for item in candidates])
    expected = np.asarray(template.marker_center_uv, dtype=float)
    if previous_marker_uv is None:
        index = int(np.argmin(np.linalg.norm(local - expected, axis=1)))
    else:
        previous = np.asarray(previous_marker_uv, dtype=float)
        index = int(np.argmin(np.linalg.norm(local - previous, axis=1)))
    error = local[index] - expected
    magnitude = float(np.linalg.norm(error))
    return MarkerAssessment(
        True,
        magnitude <= template.maximum_error_scale,
        (float(local[index, 0]), float(local[index, 1])),
        (float(error[0]), float(error[1])),
        magnitude,
        None if magnitude <= template.maximum_error_scale else "marker outside grasp window",
    )


@dataclass(frozen=True)
class PrecloseEdgeAssessment:
    accepted: bool
    bilateral_edge_fraction: float
    center_gap_fraction: float
    reason: str | None


def assess_preclose_edges(
    image_bgr: np.ndarray,
    tool: ToolImageFrame,
    *,
    minimum_bilateral_edge_fraction: float = 0.018,
    maximum_center_gap_fraction: float = 0.70,
) -> PrecloseEdgeAssessment:
    """Require lid-rim edges on both sides of the tool-forward centerline."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 100) > 0
    ys, xs = np.nonzero(edges)
    if len(xs) < 20:
        return PrecloseEdgeAssessment(False, 0.0, float("inf"), "too few rim edges")
    local = tool.image_to_tool(np.column_stack((xs, ys)))
    # Forward strip immediately beyond the pad.  Both lateral halves must have
    # support: merely putting the lid on top of one finger does not pass.
    strip = local[(local[:, 0] >= 0.0) & (local[:, 0] <= 4.0) & (np.abs(local[:, 1]) <= 2.5)]
    if len(strip) < 20:
        return PrecloseEdgeAssessment(False, 0.0, float("inf"), "rim absent from finger corridor")
    left = strip[strip[:, 1] < 0]
    right = strip[strip[:, 1] >= 0]
    denominator = max(1.0, 4.0 * 5.0 * tool.scale_px)
    bilateral = float(min(len(left), len(right)) / denominator)
    left_gap = abs(float(np.percentile(left[:, 1], 75))) if len(left) else float("inf")
    right_gap = abs(float(np.percentile(right[:, 1], 25))) if len(right) else float("inf")
    center_gap = abs(left_gap - right_gap) / 2.5
    accepted = bilateral >= minimum_bilateral_edge_fraction and center_gap <= maximum_center_gap_fraction
    reason = None if accepted else "rim is not bilaterally between fingers"
    return PrecloseEdgeAssessment(accepted, bilateral, center_gap, reason)

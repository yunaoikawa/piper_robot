"""Semantic adapter for a white cap supported by a coloured bottle neck."""

from __future__ import annotations

import cv2
import numpy as np

from rollout.grasp_window import ToolImageFrame
from rollout.realtime_sam_servo import (
    GRIPPER_CYAN_HSV_LOWER,
    GRIPPER_CYAN_HSV_UPPER,
)
from rollout.target_adapter import TargetObservation


def _components(mask: np.ndarray):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8
    )
    for index in range(1, count):
        yield index, labels, stats[index], centers[index]


def detect_media_cap(
    image_bgr: np.ndarray,
    *,
    identity_anchor_px=None,
    maximum_anchor_displacement_diagonal_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return cap mask and centers without assuming a camera or pixel ROI."""

    image = np.asarray(image_bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    neck_material = (
        (((hue <= 15) | (hue >= 150)) & (saturation >= 45) & (value >= 65))
        .astype(np.uint8)
    )
    neck_material = cv2.morphologyEx(
        neck_material, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    white_material = ((saturation <= 90) & (value >= 185)).astype(np.uint8)
    height, width = neck_material.shape
    image_area = float(height * width)
    candidates = []
    for _, _, neck_stats, neck_center in _components(neck_material):
        x, y, w, h, area = (int(item) for item in neck_stats)
        if area < max(20, int(2e-5 * image_area)):
            continue
        scale = float(max(w, h))
        if scale < 3:
            continue
        cx = float(neck_center[0])
        x0 = max(0, int(np.floor(cx - 1.65 * scale)))
        x1 = min(width, int(np.ceil(cx + 1.65 * scale)))
        y0 = max(0, int(np.floor(y - 2.2 * scale)))
        y1 = min(height, int(np.ceil(y + 0.45 * scale)))
        roi = np.zeros_like(white_material)
        roi[y0:y1, x0:x1] = white_material[y0:y1, x0:x1]
        roi = cv2.morphologyEx(
            roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        for label, labels, cap_stats, cap_center in _components(roi):
            _, _, cap_w, cap_h, cap_area = (int(item) for item in cap_stats)
            if cap_area < max(40, int(0.12 * scale * scale)):
                continue
            if cap_center[1] >= neck_center[1]:
                continue
            horizontal_error = abs(float(cap_center[0]) - cx) / scale
            vertical_gap = (
                float(neck_center[1]) - float(cap_center[1])
            ) / scale
            aspect = float(cap_w) / max(float(cap_h), 1.0)
            area_scale = float(cap_area) / (scale * scale)
            if (
                horizontal_error > 1.25
                or not 0.15 <= vertical_gap <= 2.8
                or not 0.35 <= aspect <= 2.8
                or not 0.12 <= area_scale <= 12.0
            ):
                continue
            score = (
                horizontal_error
                + 0.20 * abs(vertical_gap - 1.0)
                + 0.08 * abs(np.log(max(aspect, 1e-6)))
                + 0.02 * abs(np.log(max(area_scale, 1e-6)))
            )
            candidates.append(
                (
                    score,
                    labels == label,
                    cap_center.copy(),
                    int(cap_area),
                    neck_center.copy(),
                )
            )
    anchor = None
    selection_method = "white_above_coloured_neck"
    if identity_anchor_px is not None:
        anchor = np.asarray(identity_anchor_px, dtype=float)
        if anchor.shape != (2,) or not np.all(np.isfinite(anchor)):
            raise ValueError("media-cap identity anchor must contain finite xy")
        diagonal = float(np.hypot(width, height))
        maximum_distance = (
            float(maximum_anchor_displacement_diagonal_fraction) * diagonal
        )
        anchored = [
            item
            for item in candidates
            if float(np.linalg.norm(np.asarray(item[2]) - anchor))
            <= maximum_distance
        ]
        # A descending open jaw can hide the coloured neck while leaving most
        # of the white cap visible.  Always form a local appearance track, not
        # only after the semantic detector fails: an overlapping white gripper
        # mount can otherwise satisfy the semantic geometry by accident.
        local_radius = min(maximum_distance, 0.06 * diagonal)
        local_candidates = []
        for label, labels, stats, center in _components(white_material):
            _, _, component_w, component_h, component_area = (
                int(item) for item in stats
            )
            if component_area < max(30, int(2e-5 * image_area)):
                continue
            if component_w < 5 or component_h < 5:
                continue
            aspect = float(component_w) / max(float(component_h), 1.0)
            if not 0.35 <= aspect <= 2.8:
                continue
            distance = float(np.linalg.norm(np.asarray(center) - anchor))
            if distance > local_radius:
                continue
            local_candidates.append(
                (
                    distance / max(local_radius, 1e-6),
                    labels == label,
                    center.copy(),
                    int(component_area),
                    np.asarray([np.nan, np.nan]),
                )
            )
        semantic_distance = min(
            (
                float(np.linalg.norm(np.asarray(item[2]) - anchor))
                for item in anchored
            ),
            default=np.inf,
        )
        local_distance = min(
            (
                float(np.linalg.norm(np.asarray(item[2]) - anchor))
                for item in local_candidates
            ),
            default=np.inf,
        )
        if local_distance <= semantic_distance:
            candidates = local_candidates
            selection_method = "tap_local_white_component_under_occlusion"
        elif anchored:
            candidates = anchored
            selection_method = "tap_anchored_white_above_coloured_neck"
        else:
            raise RuntimeError(
                "no semantic or locally tracked media-cap candidate "
                "matches the tap identity"
            )
    elif not candidates:
        raise RuntimeError(
            "no compact white cap was found immediately above a coloured neck"
        )
    score, mask, cap_center, pixels, neck_center = min(
        candidates,
        key=lambda item: (
            float(np.linalg.norm(np.asarray(item[2]) - anchor))
            if anchor is not None
            else item[0],
            item[0],
        ),
    )
    return np.asarray(mask, dtype=bool), np.asarray(cap_center), {
        "candidate_count": len(candidates),
        "score": float(score),
        "cap_center_px": [float(v) for v in cap_center],
        "neck_center_px": [float(v) for v in neck_center],
        "component_pixels": int(pixels),
        "identity_anchor_px": None if anchor is None else anchor.tolist(),
        "selection_method": selection_method,
    }


class MediaCapTargetAdapter:
    name = "culture_media_cap"
    geometry = "vertical_cylinder_side_pinch"

    def __init__(
        self,
        *,
        identity_anchor_uv=None,
        maximum_anchor_displacement_diagonal_fraction: float = 0.25,
        update_identity_anchor: bool = True,
    ):
        self.identity_anchor_uv = (
            None
            if identity_anchor_uv is None
            else np.asarray(identity_anchor_uv, dtype=float)
        )
        if self.identity_anchor_uv is not None and self.identity_anchor_uv.shape != (2,):
            raise ValueError("normalized media-cap identity anchor must contain xy")
        self.maximum_anchor_displacement_diagonal_fraction = float(
            maximum_anchor_displacement_diagonal_fraction
        )
        self.update_identity_anchor = bool(update_identity_anchor)

    def observe(
        self, image_bgr: np.ndarray, tool_frame: ToolImageFrame
    ) -> TargetObservation:
        anchor = None
        if self.identity_anchor_uv is not None:
            anchor = self.identity_anchor_uv * np.asarray(
                [image_bgr.shape[1], image_bgr.shape[0]], dtype=float
            )
        mask, center, diagnostics = detect_media_cap(
            image_bgr,
            identity_anchor_px=anchor,
            maximum_anchor_displacement_diagonal_fraction=(
                self.maximum_anchor_displacement_diagonal_fraction
            ),
        )
        if self.identity_anchor_uv is not None and self.update_identity_anchor:
            self.identity_anchor_uv = np.asarray(
                [
                    center[0] / image_bgr.shape[1],
                    center[1] / image_bgr.shape[0],
                ],
                dtype=float,
            )
            diagnostics["updated_identity_anchor_uv"] = (
                self.identity_anchor_uv.tolist()
            )
        center_uv = tool_frame.image_to_tool([center])[0]
        pixels = int(np.count_nonzero(mask))
        diameter_px = 2.0 * np.sqrt(pixels / np.pi)
        return TargetObservation(
            center_px=(float(center[0]), float(center[1])),
            center_uv=(float(center_uv[0]), float(center_uv[1])),
            component_pixels=pixels,
            equivalent_diameter_tool_units=float(
                diameter_px / tool_frame.scale_px
            ),
            mask=mask,
            diagnostics=diagnostics,
        )


def detect_open_jaw_center_head(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Locate the midpoint of the two open cyan jaws in a fixed-head view.

    The two largest elongated cyan components are used; absolute image side or
    a task-specific ROI is deliberately not part of the selection.
    """

    image = np.asarray(image_bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(
        hsv,
        np.asarray(GRIPPER_CYAN_HSV_LOWER, dtype=np.uint8),
        np.asarray(GRIPPER_CYAN_HSV_UPPER, dtype=np.uint8),
    )
    image_area = float(cyan.shape[0] * cyan.shape[1])
    candidates = []
    for label, labels, stats, center in _components(cyan):
        x, y, width, height, area = (int(item) for item in stats)
        if area < max(100, int(5e-5 * image_area)):
            continue
        ys, xs = np.nonzero(labels == label)
        points = np.column_stack((xs, ys)).astype(float)
        if len(points) < 20:
            continue
        _, singular, _ = np.linalg.svd(
            points - points.mean(axis=0), full_matrices=False
        )
        elongation = float(singular[0] / max(singular[1], 1e-6))
        if elongation < 1.35:
            continue
        candidates.append(
            {
                "mask": labels == label,
                "center": np.asarray(center, dtype=float),
                "area": area,
                "elongation": elongation,
                "bbox": [x, y, width, height],
            }
        )
    if len(candidates) < 2:
        raise RuntimeError("two open cyan jaw components are not visible")
    jaws = sorted(candidates, key=lambda item: item["area"], reverse=True)[:2]
    centers = np.asarray([item["center"] for item in jaws], dtype=float)
    jaw_center = np.mean(centers, axis=0)
    mask = np.asarray(jaws[0]["mask"] | jaws[1]["mask"], dtype=bool)
    jaw_span_px = float(np.linalg.norm(centers[0] - centers[1]))
    return jaw_center, mask, {
        "jaw_centers_px": centers.tolist(),
        "jaw_center_px": jaw_center.tolist(),
        "jaw_span_px": jaw_span_px,
        "component_pixels": [int(item["area"]) for item in jaws],
        "component_elongation": [float(item["elongation"]) for item in jaws],
    }


def fixed_target_in_jaw_segment(
    target_px, jaw_centers_px, *, maximum_perpendicular_span_fraction: float
) -> dict:
    """Check that a fixed-camera target lies between the two jaw centers."""

    target = np.asarray(target_px, dtype=float)
    jaws = np.asarray(jaw_centers_px, dtype=float)
    if target.shape != (2,) or jaws.shape != (2, 2):
        raise ValueError("target and two jaw centers must be 2D points")
    axis = jaws[1] - jaws[0]
    span = float(np.linalg.norm(axis))
    if span <= 1.0:
        raise ValueError("jaw centers are coincident")
    fraction = float(np.dot(target - jaws[0], axis) / np.dot(axis, axis))
    closest = jaws[0] + np.clip(fraction, 0.0, 1.0) * axis
    perpendicular = float(np.linalg.norm(target - closest))
    return {
        "accepted": bool(
            0.0 <= fraction <= 1.0
            and perpendicular
            <= float(maximum_perpendicular_span_fraction) * span
        ),
        "along_segment_fraction": fraction,
        "perpendicular_distance_px": perpendicular,
        "perpendicular_span_fraction": perpendicular / span,
        "jaw_span_px": span,
    }


def detect_coloured_support_anchor(
    image_bgr: np.ndarray, *, target_anchor_px
) -> tuple[np.ndarray, dict]:
    """Track the lower coloured bottle body beneath a fixed-camera target."""

    image = np.asarray(image_bgr)
    anchor = np.asarray(target_anchor_px, dtype=float)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    material = (
        (((hue <= 15) | (hue >= 150)) & (saturation >= 45) & (value >= 65))
        .astype(np.uint8)
    )
    material = cv2.morphologyEx(
        material, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    candidates = []
    image_area = float(image.shape[0] * image.shape[1])
    for _, _, stats, center in _components(material):
        _, _, width, height, area = (int(item) for item in stats)
        if area < max(80, int(4e-5 * image_area)):
            continue
        if center[1] <= anchor[1]:
            continue
        if abs(float(center[0]) - float(anchor[0])) > 0.15 * image.shape[1]:
            continue
        candidates.append((float(center[1]), area, center.copy(), stats.copy()))
    if not candidates:
        raise RuntimeError("no coloured support body was found below the target")
    _, area, center, stats = max(candidates, key=lambda item: (item[0], item[1]))
    diagonal = float(np.hypot(float(stats[2]), float(stats[3])))
    return np.asarray(center, dtype=float), {
        "center_px": [float(item) for item in center],
        "component_pixels": int(area),
        "component_diagonal_px": diagonal,
        "bbox_xywh": [int(item) for item in stats[:4]],
    }

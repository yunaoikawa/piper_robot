"""Generic evidence and run bookkeeping for thin-object edge grasps.

The physical executor deliberately lives outside this module.  Everything in
this file is deterministic and replayable from images and telemetry so a run
can be audited without Codex or a connected robot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from rollout.grasp_window import (
    GraspWindowAssessment,
    GraspWindowTemplate,
    ToolImageFrame,
    assess_grasp_window,
    detect_light_pad_tool_frame,
)
from rollout.sam_segmentation import detect_blue_cross_centers


@dataclass(frozen=True)
class ClosureCalibration:
    """Calibration-relative gripper obstruction classifier.

    No target-specific aperture threshold is encoded in the controller.  A
    profile supplies empty and known non-empty samples, and the observation is
    classified by its nearest calibrated population.
    """

    empty_reference_ratios: tuple[float, ...]
    nonempty_reference_ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("empty", self.empty_reference_ratios),
            ("nonempty", self.nonempty_reference_ratios),
        ):
            array = np.asarray(values, dtype=float)
            if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} closure references must be finite and non-empty")
            if np.any((array < 0.0) | (array > 1.0)):
                raise ValueError(f"{name} closure references must lie in [0, 1]")

    def classify(self, measured_open_ratio: float) -> dict:
        measured = float(measured_open_ratio)
        if not math.isfinite(measured) or not 0.0 <= measured <= 1.0:
            raise ValueError("measured gripper ratio must lie in [0, 1]")
        empty_distance = min(
            abs(measured - reference) for reference in self.empty_reference_ratios
        )
        nonempty_distance = min(
            abs(measured - reference)
            for reference in self.nonempty_reference_ratios
        )
        return {
            "measured_open_ratio": measured,
            "empty_distance": float(empty_distance),
            "nonempty_distance": float(nonempty_distance),
            "nonempty": bool(nonempty_distance < empty_distance),
            "method": "nearest_calibrated_population",
        }


@dataclass(frozen=True)
class TargetObservation:
    center_px: tuple[float, float]
    center_uv: tuple[float, float]
    component_pixels: int
    component_area_per_tool_scale_sq: float
    component_touches_border: bool
    candidate_count: int
    tool_frame: ToolImageFrame
    grasp_window: GraspWindowAssessment
    source: str = "blue_marker"
    tracking_inlier_fraction: float | None = None
    tool_frame_source: str = "light_pad_nominal"
    marker_cross_shaped: bool | None = None
    component_interior_extent_px: float | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["tool_frame"] = asdict(self.tool_frame)
        result["grasp_window"] = self.grasp_window.to_dict()
        return result


@dataclass(frozen=True)
class RelocatedMarkerObservation:
    """A tapped marker reacquired after the movable object has translated."""

    center_px: tuple[float, float]
    center_uv: tuple[float, float]
    component_pixels: int
    area_scale_from_reference: float
    displacement_diagonal_fraction: float
    excluded_anchor_count: int
    candidate_count: int
    anchor_overlap_fallback_used: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def select_local_blue_evidence_marker(
    image_bgr: np.ndarray,
    *,
    reference_target_center_px: Sequence[float],
    reference_target_component_pixels: int,
    maximum_target_displacement_diagonal_fraction: float = 0.04,
    minimum_component_area_scale: float = 0.10,
    maximum_component_area_scale: float = 4.0,
) -> RelocatedMarkerObservation:
    """Reacquire a distant blue fiducial after its cross shape disappears.

    This is a *local continuity* fallback, not global blue-object detection.
    Record3D exposure can turn a small printed cross into several translucent
    round components.  Inside a tight envelope around the last direct target,
    rank the components by integrated blue-over-red/green evidence with a
    smooth displacement penalty.  Large cyan robot parts remain outside the
    envelope or the calibrated component-area class.
    """

    image = np.asarray(image_bgr)
    height, width = image.shape[:2]
    diagonal = float(np.hypot(width, height))
    reference_center = np.asarray(reference_target_center_px, dtype=float)
    reference_pixels = int(reference_target_component_pixels)
    if image.ndim != 3 or image.shape[2] != 3 or diagonal <= 0.0:
        raise ValueError("target image must be a non-empty BGR image")
    if reference_center.shape != (2,) or not np.all(np.isfinite(reference_center)):
        raise ValueError("reference target center must contain two finite values")
    if reference_pixels <= 0:
        raise ValueError("reference target component must contain pixels")
    maximum_distance = float(maximum_target_displacement_diagonal_fraction) * diagonal
    if maximum_distance <= 0.0:
        raise ValueError("local target displacement envelope must be positive")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    permissive = cv2.inRange(
        hsv,
        np.asarray([95, 20, 40], dtype=np.uint8),
        np.asarray([130, 255, 255], dtype=np.uint8),
    )
    permissive = cv2.morphologyEx(
        permissive, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, centers = cv2.connectedComponentsWithStats(permissive)
    candidates = []
    blue = image[:, :, 0].astype(np.int16)
    nonblue = np.maximum(image[:, :, 1], image[:, :, 2]).astype(np.int16)
    dominance = np.maximum(blue - nonblue, 0)
    for label in range(1, count):
        pixels = int(stats[label, cv2.CC_STAT_AREA])
        scale = pixels / float(reference_pixels)
        if not float(minimum_component_area_scale) <= scale <= float(
            maximum_component_area_scale
        ):
            continue
        center = np.asarray(centers[label], dtype=float)
        distance = float(np.linalg.norm(center - reference_center))
        if distance > maximum_distance:
            continue
        component = labels == label
        evidence = float(np.sum(dominance[component]))
        if evidence <= 0.0:
            continue
        normalized = distance / diagonal
        # The exponential is only a local continuity prior.  Integrated
        # chromatic evidence still lets a moved direct mark beat a faint
        # refraction at the old pixel.
        continuity_weight = math.exp(-0.5 * (distance / maximum_distance) ** 2)
        score = evidence * continuity_weight
        candidates.append((score, evidence, -distance, center, pixels, scale, normalized))
    if not candidates:
        raise ValueError("no local blue-evidence target candidate is visible")
    _, _, _, center, pixels, scale, normalized = max(
        candidates, key=lambda item: item[:3]
    )
    return RelocatedMarkerObservation(
        center_px=(float(center[0]), float(center[1])),
        center_uv=(float(center[0] / width), float(center[1] / height)),
        component_pixels=int(pixels),
        area_scale_from_reference=float(scale),
        displacement_diagonal_fraction=float(normalized),
        excluded_anchor_count=0,
        candidate_count=len(candidates),
        anchor_overlap_fallback_used=False,
    )


def observe_local_blue_evidence_target(
    image_bgr: np.ndarray,
    previous: TargetObservation,
    template: GraspWindowTemplate,
    *,
    maximum_target_displacement_diagonal_fraction: float = 0.025,
    minimum_component_area_scale: float = 0.10,
    maximum_component_area_scale: float = 4.0,
) -> TargetObservation:
    """Continue a directly established marker identity through rim merging.

    This adapter is intentionally local: it cannot initialize identity.  It
    uses the previous direct/continued target as the centre and area anchor,
    then assesses the selected permissive-blue component in the same rigid
    tool frame.  This prevents a transparent rim or cyan gripper elsewhere in
    the image from replacing the target when the printed cross briefly stops
    satisfying the direct cross-shape detector.
    """

    selected = select_local_blue_evidence_marker(
        image_bgr,
        reference_target_center_px=previous.center_px,
        reference_target_component_pixels=max(previous.component_pixels, 1),
        maximum_target_displacement_diagonal_fraction=(
            maximum_target_displacement_diagonal_fraction
        ),
        minimum_component_area_scale=minimum_component_area_scale,
        maximum_component_area_scale=maximum_component_area_scale,
    )
    image = np.asarray(image_bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    permissive = cv2.inRange(
        hsv,
        np.asarray([95, 20, 40], dtype=np.uint8),
        np.asarray([130, 255, 255], dtype=np.uint8),
    )
    permissive = cv2.morphologyEx(
        permissive, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    count, labels, _, centers = cv2.connectedComponentsWithStats(permissive)
    if count <= 1:
        raise ValueError("continued local blue target has no component")
    center = np.asarray(selected.center_px, dtype=float)
    label = int(np.argmin(np.linalg.norm(centers[1:] - center, axis=1))) + 1
    mask = labels == label
    height, width = image.shape[:2]
    ys, xs = np.nonzero(mask)
    touches_border = bool(
        len(xs)
        and (
            int(xs.min()) <= 1
            or int(ys.min()) <= 1
            or int(xs.max()) >= width - 2
            or int(ys.max()) >= height - 2
        )
    )
    border_extents = []
    if len(xs):
        if int(xs.min()) <= 1:
            border_extents.append(float(xs.max() + 1))
        if int(xs.max()) >= width - 2:
            border_extents.append(float(width - xs.min()))
        if int(ys.min()) <= 1:
            border_extents.append(float(ys.max() + 1))
        if int(ys.max()) >= height - 2:
            border_extents.append(float(height - ys.min()))
    assessment, frame = assess_grasp_window(
        image,
        mask,
        template,
        method="MASK_GEOMETRY",
        tool_frame=previous.tool_frame,
    )
    center_uv = frame.image_to_tool([center])[0]
    return TargetObservation(
        center_px=(float(center[0]), float(center[1])),
        center_uv=(float(center_uv[0]), float(center_uv[1])),
        component_pixels=int(selected.component_pixels),
        component_area_per_tool_scale_sq=float(
            selected.component_pixels / frame.scale_px**2
        ),
        component_touches_border=touches_border,
        candidate_count=int(selected.candidate_count),
        tool_frame=frame,
        grasp_window=assessment,
        source="local_blue_evidence_continuity",
        tracking_inlier_fraction=None,
        tool_frame_source="rigid_previous_tool_frame",
        marker_cross_shaped=False,
        component_interior_extent_px=(
            min(border_extents) if border_extents else None
        ),
    )


def _blue_component_mask(image_bgr: np.ndarray, center_xy: Sequence[float]) -> np.ndarray:
    hsv = cv2.cvtColor(np.asarray(image_bgr), cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(
        hsv,
        np.asarray([95, 80, 50], dtype=np.uint8),
        np.asarray([130, 255, 255], dtype=np.uint8),
    )
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, _, centers = cv2.connectedComponentsWithStats(blue)
    target = np.asarray(center_xy, dtype=float).reshape(2)
    if count <= 1:
        raise ValueError("selected blue target center has no component")
    distances = np.linalg.norm(centers[1:] - target, axis=1)
    label = int(np.argmin(distances)) + 1
    if float(distances[label - 1]) > 3.0:
        raise ValueError("selected blue target center has no component")
    return labels == label


def select_relocated_target_marker(
    image_bgr: np.ndarray,
    *,
    homography_reference_to_current: np.ndarray,
    reference_target_center_px: Sequence[float],
    reference_target_component_pixels: int,
    stationary_anchor_centers_px: Sequence[Sequence[float]] = (),
    maximum_anchor_displacement_diagonal_fraction: float = 0.02,
    maximum_target_displacement_diagonal_fraction: float = 0.12,
    minimum_component_area_scale: float = 0.35,
    maximum_component_area_scale: float = 4.0,
) -> RelocatedMarkerObservation:
    """Reacquire a tapped marker without assuming that it stayed at one pixel.

    Fixed background registration predicts where stationary distractors should
    remain.  Those anchors are removed first.  The selected target must then
    retain the tapped component's scale class and remain within a normalized
    motion envelope.  Candidate area only gates identity; it is never ranked
    by largest-area, which previously selected blue robot hardware.
    """

    image = np.asarray(image_bgr)
    height, width = image.shape[:2]
    diagonal = float(np.hypot(width, height))
    if diagonal <= 0.0:
        raise ValueError("target image is empty")
    reference_pixels = int(reference_target_component_pixels)
    if reference_pixels <= 0:
        raise ValueError("reference target component must contain pixels")
    homography = np.asarray(homography_reference_to_current, dtype=float)
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        raise ValueError("fixed-head homography must be finite 3x3")

    centers = detect_blue_cross_centers(image)
    if not centers:
        raise ValueError("no cross-shaped target candidates are visible")
    candidates = []
    for center in centers:
        mask = _blue_component_mask(image, center)
        pixels = int(np.count_nonzero(mask))
        candidates.append(
            {
                "center": np.asarray(center, dtype=float),
                "pixels": pixels,
                "area_scale": pixels / float(reference_pixels),
            }
        )

    def project(points: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if not len(values):
            return np.empty((0, 2), dtype=float)
        return cv2.perspectiveTransform(values, homography)[:, 0, :].astype(float)

    excluded: set[int] = set()
    maximum_anchor_distance = (
        float(maximum_anchor_displacement_diagonal_fraction) * diagonal
    )
    for predicted in project(stationary_anchor_centers_px):
        # Exclude the whole registered anchor neighbourhood, not only one
        # connected component.  A blue cross viewed through a transparent
        # dish can split into a direct component plus a displaced reflection.
        for index, item in enumerate(candidates):
            if np.linalg.norm(item["center"] - predicted) <= maximum_anchor_distance:
                excluded.add(index)

    predicted_target = project([reference_target_center_px])[0]
    def rank_candidates(*, include_excluded: bool):
        ranked = []
        for index, item in enumerate(candidates):
            if index in excluded and not include_excluded:
                continue
            scale = float(item["area_scale"])
            if not float(minimum_component_area_scale) <= scale <= float(
                maximum_component_area_scale
            ):
                continue
            displacement = float(np.linalg.norm(item["center"] - predicted_target))
            normalized = displacement / diagonal
            if normalized > float(maximum_target_displacement_diagonal_fraction):
                continue
            # Prefer scale consistency, with motion only breaking near ties.
            # This is deliberately not an area-maximum rule.
            score = abs(math.log(scale)) + 0.1 * normalized / max(
                float(maximum_target_displacement_diagonal_fraction), 1e-9
            )
            ranked.append((score, normalized, index, item))
        return ranked

    ranked = rank_candidates(include_excluded=False)
    fallback_used = False
    if not ranked and excluded:
        # A movable transparent target can be placed over a registered blue
        # background anchor.  In that case exclusion removes every direct and
        # reflected component.  Retry within the tapped target's independent
        # scale and displacement gates; this is still not a largest-area rule.
        ranked = rank_candidates(include_excluded=True)
        fallback_used = bool(ranked)
    if not ranked:
        raise ValueError("tapped marker could not be reacquired after anchor exclusion")
    _, normalized, _, selected = min(ranked, key=lambda value: value[:3])
    center = selected["center"]
    return RelocatedMarkerObservation(
        center_px=(float(center[0]), float(center[1])),
        center_uv=(float(center[0] / width), float(center[1] / height)),
        component_pixels=int(selected["pixels"]),
        area_scale_from_reference=float(selected["area_scale"]),
        displacement_diagonal_fraction=float(normalized),
        excluded_anchor_count=len(excluded),
        candidate_count=len(candidates),
        anchor_overlap_fallback_used=fallback_used,
    )


def observe_marked_target(
    image_bgr: np.ndarray,
    template: GraspWindowTemplate,
    *,
    maximum_candidate_distance_tool_units: float = 2.0,
    reference_component_area_per_tool_scale_sq: float | None = None,
    minimum_reference_area_fraction: float = 0.03,
    maximum_reference_area_fraction: float = 3.0,
    proximity_score_weight: float = 0.03,
    fallback_minimum_light_value: int | None = None,
    expected_tool_frame: ToolImageFrame | None = None,
    prefer_expected_tool_frame: bool = False,
    prefer_cross_shape: bool = True,
    maximum_expected_tool_angle_deg: float = 15.0,
    maximum_expected_origin_scale: float = 0.40,
    minimum_expected_scale_ratio: float = 0.65,
    maximum_expected_scale_ratio: float = 1.55,
) -> TargetObservation:
    """Associate a calibrated marker in the tool-relative grasp window.

    A successful reference image calibrates marker area divided by squared
    tool scale.  This is resolution- and distance-normalized, and prevents a
    small blue reflection on the gripper from replacing a displaced target.
    Area is an identity likelihood, never a largest-component rule.  A marker
    clipped by an image border is treated as a censored lower-bound sample.
    The blue marker remains an optional profile adapter; unmarked objects can
    provide an equivalent local-feature observation to the same executor.
    """

    # The wrist camera and gripper are rigidly mounted. Once a tool frame was
    # measured at an audited wrist orientation, illumination may change its
    # appearance but cannot change that geometry. Callers clear the expected
    # frame whenever they deliberately change wrist orientation.
    if expected_tool_frame is not None and prefer_expected_tool_frame:
        frame = expected_tool_frame
        tool_frame_source = "rigid_expected_tool_frame"
    else:
        tool_frame_source = "light_pad_nominal"
        try:
            frame = detect_light_pad_tool_frame(image_bgr)
        except ValueError as error:
            if (
                fallback_minimum_light_value is None
                or str(error) != "light gripper pad is not visible"
            ):
                raise
            frame = detect_light_pad_tool_frame(
                image_bgr,
                minimum_light_value=int(fallback_minimum_light_value),
            )
            tool_frame_source = "light_pad_dark_scene_fallback"
    if expected_tool_frame is not None and not prefer_expected_tool_frame:
        expected_forward = np.asarray(expected_tool_frame.forward_xy, dtype=float)
        observed_forward = np.asarray(frame.forward_xy, dtype=float)
        angle = math.degrees(
            math.acos(
                float(
                    np.clip(
                        expected_forward @ observed_forward
                        / (
                            np.linalg.norm(expected_forward)
                            * np.linalg.norm(observed_forward)
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        origin_error = float(
            np.linalg.norm(
                np.asarray(frame.origin_px, dtype=float)
                - np.asarray(expected_tool_frame.origin_px, dtype=float)
            )
            / expected_tool_frame.scale_px
        )
        scale_ratio = float(frame.scale_px / expected_tool_frame.scale_px)
        light_fraction = float(
            frame.light_pad_pixels / max(frame.cyan_pixels, 1)
        )
        expected_consistent = bool(
            angle <= float(maximum_expected_tool_angle_deg)
            and origin_error <= float(maximum_expected_origin_scale)
            and float(minimum_expected_scale_ratio)
            <= scale_ratio
            <= float(maximum_expected_scale_ratio)
            and light_fraction >= 0.03
        )
        if not expected_consistent:
            frame = expected_tool_frame
            tool_frame_source = "audited_expected_tool_frame"
    centers = detect_blue_cross_centers(image_bgr)
    marker_candidates = []
    for center in centers:
        marker_candidates.append(
            (
                np.asarray(center, dtype=float),
                _blue_component_mask(image_bgr, center),
                True,
            )
        )
    # A blue mark viewed through a transparent lid can lose saturation and
    # cease to look cross-shaped at hover distance.  Add normalized-area
    # components from a permissive blue mask.  The calibrated area likelihood
    # below rejects the much larger cyan gripper and tiny specular artifacts.
    if reference_component_area_per_tool_scale_sq is not None:
        hsv = cv2.cvtColor(np.asarray(image_bgr), cv2.COLOR_BGR2HSV)
        translucent = cv2.inRange(
            hsv,
            np.asarray([95, 20, 40], dtype=np.uint8),
            np.asarray([130, 255, 255], dtype=np.uint8),
        )
        translucent = cv2.morphologyEx(
            translucent, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        count, labels, stats, component_centers = cv2.connectedComponentsWithStats(
            translucent
        )
        for label in range(1, count):
            center = np.asarray(component_centers[label], dtype=float)
            if any(
                np.linalg.norm(center - existing_center) <= 4.0
                for existing_center, _, _ in marker_candidates
            ):
                continue
            marker_candidates.append((center, labels == label, False))
    if not marker_candidates:
        raise ValueError("no marked target is visible")
    goal = np.asarray(template.reference_center_uv, dtype=float)
    candidate_centers = np.asarray(
        [center for center, _, _ in marker_candidates], dtype=float
    )
    tool_centers = frame.image_to_tool(candidate_centers)
    distances = np.linalg.norm(tool_centers - goal, axis=1)
    candidates = []
    image_height, image_width = np.asarray(image_bgr).shape[:2]
    reference_scale = (
        None
        if reference_component_area_per_tool_scale_sq is None
        else float(reference_component_area_per_tool_scale_sq)
    )
    if reference_scale is not None and (
        not math.isfinite(reference_scale) or reference_scale <= 0.0
    ):
        raise ValueError("reference marker/tool area scale must be positive")
    for index, (center, supplied_mask, is_cross_shaped) in enumerate(
        marker_candidates
    ):
        distance = float(distances[index])
        if distance > float(maximum_candidate_distance_tool_units):
            continue
        mask = supplied_mask
        component_pixels = int(np.count_nonzero(mask))
        ys, xs = np.nonzero(mask)
        touches_border = bool(
            len(xs)
            and (
                int(xs.min()) <= 1
                or int(ys.min()) <= 1
                or int(xs.max()) >= image_width - 2
                or int(ys.max()) >= image_height - 2
            )
        )
        area_scale = component_pixels / float(frame.scale_px**2)
        if reference_scale is None:
            identity_score = 0.0
        else:
            fraction = area_scale / reference_scale
            if not (
                float(minimum_reference_area_fraction)
                <= fraction
                <= float(maximum_reference_area_fraction)
            ):
                continue
            log_error = abs(math.log(max(fraction, 1e-12)))
            # Border clipping only decreases the observed component area.  It
            # must not make the nearby uncensored reflection look more likely.
            identity_score = (0.20 if touches_border and fraction <= 1.0 else 1.0) * log_error
        proximity_score = float(proximity_score_weight) * distance / max(
            float(maximum_candidate_distance_tool_units), 1e-9
        )
        candidates.append(
            {
                "index": index,
                "mask": mask,
                "pixels": component_pixels,
                "area_scale": area_scale,
                "touches_border": touches_border,
                "is_cross_shaped": bool(is_cross_shaped),
                "score": identity_score + proximity_score,
            }
        )
    if not candidates:
        raise ValueError("no marked target exists near the accepted grasp window")
    if prefer_cross_shape:
        candidate_key = lambda item: (
            not item["is_cross_shaped"],
            item["score"],
            float(distances[item["index"]]),
        )
    else:
        candidate_key = lambda item: (
            item["score"],
            not item["is_cross_shaped"],
            float(distances[item["index"]]),
        )
    selected_candidate = min(candidates, key=candidate_key)
    selected = int(selected_candidate["index"])
    mask = selected_candidate["mask"]
    component_pixels = int(selected_candidate["pixels"])
    ys, xs = np.nonzero(mask)
    border_extents = []
    if len(xs):
        if int(xs.min()) <= 1:
            border_extents.append(float(xs.max() + 1))
        if int(xs.max()) >= image_width - 2:
            border_extents.append(float(image_width - xs.min()))
        if int(ys.min()) <= 1:
            border_extents.append(float(ys.max() + 1))
        if int(ys.max()) >= image_height - 2:
            border_extents.append(float(image_height - ys.min()))
    # Wrist perspective can shrink a valid marker below the geometry helper's
    # 100-point numerical minimum.  Expand only its support for percentile
    # estimation; retain the original component size as evidence.
    assessment_mask = mask
    for _ in range(4):
        if int(np.count_nonzero(assessment_mask)) >= 100:
            break
        assessment_mask = cv2.dilate(
            assessment_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=1,
        ).astype(bool)
    assessment, assessed_frame = assess_grasp_window(
        image_bgr,
        assessment_mask,
        template,
        method="MASK_GEOMETRY",
        tool_frame=frame,
    )
    return TargetObservation(
        center_px=tuple(float(value) for value in candidate_centers[selected]),
        center_uv=tuple(float(value) for value in tool_centers[selected]),
        component_pixels=component_pixels,
        component_area_per_tool_scale_sq=float(selected_candidate["area_scale"]),
        component_touches_border=bool(selected_candidate["touches_border"]),
        candidate_count=len(candidates),
        tool_frame=assessed_frame,
        grasp_window=assessment,
        tool_frame_source=tool_frame_source,
        marker_cross_shaped=bool(selected_candidate["is_cross_shaped"]),
        component_interior_extent_px=(
            min(border_extents) if border_extents else None
        ),
    )


def track_target_center_lk(
    previous_image_bgr: np.ndarray,
    current_image_bgr: np.ndarray,
    previous: TargetObservation,
    template: GraspWindowTemplate,
    *,
    minimum_inliers: int = 8,
    minimum_inlier_fraction: float = 0.45,
) -> TargetObservation:
    """Bridge a short semantic-mask dropout with local optical flow.

    The fallback is deliberately local and cannot authorize closure.  It is
    intended for the last hover steps where a target mark can temporarily
    merge with a similarly coloured tool mask.  Geometry is normalized by the
    observed tool scale, so the tracker contains no task-specific pixel box.
    """

    previous_image = np.asarray(previous_image_bgr)
    current_image = np.asarray(current_image_bgr)
    if previous_image.shape != current_image.shape or previous_image.ndim != 3:
        raise ValueError("optical-flow images must have one matching BGR shape")
    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    height, width = previous_gray.shape
    center = np.asarray(previous.center_px, dtype=float)
    radius = max(12, int(round(0.45 * previous.tool_frame.scale_px)))
    yy, xx = np.ogrid[:height, :width]
    roi = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2).astype(
        np.uint8
    )
    hsv = cv2.cvtColor(previous_image, cv2.COLOR_BGR2HSV)
    # Reject the saturated cyan tool body while retaining low-saturation lid
    # rim texture and the darker portions of the target mark.
    tool = cv2.inRange(
        hsv,
        np.asarray([80, 100, 70], dtype=np.uint8),
        np.asarray([103, 255, 255], dtype=np.uint8),
    )
    roi[tool > 0] = 0
    features = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=120,
        qualityLevel=0.01,
        minDistance=max(3.0, 0.025 * previous.tool_frame.scale_px),
        mask=roi,
        blockSize=5,
    )
    if features is None or len(features) < int(minimum_inliers):
        raise ValueError("target optical-flow ROI has too few features")
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        features,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            40,
            0.01,
        ),
    )
    valid = status.reshape(-1).astype(bool)
    if moved is None or int(np.count_nonzero(valid)) < int(minimum_inliers):
        raise ValueError("target optical flow has too few tracked features")
    displacement = moved.reshape(-1, 2)[valid] - features.reshape(-1, 2)[valid]
    median = np.median(displacement, axis=0)
    residual = np.linalg.norm(displacement - median, axis=1)
    tolerance = max(2.0, 0.05 * previous.tool_frame.scale_px)
    inliers = residual <= tolerance
    inlier_count = int(np.count_nonzero(inliers))
    inlier_fraction = inlier_count / float(len(displacement))
    if (
        inlier_count < int(minimum_inliers)
        or inlier_fraction < float(minimum_inlier_fraction)
    ):
        raise ValueError("target optical flow failed robust-consensus gate")
    delta = np.median(displacement[inliers], axis=0)
    if float(np.linalg.norm(delta)) > 0.25 * float(np.hypot(width, height)):
        raise ValueError("target optical-flow displacement is implausibly large")
    tracked_center = center + delta
    if not (0 <= tracked_center[0] < width and 0 <= tracked_center[1] < height):
        raise ValueError("tracked target center left the image")
    frame = detect_light_pad_tool_frame(current_image)
    tracked_uv = frame.image_to_tool([tracked_center])[0]
    center_error = float(
        np.linalg.norm(tracked_uv - np.asarray(template.reference_center_uv))
        / template.square_side_u
    )
    assessment = GraspWindowAssessment(
        selected_method="MASK_GEOMETRY",
        allowed_to_close=False,
        white_window_ready=False,
        mask_geometry_ready=False,
        target_inside_fraction=0.0,
        normalized_center_error=center_error,
        normalized_quantile_error=center_error,
        target_center_uv=(float(tracked_uv[0]), float(tracked_uv[1])),
        failure_reasons=("semantic_mask_temporarily_unavailable",),
    )
    return TargetObservation(
        center_px=(float(tracked_center[0]), float(tracked_center[1])),
        center_uv=(float(tracked_uv[0]), float(tracked_uv[1])),
        component_pixels=0,
        component_area_per_tool_scale_sq=0.0,
        component_touches_border=False,
        candidate_count=0,
        tool_frame=frame,
        grasp_window=assessment,
        source="pyramidal_lk_semantic_dropout_bridge",
        tracking_inlier_fraction=float(inlier_fraction),
        marker_cross_shaped=previous.marker_cross_shaped,
    )


def target_follow_evidence(
    before_center_px: Sequence[float],
    after_center_px: Sequence[float],
    image_shape_hw: Sequence[int],
    *,
    maximum_displacement_diagonal_fraction: float,
    closure_before: dict,
    closure_after: dict,
) -> dict:
    """Verify that a camera-rigid target stayed with the gripper during lift."""

    shape = np.asarray(image_shape_hw, dtype=float)
    if shape.shape != (2,) or np.any(shape <= 0) or not np.all(np.isfinite(shape)):
        raise ValueError("image_shape_hw must contain positive height and width")
    limit = float(maximum_displacement_diagonal_fraction)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("follow displacement limit must be positive")
    displacement_px = float(
        np.linalg.norm(
            np.asarray(after_center_px, dtype=float)
            - np.asarray(before_center_px, dtype=float)
        )
    )
    normalized = displacement_px / float(np.linalg.norm(shape))
    accepted = bool(
        normalized <= limit
        and closure_before.get("nonempty") is True
        and closure_after.get("nonempty") is True
    )
    return {
        "accepted": accepted,
        "displacement_px": displacement_px,
        "displacement_diagonal_fraction": normalized,
        "maximum_displacement_diagonal_fraction": limit,
        "closure_nonempty_before": bool(closure_before.get("nonempty")),
        "closure_nonempty_after": bool(closure_after.get("nonempty")),
    }


@dataclass
class ConsecutiveSuccessLedger:
    required: int
    consecutive: int = 0
    attempts: int = 0

    def __post_init__(self) -> None:
        if int(self.required) <= 0:
            raise ValueError("required consecutive successes must be positive")

    def record(self, success: bool) -> dict:
        self.attempts += 1
        self.consecutive = self.consecutive + 1 if success else 0
        return {
            "attempts": self.attempts,
            "consecutive_successes": self.consecutive,
            "required_consecutive_successes": self.required,
            "complete": self.consecutive >= self.required,
        }

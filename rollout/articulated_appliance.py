"""Endpoint-state perception and policy for demonstrated articulated appliances.

The motion policy deliberately knows nothing about incubators.  An appliance
profile supplies two verified endpoint observations (open and closed), fixed
registration tags, and an optional marker rigidly attached to the moving
panel.  Live RGB-D is registered to the endpoint observations and compared in
a region whose size is expressed in *marker lengths*, not image pixels.

This keeps shadows and robot appearance out of the state decision.  A missing
marker is never, by itself, accepted as proof that the appliance is open.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from rollout.apriltag_retarget import detect_tags


class ApplianceState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EndpointObservation:
    image_bgr: np.ndarray
    depth_m: np.ndarray
    source: str


@dataclass(frozen=True)
class EndpointModel:
    canonical_image_bgr: np.ndarray
    open_depth_m: np.ndarray
    closed_depth_m: np.ndarray
    dynamic_mask: np.ndarray
    endpoint_separation_m: float
    closed_marker_corners: np.ndarray
    closed_marker_length_depth_px: float
    open_source: str
    closed_source: str


def load_endpoint(image_path: str | Path, depth_path: str | Path) -> EndpointObservation:
    image_path = Path(image_path)
    depth_path = Path(depth_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read endpoint image: {image_path}")
    depth = np.asarray(np.load(depth_path), dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"endpoint depth must be 2-D: {depth_path}")
    return EndpointObservation(image, depth, str(image_path.resolve()))


def load_bundle_endpoint(capture_dir: str | Path, camera: str = "head") -> EndpointObservation:
    """Load the middle frame from a lossless Record3D capture bundle."""

    root = Path(capture_dir) / "raw" / camera
    frames = sorted(path for path in root.iterdir() if path.is_dir())
    if not frames:
        raise RuntimeError(f"no {camera} frames in capture bundle: {capture_dir}")
    frame = frames[len(frames) // 2]
    image = cv2.imread(str(frame / "rgb.png"), cv2.IMREAD_COLOR)
    depth = np.asarray(np.load(frame / "depth.npy"), dtype=float)
    if image is None or depth.ndim != 2:
        raise RuntimeError(f"incomplete RGB-D frame: {frame}")
    # Record3D bundle data is native portrait; all door references and robot
    # camera views use landscape clockwise orientation.
    return EndpointObservation(
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE),
        str(frame.resolve()),
    )


def _tag_map(image: np.ndarray, settings: Mapping) -> dict[int, object]:
    detections = detect_tags(
        image,
        str(settings["tag_family"]),
        scales=tuple(settings.get("tag_detection_scales", (1, 2))),
    )
    return {int(item.tag_id): item for item in detections}


def _mean_tag_edge(corners: np.ndarray) -> float:
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    return float(
        np.mean(np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1))
    )


def _registration(
    live_image: np.ndarray,
    canonical_image: np.ndarray,
    settings: Mapping,
) -> tuple[np.ndarray, dict]:
    live = _tag_map(live_image, settings)
    canonical = _tag_map(canonical_image, settings)
    configured = tuple(int(value) for value in settings["fixed_tag_ids"])
    common = tuple(value for value in configured if value in live and value in canonical)
    minimum = int(settings.get("minimum_fixed_tags", 2))
    if len(common) < minimum:
        raise RuntimeError(
            f"need {minimum} fixed tags for endpoint registration; "
            f"common={list(common)}, live={sorted(live)}"
        )
    source = np.concatenate([live[value].corners for value in common]).astype(np.float32)
    target = np.concatenate([canonical[value].corners for value in common]).astype(np.float32)
    live_tag_length = float(
        np.median([_mean_tag_edge(live[value].corners) for value in common])
    )
    ransac_threshold = live_tag_length * float(
        settings.get("registration_ransac_threshold_tag_lengths", 0.08)
    )
    homography, inliers = cv2.findHomography(
        source, target, cv2.RANSAC, ransac_threshold
    )
    if homography is None:
        raise RuntimeError("fixed-tag endpoint registration failed")
    projected = cv2.perspectiveTransform(source.reshape(1, -1, 2), homography)[0]
    tag_length = float(np.median([_mean_tag_edge(canonical[value].corners) for value in common]))
    error_tag_lengths = float(np.median(np.linalg.norm(projected - target, axis=1)) / tag_length)
    maximum = float(settings.get("maximum_registration_error_tag_lengths", 0.20))
    if error_tag_lengths > maximum:
        raise RuntimeError(
            "fixed-tag endpoint registration residual is too large: "
            f"{error_tag_lengths:.3f} tag lengths > {maximum:.3f}"
        )
    return homography, {
        "common_fixed_tag_ids": list(common),
        "live_tag_ids": sorted(live),
        "registration_error_tag_lengths": error_tag_lengths,
        "inlier_corner_fraction": (
            float(np.mean(inliers)) if inliers is not None else None
        ),
    }


def _depth_homography(
    image_homography: np.ndarray,
    source_image_shape: tuple[int, ...],
    source_depth_shape: tuple[int, ...],
    target_image_shape: tuple[int, ...],
    target_depth_shape: tuple[int, ...],
) -> np.ndarray:
    source_scale = np.diag(
        [
            source_depth_shape[1] / source_image_shape[1],
            source_depth_shape[0] / source_image_shape[0],
            1.0,
        ]
    )
    target_scale = np.diag(
        [
            target_depth_shape[1] / target_image_shape[1],
            target_depth_shape[0] / target_image_shape[0],
            1.0,
        ]
    )
    return target_scale @ image_homography @ np.linalg.inv(source_scale)


def _warp_depth(
    observation: EndpointObservation,
    canonical: EndpointObservation,
    homography: np.ndarray,
) -> np.ndarray:
    depth_h = _depth_homography(
        homography,
        observation.image_bgr.shape,
        observation.depth_m.shape,
        canonical.image_bgr.shape,
        canonical.depth_m.shape,
    )
    return cv2.warpPerspective(
        observation.depth_m.astype(np.float32),
        depth_h,
        (canonical.depth_m.shape[1], canonical.depth_m.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    ).astype(float)


def build_endpoint_model(
    opened: EndpointObservation,
    closed: EndpointObservation,
    settings: Mapping,
) -> EndpointModel:
    """Learn the moving-panel comparison mask from verified endpoints."""

    open_to_closed, _ = _registration(opened.image_bgr, closed.image_bgr, settings)
    open_depth = _warp_depth(opened, closed, open_to_closed)
    closed_depth = np.asarray(closed.depth_m, dtype=float)
    closed_tags = _tag_map(closed.image_bgr, settings)
    marker_id = int(settings["closed_state_marker_tag_id"])
    if marker_id not in closed_tags:
        raise RuntimeError(f"closed endpoint is missing moving-panel marker {marker_id}")
    marker = closed_tags[marker_id]
    image_to_depth_x = closed_depth.shape[1] / closed.image_bgr.shape[1]
    image_to_depth_y = closed_depth.shape[0] / closed.image_bgr.shape[0]
    center = np.asarray(marker.center, dtype=float) * [image_to_depth_x, image_to_depth_y]
    marker_length = _mean_tag_edge(marker.corners) * image_to_depth_x
    yy, xx = np.indices(closed_depth.shape)
    radius_x = float(settings.get("roi_horizontal_marker_lengths", 4.0)) * marker_length
    radius_y = float(settings.get("roi_vertical_marker_lengths", 5.2)) * marker_length
    region = ((xx - center[0]) / radius_x) ** 2 + ((yy - center[1]) / radius_y) ** 2 <= 1.0
    minimum_depth = float(settings.get("minimum_depth_m", 0.15))
    maximum_depth = float(settings.get("maximum_depth_m", 2.0))
    valid = (
        region
        & np.isfinite(open_depth)
        & np.isfinite(closed_depth)
        & (open_depth >= minimum_depth)
        & (closed_depth >= minimum_depth)
        & (open_depth <= maximum_depth)
        & (closed_depth <= maximum_depth)
    )
    delta = np.abs(open_depth - closed_depth)
    dynamic = valid & (delta >= float(settings.get("minimum_endpoint_depth_change_m", 0.025)))
    minimum_tag_areas = float(settings.get("minimum_dynamic_tag_areas", 4.0))
    minimum_points = int(np.ceil(minimum_tag_areas * marker_length**2))
    if int(dynamic.sum()) < minimum_points:
        raise RuntimeError(
            "verified endpoints do not contain enough changed depth near the "
            f"moving-panel marker: {int(dynamic.sum())} < {minimum_points}"
        )
    separation = float(np.median(delta[dynamic]))
    if separation <= 0.0 or not np.isfinite(separation):
        raise RuntimeError("endpoint depth separation is invalid")
    return EndpointModel(
        canonical_image_bgr=closed.image_bgr,
        open_depth_m=open_depth,
        closed_depth_m=closed_depth,
        dynamic_mask=dynamic,
        endpoint_separation_m=separation,
        closed_marker_corners=np.asarray(marker.corners, dtype=float),
        closed_marker_length_depth_px=marker_length,
        open_source=opened.source,
        closed_source=closed.source,
    )


def classify_endpoint_state(
    live: EndpointObservation,
    model: EndpointModel,
    settings: Mapping,
) -> dict:
    """Classify one registered live RGB-D frame; ambiguous input stays unknown."""

    homography, registration = _registration(
        live.image_bgr, model.canonical_image_bgr, settings
    )
    canonical = EndpointObservation(
        model.canonical_image_bgr,
        model.closed_depth_m,
        model.closed_source,
    )
    depth = _warp_depth(live, canonical, homography)
    valid = model.dynamic_mask & np.isfinite(depth)
    required_fraction = float(settings.get("minimum_dynamic_depth_fraction", 0.70))
    valid_fraction = float(valid.sum() / max(int(model.dynamic_mask.sum()), 1))
    if valid_fraction < required_fraction:
        state = ApplianceState.UNKNOWN
        open_error = closed_error = float("inf")
        reason = "insufficient registered depth in learned moving-panel region"
    else:
        scale = model.endpoint_separation_m
        open_error = float(np.median(np.abs(depth[valid] - model.open_depth_m[valid])) / scale)
        closed_error = float(np.median(np.abs(depth[valid] - model.closed_depth_m[valid])) / scale)
        best = min(open_error, closed_error)
        margin = abs(open_error - closed_error)
        maximum_error = float(settings.get("maximum_endpoint_relative_error", 0.45))
        minimum_margin = float(settings.get("minimum_endpoint_relative_margin", 0.20))
        if best > maximum_error or margin < minimum_margin:
            state = ApplianceState.UNKNOWN
            reason = "live depth is between or outside verified endpoint states"
        elif open_error < closed_error:
            state = ApplianceState.OPEN
            reason = "registered depth matches verified open endpoint"
        else:
            state = ApplianceState.CLOSED
            reason = "registered depth matches verified closed endpoint"

    marker_id = int(settings["closed_state_marker_tag_id"])
    live_tags = _tag_map(live.image_bgr, settings)
    marker_confirmed = False
    marker_error_tag_lengths = None
    if marker_id in live_tags:
        projected = cv2.perspectiveTransform(
            np.asarray(live_tags[marker_id].corners, dtype=np.float32).reshape(1, 4, 2),
            homography,
        )[0]
        marker_length = _mean_tag_edge(model.closed_marker_corners)
        marker_error_tag_lengths = float(
            np.median(np.linalg.norm(projected - model.closed_marker_corners, axis=1))
            / marker_length
        )
        marker_confirmed = marker_error_tag_lengths <= float(
            settings.get("maximum_closed_marker_error_tag_lengths", 0.75)
        )
        # The marker is affirmative evidence, but it does not override grossly
        # incompatible depth (for example a loose paper tag on the table).
        if marker_confirmed and closed_error <= float(
            settings.get("maximum_marker_confirmed_relative_error", 0.65)
        ):
            state = ApplianceState.CLOSED
            reason = "moving-panel marker and registered depth confirm closed endpoint"

    return {
        "schema": "piper_robot.articulated_appliance_state/v1",
        "state": state.value,
        "reason": reason,
        "source": live.source,
        "registration": registration,
        "dynamic_depth_fraction": valid_fraction,
        "dynamic_point_count": int(valid.sum()),
        "endpoint_separation_m": model.endpoint_separation_m,
        "relative_open_error": open_error,
        "relative_closed_error": closed_error,
        "closed_marker_tag_id": marker_id,
        "closed_marker_visible": marker_id in live_tags,
        "closed_marker_confirmed": marker_confirmed,
        "closed_marker_error_tag_lengths": marker_error_tag_lengths,
        "reference_sources": {
            "open": model.open_source,
            "closed": model.closed_source,
        },
    }


def render_endpoint_evidence(
    live: EndpointObservation,
    model: EndpointModel,
    report: Mapping,
) -> np.ndarray:
    """Render the learned comparison region and the accepted endpoint state."""

    image = live.image_bgr.copy()
    mask = cv2.resize(
        np.uint8(model.dynamic_mask) * 255,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    tint = np.zeros_like(image)
    tint[:, :, 1] = mask
    image = cv2.addWeighted(image, 1.0, tint, 0.25, 0.0)
    label = (
        f"state={report['state']} open={report['relative_open_error']:.2f} "
        f"closed={report['relative_closed_error']:.2f}"
    )
    cv2.putText(image, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(image, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return image


def workflow_stages(goal: str) -> tuple[str, ...]:
    """Return the evidence-gated stage graph for a requested endpoint."""

    if goal == ApplianceState.OPEN.value:
        return (
            "observe-closed",
            "register-moving-panel",
            "open-jaw-preclose",
            "bounded-visual-align",
            "open-jaw-contact",
            "close-once",
            "proof-pull-5mm",
            "stationary-proof-reverify",
            "demonstrated-open-pull",
            "clearance-retreat-and-release",
            "verify-open",
        )
    if goal == ApplianceState.CLOSED.value:
        return (
            "observe-open",
            "dedicated-open-jaw-close-demo",
            "verify-closed",
        )
    raise ValueError(f"unsupported endpoint goal: {goal}")

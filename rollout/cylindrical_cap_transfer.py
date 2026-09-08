"""Evidence gates for a vertical cylindrical-cap side-pinch transfer.

This module has no robot I/O.  It turns fixed-head images and measured robot
state into scale-normalized lift/transport evidence.  A route may be promoted
for replay only when the cap has cleared its source, the bottle/support stayed
fixed, and the non-empty gripper obstruction persisted through transport.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from rollout.media_cap_target import (
    detect_coloured_support_anchor,
    detect_open_jaw_center_head,
    fixed_target_in_jaw_segment,
)


AUDIT_SCHEMA = "piper_robot.cylindrical_cap_transfer_audit/v1"


def waypoint_route_sha256(waypoints) -> str:
    """Bind a hardware-evidence exception to one exact waypoint list."""

    encoded = json.dumps(
        waypoints,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_hardware_route_replay(
    path,
    *,
    lower,
    upper,
    model_collision_audit: Mapping,
    hardware_evidence_audit: Mapping,
    actual_waypoints_sha256: str,
    expected_waypoints_sha256: str,
    maximum_route_joint_step_rad: float,
    allow_model_false_positive: bool,
) -> dict:
    """Allow a model false positive only for exact successful hardware data."""

    path = np.asarray(path, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if path.ndim != 2 or path.shape[1] != 6:
        raise ValueError("cap transfer path must be Nx6")
    within_joint_bounds = bool(
        np.all(path >= lower[None, :]) and np.all(path <= upper[None, :])
    )
    maximum_joint_step = float(np.max(np.abs(np.diff(path, axis=0))))
    exact_hardware_route = bool(
        actual_waypoints_sha256 == expected_waypoints_sha256
        and hardware_evidence_audit.get("accepted") is True
    )
    model_override = bool(
        model_collision_audit.get("accepted") is not True
        and allow_model_false_positive
        and exact_hardware_route
    )
    accepted = bool(
        within_joint_bounds
        and maximum_joint_step <= float(maximum_route_joint_step_rad)
        and (model_collision_audit.get("accepted") is True or model_override)
    )
    return {
        "accepted": accepted,
        "within_joint_bounds": within_joint_bounds,
        "maximum_joint_step_rad": maximum_joint_step,
        "approach_waypoints_sha256": actual_waypoints_sha256,
        "exact_hardware_route": exact_hardware_route,
        "hardware_evidence_audit": hardware_evidence_audit,
        "mujoco_collision_audit": model_collision_audit,
        "model_false_positive_override": model_override,
        "override_scope": (
            "exact_hardware_observed_reverse_egress_only"
            if model_override
            else "none"
        ),
    }


@dataclass(frozen=True)
class CapTransferFrame:
    capture: str
    target_anchor_px: tuple[float, float]
    jaw_center_px: tuple[float, float]
    jaw_centers_px: tuple[tuple[float, float], tuple[float, float]]
    jaw_span_px: float
    source_local_white_pixels: int
    source_local_area_pixels: int
    support_center_px: tuple[float, float]
    support_scale_px: float
    right_q_physical_rad: tuple[float, ...]
    right_ee_xyz_m: tuple[float, float, float]
    gripper_open_ratio: float

    @property
    def source_local_white_fraction(self) -> float:
        return self.source_local_white_pixels / max(
            self.source_local_area_pixels, 1
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        value["source_local_white_fraction"] = (
            self.source_local_white_fraction
        )
        return value


def _state_from_manifest(manifest: Mapping) -> Mapping:
    state = manifest.get("robot_state", {}).get("before")
    if not isinstance(state, Mapping):
        raise ValueError("capture manifest has no robot_state.before")
    return state


def capture_frame_evidence(
    capture: str | Path,
    *,
    target_anchor_uv,
    source_radius_jaw_spans: float = 0.30,
) -> CapTransferFrame:
    """Extract evidence without an absolute pixel ROI or cap colour label."""

    capture = Path(capture).resolve()
    image_path = capture / "derived/head_rgb_landscape.png"
    manifest_path = capture / "manifest.json"
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"head image is unreadable: {image_path}")
    manifest = json.loads(manifest_path.read_text())
    state = _state_from_manifest(manifest)
    uv = np.asarray(target_anchor_uv, dtype=float)
    if uv.shape != (2,) or np.any(~np.isfinite(uv)):
        raise ValueError("target_anchor_uv must contain two finite values")
    anchor = uv * np.asarray([image.shape[1], image.shape[0]], dtype=float)
    jaw_center, _, jaw = detect_open_jaw_center_head(image)
    span = float(jaw["jaw_span_px"])
    radius = float(source_radius_jaw_spans) * span
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    source_roi = (xx - anchor[0]) ** 2 + (yy - anchor[1]) ** 2 <= radius**2
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    local_white = (hsv[:, :, 1] <= 90) & (hsv[:, :, 2] >= 185)
    support_center, support = detect_coloured_support_anchor(
        image, target_anchor_px=anchor
    )
    pose = state.get("right_ee_pose", {})
    xyz = pose.get("translation_xyz_m")
    q = state.get("right_joint_positions_rad")
    aperture = state.get("right_gripper_open_ratio")
    if xyz is None or q is None or aperture is None:
        raise ValueError("capture manifest lacks measured right-arm state")
    return CapTransferFrame(
        capture=str(capture),
        target_anchor_px=(float(anchor[0]), float(anchor[1])),
        jaw_center_px=(float(jaw_center[0]), float(jaw_center[1])),
        jaw_centers_px=tuple(
            tuple(float(item) for item in center)
            for center in jaw["jaw_centers_px"]
        ),
        jaw_span_px=span,
        source_local_white_pixels=int(np.count_nonzero(local_white & source_roi)),
        source_local_area_pixels=int(np.count_nonzero(source_roi)),
        support_center_px=(float(support_center[0]), float(support_center[1])),
        support_scale_px=float(support["component_diagonal_px"]),
        right_q_physical_rad=tuple(float(item) for item in q),
        right_ee_xyz_m=tuple(float(item) for item in xyz),
        gripper_open_ratio=float(aperture),
    )


def _support_shift_fraction(first: CapTransferFrame, second: CapTransferFrame):
    shift = float(
        np.linalg.norm(
            np.asarray(second.support_center_px)
            - np.asarray(first.support_center_px)
        )
    )
    scale = max(first.support_scale_px, second.support_scale_px, 1.0)
    return shift / scale


def _source_appearance_ratio(first: CapTransferFrame, second: CapTransferFrame):
    return second.source_local_white_fraction / max(
        first.source_local_white_fraction, 1e-6
    )


def validate_lift_transition(
    before: CapTransferFrame,
    after: CapTransferFrame,
    settings: Mapping,
) -> dict:
    """Require source clearance, stationary support, and persistent grasp."""

    source_gate = fixed_target_in_jaw_segment(
        before.target_anchor_px,
        before.jaw_centers_px,
        maximum_perpendicular_span_fraction=float(
            settings["maximum_target_perpendicular_span_fraction"]
        ),
    )
    lift = float(after.right_ee_xyz_m[2] - before.right_ee_xyz_m[2])
    jaw_motion = float(
        np.linalg.norm(
            np.asarray(after.jaw_center_px) - np.asarray(before.jaw_center_px)
        )
        / max(before.jaw_span_px, after.jaw_span_px, 1.0)
    )
    support_shift = _support_shift_fraction(before, after)
    source_ratio = _source_appearance_ratio(before, after)
    aperture_drift = abs(
        after.gripper_open_ratio - before.gripper_open_ratio
    )
    accepted = bool(
        source_gate["accepted"]
        and lift >= float(settings["minimum_measured_lift_m"])
        and jaw_motion >= float(settings["minimum_jaw_motion_spans"])
        and support_shift <= float(settings["maximum_support_shift_fraction"])
        and source_ratio <= float(settings["maximum_source_appearance_ratio"])
        and float(settings["minimum_contact_open_ratio"])
        <= after.gripper_open_ratio
        <= float(settings["maximum_contact_open_ratio"])
        and aperture_drift <= float(settings["maximum_aperture_drift"])
    )
    return {
        "accepted": accepted,
        "source_target_between_jaws_before_lift": source_gate,
        "measured_lift_m": lift,
        "jaw_motion_spans": jaw_motion,
        "support_shift_fraction": support_shift,
        "source_appearance_ratio": source_ratio,
        "aperture_before": before.gripper_open_ratio,
        "aperture_after": after.gripper_open_ratio,
        "aperture_drift": aperture_drift,
    }


def validate_transport_transition(
    lift: CapTransferFrame,
    transported: CapTransferFrame,
    source_before_lift: CapTransferFrame,
    settings: Mapping,
) -> dict:
    """Require the same non-empty obstruction to survive a useful transfer."""

    distance = float(
        np.linalg.norm(
            np.asarray(transported.right_ee_xyz_m)
            - np.asarray(lift.right_ee_xyz_m)
        )
    )
    aperture_drift = abs(
        transported.gripper_open_ratio - lift.gripper_open_ratio
    )
    support_shift = _support_shift_fraction(source_before_lift, transported)
    source_ratio = _source_appearance_ratio(
        source_before_lift, transported
    )
    accepted = bool(
        distance >= float(settings["minimum_transport_distance_m"])
        and float(settings["minimum_contact_open_ratio"])
        <= transported.gripper_open_ratio
        <= float(settings["maximum_contact_open_ratio"])
        and aperture_drift <= float(settings["maximum_aperture_drift"])
        and support_shift <= float(settings["maximum_support_shift_fraction"])
        and source_ratio <= float(settings["maximum_source_appearance_ratio"])
    )
    return {
        "accepted": accepted,
        "measured_transport_distance_m": distance,
        "aperture_lift": lift.gripper_open_ratio,
        "aperture_transport": transported.gripper_open_ratio,
        "aperture_drift": aperture_drift,
        "support_shift_fraction": support_shift,
        "source_appearance_ratio": source_ratio,
    }


def audit_cap_transfer_captures(
    before_capture: str | Path,
    lift_capture: str | Path,
    transported_capture: str | Path,
    *,
    target_anchor_uv,
    settings: Mapping,
) -> dict:
    """Audit three immutable captures and return a replay-promotion record."""

    frames = [
        capture_frame_evidence(
            item,
            target_anchor_uv=target_anchor_uv,
            source_radius_jaw_spans=float(
                settings.get("source_radius_jaw_spans", 0.30)
            ),
        )
        for item in (before_capture, lift_capture, transported_capture)
    ]
    lift = validate_lift_transition(frames[0], frames[1], settings)
    transport = validate_transport_transition(
        frames[1], frames[2], frames[0], settings
    )
    accepted = bool(lift["accepted"] and transport["accepted"])
    return {
        "schema": AUDIT_SCHEMA,
        "accepted": accepted,
        "promotion_scope": (
            "cap_side_pinch_lift_and_held_transport"
            if accepted
            else "none"
        ),
        "placement_promoted": False,
        "placement_reason": (
            "release destination was not visually/depth verified"
        ),
        "frames": [frame.to_dict() for frame in frames],
        "lift": lift,
        "transport": transport,
    }

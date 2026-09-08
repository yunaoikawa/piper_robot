#!/usr/bin/env python3
"""Run a tap-initialized thin-object edge grasp without Codex decisions.

The task policy is configuration driven.  The executor uses the same 30 Hz
Cartesian RPC path as human teleoperation, checks jaw level only at semantic
checkpoints, closes once at the lowest pose, verifies a mechanical
obstruction, and requires the selected target to follow a straight lift.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import mink
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.arm.home import (
    physical_home_q,
    physical_to_semantic_model_q_offset,
)
from robot.camera_id import configure_camera_map_by_udid, load_camera_map
from robot.rpc import RPCClient
from rollout.camera import USBWristCameraFeedManager
from rollout.fast_lid_grasp import ClosureEvidence, FastLidGraspMachine
from rollout.grasp_window import (
    GraspWindowAssessment,
    GraspWindowTemplate,
    ToolImageFrame,
)
from rollout.gripper_level import (
    JawLevelReference,
    RightJawLevelCheckpoint,
    assess_jaw_level,
    leveled_pose,
)
from rollout.grasp_orchestration import ControllerLease
from rollout.grasp_orchestration import seating_distance_m
from rollout.sam_segmentation import detect_blue_cross_centers
from rollout.tapped_lid_target import register_fixed_head
from rollout.scene_registration import (
    bridge_camera_from_fixed_tag,
    intersect_pixel_with_horizontal_plane,
    tag_pose_camera,
)
from rollout.teleop_trajectory_stream import (
    CONTROL_HZ,
    JointTrajectorySample,
    ProductionRightFK,
    TeleopTrajectoryStreamer,
    TrajectoryStreamError,
    sample_joint_knots,
)
from rollout.thin_object_grasp import (
    ClosureCalibration,
    ConsecutiveSuccessLedger,
    TargetObservation,
    observe_local_blue_evidence_target,
    observe_marked_target,
    select_local_blue_evidence_marker,
    select_relocated_target_marker,
    track_target_center_lk,
    target_follow_evidence,
)
from src.optimize_lid_grasp_trajectory import GraspKinematics


SCHEMA = "piper_robot.codexless_thin_object_grasp_profile/v1"
RUN_SCHEMA = "piper_robot.codexless_thin_object_grasp_run/v1"
RUNTIME_ALIGNMENT_SCHEMA = "piper_robot.thin_object_runtime_alignment/v1"
PENDING_ALIGNMENT_SCHEMA = "piper_robot.thin_object_pending_alignment/v1"
PENDING_HOVER_SCHEMA = "piper_robot.thin_object_pending_hover/v1"


class DescentPlanRejected(RuntimeError):
    """Raised before any low/descent command has been sent."""


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).resolve().read_text())


def _q(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain six finite values")
    return result


def _target_observation_from_dict(value: dict) -> TargetObservation:
    """Restore a persisted observation as a continuity anchor, not evidence."""

    data = dict(value)
    return TargetObservation(
        center_px=tuple(data["center_px"]),
        center_uv=tuple(data["center_uv"]),
        component_pixels=int(data["component_pixels"]),
        component_area_per_tool_scale_sq=float(
            data["component_area_per_tool_scale_sq"]
        ),
        component_touches_border=bool(data["component_touches_border"]),
        candidate_count=int(data["candidate_count"]),
        tool_frame=ToolImageFrame(**data["tool_frame"]),
        grasp_window=GraspWindowAssessment(**data["grasp_window"]),
        source=str(data.get("source", "persisted_preclose_identity_anchor")),
        tracking_inlier_fraction=data.get("tracking_inlier_fraction"),
        tool_frame_source=str(
            data.get("tool_frame_source", "persisted_preclose_identity_anchor")
        ),
        marker_cross_shaped=data.get("marker_cross_shaped"),
        component_interior_extent_px=data.get("component_interior_extent_px"),
    )


def _pose(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite wxyz+xyz")
    norm = float(np.linalg.norm(result[:4]))
    if norm < 1e-9:
        raise ValueError(f"{name} has an invalid quaternion")
    result = result.copy()
    result[:4] /= norm
    return result


def load_profile(path: str | Path) -> dict:
    profile = _load(path)
    if profile.get("schema") != SCHEMA:
        raise ValueError(f"profile must use {SCHEMA}")
    if profile.get("physical_arm") != "right":
        raise ValueError("the reviewed executor is physical-right only")
    if profile.get("execution", {}).get("left_arm_commands") != 0:
        raise ValueError("profile must prohibit left-arm commands")
    trajectory = profile["trajectory"]
    _q(trajectory["safe_high_q_physical_rad"], "safe high q")
    _q(trajectory["verified_preclose_q_physical_rad"], "preclose q")
    _q(trajectory["canonical_hover_q_physical_rad"], "canonical hover q")
    _pose(trajectory["verified_preclose_pose_wxyz_xyz"], "preclose pose")
    ClosureCalibration(
        tuple(profile["closure"]["empty_reference_ratios"]),
        tuple(profile["closure"]["nonempty_reference_ratios"]),
    )
    seating_distance_m(profile)
    if profile["target_identity"].get("feature_adapter") != "blue_cross":
        raise ValueError("this executable currently requires the blue_cross adapter")
    for key in ("canonical_hover_goal_uv", "canonical_preclose_goal_uv"):
        goal = np.asarray(profile["target_identity"].get(key), dtype=float)
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError(f"{key} must contain two finite tool-frame values")
    return profile


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.monotonic_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _runtime_alignment_geometry_audit(
    profile: dict, target_center_scene_xyz_m, low_q_physical_rad
) -> dict:
    """Cross-check a learned arm pose against the fixed-head scene motion.

    Matching the same head-camera marker is insufficient: a wrist camera can
    lock onto a neighbouring transparent dish while the head still tracks the
    correct blue cross.  The learned low pose must therefore move from the
    configured physical anchor by approximately the mapped head-scene delta.
    This catches a semantically poisoned cache without deleting its evidence.
    """

    target = profile["target_identity"]
    settings = profile["head_localization"]
    relocation = target.get("relocation") or {}
    scene = np.asarray(target_center_scene_xyz_m, dtype=float)
    reference_scene = np.asarray(
        settings.get("reference_target_center_scene_xyz_m"), dtype=float
    )
    if (
        scene.shape != (3,)
        or reference_scene.shape != (3,)
        or not np.all(np.isfinite(np.r_[scene, reference_scene]))
    ):
        return {
            "accepted": False,
            "applicable": True,
            "reason": "head-scene anchor is invalid",
        }
    scene_delta = scene - reference_scene
    scene_distance = float(np.linalg.norm(scene_delta[:2]))
    maximum_scene = float(relocation.get("maximum_scene_displacement_m", 0.0))
    yaw = float(relocation.get("scene_delta_to_production_xy_yaw_rad", math.nan))
    if (
        relocation.get("accepted") is not True
        or not np.isfinite(yaw)
        or maximum_scene <= 0.0
        or scene_distance > maximum_scene
    ):
        # Synthetic/unit-test anchors and scenes outside the calibrated
        # relocation envelope cannot be judged by this physical invariant.
        return {
            "accepted": True,
            "applicable": False,
            "reason": "scene lies outside the calibrated cross-modal envelope",
            "target_scene_displacement_m": scene_distance,
        }
    fk = ProductionRightFK(profile["production_model"])
    reference_q = _q(
        profile["trajectory"]["verified_preclose_q_physical_rad"],
        "configured cross-modal reference q",
    )
    learned_q = _q(low_q_physical_rad, "learned runtime low q")
    reference_xyz = np.asarray(fk.pose(reference_q).parameters()[4:], dtype=float)
    learned_xyz = np.asarray(fk.pose(learned_q).parameters()[4:], dtype=float)
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    expected_xyz = reference_xyz.copy()
    expected_xyz[:2] += rotation @ scene_delta[:2]
    residual = float(np.linalg.norm(learned_xyz - expected_xyz))
    maximum_residual = float(
        target.get("maximum_runtime_alignment_geometry_residual_m", 0.03)
    )
    return {
        "accepted": residual <= maximum_residual,
        "applicable": True,
        "residual_m": residual,
        "maximum_residual_m": maximum_residual,
        "target_scene_displacement_m": scene_distance,
        "reference_production_xyz_m": reference_xyz.tolist(),
        "expected_production_xyz_m": expected_xyz.tolist(),
        "learned_production_xyz_m": learned_xyz.tolist(),
        "reason": (
            None
            if residual <= maximum_residual
            else "learned wrist pose is inconsistent with fixed-head 3D motion"
        ),
    }


def _load_runtime_alignment(profile: dict, localization: dict) -> tuple[np.ndarray | None, dict]:
    target = profile["target_identity"]
    configured = target.get("runtime_alignment_calibration")
    if not configured:
        return None, {"accepted": False, "reason": "runtime alignment is disabled"}
    path = Path(configured).resolve()
    if not path.is_file():
        return None, {"accepted": False, "reason": "runtime alignment is absent", "path": str(path)}
    value = _load(path)
    if value.get("schema") != RUNTIME_ALIGNMENT_SCHEMA:
        return None, {"accepted": False, "reason": "runtime alignment schema mismatch", "path": str(path)}
    configured_goal = np.asarray(target["canonical_hover_goal_uv"], dtype=float)
    saved_goal = np.asarray(value.get("canonical_hover_goal_uv"), dtype=float)
    configured_preclose_goal = np.asarray(
        target.get("canonical_preclose_goal_uv"), dtype=float
    )
    saved_preclose_goal = np.asarray(
        value.get("canonical_preclose_goal_uv"), dtype=float
    )
    if (
        saved_goal.shape != (2,)
        or not np.allclose(saved_goal, configured_goal, atol=1e-9)
        or configured_preclose_goal.shape != (2,)
        or saved_preclose_goal.shape != (2,)
        or not np.allclose(
            saved_preclose_goal, configured_preclose_goal, atol=1e-9
        )
    ):
        return None, {"accepted": False, "reason": "stage-specific image goal calibration changed", "path": str(path)}
    current_scene = np.asarray(localization["target_center_scene_xyz_m"], dtype=float)
    saved_scene = np.asarray(value.get("target_center_scene_xyz_m"), dtype=float)
    if current_scene.shape != (3,) or saved_scene.shape != (3,):
        return None, {"accepted": False, "reason": "runtime target scene point is invalid", "path": str(path)}
    displacement = float(np.linalg.norm(current_scene - saved_scene))
    maximum = float(target.get("maximum_runtime_target_scene_displacement_m", 0.008))
    report = {
        "accepted": displacement <= maximum,
        "path": str(path),
        "target_scene_displacement_m": displacement,
        "maximum_target_scene_displacement_m": maximum,
        "source_run": value.get("source_run"),
        "normalized_hover_error": value.get("normalized_hover_error"),
    }
    saved_seed = value.get("servo_seed")
    if saved_seed is not None:
        jacobian = np.asarray(saved_seed.get("jacobian"), dtype=float)
        best_error = float(saved_seed.get("best_error_norm", math.nan))
        if (
            jacobian.shape != (2, 2)
            or not np.all(np.isfinite(jacobian))
            or not np.isfinite(best_error)
            or best_error < 0.0
        ):
            return None, {
                **report,
                "accepted": False,
                "reason": "runtime visual-servo seed is invalid",
            }
        # Deliberately omit all previous sample/best-state fields.  They would
        # either perform a bogus first Broyden update or jump straight to the
        # fully occluded aligned pose.  Start at the last semantic observation
        # and reuse only the locally learned image Jacobian.
        report["servo_state_seed"] = {
            "jacobian": jacobian.tolist(),
            "adaptive_trust_from_first_observation": True,
        }
        learned_parallax = saved_seed.get(
            "vertical_descent_target_pixel_delta_px"
        )
        if learned_parallax is not None:
            learned_parallax = np.asarray(learned_parallax, dtype=float)
            if learned_parallax.shape != (2,) or not np.all(
                np.isfinite(learned_parallax)
            ):
                return None, {
                    **report,
                    "accepted": False,
                    "reason": "runtime vertical descent parallax is invalid",
                }
            report["servo_state_seed"][
                "vertical_descent_target_pixel_delta_px"
            ] = learned_parallax.tolist()
        identity_anchor = saved_seed.get("hover_identity_anchor")
        if identity_anchor is not None:
            center = np.asarray(identity_anchor.get("center_px"), dtype=float)
            pixels = int(identity_anchor.get("component_pixels", 0))
            if (
                center.shape != (2,)
                or not np.all(np.isfinite(center))
                or pixels <= 0
            ):
                return None, {
                    **report,
                    "accepted": False,
                    "reason": "runtime hover identity anchor is invalid",
                }
            report["servo_state_seed"]["hover_identity_anchor"] = {
                "center_px": center.tolist(),
                "component_pixels": pixels,
            }
    if not report["accepted"]:
        report["reason"] = "tapped target moved outside the runtime calibration envelope"
        return None, report
    low_q = _q(value["low_q_physical_rad"], "runtime aligned low q")
    reacquisition_q = _q(
        value.get("semantic_reacquisition_low_q_physical_rad", low_q),
        "runtime semantic reacquisition low q",
    )
    geometry = _runtime_alignment_geometry_audit(
        profile, saved_scene, reacquisition_q
    )
    report["cross_modal_geometry"] = geometry
    if not geometry["accepted"]:
        return None, {
            **report,
            "accepted": False,
            "reason": "runtime alignment failed fixed-head/FK geometry audit",
        }
    report["aligned_low_q_physical_rad"] = low_q.tolist()
    report["semantic_reacquisition_low_q_physical_rad"] = reacquisition_q.tolist()
    if value.get("hover_q_physical_rad") is not None:
        report["hover_q_physical_rad"] = _q(
            value["hover_q_physical_rad"], "runtime camera-visible hover q"
        ).tolist()
        tool_frame = value.get("tool_frame")
        if not isinstance(tool_frame, dict):
            return None, {
                **report,
                "accepted": False,
                "reason": "runtime hover tool frame is invalid",
            }
        # Construction performs shape/range validation without coupling this
        # persisted state to a particular image size.
        ToolImageFrame(**tool_frame)
        report["tool_frame"] = dict(tool_frame)
    return reacquisition_q, report


def _accepted_runtime_servo_seed(runtime_q, runtime_report: dict) -> dict:
    """Install cached image dynamics only with its accepted physical pose."""

    saved = dict(runtime_report.get("servo_state_seed", {}))
    # Identity remains useful when a moved target makes the old joint seed fail
    # its cross-modal audit.  Vertical parallax does *not*: it depends on the
    # exact eye-in-hand hover/low orientations and was observed to reverse sign
    # after a descent-branch change.  Never retain parallax, a Jacobian, or a
    # best joint state without the accepted physical pose.
    pose_independent = {
        key: saved[key]
        for key in ("hover_identity_anchor",)
        if key in saved
    }
    if runtime_q is None:
        return pose_independent
    return saved


def _hover_alignment_policy(
    profile: dict,
    observation,
    *,
    canonical_goal_aligned: bool,
    direct_cross_verified: bool,
    replaying_low_pose_correction: bool = False,
) -> tuple[bool, str]:
    """Keep camera-view identity and preclose alignment as separate stages.

    The successful demonstration's approach image was captured on a planar
    camera-view leg, not at the pose vertically above the eventual low grasp.
    Applying that pixel goal to a vertical hover moved an independently
    localized low pose several centimetres away from the target.  In the
    head-seeded mode, the high right-camera frame proves only that the tapped
    marked object is still visible.  Planar grasp correction is deferred to a
    fresh image at the lowest open-jaw pose, where the calibrated preclose goal
    is geometrically meaningful and closure is still forbidden until it
    passes.
    """

    mode = str(
        profile["perception"].get("hover_alignment_mode", "canonical_goal")
    )
    if mode == "canonical_goal":
        return bool(canonical_goal_aligned), mode
    if mode != "identity_only_before_fresh_preclose":
        raise ValueError(f"unknown hover alignment mode: {mode}")
    # A persisted low-pose correction is geometry, not fresh semantic proof.
    # Keep the argument explicit so callers cannot accidentally make replay an
    # identity bypass again: every descent still needs a newly observed,
    # directly segmented blue cross in the current hover frame.
    del replaying_low_pose_correction
    direct_marker = bool(
        direct_cross_verified
        and getattr(observation, "marker_cross_shaped", None) is True
        and str(getattr(observation, "source", "")) == "blue_marker"
    )
    persisted_identity_continuity = bool(
        str(getattr(observation, "source", ""))
        == "persisted_direct_cross_identity_continuity"
    )
    return bool(direct_marker or persisted_identity_continuity), mode


def _may_skip_high_view_for_fresh_low_replay(
    profile: dict, cached_alignment: dict | None
) -> bool:
    """Skip a redundant wrist high view, never the fresh low close gate.

    A pending preclose artifact has already been revalidated against the
    current fixed-head target identity before ``run_attempt`` is called.  Its
    low pose is only an untrusted proposal: closure remains forbidden and a
    fresh low wrist image is still mandatory.  Requiring the target to also
    appear in a vertically offset wrist frame can deadlock when the eye-in-hand
    camera's parallax puts the target outside that high image, even though the
    proposed low view is the exact state that must be evaluated next.
    """

    if cached_alignment is None:
        return False
    if str(
        profile["perception"].get("hover_alignment_mode", "canonical_goal")
    ) != "identity_only_before_fresh_preclose":
        return False
    return bool(
        cached_alignment.get("accepted") is True
        and cached_alignment.get("fresh_preclose_required") is True
        and cached_alignment.get("closure_authorized") is False
        and cached_alignment.get("method")
        in (
            "semantically_verified_preclose_level_q_correction",
            "semantically_verified_same_xy_level_retry",
            "semantically_verified_preclose_axis_probe",
            "semantically_detected_regression_backtrack",
        )
    )


def _save_runtime_alignment(
    profile: dict,
    localization: dict,
    attempt: dict,
    low_q,
    *,
    source_run: str,
    servo_state: dict | None = None,
) -> dict:
    target = profile["target_identity"]
    path = Path(target["runtime_alignment_calibration"]).resolve()
    hover = attempt["hover"]
    executed_low_q = _q(low_q, "runtime alignment executed low q")
    state = {} if servo_state is None else servo_state
    jacobian = np.asarray(
        state.get(
            "jacobian",
            profile["perception"]["hover_error_jacobian_uv_per_physical_m"],
        ),
        dtype=float,
    )
    if jacobian.shape != (2, 2) or not np.all(np.isfinite(jacobian)):
        raise ValueError("runtime alignment Jacobian must be a finite 2x2 matrix")
    best_error = float(np.linalg.norm(np.asarray(hover["error_uv"], dtype=float)))
    semantic_reacquisition_q = _q(
        state.get("last_semantic_low_q_physical_rad", executed_low_q),
        "runtime semantic reacquisition low q",
    )
    camera_hover_q = _q(
        hover.get("measured_hover_q_physical_rad", hover["hover_q_physical_rad"]),
        "runtime camera-visible hover q",
    )
    tool_frame = dict(hover["observation"]["tool_frame"])
    ToolImageFrame(**tool_frame)
    value = {
        "schema": RUNTIME_ALIGNMENT_SCHEMA,
        "canonical_hover_goal_uv": list(target["canonical_hover_goal_uv"]),
        "canonical_preclose_goal_uv": list(
            target["canonical_preclose_goal_uv"]
        ),
        "target_center_scene_xyz_m": list(localization["target_center_scene_xyz_m"]),
        "head_marker": localization["marker"],
        # This primary replay seed must retain the wrist-camera view used to
        # compute the image Jacobian.  Dense vertical descent intentionally
        # ends on another IK branch; replaying that branch hid the gripper pad
        # on the next process and made visual reacquisition impossible.
        "low_q_physical_rad": semantic_reacquisition_q.tolist(),
        "semantic_reacquisition_low_q_physical_rad": semantic_reacquisition_q.tolist(),
        "executed_descent_low_q_physical_rad": executed_low_q.tolist(),
        "hover_q_physical_rad": camera_hover_q.tolist(),
        "tool_frame": tool_frame,
        "normalized_hover_error": float(hover["normalized_center_error"]),
        "hover_observation_source": hover["observation"].get("source", "blue_marker"),
        "servo_seed": {
            "jacobian": jacobian.tolist(),
            "best_error_norm": best_error,
            "best_low_q_physical_rad": semantic_reacquisition_q.tolist(),
        },
        "source_run": str(source_run),
        "updated_at_s": time.time(),
    }
    learned_parallax = state.get("vertical_descent_target_pixel_delta_px")
    if learned_parallax is not None:
        learned_parallax = np.asarray(learned_parallax, dtype=float)
        if learned_parallax.shape != (2,) or not np.all(
            np.isfinite(learned_parallax)
        ):
            raise ValueError("learned vertical descent parallax is invalid")
        value["servo_seed"]["vertical_descent_target_pixel_delta_px"] = (
            learned_parallax.tolist()
        )
    identity_anchor = state.get("hover_identity_anchor")
    if identity_anchor is None and bool(
        hover["observation"].get("marker_cross_shaped") is True
        and hover["observation"].get("source") == "blue_marker"
    ):
        identity_anchor = {
            "center_px": hover["observation"]["center_px"],
            "component_pixels": hover["observation"]["component_pixels"],
        }
    if identity_anchor is not None:
        center = np.asarray(identity_anchor.get("center_px"), dtype=float)
        pixels = int(identity_anchor.get("component_pixels", 0))
        if center.shape != (2,) or not np.all(np.isfinite(center)) or pixels <= 0:
            raise ValueError("runtime hover identity anchor is invalid")
        value["servo_seed"]["hover_identity_anchor"] = {
            "center_px": center.tolist(),
            "component_pixels": pixels,
        }
    _atomic_json(path, value)
    return {"saved": True, "path": str(path), "value": value}


def _pending_alignment_path(profile: dict) -> Path:
    target = profile["target_identity"]
    configured = target.get("pending_runtime_alignment_calibration")
    if configured:
        return Path(configured).resolve()
    stable = Path(target["runtime_alignment_calibration"]).resolve()
    return stable.with_name(f"{stable.stem}.pending{stable.suffix}")


def _fresh_preclose_retry_servo_seed(
    state: dict | None,
    *,
    preserve_runtime_axis_calibration: bool = False,
    minimum_completed_axis_probe_span_m: float = 0.0,
) -> dict:
    """Carry calibration direction across retries, never a stale best pose.

    A failed attempt has already retreated and will rebuild the eye-in-hand
    pose.  Best/last samples are meaningful only in the previous target and
    camera frame; restoring them caused every new run to backtrack to an old
    lid location.  A completed cardinal-axis calibration is different: its
    origin and validity radius make it self-invalidating, while its high-SNR
    Jacobian should survive process boundaries.  The first fresh low image
    still establishes the new best sample.
    """

    source = {} if state is None else dict(state)
    seed = {}
    axis = source.get("runtime_axis_calibration")
    completed_axis = bool(
        isinstance(axis, dict) and axis.get("completed") is True
    )
    completed_axis_underexcited = False
    if completed_axis and minimum_completed_axis_probe_span_m > 0.0:
        probe_spans = []
        for key in ("x_delta_xy_m", "y_delta_xy_m"):
            delta = np.asarray(axis.get(key), dtype=float)
            if delta.shape == (2,) and np.all(np.isfinite(delta)):
                probe_spans.append(float(np.linalg.norm(delta)))
        completed_axis_underexcited = bool(
            len(probe_spans) != 2
            or min(probe_spans) < minimum_completed_axis_probe_span_m
        )
        if completed_axis_underexcited:
            completed_axis = False
    jacobian = (
        axis.get("jacobian_uv_per_physical_m")
        if completed_axis
        else (
            None
            if completed_axis_underexcited
            else source.get("jacobian")
        )
    )
    if jacobian is not None:
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.shape == (2, 2) and np.all(np.isfinite(jacobian)):
            seed["jacobian"] = jacobian.tolist()
            seed["adaptive_trust_from_first_observation"] = True
            if completed_axis:
                baseline_xy = np.asarray(axis.get("baseline_xy_m"), dtype=float)
                if baseline_xy.shape == (2,) and np.all(
                    np.isfinite(baseline_xy)
                ):
                    seed["runtime_axis_calibration"] = {
                        "completed": True,
                        "stage": str(axis.get("stage", "completed")),
                        "baseline_xy_m": baseline_xy.tolist(),
                        "jacobian_uv_per_physical_m": jacobian.tolist(),
                        "jacobian_condition": axis.get("jacobian_condition"),
                        "motion_condition": axis.get("motion_condition"),
                    }
                    for key in ("x_delta_xy_m", "y_delta_xy_m"):
                        delta = np.asarray(axis.get(key), dtype=float)
                        if delta.shape == (2,) and np.all(np.isfinite(delta)):
                            seed["runtime_axis_calibration"][key] = delta.tolist()
    if (
        preserve_runtime_axis_calibration
        and isinstance(axis, dict)
        and axis.get("completed") is not True
    ):
        # With level_fixed_yaw, the pending alignment stores the exact next
        # probe pose and reconstructs the same eye-in-hand frame after a
        # process restart.  Keeping the incomplete cardinal-probe witness lets
        # the first fresh image score that commanded probe instead of wasting
        # two more moves.  Other orientation modes retain the historical reset.
        seed["runtime_axis_calibration"] = json.loads(json.dumps(axis))
    learned_parallax = source.get("vertical_descent_target_pixel_delta_px")
    if learned_parallax is not None:
        learned_parallax = np.asarray(learned_parallax, dtype=float)
        if learned_parallax.shape == (2,) and np.all(
            np.isfinite(learned_parallax)
        ):
            seed["vertical_descent_target_pixel_delta_px"] = (
                learned_parallax.tolist()
            )
    if source.get("enable_runtime_axis_calibration") is True:
        seed["enable_runtime_axis_calibration"] = True
    return seed


def _load_pending_preclose_alignment(
    profile: dict, localization: dict
) -> tuple[dict | None, dict]:
    """Load an unclosed low-pose proposal without ever authorizing closure."""

    path = _pending_alignment_path(profile)
    if not path.is_file():
        return None, {"accepted": False, "reason": "pending alignment is absent", "path": str(path)}
    value = _load(path)
    if value.get("schema") != PENDING_ALIGNMENT_SCHEMA:
        return None, {"accepted": False, "reason": "pending alignment schema mismatch", "path": str(path)}
    if value.get("active") is not True:
        return None, {"accepted": False, "reason": "pending alignment is inactive", "path": str(path)}
    configured_goal = np.asarray(
        profile["target_identity"]["canonical_hover_goal_uv"], dtype=float
    )
    saved_goal = np.asarray(value.get("canonical_hover_goal_uv"), dtype=float)
    configured_preclose_goal = np.asarray(
        profile["target_identity"].get("canonical_preclose_goal_uv"),
        dtype=float,
    )
    saved_preclose_goal = np.asarray(
        value.get("canonical_preclose_goal_uv"), dtype=float
    )
    current_scene = np.asarray(localization["target_center_scene_xyz_m"], dtype=float)
    saved_scene = np.asarray(value.get("target_center_scene_xyz_m"), dtype=float)
    current_uv = np.asarray(localization["marker"]["center_uv"], dtype=float)
    saved_uv = np.asarray((value.get("head_marker") or {}).get("center_uv"), dtype=float)
    if (
        saved_goal.shape != (2,)
        or not np.allclose(saved_goal, configured_goal, atol=1e-9)
        or configured_preclose_goal.shape != (2,)
        or saved_preclose_goal.shape != (2,)
        or not np.allclose(
            saved_preclose_goal, configured_preclose_goal, atol=1e-9
        )
        or current_scene.shape != (3,)
        or saved_scene.shape != (3,)
        or current_uv.shape != (2,)
        or saved_uv.shape != (2,)
        or not np.all(np.isfinite(np.r_[current_scene, saved_scene, current_uv, saved_uv]))
    ):
        return None, {"accepted": False, "reason": "pending alignment identity is invalid", "path": str(path)}
    scene_displacement = float(np.linalg.norm(current_scene - saved_scene))
    pixel_displacement = float(np.linalg.norm(current_uv - saved_uv))
    maximum_scene = float(
        profile["target_identity"].get(
            "maximum_runtime_target_scene_displacement_m", 0.008
        )
    )
    maximum_pixel = float(
        profile["head_localization"].get(
            "maximum_fast_replay_proxy_displacement_diagonal_fraction", 0.01
        )
    )
    report = {
        "accepted": bool(
            scene_displacement <= maximum_scene
            and pixel_displacement <= maximum_pixel
        ),
        "path": str(path),
        "target_scene_displacement_m": scene_displacement,
        "maximum_target_scene_displacement_m": maximum_scene,
        "head_marker_uv_displacement": pixel_displacement,
        "maximum_head_marker_uv_displacement": maximum_pixel,
        "source_run": value.get("source_run"),
        "source_attempt": value.get("source_attempt"),
    }
    if not report["accepted"]:
        report["reason"] = "pending alignment no longer matches the tapped target"
        return None, report
    alignment = dict(value["alignment"])
    alignment["preclose_servo_state"] = _fresh_preclose_retry_servo_seed(
        alignment.get("preclose_servo_state"),
        preserve_runtime_axis_calibration=(
            profile["execution"].get("descent_samplewise_orientation_mode")
            == "level_fixed_yaw"
        ),
        minimum_completed_axis_probe_span_m=float(
            profile["perception"].get(
                "minimum_persisted_runtime_axis_probe_span_m", 0.0
            )
        ),
    )
    alignment["accepted"] = True
    alignment["aligned_low_q_physical_rad"] = _q(
        alignment["aligned_low_q_physical_rad"], "pending aligned low q"
    ).tolist()
    alignment["closure_authorized"] = False
    alignment["fresh_preclose_required"] = True
    if "source_hover_q_physical_rad" not in alignment:
        source_attempt = (
            Path(str(value.get("source_run", "")))
            / str(value.get("source_attempt", ""))
            / "attempt.json"
        )
        if source_attempt.is_file():
            source_value = _load(source_attempt)
            source_hover_q = (source_value.get("hover") or {}).get(
                "hover_q_physical_rad"
            )
            if source_hover_q is not None:
                alignment["source_hover_q_physical_rad"] = _q(
                    source_hover_q, "migrated pending source hover q"
                ).tolist()
    if alignment.get("source_hover_orientation_seed_q_physical_rad") is None:
        source_attempt = (
            Path(str(value.get("source_run", "")))
            / str(value.get("source_attempt", ""))
            / "attempt.json"
        )
        if source_attempt.is_file():
            source_value = _load(source_attempt)
            source_hover_q = (source_value.get("hover") or {}).get(
                "hover_q_physical_rad"
            )
            if source_hover_q is not None:
                alignment["source_hover_orientation_seed_q_physical_rad"] = _q(
                    source_hover_q,
                    "migrated pending hover orientation seed",
                ).tolist()
    if alignment.get("source_tool_frame") is None:
        source_attempt = (
            Path(str(value.get("source_run", "")))
            / str(value.get("source_attempt", ""))
            / "attempt.json"
        )
        if source_attempt.is_file():
            source_value = _load(source_attempt)
            source_tool_frame = (
                ((source_value.get("preclose") or {}).get("observation") or {}).get(
                    "tool_frame"
                )
            )
            if source_tool_frame is not None:
                alignment["source_tool_frame"] = dict(source_tool_frame)
    if alignment.get("source_preclose_identity_observation") is None:
        source_attempt = (
            Path(str(value.get("source_run", "")))
            / str(value.get("source_attempt", ""))
            / "attempt.json"
        )
        if source_attempt.is_file():
            source_value = _load(source_attempt)
            source_observation = (
                (source_value.get("preclose") or {}).get("observation") or {}
            )
            if (
                source_observation.get("source")
                in ("blue_marker", "local_blue_evidence_continuity")
                and source_observation.get("component_touches_border") is False
            ):
                alignment["source_preclose_identity_observation"] = dict(
                    source_observation
                )
    report["method"] = alignment.get("method")
    return alignment, report


def _save_pending_preclose_alignment(
    profile: dict,
    localization: dict,
    alignment: dict,
    *,
    source_run: str,
    source_attempt: str,
) -> dict:
    """Persist only a locally audited proposal that still needs fresh vision."""

    if (
        alignment.get("accepted") is not True
        or alignment.get("closure_authorized") is not False
        or alignment.get("fresh_preclose_required") is not True
    ):
        raise ValueError("pending alignment must require a fresh preclose gate")
    alignment = dict(alignment)
    alignment["aligned_low_q_physical_rad"] = _q(
        alignment["aligned_low_q_physical_rad"], "pending aligned low q"
    ).tolist()
    alignment["closure_authorized"] = False
    alignment["fresh_preclose_required"] = True
    value = {
        "schema": PENDING_ALIGNMENT_SCHEMA,
        "active": True,
        "canonical_hover_goal_uv": list(
            profile["target_identity"]["canonical_hover_goal_uv"]
        ),
        "canonical_preclose_goal_uv": list(
            profile["target_identity"]["canonical_preclose_goal_uv"]
        ),
        "target_center_scene_xyz_m": list(
            localization["target_center_scene_xyz_m"]
        ),
        "head_marker": localization["marker"],
        "alignment": alignment,
        "source_run": str(source_run),
        "source_attempt": str(source_attempt),
        "updated_at_s": time.time(),
    }
    path = _pending_alignment_path(profile)
    _atomic_json(path, value)
    return {"saved": True, "path": str(path), "value": value}


def _deactivate_pending_preclose_alignment(profile: dict, *, source_run: str) -> dict:
    path = _pending_alignment_path(profile)
    if not path.is_file():
        return {"deactivated": False, "reason": "pending alignment is absent"}
    value = _load(path)
    value["active"] = False
    value["consumed_by_successful_run"] = str(source_run)
    value["consumed_at_s"] = time.time()
    _atomic_json(path, value)
    return {"deactivated": True, "path": str(path)}


def _invalidate_pending_preclose_alignment(
    profile: dict, *, source_run: str, reason: str
) -> dict:
    """Invalidate a replay proposal after fresh low vision rejects identity."""

    path = _pending_alignment_path(profile)
    if not path.is_file():
        return {"invalidated": False, "reason": "pending alignment is absent"}
    value = _load(path)
    value["active"] = False
    value["invalidated_by_run"] = str(source_run)
    value["invalidation_reason"] = str(reason)
    value["invalidated_at_s"] = time.time()
    _atomic_json(path, value)
    return {"invalidated": True, "path": str(path), "reason": str(reason)}


def _pending_hover_path(profile: dict) -> Path:
    target = profile["target_identity"]
    configured = target.get("pending_hover_alignment_calibration")
    if configured:
        return Path(configured).resolve()
    stable = Path(target["runtime_alignment_calibration"]).resolve()
    return stable.with_name(f"{stable.stem}.pending_hover{stable.suffix}")


def _save_pending_hover_progress(
    profile: dict,
    localization: dict,
    progress: dict,
    servo_state: dict,
    *,
    source_run: str,
    source_attempt: str,
) -> dict:
    """Persist visual-servo progress, never a closure authorization."""

    low_q = _q(progress["low_q_physical_rad"], "pending hover low q")
    hover_q = _q(progress["hover_q_physical_rad"], "pending hover q")
    tool_frame = ToolImageFrame(**progress["tool_frame"])
    value = {
        "schema": PENDING_HOVER_SCHEMA,
        "active": True,
        "canonical_hover_goal_uv": list(
            profile["target_identity"]["canonical_hover_goal_uv"]
        ),
        "target_center_scene_xyz_m": list(
            localization["target_center_scene_xyz_m"]
        ),
        "head_marker": localization["marker"],
        "progress": {
            "method": "best_observed_hover_progress",
            "low_q_physical_rad": low_q.tolist(),
            "hover_q_physical_rad": hover_q.tolist(),
            "tool_frame": {
                "origin_px": list(tool_frame.origin_px),
                "forward_xy": list(tool_frame.forward_xy),
                "lateral_xy": list(tool_frame.lateral_xy),
                "scale_px": float(tool_frame.scale_px),
                "cyan_pixels": int(tool_frame.cyan_pixels),
                "light_pad_pixels": int(tool_frame.light_pad_pixels),
            },
            "normalized_center_error": float(
                progress["normalized_center_error"]
            ),
            "target_observation": dict(progress.get("target_observation") or {}),
            "direct_cross_verified": bool(
                progress.get("direct_cross_verified", False)
            ),
            "branch_identity_verified": bool(
                progress.get("branch_identity_verified", False)
            ),
            "servo_seed": {
                "jacobian": np.asarray(
                    servo_state.get(
                        "jacobian",
                        profile["perception"][
                            "hover_error_jacobian_uv_per_physical_m"
                        ],
                    ),
                    dtype=float,
                ).tolist(),
                "best_error_norm": float(
                    servo_state.get(
                        "best_error_norm", progress["normalized_center_error"]
                    )
                ),
                "best_low_q_physical_rad": low_q.tolist(),
            },
            "closure_authorized": False,
            "fresh_hover_observation_required": True,
        },
        "source_run": str(source_run),
        "source_attempt": str(source_attempt),
        "updated_at_s": time.time(),
    }
    path = _pending_hover_path(profile)
    _atomic_json(path, value)
    return {"saved": True, "path": str(path), "value": value}


def _load_pending_hover_progress(
    profile: dict, localization: dict
) -> tuple[dict | None, dict]:
    path = _pending_hover_path(profile)
    if not path.is_file():
        return None, {
            "accepted": False,
            "reason": "pending hover progress is absent",
            "path": str(path),
        }
    value = _load(path)
    if value.get("schema") != PENDING_HOVER_SCHEMA or value.get("active") is not True:
        return None, {
            "accepted": False,
            "reason": "pending hover progress is inactive or has wrong schema",
            "path": str(path),
        }
    current_scene = np.asarray(localization["target_center_scene_xyz_m"], dtype=float)
    saved_scene = np.asarray(value.get("target_center_scene_xyz_m"), dtype=float)
    current_uv = np.asarray(localization["marker"]["center_uv"], dtype=float)
    saved_uv = np.asarray((value.get("head_marker") or {}).get("center_uv"), dtype=float)
    configured_goal = np.asarray(
        profile["target_identity"]["canonical_hover_goal_uv"], dtype=float
    )
    saved_goal = np.asarray(value.get("canonical_hover_goal_uv"), dtype=float)
    if (
        current_scene.shape != (3,)
        or saved_scene.shape != (3,)
        or current_uv.shape != (2,)
        or saved_uv.shape != (2,)
        or saved_goal.shape != (2,)
        or not np.allclose(saved_goal, configured_goal, atol=1e-9)
        or not np.all(np.isfinite(np.r_[current_scene, saved_scene, current_uv, saved_uv]))
    ):
        return None, {
            "accepted": False,
            "reason": "pending hover identity is invalid",
            "path": str(path),
        }
    scene_displacement = float(np.linalg.norm(current_scene - saved_scene))
    pixel_displacement = float(np.linalg.norm(current_uv - saved_uv))
    maximum_scene = float(
        profile["target_identity"].get(
            "maximum_runtime_target_scene_displacement_m", 0.008
        )
    )
    maximum_pixel = float(
        profile["head_localization"].get(
            "maximum_fast_replay_proxy_displacement_diagonal_fraction", 0.01
        )
    )
    accepted = bool(
        scene_displacement <= maximum_scene and pixel_displacement <= maximum_pixel
    )
    report = {
        "accepted": accepted,
        "path": str(path),
        "target_scene_displacement_m": scene_displacement,
        "head_marker_uv_displacement": pixel_displacement,
        "source_run": value.get("source_run"),
        "source_attempt": value.get("source_attempt"),
    }
    if not accepted:
        report["reason"] = "pending hover no longer matches the tapped target"
        return None, report
    progress = dict(value["progress"])
    progress["low_q_physical_rad"] = _q(
        progress["low_q_physical_rad"], "loaded pending hover low q"
    ).tolist()
    progress["hover_q_physical_rad"] = _q(
        progress["hover_q_physical_rad"], "loaded pending hover q"
    ).tolist()
    # A wrist image can look locally plausible while belonging to the wrong
    # dish (or to a camera branch whose associated low pose has drifted far
    # from the fixed-head 3-D target).  Identity agreement in head pixels is
    # therefore necessary but not sufficient for replay.  Apply the same
    # fixed-head/FK cross-modal audit used by the stable runtime alignment
    # before any saved hover can replace the independently relocated seed.
    geometry = _runtime_alignment_geometry_audit(
        profile,
        saved_scene,
        progress["low_q_physical_rad"],
    )
    report["cross_modal_geometry"] = geometry
    if not geometry["accepted"]:
        return None, {
            **report,
            "accepted": False,
            "reason": "pending hover failed fixed-head/FK geometry audit",
        }
    tool_frame = ToolImageFrame(**progress["tool_frame"])
    progress["tool_frame"] = {
        "origin_px": list(tool_frame.origin_px),
        "forward_xy": list(tool_frame.forward_xy),
        "lateral_xy": list(tool_frame.lateral_xy),
        "scale_px": float(tool_frame.scale_px),
        "cyan_pixels": int(tool_frame.cyan_pixels),
        "light_pad_pixels": int(tool_frame.light_pad_pixels),
    }
    # The saved pose and locally learned Jacobian are useful across processes,
    # but a scalar "best" image error is not: lighting, actuator following,
    # and transparent-edge reflections can change at the same commanded q.
    # A fresh first hover must establish the new run's best anchor.
    saved_seed = dict(progress.get("servo_seed") or {})
    progress["servo_seed"] = {
        "jacobian": np.asarray(
            saved_seed.get(
                "jacobian",
                profile["perception"][
                    "hover_error_jacobian_uv_per_physical_m"
                ],
            ),
            dtype=float,
        ).tolist(),
        "adaptive_trust_from_first_observation": True,
    }
    progress["closure_authorized"] = False
    progress["fresh_hover_observation_required"] = True
    progress["target_observation"] = dict(
        progress.get("target_observation") or {}
    )
    progress["direct_cross_verified"] = bool(
        progress.get("direct_cross_verified", False)
    )
    progress["branch_identity_verified"] = bool(
        progress.get("branch_identity_verified", False)
    )
    if progress["branch_identity_verified"]:
        progress["servo_seed"]["branch_identity_verified"] = True
    return progress, report


def _deactivate_pending_hover_progress(profile: dict, *, source_run: str) -> dict:
    path = _pending_hover_path(profile)
    if not path.is_file():
        return {"deactivated": False, "reason": "pending hover progress is absent"}
    value = _load(path)
    value["active"] = False
    value["consumed_by_successful_run"] = str(source_run)
    value["consumed_at_s"] = time.time()
    _atomic_json(path, value)
    return {"deactivated": True, "path": str(path)}


def _invalidate_pending_hover_progress(
    profile: dict, *, source_run: str, reason: str
) -> dict:
    """Keep the evidence, but prevent replay of a camera view that lost identity."""

    path = _pending_hover_path(profile)
    if not path.is_file():
        return {"invalidated": False, "reason": "pending hover progress is absent"}
    value = _load(path)
    value["active"] = False
    value["invalidated_by_run"] = str(source_run)
    value["invalidation_reason"] = str(reason)
    value["invalidated_at_s"] = time.time()
    _atomic_json(path, value)
    return {"invalidated": True, "path": str(path), "reason": str(reason)}


_STALE_PRECISE_CAMERA_REPLAY_MESSAGES = (
    "no marked target exists near the accepted grasp window",
    "saved camera replay does not directly observe the blue cross",
    "no local blue-evidence target candidate is visible",
)

_BRANCH_ESCAPE_IDENTITY_ERROR = (
    "occlusion recovery changed the wrist camera branch; "
    "retry from fixed-head target identity"
)


def _identity_consistent_hover_progress(progress: dict | None) -> bool:
    """Whether a recoverable camera checkpoint still proves target identity.

    A joint-branch escape changes the wrist view enough that global blue-region
    ranking can jump to the neighbouring dish.  Replaying an earlier view is
    safe only when it descends from the explicitly verified blue cross and its
    saved observation came from direct/local tracking (not a fresh global
    semantic selection after the branch change).
    """

    if not isinstance(progress, dict) or not bool(
        progress.get("direct_cross_verified", False)
    ):
        return False
    observation = progress.get("target_observation") or {}
    source = str(observation.get("source", ""))
    return source in {
        "blue_marker",
        "local_blue_evidence_continuity",
        "persisted_direct_cross_identity_anchor",
    }


def _stale_precise_camera_replay_reason(attempt: dict) -> str | None:
    """Return why an exact saved wrist view must no longer be replayed.

    A camera-visible hover is a useful resumable checkpoint, but not a pose
    authorization.  Transparent objects or a neighbouring dish can move while
    the robot joint replay remains exact.  If the very first fresh image has no
    identity-consistent target, preserve the old evidence for audit and force a
    new head-geometry approach instead of repeatedly revisiting an occluded
    camera pose.
    """

    precise = bool(
        (attempt.get("stages", {}).get("approach") or {}).get(
            "precise_camera_replay"
        )
    )
    if not precise or attempt.get("hover_iterations"):
        return None
    error = str(attempt.get("hover_perception_or_servo_error", ""))
    for message in _STALE_PRECISE_CAMERA_REPLAY_MESSAGES:
        if message in error:
            return message
    return None


def _runtime_head_continuity_marker(profile: dict, image: np.ndarray):
    """Keep a once-tapped marker identity through transient reflections.

    The fixed head sometimes sees only a small refracted satellite of the blue
    cross on the transparent lid.  A second marked dish is nearby, so global
    area ranking can switch objects.  Within a short fixed-camera continuity
    envelope, a weak satellite proves that the old object is still present;
    retain its last direct center until the direct component returns.
    """
    target = profile["target_identity"]
    configured = target.get("runtime_alignment_calibration")
    if not configured:
        return None, {"accepted": False, "reason": "runtime alignment is disabled"}
    path = Path(configured).resolve()
    if not path.is_file():
        return None, {"accepted": False, "reason": "runtime alignment is absent"}
    value = _load(path)
    if value.get("schema") != RUNTIME_ALIGNMENT_SCHEMA:
        return None, {"accepted": False, "reason": "runtime alignment schema mismatch"}
    saved = value.get("head_marker") or {}
    previous_center = np.asarray(saved.get("center_px"), dtype=float)
    previous_pixels = int(saved.get("component_pixels", 0))
    if previous_center.shape != (2,) or not np.all(np.isfinite(previous_center)) or previous_pixels <= 0:
        return None, {"accepted": False, "reason": "runtime head marker is invalid"}
    settings = profile["head_localization"]
    reacquisition_method = "cross_shape_continuity"
    maximum_continuity = float(
        settings.get("maximum_runtime_head_continuity_diagonal_fraction", 0.02)
    )
    try:
        observed = select_relocated_target_marker(
            image,
            homography_reference_to_current=np.eye(3),
            reference_target_center_px=previous_center,
            reference_target_component_pixels=previous_pixels,
            maximum_target_displacement_diagonal_fraction=float(
                settings.get("maximum_runtime_head_continuity_diagonal_fraction", 0.02)
            ),
            minimum_component_area_scale=float(
                settings.get("minimum_runtime_head_continuity_area_scale", 0.1)
            ),
            maximum_component_area_scale=float(
                settings.get("maximum_runtime_head_continuity_area_scale", 4.0)
            ),
        )
    except ValueError as cross_error:
        try:
            observed = select_local_blue_evidence_marker(
                image,
                reference_target_center_px=previous_center,
                reference_target_component_pixels=previous_pixels,
                maximum_target_displacement_diagonal_fraction=float(
                    settings.get(
                        "maximum_runtime_head_color_continuity_diagonal_fraction",
                        0.04,
                    )
                ),
                minimum_component_area_scale=float(
                    settings.get("minimum_runtime_head_continuity_area_scale", 0.1)
                ),
                maximum_component_area_scale=float(
                    settings.get("maximum_runtime_head_continuity_area_scale", 4.0)
                ),
            )
            reacquisition_method = "local_integrated_blue_evidence"
            maximum_continuity = float(
                settings.get(
                    "maximum_runtime_head_color_continuity_diagonal_fraction",
                    0.04,
                )
            )
        except ValueError as color_error:
            return None, {
                "accepted": False,
                "reason": str(color_error),
                "cross_shape_reason": str(cross_error),
            }
    direct_minimum = float(
        settings.get("minimum_direct_runtime_head_component_area_scale", 0.5)
    )
    proxy_used = observed.area_scale_from_reference < direct_minimum
    proxy_candidate = observed if proxy_used else None
    if proxy_used:
        # A weak reflection near the last center must not hide a newly visible
        # full marker slightly farther away.  Search the wider color-
        # continuity envelope using the direct-component area floor before
        # freezing the old center as a proxy.
        try:
            direct = select_local_blue_evidence_marker(
                image,
                reference_target_center_px=previous_center,
                reference_target_component_pixels=previous_pixels,
                maximum_target_displacement_diagonal_fraction=float(
                    settings.get(
                        "maximum_runtime_head_color_continuity_diagonal_fraction",
                        0.04,
                    )
                ),
                minimum_component_area_scale=direct_minimum,
                maximum_component_area_scale=float(
                    settings.get(
                        "maximum_runtime_head_continuity_area_scale", 4.0
                    )
                ),
            )
        except ValueError:
            direct = None
        if direct is not None:
            observed = direct
            proxy_used = False
            reacquisition_method = "full_marker_supersedes_nearby_reflection"
            maximum_continuity = float(
                settings.get(
                    "maximum_runtime_head_color_continuity_diagonal_fraction",
                    0.04,
                )
            )
    marker = observed
    if proxy_used:
        height, width = image.shape[:2]
        marker = replace(
            observed,
            center_px=(float(previous_center[0]), float(previous_center[1])),
            center_uv=(
                float(previous_center[0] / width),
                float(previous_center[1] / height),
            ),
            displacement_diagonal_fraction=0.0,
        )
    return marker, {
        "accepted": True,
        "reacquisition_method": reacquisition_method,
        "proxy_used": proxy_used,
        "previous_direct_center_px": previous_center.tolist(),
        "observed_component_center_px": list(observed.center_px),
        "observed_component_area_scale": observed.area_scale_from_reference,
        "observed_component_displacement_diagonal_fraction": (
            observed.displacement_diagonal_fraction
        ),
        "maximum_continuity_diagonal_fraction": maximum_continuity,
        "rejected_proxy_component_center_px": (
            None if proxy_candidate is None else list(proxy_candidate.center_px)
        ),
        "rejected_proxy_component_area_scale": (
            None
            if proxy_candidate is None
            else proxy_candidate.area_scale_from_reference
        ),
    }


def _runtime_fast_replay_alignment(
    profile: dict, runtime_report: dict, localization: dict
) -> dict | None:
    """Authorize only coarse hover replay; never authorize gripper closure."""
    if not runtime_report.get("accepted"):
        return None
    continuity = localization.get("runtime_head_continuity") or {}
    continuity_displacement = float(
        continuity.get(
            "observed_component_displacement_diagonal_fraction", math.inf
        )
    )
    maximum_pixel_displacement = float(
        profile["head_localization"].get(
            "maximum_fast_replay_proxy_displacement_diagonal_fraction",
            0.01,
        )
    )
    if continuity.get("proxy_used") is True:
        if continuity_displacement > maximum_pixel_displacement:
            return None
    displacement = float(runtime_report["target_scene_displacement_m"])
    maximum = float(
        profile["target_identity"].get(
            "maximum_runtime_fast_replay_target_displacement_m", 0.0025
        )
    )
    # On a fixed support plane, near-identical head pixels are stronger
    # evidence than a few millimetres of tag-pose reprojection noise.  Keep the
    # local 8 mm calibration envelope, but never broaden beyond it.
    strong_fixed_head_identity = (
        continuity.get("accepted") is True
        and continuity_displacement <= maximum_pixel_displacement
    )
    if strong_fixed_head_identity:
        maximum = max(
            maximum,
            float(
                profile["target_identity"].get(
                    "maximum_runtime_target_scene_displacement_m", maximum
                )
            ),
        )
    normalized = float(runtime_report.get("normalized_hover_error", math.inf))
    gate = float(profile["perception"]["maximum_hover_center_error_scale"])
    aligned_q = runtime_report.get("aligned_low_q_physical_rad")
    if displacement > maximum or normalized > gate or aligned_q is None:
        return None
    return {
        "accepted": True,
        "method": "fixed_head_identity_plus_prior_wrist_alignment",
        "target_scene_displacement_m": displacement,
        "maximum_target_scene_displacement_m": maximum,
        "strong_fixed_head_pixel_identity": strong_fixed_head_identity,
        "head_component_displacement_diagonal_fraction": continuity_displacement,
        "prior_normalized_hover_error": normalized,
        "aligned_low_q_physical_rad": _q(
            aligned_q, "runtime fast-replay aligned low q"
        ).tolist(),
        "closure_authorized": False,
    }


def _preclose_correction_replay_alignment(
    profile: dict, attempt: dict, corrected_low_q
) -> dict | None:
    """Reuse a semantically verified low-pose correction for one next descent.

    At hover, the transparent target marker is often occluded by the gripper.
    A low-pose observation is more informative: it has already passed target
    identity, measured jaw level, and configured metric trust-region checks.
    Replaying its corrected, level joint branch avoids replacing that evidence
    with a cyan-arm false positive.  This only skips hover perception; closure
    still requires a fresh low-pose semantic and geometry check.
    """

    preclose = attempt.get("preclose") or {}
    observation = preclose.get("observation") or {}
    level = preclose.get("level") or {}
    correction = attempt.get("visual_replan") or {}
    ik = correction.get("ik") or {}
    delta_xy = np.asarray(correction.get("selected_delta_xy_m", []), dtype=float)
    target_low_xy = np.asarray(correction.get("target_low_xy_m", []), dtype=float)
    method = correction.get("method")
    ik_role = ik.get("role")
    correction_is_new_probe = bool(
        method
        in (
            "fixed_camera_physical_xy_trust_region_broyden",
            "fixed_camera_runtime_cartesian_axis_probe",
        )
        and ik_role
        in (
            "level_yaw_free_low_visual_servo",
            "level_fixed_yaw_low_visual_servo",
            "fixed_orientation_low_visual_servo",
        )
    )
    correction_is_backtrack = bool(
        method == "fixed_camera_physical_xy_broyden_backtrack"
        and ik_role
        in (
            "previously_observed_level_low_visual_servo_best",
            "fixed_orientation_low_visual_servo",
            "level_fixed_yaw_low_visual_servo",
            "level_yaw_free_low_visual_servo",
        )
    )
    correction_is_same_xy_level_retry = bool(
        method == "preclose_same_xy_level_retry"
        and ik_role
        in (
            "level_yaw_free_low_level_retry",
            "level_fixed_yaw_low_level_retry",
        )
    )
    maximum_seed_tilt_deg = float(
        profile["execution"].get(
            "maximum_pending_seed_source_tilt_deg", 3.0
        )
    )
    maximum_seed_tip_delta_m = float(
        profile["execution"].get(
            "maximum_pending_seed_source_tip_height_difference_m", 0.004
        )
    )

    def seed_level_usable(assessment) -> bool:
        return bool(
            assessment.accepted
            or (
                assessment.combined_tilt_deg <= maximum_seed_tilt_deg
                and assessment.tip_height_difference_m
                <= maximum_seed_tip_delta_m
            )
        )

    source_level_accepted = level.get("accepted") is True
    source_level_seed_usable = source_level_accepted
    if not source_level_accepted:
        corrections = (
            (attempt.get("preclose_level_correction") or {}).get("corrections")
            or []
        )
        latest_motion = (corrections[-1].get("motion") or {}) if corrections else {}
        latest_pose = latest_motion.get("final_right_ee_wxyz_xyz")
        if latest_pose is not None:
            source_assessment = assess_jaw_level(
                latest_pose,
                _load_level_reference(profile["level_config"]),
            )
            source_level_accepted = source_assessment.accepted
            source_level_seed_usable = seed_level_usable(source_assessment)
    if not source_level_accepted:
        descent_pose = (
            ((attempt.get("stages") or {}).get("descent") or {}).get(
                "final_right_ee_wxyz_xyz"
            )
        )
        if descent_pose is not None:
            source_assessment = assess_jaw_level(
                descent_pose,
                _load_level_reference(profile["level_config"]),
            )
            source_level_accepted = source_assessment.accepted
            source_level_seed_usable = seed_level_usable(source_assessment)
    if (
        preclose.get("allowed_to_close") is not False
        or observation.get("source")
        not in ("blue_marker", "local_blue_evidence_continuity")
        or bool(observation.get("component_touches_border", True))
        or not source_level_seed_usable
        or not (
            correction_is_new_probe
            or correction_is_backtrack
            or correction_is_same_xy_level_retry
        )
        or ik.get("accepted") is not True
        or delta_xy.shape != (2,)
        or not np.all(np.isfinite(delta_xy))
        or target_low_xy.shape != (2,)
        or not np.all(np.isfinite(target_low_xy))
        or float(np.linalg.norm(delta_xy))
        > float(profile["perception"]["maximum_planar_correction_m"]) + 1e-9
    ):
        return None
    grasp_window = observation.get("grasp_window") or {}
    source_error = float(grasp_window.get("normalized_center_error", math.inf))
    if not np.isfinite(source_error):
        return None
    source_hover_observation = (attempt.get("hover") or {}).get(
        "observation"
    ) or {}
    source_hover_identity_anchor = None
    if bool(
        source_hover_observation.get("source") == "blue_marker"
        and source_hover_observation.get("marker_cross_shaped") is True
    ):
        source_hover_identity_anchor = {
            "center_px": list(source_hover_observation["center_px"]),
            "component_pixels": int(
                source_hover_observation["component_pixels"]
            ),
        }
    return {
        "accepted": True,
        "method": (
            (
                "semantically_verified_preclose_axis_probe"
                if method == "fixed_camera_runtime_cartesian_axis_probe"
                else "semantically_verified_preclose_level_q_correction"
            )
            if correction_is_new_probe
            else (
                "semantically_detected_regression_backtrack"
                if correction_is_backtrack
                else "semantically_verified_same_xy_level_retry"
            )
        ),
        "source_preclose_normalized_center_error": source_error,
        "source_level_strictly_accepted": source_level_accepted,
        "source_level_seed_usable": source_level_seed_usable,
        # The metric update predicts a zero residual at the corrected pose.
        # This number is only an audit field; it never authorizes closure.
        "prior_normalized_hover_error": 0.0,
        "selected_delta_xy_m": delta_xy.tolist(),
        "target_low_xy_m": target_low_xy.tolist(),
        "aligned_low_q_physical_rad": _q(
            corrected_low_q, "preclose-corrected replay low q"
        ).tolist(),
        "source_low_q_physical_rad": _q(
            attempt["hover"]["low_q_physical_rad"],
            "source low q for preclose correction",
        ).tolist(),
        # The corrected low pose is at a different support-plane XY. Reusing
        # the old hover joints would place the wrist camera above the old
        # point, and reusing its pixel frame would silently assume the same
        # wrist orientation. The next attempt must instead plan a fresh
        # vertical hover over this low proposal and measure a fresh tool frame.
        "source_hover_q_physical_rad": None,
        "source_hover_orientation_seed_q_physical_rad": _q(
            attempt["hover"]["hover_q_physical_rad"],
            "source hover orientation seed for preclose correction",
        ).tolist(),
        "source_tool_frame": None,
        # This previous low-view mask is only a local identity anchor for the
        # next freshly captured low image. It cannot authorize closure.
        "source_preclose_identity_observation": dict(observation),
        "source_hover_identity_anchor": source_hover_identity_anchor,
        # This object is also passed directly to the next attempt in the same
        # process.  Preserve an in-progress X/Y probe there.  The disk loader
        # calls _fresh_preclose_retry_servo_seed before a later process may
        # reuse it, which is where stale best/last samples are discarded.
        "preclose_servo_state": dict(
            attempt.get("preclose_servo_state") or {}
        ),
        "planned_collision_low_q_physical_rad": _q(
            correction["corrected_q_physical_rad"],
            "preclose-corrected collision-audit q",
        ).tolist(),
        "closure_authorized": False,
        "fresh_preclose_required": True,
    }


def _minimum_jerk(fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5


def _nlerp_wxyz(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first = np.asarray(first, dtype=float).copy()
    second = np.asarray(second, dtype=float).copy()
    if float(first @ second) < 0.0:
        second *= -1.0
    result = first + float(fraction) * (second - first)
    result /= np.linalg.norm(result)
    return result


class LiveCamera:
    def __init__(self, name: str):
        self.name = str(name)
        self.stop_event = threading.Event()
        index = int(load_camera_map()[self.name])
        self.manager = USBWristCameraFeedManager(
            self.stop_event,
            device_index=index,
            label=f"codexless {self.name}",
        )
        self.last_timestamp = 0.0

    def __enter__(self):
        self.manager.start()
        self.frame(timeout_s=8.0)
        return self

    def __exit__(self, _type, _value, _traceback):
        self.manager.stop()

    def frame(self, *, timeout_s: float = 5.0, fresh_after_s: float = 0.0):
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            rgb, timestamp, _ = self.manager.get_latest_frame()
            if (
                rgb is not None
                and rgb.size
                and timestamp is not None
                and float(timestamp) > max(self.last_timestamp, fresh_after_s)
            ):
                self.last_timestamp = float(timestamp)
                rotated = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                return cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR), float(timestamp)
            time.sleep(0.03)
        raise RuntimeError(f"{self.name} camera did not produce a fresh frame")


def capture_named_camera(name: str) -> tuple[np.ndarray, float]:
    with LiveCamera(name) as camera:
        return camera.frame(timeout_s=5.0)


def _load_level_reference(path: str | Path) -> JawLevelReference:
    value = _load(path)
    fields = {
        "support_up_robot",
        "tip_baseline_ee",
        "approach_axis_ee",
        "open_tip_span_m",
        "maximum_checkpoint_tilt_deg",
        "maximum_planned_tilt_deg",
        "maximum_tip_height_difference_m",
    }
    kwargs = {field: value[field] for field in fields if field in value}
    kwargs["source"] = str(value.get("schema", path))
    return JawLevelReference(**kwargs)


def _streamer(
    profile: dict,
    rpc,
    fk,
    *,
    maximum_start_fk_position_error_m: float | None = None,
    maximum_start_fk_rotation_error_rad: float | None = None,
) -> TeleopTrajectoryStreamer:
    torque = _load(profile["torque_config"])
    execution = profile["execution"]
    return TeleopTrajectoryStreamer(
        rpc,
        fk,
        torque_limit_nm=torque["thresholds"]["right"],
        consecutive_torque_samples=int(torque.get("consecutive_samples", 5)),
        enforce_torque_stop=False,
        maximum_start_fk_position_error_m=float(
            execution.get("maximum_start_fk_position_error_m", 0.001)
            if maximum_start_fk_position_error_m is None
            else maximum_start_fk_position_error_m
        ),
        maximum_start_fk_rotation_error_rad=float(
            execution.get("maximum_start_fk_rotation_error_rad", 0.01)
            if maximum_start_fk_rotation_error_rad is None
            else maximum_start_fk_rotation_error_rad
        ),
        maximum_tracking_position_error_m=float(
            execution["maximum_tracking_position_error_m"]
        ),
        maximum_tracking_rotation_error_rad=float(
            execution["maximum_tracking_rotation_error_rad"]
        ),
        maximum_tracking_joint_error_rad=float(
            execution.get("maximum_tracking_joint_error_rad", 0.65)
        ),
        tracking_check_interval=int(execution["tracking_check_interval"]),
    )


def _execute_direct_joint_samples(
    profile,
    rpc,
    fk,
    samples,
    *,
    final_tolerance_rad: float | None = None,
    endpoint_correction_gain: float | None = None,
    maximum_endpoint_correction_rad: float | None = None,
    settle_timeout_s: float | None = None,
    accumulate_endpoint_correction: bool = False,
    require_final_convergence: bool = True,
) -> dict:
    """Stream audited physical joints with teleop's MIT/gain preparation.

    EE IK has multiple wrist branches at the low rim pose.  Re-sending the EE
    pose repeatedly stayed on the wrong branch, whereas this path reproduces
    the measured successful branch directly.  It retains the same 30 Hz
    timing, measured-pose latch, gain ramp, and observe-only torque policy.
    """

    samples = list(samples)
    if not samples:
        raise ValueError("cannot execute an empty direct-joint trajectory")
    streamer = _streamer(profile, rpc, fk)
    prepared = None
    motion_error = None
    stages = []
    maximum_joint_error = 0.0
    final_settle_error = math.inf
    final_settle_samples = 0
    final_convergence_accepted = False
    maximum_final_command_correction = 0.0
    try:
        prepared = streamer._prepare()
        start_error = float(
            np.max(
                np.abs(
                    prepared["start_q"]
                    - np.asarray(samples[0].right_q_physical_rad, dtype=float)
                )
            )
        )
        if start_error > streamer.maximum_start_joint_error_rad:
            raise TrajectoryStreamError(
                "right arm is not at direct-joint trajectory start: "
                f"max_joint_error={start_error:.3f}rad"
            )
        started = time.monotonic()
        previous_time = 0.0
        for index, sample in enumerate(samples, start=1):
            if sample.t_s <= previous_time:
                raise ValueError("direct-joint sample times must increase")
            previous_time = sample.t_s
            streamer._check_torque(f"during {sample.stage}")
            rpc.set_right_joint_target(
                np.asarray(sample.right_q_physical_rad, dtype=float),
                gripper_target=float(sample.right_gripper_open_ratio),
                preview_time=0.05,
            )
            if not stages or stages[-1] != sample.stage:
                stages.append(sample.stage)
            if index % streamer.tracking_check_interval == 0:
                measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
                error = float(
                    np.max(
                        np.abs(
                            measured
                            - np.asarray(sample.right_q_physical_rad, dtype=float)
                        )
                    )
                )
                maximum_joint_error = max(maximum_joint_error, error)
                if error > streamer.maximum_tracking_joint_error_rad:
                    raise TrajectoryStreamError(
                        "right arm stopped following direct-joint trajectory: "
                        f"joint_error={error:.3f}rad"
                    )
            remaining = started + float(sample.t_s) - time.monotonic()
            if remaining < -2.0 / CONTROL_HZ:
                raise TrajectoryStreamError(
                    f"direct-joint stream missed deadline at {sample.stage}"
                )
            if remaining > 0.0:
                time.sleep(remaining)
        final_sample = samples[-1]
        final_target = np.asarray(final_sample.right_q_physical_rad, dtype=float)
        execution = profile["execution"]
        final_tolerance = float(
            execution.get("direct_joint_final_tolerance_rad", 0.008)
            if final_tolerance_rad is None
            else final_tolerance_rad
        )
        settle_timeout = float(
            execution.get("direct_joint_settle_timeout_s", 3.0)
            if settle_timeout_s is None
            else settle_timeout_s
        )
        required_settle_samples = int(
            execution.get("direct_joint_settle_consecutive_samples", 3)
        )
        correction_gain = float(
            execution.get("direct_joint_endpoint_correction_gain", 1.0)
            if endpoint_correction_gain is None
            else endpoint_correction_gain
        )
        maximum_correction = float(
            execution.get("direct_joint_maximum_endpoint_correction_rad", 0.10)
            if maximum_endpoint_correction_rad is None
            else maximum_endpoint_correction_rad
        )
        if final_tolerance <= 0.0 or settle_timeout <= 0.0:
            raise ValueError("direct-joint final settling limits must be positive")
        if required_settle_samples <= 0:
            raise ValueError("direct-joint settle sample count must be positive")
        if correction_gain < 0.0 or maximum_correction <= 0.0:
            raise ValueError("direct-joint endpoint correction is outside range")
        settle_deadline = time.monotonic() + settle_timeout
        consecutive = 0
        correction_offset = np.zeros(6, dtype=float)
        while time.monotonic() < settle_deadline:
            streamer._check_torque(
                f"while converging at {final_sample.stage}"
            )
            measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
            measured_error = final_target - measured
            final_settle_error = float(np.max(np.abs(measured_error)))
            maximum_joint_error = max(maximum_joint_error, final_settle_error)
            final_settle_samples += 1
            if final_settle_error <= final_tolerance:
                consecutive += 1
                if consecutive >= required_settle_samples:
                    final_convergence_accepted = True
                    break
            else:
                consecutive = 0
            correction_step = correction_gain * measured_error
            if accumulate_endpoint_correction:
                # Piper's low/contact-adjacent posture has a repeatable motor
                # deadband. A proportional command leaves a steady residual;
                # integrate only at this semantic endpoint until measured
                # joints, not the command, satisfy the closure prerequisite.
                correction_offset = np.clip(
                    correction_offset + correction_step,
                    -maximum_correction,
                    maximum_correction,
                )
                correction = correction_offset
            else:
                correction = np.clip(
                    correction_step,
                    -maximum_correction,
                    maximum_correction,
                )
            maximum_final_command_correction = max(
                maximum_final_command_correction,
                float(np.max(np.abs(correction))),
            )
            rpc.set_right_joint_target(
                final_target + correction,
                gripper_target=float(final_sample.right_gripper_open_ratio),
                preview_time=0.05,
            )
            time.sleep(1.0 / CONTROL_HZ)
        else:
            if require_final_convergence:
                raise TrajectoryStreamError(
                    "right arm did not converge to direct-joint endpoint: "
                    f"joint_error={final_settle_error:.4f}rad, "
                    f"limit={final_tolerance:.4f}rad"
                )
    except BaseException as error:
        motion_error = error
        raise
    finally:
        try:
            streamer._finish()
        except BaseException as cleanup_error:
            if motion_error is None:
                raise
            note = getattr(motion_error, "add_note", None)
            if note is not None:
                note(f"direct-joint cleanup also failed: {cleanup_error!r}")
    final_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    final_ee = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    return {
        "commands_sent": True,
        "command_path": "set_right_joint_target",
        "control_hz": CONTROL_HZ,
        "sample_count": len(samples),
        "stages": stages,
        "fk_validation": prepared["fk_check"],
        "maximum_tracking_joint_error_rad": maximum_joint_error,
        "final_settle_joint_error_rad": final_settle_error,
        "final_settle_samples": final_settle_samples,
        "final_convergence_accepted": final_convergence_accepted,
        "final_convergence_required": require_final_convergence,
        "maximum_final_command_correction_rad": maximum_final_command_correction,
        "final_tolerance_rad": final_tolerance,
        "settle_timeout_s": settle_timeout,
        "endpoint_correction_gain": correction_gain,
        "endpoint_correction_mode": (
            "integral" if accumulate_endpoint_correction else "proportional"
        ),
        "torque_stop_enforced": streamer.enforce_torque_stop,
        "torque_warning_count": streamer.torque_warning_count,
        "last_torque_warning": streamer.last_torque_warning,
        "final_right_q_physical_rad": final_q.tolist(),
        "final_right_ee_wxyz_xyz": final_ee.tolist(),
        "final_right_gripper_open_ratio": float(rpc.get_right_gripper_exact()),
    }


def _joint_bounds(profile, fk, *, center_q=None, maximum_delta_rad=None):
    joint_ids = fk.solver.dof_ids
    lower = np.asarray(fk.solver.model.jnt_range[joint_ids, 0], dtype=float) + 1e-4
    upper = np.asarray(fk.solver.model.jnt_range[joint_ids, 1], dtype=float) - 1e-4
    upper[5] = min(
        upper[5],
        float(profile["head_localization"].get("maximum_joint6_rad", 2.98)),
    )
    if center_q is not None and maximum_delta_rad is not None:
        center_q = _q(center_q, "local IK center")
        maximum_delta_rad = float(maximum_delta_rad)
        lower = np.maximum(lower, center_q - maximum_delta_rad)
        upper = np.minimum(upper, center_q + maximum_delta_rad)
    return lower, upper


def _rotation_error_vector(target_rotation, measured_rotation) -> np.ndarray:
    return Rotation.from_matrix(
        np.asarray(target_rotation, dtype=float).T
        @ np.asarray(measured_rotation, dtype=float)
    ).as_rotvec()


def _plan_fixed_orientation_pose(
    profile,
    fk,
    *,
    target_position,
    orientation_q,
    seed_q,
    role: str,
) -> tuple[np.ndarray, dict]:
    """Solve one local IK branch while preserving the calibrated jaw pose.

    A merely *level* gripper can still spin around the support normal.  That
    changes the wrist-camera projection and was the cause of divergent visual
    corrections.  This solver fixes all three orientation axes and also keeps
    the solution in the calibrated local joint branch.
    """

    target_position = np.asarray(target_position, dtype=float)
    if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
        raise ValueError("fixed-orientation target must be a finite xyz vector")
    orientation_q = _q(orientation_q, f"{role} orientation q")
    seed_q = _q(seed_q, f"{role} seed q")
    target_rotation = fk.pose(orientation_q).as_matrix()[:3, :3]
    perception = profile["perception"]
    maximum_joint_delta = float(
        perception.get("maximum_fixed_orientation_joint_delta_rad", 0.65)
    )
    maximum_solution_joint_delta = float(
        perception.get(
            "maximum_fixed_orientation_solution_joint_delta_rad", 0.35
        )
    )
    lower, upper = _joint_bounds(
        profile,
        fk,
        center_q=orientation_q,
        maximum_delta_rad=maximum_joint_delta,
    )
    seed_q = np.clip(seed_q, lower + 1e-8, upper - 1e-8)

    def residual(q_physical):
        pose = fk.pose(q_physical)
        parameters = np.asarray(pose.parameters(), dtype=float)
        rotation = pose.as_matrix()[:3, :3]
        return np.concatenate(
            [
                (parameters[4:] - target_position) / float(
                    perception.get("fixed_orientation_position_scale_m", 0.001)
                ),
                _rotation_error_vector(target_rotation, rotation) / float(
                    perception.get("fixed_orientation_rotation_scale_rad", 0.03)
                ),
                0.01 * (q_physical - seed_q),
            ]
        )

    solved = least_squares(
        residual,
        # Motor feedback may settle a few 1e-4 rad beyond a nominal software
        # bound. SciPy rejects such an initial point before optimization;
        # clipping only x0 keeps every solved waypoint inside the real audit
        # bounds without changing the measured pose or widening a limit.
        np.clip(seed_q, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=500,
    )
    parameters = np.asarray(fk.pose(solved.x).parameters(), dtype=float)
    position_error = float(np.linalg.norm(parameters[4:] - target_position))
    rotation_error = float(
        np.linalg.norm(
            _rotation_error_vector(
                target_rotation,
                fk.pose(solved.x).as_matrix()[:3, :3],
            )
        )
    )
    level_reference = _load_level_reference(profile["level_config"])
    level = assess_jaw_level(parameters, level_reference, planned=True)
    collision = _trajectory_contact_audit(profile, solved.x)
    solution_joint_delta = float(
        np.max(np.abs(np.asarray(solved.x, dtype=float) - orientation_q))
    )
    accepted = bool(
        solved.success
        and position_error <= 0.001
        and rotation_error
        <= float(perception.get("maximum_fixed_orientation_error_rad", 0.05))
        and solution_joint_delta <= maximum_solution_joint_delta
        and level.accepted
        and collision["accepted"]
    )
    report = {
        "accepted": accepted,
        "role": role,
        "method": "position_plus_fixed_orientation",
        "target_position_m": target_position.tolist(),
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
        "q_physical_rad": solved.x.tolist(),
        "orientation_source_q_physical_rad": orientation_q.tolist(),
        "maximum_local_joint_delta_rad": maximum_joint_delta,
        "solution_joint_delta_rad": solution_joint_delta,
        "maximum_solution_joint_delta_rad": maximum_solution_joint_delta,
        "level": level.to_dict(),
        "collision": collision,
    }
    if not accepted:
        raise RuntimeError(f"fixed-orientation {role} pose failed offline audit: {report}")
    return np.asarray(solved.x, dtype=float), report


def _plan_level_yaw_free_pose(
    profile, fk, *, target_position, seed_q, role: str
) -> tuple[np.ndarray, dict]:
    """Reach one position while preserving jaw level and allowing only yaw."""

    seed_q = _q(seed_q, f"{role} seed q")
    target_position = np.asarray(target_position, dtype=float).reshape(3)
    level_reference = _load_level_reference(profile["level_config"])
    up = np.asarray(level_reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    angle_scale = math.sin(
        math.radians(level_reference.maximum_checkpoint_tilt_deg)
    )
    lower, upper = _joint_bounds(profile, fk)

    def residual(q_physical):
        pose = fk.pose(q_physical)
        parameters = np.asarray(pose.parameters(), dtype=float)
        rotation = pose.as_matrix()[:3, :3]
        return np.concatenate(
            [
                (parameters[4:] - target_position) / 0.002,
                (rotation[:, 1] - up) / max(angle_scale, 1e-6),
                0.003 * (q_physical - seed_q),
            ]
        )

    solved = least_squares(
        residual,
        # Real joint feedback can differ from the commanded software limit by
        # a few 1e-4 rad. Keep the measured seed for residual regularization,
        # but place SciPy's x0 just inside its strict bound requirement.
        np.clip(seed_q, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=750,
    )
    q_physical = np.asarray(solved.x, dtype=float)
    parameters = np.asarray(fk.pose(q_physical).parameters(), dtype=float)
    position_error = float(np.linalg.norm(parameters[4:] - target_position))
    level = assess_jaw_level(parameters, level_reference, planned=True)
    collision = _trajectory_contact_audit(profile, q_physical)
    accepted = bool(
        solved.success
        and position_error <= 0.001
        and level.accepted
        and collision["accepted"]
    )
    report = {
        "accepted": accepted,
        "role": role,
        "method": "position_plus_level_yaw_free",
        "target_position_m": target_position.tolist(),
        "position_error_m": position_error,
        "q_physical_rad": q_physical.tolist(),
        "level": level.to_dict(),
        "collision": collision,
    }
    if not accepted:
        raise RuntimeError(f"level-yaw-free {role} pose failed audit: {report}")
    return q_physical, report


def _plan_level_fixed_yaw_pose(
    profile, fk, *, target_position, seed_q, role: str
) -> tuple[np.ndarray, dict]:
    """Reach xyz while keeping the seed's horizontal wrist-camera yaw."""

    seed_q = _q(seed_q, f"{role} seed q")
    target_position = np.asarray(target_position, dtype=float).reshape(3)
    level_reference = _load_level_reference(profile["level_config"])
    up = np.asarray(level_reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    seed_rotation = fk.pose(seed_q).as_matrix()[:3, :3]
    target_approach = seed_rotation[:, 0]
    target_approach -= float(target_approach @ up) * up
    target_approach /= np.linalg.norm(target_approach)
    angle_scale = math.sin(
        math.radians(level_reference.maximum_checkpoint_tilt_deg)
    )
    maximum_yaw_error = math.radians(
        float(profile["perception"].get("maximum_hover_yaw_error_deg", 3.0))
    )
    lower, upper = _joint_bounds(
        profile,
        fk,
        center_q=seed_q,
        maximum_delta_rad=float(
            profile["perception"].get(
                "maximum_fixed_yaw_hover_joint_delta_rad", 0.65
            )
        ),
    )

    def residual(q_physical):
        pose = fk.pose(q_physical)
        parameters = np.asarray(pose.parameters(), dtype=float)
        rotation = pose.as_matrix()[:3, :3]
        return np.concatenate(
            [
                (parameters[4:] - target_position) / 0.002,
                (rotation[:, 1] - up) / max(angle_scale, 1e-6),
                (rotation[:, 0] - target_approach)
                / max(math.sin(maximum_yaw_error), 1e-6),
                0.003 * (q_physical - seed_q),
            ]
        )

    solved = least_squares(
        residual,
        np.clip(seed_q, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=750,
    )
    q_physical = np.asarray(solved.x, dtype=float)
    pose = fk.pose(q_physical)
    parameters = np.asarray(pose.parameters(), dtype=float)
    rotation = pose.as_matrix()[:3, :3]
    approach = rotation[:, 0]
    approach -= float(approach @ up) * up
    approach /= np.linalg.norm(approach)
    yaw_error = float(
        math.acos(float(np.clip(target_approach @ approach, -1.0, 1.0)))
    )
    position_error = float(np.linalg.norm(parameters[4:] - target_position))
    level = assess_jaw_level(parameters, level_reference, planned=True)
    collision = _trajectory_contact_audit(profile, q_physical)
    accepted = bool(
        solved.success
        and position_error <= 0.001
        and yaw_error <= maximum_yaw_error
        and level.accepted
        and collision["accepted"]
    )
    report = {
        "accepted": accepted,
        "role": role,
        "method": "position_plus_level_plus_fixed_horizontal_yaw",
        "target_position_m": target_position.tolist(),
        "position_error_m": position_error,
        "yaw_error_rad": yaw_error,
        "maximum_yaw_error_rad": maximum_yaw_error,
        "q_physical_rad": q_physical.tolist(),
        "level": level.to_dict(),
        "collision": collision,
    }
    if not accepted:
        raise RuntimeError(f"level-fixed-yaw {role} pose failed audit: {report}")
    return q_physical, report


def _plan_common_level_vertical_endpoints(
    profile,
    fk,
    *,
    hover_position,
    low_position,
    hover_seed_q,
    low_seed_q,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Find level hover/low endpoints with one identical camera orientation.

    Piper can represent the same Cartesian wrist orientation with very
    different joint-6 values.  Solving the two heights independently therefore
    produces a drifting hand-camera projection, and constraining either solve
    to the other endpoint's *joint-local* branch can incorrectly report the
    shared orientation unreachable.  This coupled solve uses the full audited
    joint limits and requires the two endpoint rotations to agree directly.
    """

    hover_position = np.asarray(hover_position, dtype=float).reshape(3)
    low_position = np.asarray(low_position, dtype=float).reshape(3)
    hover_seed_q = _q(hover_seed_q, "common-orientation hover seed")
    low_seed_q = _q(low_seed_q, "common-orientation low seed")
    reference = _load_level_reference(profile["level_config"])
    up = np.asarray(reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    execution = profile["execution"]
    position_scale = float(
        execution.get("descent_common_orientation_position_scale_m", 0.0005)
    )
    level_scale = float(
        execution.get("descent_common_orientation_level_scale_rad", 0.02)
    )
    rotation_scale = float(
        execution.get("descent_common_orientation_rotation_scale_rad", 0.01)
    )
    continuity_weight = float(
        execution.get("descent_common_orientation_continuity_weight", 0.0002)
    )
    path_span_weight = float(
        execution.get("descent_common_orientation_path_span_weight", 0.01)
    )
    lower, upper = _joint_bounds(profile, fk)
    stacked_lower = np.concatenate((lower, lower))
    stacked_upper = np.concatenate((upper, upper))
    initial = np.concatenate((hover_seed_q, low_seed_q))

    def residual(stacked_q):
        hover_q = stacked_q[:6]
        low_q = stacked_q[6:]
        hover_pose = fk.pose(hover_q)
        low_pose = fk.pose(low_q)
        hover_parameters = np.asarray(hover_pose.parameters(), dtype=float)
        low_parameters = np.asarray(low_pose.parameters(), dtype=float)
        hover_rotation = hover_pose.as_matrix()[:3, :3]
        low_rotation = low_pose.as_matrix()[:3, :3]
        return np.concatenate(
            [
                (hover_parameters[4:] - hover_position) / position_scale,
                (low_parameters[4:] - low_position) / position_scale,
                (hover_rotation[:, 1] - up) / level_scale,
                (low_rotation[:, 1] - up) / level_scale,
                _rotation_error_vector(hover_rotation, low_rotation)
                / rotation_scale,
                continuity_weight * (hover_q - hover_seed_q),
                continuity_weight * (low_q - low_seed_q),
                path_span_weight * (low_q - hover_q),
            ]
        )

    solved = least_squares(
        residual,
        np.clip(initial, stacked_lower + 1e-8, stacked_upper - 1e-8),
        bounds=(stacked_lower, stacked_upper),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=int(
            execution.get("descent_common_orientation_max_nfev", 2500)
        ),
    )
    hover_q = np.asarray(solved.x[:6], dtype=float)
    low_q = np.asarray(solved.x[6:], dtype=float)
    hover_pose = fk.pose(hover_q)
    low_pose = fk.pose(low_q)
    hover_parameters = np.asarray(hover_pose.parameters(), dtype=float)
    low_parameters = np.asarray(low_pose.parameters(), dtype=float)
    position_errors = [
        float(np.linalg.norm(hover_parameters[4:] - hover_position)),
        float(np.linalg.norm(low_parameters[4:] - low_position)),
    ]
    rotation_error = float(
        np.linalg.norm(
            _rotation_error_vector(
                hover_pose.as_matrix()[:3, :3],
                low_pose.as_matrix()[:3, :3],
            )
        )
    )
    hover_level = assess_jaw_level(hover_parameters, reference, planned=True)
    low_level = assess_jaw_level(low_parameters, reference, planned=True)
    hover_contact = _trajectory_contact_audit(profile, hover_q)
    low_contact = _trajectory_contact_audit(profile, low_q)
    maximum_position_error = float(
        execution.get("descent_maximum_ik_position_error_m", 0.0003)
    )
    maximum_rotation_error = float(
        execution.get("descent_maximum_ik_rotation_error_rad", 0.003)
    )
    accepted = bool(
        solved.success
        and max(position_errors) <= maximum_position_error
        and rotation_error <= maximum_rotation_error
        and hover_level.accepted
        and low_level.accepted
        and hover_contact["accepted"]
        and low_contact["accepted"]
    )
    report = {
        "accepted": accepted,
        "method": "coupled_level_identical_orientation_endpoints",
        "position_errors_m": position_errors,
        "rotation_error_rad": rotation_error,
        "maximum_endpoint_joint_span_rad": float(
            np.max(np.abs(low_q - hover_q))
        ),
        "maximum_position_error_m": maximum_position_error,
        "maximum_rotation_error_rad": maximum_rotation_error,
        "hover_q_physical_rad": hover_q.tolist(),
        "low_q_physical_rad": low_q.tolist(),
        "hover_level": hover_level.to_dict(),
        "low_level": low_level.to_dict(),
        "hover_contact": hover_contact,
        "low_contact": low_contact,
    }
    if not accepted:
        raise RuntimeError(
            "common level vertical endpoints failed offline audit: "
            f"{report}"
        )
    return hover_q, low_q, report


def _plan_level_vertical_offset(
    profile, fk, low_q, clearance_m: float, *, seed_hover_q=None
) -> tuple[np.ndarray, dict]:
    """Plan a canonical-camera hover directly above a low grasp pose.

    Target acquisition must keep the demonstrated wrist-camera direction.
    The later descent planner is responsible for rotating, while still in
    free space, to the low pose's feasible level orientation before moving
    vertically.  Reusing an arbitrary yaw-free low IK orientation here can
    turn the camera away from the object and make visual servo self-blind.
    """

    low_q = _q(low_q, "low q for vertical offset")
    low_parameters = np.asarray(fk.pose(low_q).parameters(), dtype=float)
    level_reference = _load_level_reference(profile["level_config"])
    up = np.asarray(level_reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    target_position = low_parameters[4:] + float(clearance_m) * up
    canonical_q = _q(
        profile["trajectory"]["canonical_hover_q_physical_rad"],
        "canonical hover q",
    )
    try:
        solved, report = _plan_fixed_orientation_pose(
            profile,
            fk,
            target_position=target_position,
            orientation_q=canonical_q,
            seed_q=canonical_q if seed_hover_q is None else seed_hover_q,
            role="canonical_hover",
        )
    except RuntimeError as fixed_error:
        if seed_hover_q is not None:
            try:
                solved, report = _plan_level_fixed_yaw_pose(
                    profile,
                    fk,
                    target_position=target_position,
                    seed_q=seed_hover_q,
                    role="level_fixed_yaw_hover",
                )
            except RuntimeError as fixed_yaw_error:
                # At a previously unseen workspace location q6 can reach its
                # physical limit before the old camera yaw is reproducible.
                # The grasp error is expressed in the detected tool frame, so
                # horizontal yaw is not part of the closure geometry.  Keep
                # the jaws level and continue locally instead of abandoning a
                # target that is already visible at the image border.
                solved, report = _plan_level_yaw_free_pose(
                    profile,
                    fk,
                    target_position=target_position,
                    seed_q=low_q,
                    role="level_yaw_free_hover_after_fixed_yaw_limit",
                )
                report["fixed_yaw_error"] = str(fixed_yaw_error)
                report["orientation_mode_change"] = (
                    "horizontal_yaw_released_at_joint_limit"
                )
        else:
            solved, report = _plan_level_yaw_free_pose(
                profile,
                fk,
                target_position=target_position,
                seed_q=low_q,
                role="level_yaw_free_hover",
            )
        report["fixed_orientation_error"] = str(fixed_error)
    lower, upper = _joint_bounds(profile, fk)
    joint6_margin = float(
        profile["head_localization"].get(
            "joint6_hover_limit_avoidance_margin_rad", 0.03
        )
    )
    if (
        bool(
            profile["head_localization"].get(
                "release_hover_yaw_near_joint6_limit", True
            )
        )
        and min(solved[5] - lower[5], upper[5] - solved[5]) <= joint6_margin
    ):
        saturated_report = report
        solved, report = _plan_level_yaw_free_pose(
            profile,
            fk,
            target_position=target_position,
            seed_q=low_q,
            role="level_yaw_free_hover_away_from_joint6_limit",
        )
        report["joint_limit_avoidance"] = {
            "joint": 6,
            "configured_margin_rad": joint6_margin,
            "rejected_q_physical_rad": np.asarray(
                saturated_report["q_physical_rad"], dtype=float
            ).tolist(),
            "rejected_plan_method": saturated_report.get("method"),
        }
        report["orientation_mode_change"] = (
            "horizontal_yaw_released_to_avoid_joint6_limit"
        )
    report["clearance_m"] = float(clearance_m)
    return solved, report


def _audit_camera_visible_hover_seed(profile, fk, *, low_q, hover_q) -> dict:
    """Validate an exact prior wrist-camera viewpoint for visual replay."""

    low_q = _q(low_q, "camera-visible hover low q")
    hover_q = _q(hover_q, "camera-visible hover q")
    reference = _load_level_reference(profile["level_config"])
    up = np.asarray(reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    low_pose = np.asarray(fk.pose(low_q).parameters(), dtype=float)
    hover_pose = np.asarray(fk.pose(hover_q).parameters(), dtype=float)
    delta = hover_pose[4:] - low_pose[4:]
    height = float(delta @ up)
    planar = delta - height * up
    planar_distance = float(np.linalg.norm(planar))
    hover_level = assess_jaw_level(hover_pose, reference, planned=False)
    execution = profile["execution"]
    nominal = float(profile["trajectory"]["verification_lift_m"])
    minimum_height = float(
        execution.get("camera_replay_minimum_clearance_m", 0.010)
    )
    maximum_height = float(
        execution.get("camera_replay_maximum_clearance_m", 0.080)
    )
    maximum_planar = float(
        execution.get("camera_replay_maximum_planar_offset_m", 0.015)
    )
    maximum_start_tilt = float(
        execution.get("camera_replay_maximum_tilt_deg", 12.0)
    )
    maximum_start_tip_delta = float(
        execution.get("camera_replay_maximum_tip_height_difference_m", 0.025)
    )
    collision = _trajectory_contact_audit(profile, hover_q)
    accepted = bool(
        minimum_height <= height <= maximum_height
        and planar_distance <= maximum_planar
        and hover_level.combined_tilt_deg <= maximum_start_tilt
        and hover_level.tip_height_difference_m <= maximum_start_tip_delta
        and collision["accepted"]
    )
    report = {
        "accepted": accepted,
        "role": "persisted_camera_visible_hover",
        "method": "exact_runtime_hover_replay_before_fresh_visual_servo",
        "q_physical_rad": hover_q.tolist(),
        "height_above_low_m": height,
        "nominal_height_m": nominal,
        "minimum_camera_clearance_m": minimum_height,
        "maximum_camera_clearance_m": maximum_height,
        "maximum_camera_planar_offset_m": maximum_planar,
        "maximum_camera_tilt_deg": maximum_start_tilt,
        "maximum_camera_tip_height_difference_m": maximum_start_tip_delta,
        "planar_offset_from_low_m": planar_distance,
        "hover_level": hover_level.to_dict(),
        "collision": collision,
        "closure_authorized": False,
        "fresh_wrist_alignment_required": True,
        "descent_level_authorized": False,
    }
    if not accepted:
        raise RuntimeError(f"persisted camera-visible hover audit rejected: {report}")
    return report


def _joint_approach(
    profile,
    rpc,
    fk,
    hover_q,
    *,
    precise_camera_replay: bool = False,
) -> dict:
    current = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    trajectory = profile["trajectory"]
    safe = _q(trajectory["safe_high_q_physical_rad"], "safe q")
    samples = sample_joint_knots(
        [
            {
                "stage": "measured_start",
                "right_q_physical_rad": current.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "normalize_safe_high",
                "right_q_physical_rad": safe.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": float(trajectory["normalize_to_safe_duration_s"]),
            },
            {
                "stage": "transit_to_hover",
                "right_q_physical_rad": np.asarray(hover_q, dtype=float).tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": float(trajectory["descend_duration_s"]),
            },
            {
                "stage": "settle_at_fixed_orientation_hover",
                "right_q_physical_rad": np.asarray(hover_q, dtype=float).tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": float(
                    trajectory.get("hover_settle_s", 1.0)
                ),
            },
        ]
    )
    execution = profile["execution"]
    return _execute_direct_joint_samples(
        profile,
        rpc,
        fk,
        samples,
        final_tolerance_rad=float(
            execution.get(
                "persisted_hover_replay_final_tolerance_rad", 0.02
            )
            if precise_camera_replay
            else execution.get(
                "hover_visual_final_tolerance_rad", 0.02
            )
        ),
        endpoint_correction_gain=float(
            execution.get(
                "hover_visual_endpoint_correction_gain", 1.0
            )
        ),
        maximum_endpoint_correction_rad=float(
            execution.get(
                "persisted_hover_replay_maximum_endpoint_correction_rad", 0.12
            )
        ) if precise_camera_replay else None,
        settle_timeout_s=float(
            execution.get("persisted_hover_replay_settle_timeout_s", 5.0)
        ) if precise_camera_replay else None,
        accumulate_endpoint_correction=precise_camera_replay,
    )


def _transition_to_level_hover(profile, rpc, fk, low_q) -> dict:
    """Move an aligned free-space camera view onto a descent-ready level hover."""

    low_q = _q(low_q, "aligned low q before level-hover transition")
    current_q = _q(rpc.get_right_joint_positions(), "current camera-view q")
    level_hover_q, plan = _plan_level_vertical_offset(
        profile,
        fk,
        low_q,
        float(profile["trajectory"]["verification_lift_m"]),
        seed_hover_q=current_q,
    )
    sample_count = int(
        profile["execution"].get("level_hover_transition_audit_samples", 121)
    )
    q_path = np.linspace(current_q, level_hover_q, sample_count)
    collision = _right_joint_path_contact_audit(profile, q_path)
    if not collision["accepted"]:
        raise RuntimeError(
            f"camera-to-level hover transition collision audit rejected: {collision}"
        )
    samples = sample_joint_knots(
        [
            {
                "stage": "measured_camera_view_start",
                "right_q_physical_rad": current_q.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "transition_to_level_hover",
                "right_q_physical_rad": level_hover_q.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": float(
                    profile["execution"].get(
                        "level_hover_transition_duration_s", 2.0
                    )
                ),
            },
            {
                "stage": "settle_at_level_hover",
                "right_q_physical_rad": level_hover_q.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": float(
                    profile["execution"].get(
                        "level_hover_transition_settle_s", 0.8
                    )
                ),
            },
        ]
    )
    execution = _execute_direct_joint_samples(
        profile,
        rpc,
        fk,
        samples,
        final_tolerance_rad=float(
            profile["execution"].get(
                "level_hover_transition_final_tolerance_rad", 0.008
            )
        ),
        endpoint_correction_gain=float(
            profile["execution"].get(
                "level_hover_transition_endpoint_correction_gain", 1.0
            )
        ),
        maximum_endpoint_correction_rad=float(
            profile["execution"].get(
                "level_hover_transition_maximum_endpoint_correction_rad", 0.12
            )
        ),
        settle_timeout_s=float(
            profile["execution"].get(
                "level_hover_transition_settle_timeout_s", 5.0
            )
        ),
        accumulate_endpoint_correction=True,
    )
    reference = _load_level_reference(profile["level_config"])
    measured_pose = _pose(
        rpc.get_right_ee_pose().parameters(), "measured transitioned level hover"
    )
    measured_level = assess_jaw_level(measured_pose, reference, planned=False)
    if not measured_level.accepted:
        raise RuntimeError(
            "transitioned hover is not descent-level: "
            f"{measured_level.to_dict()}"
        )
    return {
        "accepted": True,
        "method": "collision_audited_direct_joint_level_hover_transition",
        "plan": plan,
        "collision": collision,
        "execution": execution,
        "measured_level": measured_level.to_dict(),
        "measured_pose_wxyz_xyz": measured_pose.tolist(),
        "level_hover_q_physical_rad": level_hover_q.tolist(),
        "closure_authorized": False,
        "fresh_hover_observation_required": True,
    }


def _descend_from_hover(profile, rpc, fk, low_q) -> dict:
    """Descend vertically with an offline-audited, level joint trajectory.

    The teleop-equivalent Cartesian controller reaches the requested position,
    but the physical wrist can settle on a different IK branch and leave the
    two fingertips at different heights.  Plan the same fixed-pose motion
    densely in joint space, audit every sample, then stream those samples via
    the same 30 Hz MIT/gain path used by teleoperation.  Level is checked once
    before and once after descent; there is no slow 30 Hz visual or pose gate.
    """

    low_q = _q(low_q, "vertical descent low q")
    level_reference = _load_level_reference(profile["level_config"])
    up = np.asarray(level_reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    clearance = float(profile["trajectory"]["verification_lift_m"])
    if clearance <= 0.0:
        raise ValueError("vertical descent clearance must be positive")
    start_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    start_level = assess_jaw_level(start_pose, level_reference, planned=False)
    if not start_level.accepted:
        raise RuntimeError(
            f"measured Cartesian descent start is not level: {start_level.to_dict()}"
        )
    start_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    try:
        samples, audit = _plan_level_vertical_joint_descent(
            profile,
            fk,
            start_q=start_q,
            reference_low_q=low_q,
        )
    except Exception as error:
        raise DescentPlanRejected(str(error)) from error
    result = _execute_direct_joint_samples(
        profile,
        rpc,
        fk,
        samples,
        final_tolerance_rad=float(
            profile["execution"].get(
                "descent_direct_joint_final_tolerance_rad", 0.008
            )
        ),
        endpoint_correction_gain=float(
            profile["execution"].get(
                "descent_direct_joint_endpoint_correction_gain", 1.0
            )
        ),
        maximum_endpoint_correction_rad=float(
            profile["execution"].get(
                "descent_direct_joint_maximum_endpoint_correction_rad", 0.03
            )
        ),
        accumulate_endpoint_correction=bool(
            profile["execution"].get(
                "descent_direct_joint_integral_endpoint_correction", True
            )
        ),
        # Joint endpoint error is only a controller diagnostic here.  Closure
        # is authorized later from a fresh wrist image plus measured FK gates
        # for jaw level and support height.  Requiring the Piper motor proxy to
        # settle first caused safe, inspectable poses to be abandoned before
        # those more direct checks could run.
        require_final_convergence=bool(
            profile["execution"].get(
                "descent_require_final_joint_convergence", False
            )
        ),
    )
    if float(profile["trajectory"].get("preclose_settle_s", 0.0)) > 0.0:
        time.sleep(float(profile["trajectory"]["preclose_settle_s"]))
    result["offline_vertical_audit"] = audit
    result["planned_low_q_physical_rad"] = np.asarray(
        samples[-1].right_q_physical_rad, dtype=float
    ).tolist()
    return result


def _right_joint_path_contact_audit(profile: dict, q_path) -> dict:
    """Check the exact physical-right joint path in the calibrated scene."""

    import mujoco

    q_path = np.asarray(q_path, dtype=float)
    if q_path.ndim != 2 or q_path.shape[1] != 6 or not np.all(np.isfinite(q_path)):
        raise ValueError("right joint path must be a finite Nx6 array")
    if len(q_path) < 2:
        raise ValueError("right joint path must contain at least two samples")
    model = mujoco.MjModel.from_xml_path(
        str(Path(profile["planning_model"]).resolve())
    )
    data = mujoco.MjData(model)
    right_ids = np.asarray(
        [model.joint(f"right/joint{index}").qposadr[0] for index in range(1, 7)],
        dtype=int,
    )
    left_ids = np.asarray(
        [model.joint(f"left/joint{index}").qposadr[0] for index in range(1, 7)],
        dtype=int,
    )
    data.qpos[left_ids] = physical_home_q(
        "left"
    ) + physical_to_semantic_model_q_offset("left")
    right_offset = physical_to_semantic_model_q_offset("right")
    disallowed = set()
    expected_support = set()
    first_disallowed_sample = None
    for sample_index, q_physical in enumerate(q_path):
        data.qpos[right_ids] = q_physical + right_offset
        mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = model.geom(int(contact.geom1))
            geom2 = model.geom(int(contact.geom2))
            body1 = model.body(int(geom1.bodyid[0])).name
            body2 = model.body(int(geom2.bodyid[0])).name
            if not (body1.startswith("right/") or body2.startswith("right/")):
                continue
            pair = tuple(sorted((geom1.name, geom2.name)))
            text = " ".join((*pair, body1, body2))
            if "right/nyu_gripper_collision" in text and "support-platform" in text:
                expected_support.add(pair)
            else:
                disallowed.add(pair)
                if first_disallowed_sample is None:
                    first_disallowed_sample = sample_index
    return {
        "accepted": not disallowed,
        "sample_count": int(len(q_path)),
        "first_disallowed_sample": first_disallowed_sample,
        "expected_target_support_contacts": [
            list(pair) for pair in sorted(expected_support)
        ],
        "non_support_contacts": [list(pair) for pair in sorted(disallowed)],
    }


def _signed_tip_height_difference_m(
    pose_wxyz_xyz, reference: JawLevelReference
) -> float:
    """Return signed left/right tip height for controller-bias compensation."""

    pose = _pose(pose_wxyz_xyz, "signed tip-height pose")
    rotation = mink.SE3(pose).as_matrix()[:3, :3]
    up = np.asarray(reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    baseline_ee = np.asarray(reference.tip_baseline_ee, dtype=float)
    baseline_ee /= np.linalg.norm(baseline_ee)
    return float(reference.open_tip_span_m * ((rotation @ baseline_ee) @ up))


def _next_preclose_tip_height_bias_m(
    *,
    current_command_m: float,
    measured_error_m: float,
    maximum_bias_m: float,
    fallback_gain: float,
    previous_command_m: float | None = None,
    previous_error_m: float | None = None,
) -> tuple[float, str]:
    """Choose the next roll target from measured fingertip-height response.

    Motor deadband makes a nominal one-millimetre roll target produce a
    different physical fingertip response on every settle.  Once two measured
    command/error pairs exist, a bounded secant step estimates the zero
    crossing.  The first step (or a degenerate secant) uses residual feedback.
    """

    current = float(current_command_m)
    error = float(measured_error_m)
    maximum = float(maximum_bias_m)
    if not all(math.isfinite(value) for value in (current, error, maximum)):
        raise ValueError("preclose tip-height bias inputs must be finite")
    if maximum <= 0:
        raise ValueError("maximum preclose tip-height bias must be positive")
    candidate = None
    method = "bounded_residual"
    if previous_command_m is not None and previous_error_m is not None:
        previous_command = float(previous_command_m)
        previous_error = float(previous_error_m)
        denominator = error - previous_error
        command_delta = current - previous_command
        if (
            math.isfinite(previous_command)
            and math.isfinite(previous_error)
            and abs(denominator) >= 1e-6
            and abs(command_delta) >= 1e-6
        ):
            secant = current - error * command_delta / denominator
            if math.isfinite(secant):
                candidate = secant
                method = "bounded_secant_zero_crossing"
    if candidate is None:
        candidate = current - float(fallback_gain) * error
    return float(np.clip(candidate, -maximum, maximum)), method


def _plan_same_position_level_joint_samples(
    profile,
    fk,
    *,
    start_q,
    target_signed_tip_height_difference_m: float = 0.0,
    target_orientation_q=None,
) -> tuple[list[JointTrajectorySample], dict]:
    """Level the open jaws without translating, optionally biasing tip roll.

    The optional signed target compensates repeatable motor deadband.  It is
    never itself a closure authorization; a fresh measured strict checkpoint
    still follows execution.
    """

    start_q = _q(start_q, "same-position level start q")
    trajectory = profile["trajectory"]
    execution = profile["execution"]
    reference = _load_level_reference(profile["level_config"])
    start_pose = np.asarray(fk.pose(start_q).parameters(), dtype=float)
    start_assessment = assess_jaw_level(start_pose, reference, planned=False)
    maximum_start_tilt = float(
        execution.get("descent_maximum_prelevel_start_tilt_deg", 5.0)
    )
    maximum_start_tip_delta = float(
        execution.get("descent_maximum_prelevel_start_tip_delta_m", 0.012)
    )
    if (
        start_assessment.combined_tilt_deg > maximum_start_tilt
        or start_assessment.tip_height_difference_m > maximum_start_tip_delta
    ):
        raise RuntimeError(
            "hover is too tilted for a free-space level correction: "
            f"{start_assessment.to_dict()}"
        )
    target_tip_bias = float(target_signed_tip_height_difference_m)
    maximum_target_tip_bias = float(
        execution.get("preclose_maximum_commanded_tip_height_bias_m", 0.001)
    )
    if (
        not math.isfinite(target_tip_bias)
        or abs(target_tip_bias) > maximum_target_tip_bias
    ):
        raise ValueError(
            "same-position target tip-height bias is outside the configured range"
        )
    if target_orientation_q is None:
        target_pose = leveled_pose(start_pose, reference)
        target_orientation_source = "level_start_pose_preserve_yaw"
    else:
        if abs(target_tip_bias) > 0.0:
            raise ValueError(
                "an explicit target orientation cannot be combined with a "
                "tip-height bias"
            )
        target_orientation_q = _q(
            target_orientation_q, "same-position target orientation q"
        )
        target_pose = start_pose.copy()
        target_pose[:4] = np.asarray(
            fk.pose(target_orientation_q).parameters(), dtype=float
        )[:4]
        target_orientation_source = "explicit_reference_low_orientation"
    if abs(target_tip_bias) > 0.0:
        level_rotation = mink.SE3(target_pose).as_matrix()[:3, :3]
        up = np.asarray(reference.support_up_robot, dtype=float)
        up /= np.linalg.norm(up)
        approach = level_rotation @ np.asarray(reference.approach_axis_ee, dtype=float)
        approach -= up * float(approach @ up)
        approach /= np.linalg.norm(approach)
        horizontal_baseline = np.cross(approach, up)
        horizontal_baseline /= np.linalg.norm(horizontal_baseline)
        sine = float(
            np.clip(
                target_tip_bias / reference.open_tip_span_m,
                -0.25,
                0.25,
            )
        )
        baseline = (
            math.sqrt(max(0.0, 1.0 - sine * sine)) * horizontal_baseline
            + sine * up
        )
        baseline /= np.linalg.norm(baseline)
        ee_up = np.cross(baseline, approach)
        ee_up /= np.linalg.norm(ee_up)
        desired = np.column_stack((approach, ee_up, baseline))
        xyzw = Rotation.from_matrix(desired).as_quat()
        target_pose[:4] = xyzw[[3, 0, 1, 2]]
    duration = float(trajectory.get("descent_prelevel_duration_s", 0.6))
    count = max(1, int(math.ceil(duration * CONTROL_HZ)))
    if target_orientation_q is not None:
        # Minimum jerk peaks at 1.875x its average speed.  Allocate enough
        # free-space samples that even a redundant wrist representation can
        # rotate to the common descent orientation without a large joint step.
        maximum_step = float(
            execution.get("descent_maximum_joint_step_rad", 0.06)
        ) * 0.9
        estimated_span = float(
            np.max(np.abs(target_orientation_q - start_q))
        )
        count = max(
            count,
            int(math.ceil(1.875 * estimated_span / maximum_step)),
        )
    lower, upper = _joint_bounds(
        profile,
        fk,
        center_q=start_q,
        maximum_delta_rad=float(
            execution.get("descent_maximum_local_joint_delta_rad", 1.5)
        ),
    )
    position_scale = float(execution.get("descent_ik_position_scale_m", 0.0001))
    rotation_scale = float(
        execution.get("descent_ik_rotation_scale_rad", 0.001)
    )
    continuity_weight = float(
        execution.get("descent_ik_continuity_weight", 0.01)
    )
    q_path = [start_q.copy()]
    pose_path = [start_pose.copy()]
    samples = []
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0
    for index in range(1, count + 1):
        fraction = _minimum_jerk(index / count)
        desired_pose = start_pose.copy()
        desired_pose[:4] = _nlerp_wxyz(
            start_pose[:4], target_pose[:4], fraction
        )
        desired_rotation = mink.SE3(desired_pose).as_matrix()[:3, :3]
        previous_q = q_path[-1]
        lower, upper = _joint_bounds(
            profile,
            fk,
            center_q=previous_q,
            maximum_delta_rad=float(
                execution.get("descent_maximum_joint_step_rad", 0.06)
            )
            * 0.9,
        )

        def residual(q_physical):
            pose = fk.pose(q_physical)
            parameters = np.asarray(pose.parameters(), dtype=float)
            return np.concatenate(
                [
                    (parameters[4:] - start_pose[4:]) / position_scale,
                    _rotation_error_vector(
                        desired_rotation, pose.as_matrix()[:3, :3]
                    )
                    / rotation_scale,
                    continuity_weight * (q_physical - previous_q),
                ]
            )

        solved = least_squares(
            residual,
            np.clip(previous_q, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            max_nfev=int(execution.get("descent_ik_max_nfev", 150)),
        )
        if not solved.success:
            raise RuntimeError(
                f"same-position level IK failed at sample {index}: {solved.message}"
            )
        solved_q = np.asarray(solved.x, dtype=float)
        solved_pose = np.asarray(fk.pose(solved_q).parameters(), dtype=float)
        maximum_position_error = max(
            maximum_position_error,
            float(np.linalg.norm(solved_pose[4:] - start_pose[4:])),
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            float(
                np.linalg.norm(
                    _rotation_error_vector(
                        desired_rotation,
                        fk.pose(solved_q).as_matrix()[:3, :3],
                    )
                )
            ),
        )
        q_path.append(solved_q)
        pose_path.append(solved_pose)
        samples.append(
            JointTrajectorySample(
                t_s=index / CONTROL_HZ,
                stage="level_jaws_above_target",
                right_q_physical_rad=solved_q.copy(),
                right_gripper_open_ratio=1.0,
            )
        )
    q_path_array = np.asarray(q_path, dtype=float)
    pose_path_array = np.asarray(pose_path, dtype=float)
    final_level = assess_jaw_level(
        pose_path_array[-1], reference, planned=True
    )
    maximum_planar_motion = float(
        np.max(
            np.linalg.norm(
                pose_path_array[:, 4:6] - pose_path_array[0, 4:6], axis=1
            )
        )
    )
    maximum_height_motion = float(
        np.max(np.abs(pose_path_array[:, 6] - pose_path_array[0, 6]))
    )
    maximum_same_position_motion = float(
        execution.get("descent_prelevel_maximum_position_motion_m", 0.0003)
    )
    maximum_joint_step = float(np.max(np.abs(np.diff(q_path_array, axis=0))))
    contact = _right_joint_path_contact_audit(profile, q_path_array)
    accepted = bool(
        maximum_position_error
        <= float(execution.get("descent_maximum_ik_position_error_m", 0.0003))
        and maximum_rotation_error
        <= float(
            execution.get("descent_maximum_ik_rotation_error_rad", 0.003)
        )
        and maximum_planar_motion <= maximum_same_position_motion
        and maximum_height_motion <= maximum_same_position_motion
        and maximum_joint_step
        <= float(execution.get("descent_maximum_joint_step_rad", 0.06))
        and final_level.accepted
        and contact["accepted"]
    )
    audit = {
        "accepted": accepted,
        "method": (
            "dense_same_position_joint_orientation_alignment"
            if target_orientation_q is not None
            else "dense_same_position_joint_leveling"
        ),
        "target_orientation_source": target_orientation_source,
        "sample_count": count,
        "start_level": start_assessment.to_dict(),
        "final_level": final_level.to_dict(),
        "maximum_ik_position_error_m": maximum_position_error,
        "maximum_ik_rotation_error_rad": maximum_rotation_error,
        "maximum_planar_motion_m": maximum_planar_motion,
        "maximum_height_motion_m": maximum_height_motion,
        "maximum_same_position_motion_m": maximum_same_position_motion,
        "maximum_joint_step_rad": maximum_joint_step,
        "planned_level_q_physical_rad": q_path_array[-1].tolist(),
        "target_signed_tip_height_difference_m": target_tip_bias,
        "planned_signed_tip_height_difference_m": (
            _signed_tip_height_difference_m(pose_path_array[-1], reference)
        ),
        "contact": contact,
    }
    if not accepted:
        raise RuntimeError(f"same-position joint level audit rejected: {audit}")
    return samples, audit


def _plan_level_vertical_joint_descent(
    profile,
    fk,
    *,
    start_q,
    reference_low_q,
) -> tuple[list[JointTrajectorySample], dict]:
    """Replay the proven level hover-to-low joint branch and audit its path.

    The endpoint solve stays near the preceding measured low joint branch and
    preserves its full camera orientation whenever reachable.  Dense
    minimum-jerk joint interpolation is retained because it has demonstrated
    reliable motor tracking on this hardware.  The result is accepted only
    when FK proves monotonic downward progress, level fingertips, small planar
    drift, bounded joint steps, and collision-safe geometry.
    """

    start_q = _q(start_q, "level vertical descent start q")
    reference_low_q = _q(reference_low_q, "level vertical reference low q")
    trajectory = profile["trajectory"]
    execution = profile["execution"]
    samplewise_orientation_mode = str(
        execution.get(
            "descent_samplewise_orientation_mode",
            "interpolate_to_reference_low",
        )
    )
    level_reference = _load_level_reference(profile["level_config"])
    up = np.asarray(level_reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    start_pose = np.asarray(fk.pose(start_q).parameters(), dtype=float)
    reference_low_pose = np.asarray(
        fk.pose(reference_low_q).parameters(), dtype=float
    )
    low_height_bias = float(
        execution.get("descent_low_height_bias_m", 0.0)
    )
    if low_height_bias < 0.0:
        raise ValueError("descent low-height bias must be non-negative")
    if low_height_bias > 1e-9:
        raise ValueError(
            "branch-locked descent requires descent_low_height_bias_m=0; "
            "calibrate a new verified low endpoint instead of extrapolating "
            "below it"
        )
    biased_reference_low_position = reference_low_pose[4:]
    nominal_clearance = float(trajectory["verification_lift_m"])
    clearance = float(
        (start_pose[4:] - biased_reference_low_position) @ up
    )
    maximum_clearance_adjustment = float(
        execution.get("descent_maximum_clearance_adjustment_m", 0.003)
    )
    if (
        clearance <= 0.0
        or abs(clearance - nominal_clearance) > maximum_clearance_adjustment
    ):
        raise RuntimeError(
            "measured hover-to-reference-low height is outside the audited "
            f"range: clearance={clearance:.6f}m, nominal={nominal_clearance:.6f}m"
        )
    verified_low_level = assess_jaw_level(
        reference_low_pose, level_reference, planned=True
    )
    if not verified_low_level.accepted:
        raise RuntimeError(
            "reference low pose is not level enough for a reversible descent: "
            f"{verified_low_level.to_dict()}"
        )
    start_level = assess_jaw_level(
        start_pose, level_reference, planned=False
    )
    prelevel_trigger_tip_delta = float(
        profile["perception"]["maximum_preclose_tip_height_difference_m"]
    )
    prelevel_trigger_tilt_deg = float(
        execution.get("descent_prelevel_trigger_tilt_deg", 0.25)
    )
    yaw_free_descent_can_level_above_target = bool(
        execution.get("descent_use_samplewise_cartesian_path", True)
        and samplewise_orientation_mode in {"level_yaw_free", "level_fixed_yaw"}
        and start_level.accepted
    )
    if (
        not yaw_free_descent_can_level_above_target
        and (
            start_level.tip_height_difference_m > prelevel_trigger_tip_delta
            or start_level.combined_tilt_deg > prelevel_trigger_tilt_deg
        )
    ):
        prelevel_samples, prelevel_audit = (
            _plan_same_position_level_joint_samples(
                profile,
                fk,
                start_q=start_q,
            )
        )
        oriented_start_q = np.asarray(
            prelevel_samples[-1].right_q_physical_rad, dtype=float
        )
        oriented_start_pose = np.asarray(
            fk.pose(oriented_start_q).parameters(), dtype=float
        )
    else:
        prelevel_samples = []
        oriented_start_q = start_q.copy()
        oriented_start_pose = start_pose.copy()
        prelevel_audit = {
            "accepted": True,
            "skipped": True,
            "reason": (
                "level_constrained_descent_levels_before_lowering"
                if yaw_free_descent_can_level_above_target
                else "measured_hover_already_within_strict_tip_height_gate"
            ),
            "start_level": start_level.to_dict(),
            "maximum_preclose_tip_height_difference_m": (
                prelevel_trigger_tip_delta
            ),
            "descent_prelevel_trigger_tilt_deg": prelevel_trigger_tilt_deg,
        }
    # Build the low endpoint directly below the measured hover, but preserve
    # the preceding *measured low* camera orientation.  This stays on the
    # motor-proven nearby joint branch while making successive low-pose pixel
    # errors comparable.  If that exact orientation becomes unreachable, use
    # a level yaw-free endpoint seeded by the prior low branch; the dense path
    # audit below must still prove verticality, level jaws, and no collision.
    dynamic_low_position = oriented_start_pose[4:] - clearance * up
    try:
        dynamic_low_q, dynamic_low_plan = _plan_fixed_orientation_pose(
            profile,
            fk,
            target_position=dynamic_low_position,
            orientation_q=reference_low_q,
            seed_q=reference_low_q,
            role="measured_hover_vertical_low_preserve_previous_low_orientation",
        )
        dynamic_low_plan["orientation_fallback_used"] = False
    except RuntimeError as fixed_orientation_error:
        dynamic_low_q, dynamic_low_plan = _plan_level_yaw_free_pose(
            profile,
            fk,
            target_position=dynamic_low_position,
            seed_q=reference_low_q,
            role="measured_hover_vertical_low_near_previous_branch",
        )
        dynamic_low_plan["orientation_fallback_used"] = True
        dynamic_low_plan["fixed_orientation_error"] = str(
            fixed_orientation_error
        )
    dynamic_low_pose = np.asarray(
        fk.pose(dynamic_low_q).parameters(), dtype=float
    )
    dynamic_low_level = assess_jaw_level(
        dynamic_low_pose, level_reference, planned=True
    )
    descend_duration = float(trajectory["descend_duration_s"])
    descend_count = max(1, int(math.ceil(descend_duration * CONTROL_HZ)))
    samples: list[JointTrajectorySample] = list(prelevel_samples)
    prelevel_duration = (
        float(prelevel_samples[-1].t_s) if prelevel_samples else 0.0
    )
    q_path = [oriented_start_q.copy()]
    pose_path = [oriented_start_pose.copy()]
    samplewise_audit = None
    if bool(execution.get("descent_use_samplewise_cartesian_path", True)):
        # A yaw-free endpoint can be perfectly level while straight joint
        # interpolation to it bows sideways and temporarily tilts the jaws.
        # Follow the Cartesian vertical line sample-by-sample instead.  The
        # desired camera orientation is interpolated to the previously
        # calibrated low orientation, while a strict per-sample joint bound
        # keeps the IK on one motor-trackable continuous branch.
        target_pose = reference_low_pose.copy()
        target_pose[4:] = dynamic_low_position
        maximum_step_bound = float(
            execution.get("descent_samplewise_maximum_joint_step_rad", 0.04)
        )
        position_scale = float(
            execution.get("descent_ik_position_scale_m", 0.0001)
        )
        rotation_scale = float(
            execution.get("descent_ik_rotation_scale_rad", 0.001)
        )
        continuity_weight = float(
            execution.get("descent_ik_continuity_weight", 0.01)
        )
        orientation_mode = samplewise_orientation_mode
        if orientation_mode not in {
            "interpolate_to_reference_low",
            "level_yaw_free",
            "level_fixed_yaw",
        }:
            raise ValueError(
                "descent_samplewise_orientation_mode must be "
                "'interpolate_to_reference_low', 'level_yaw_free', or "
                "'level_fixed_yaw'"
            )
        level_axis_scale = float(
            execution.get("descent_samplewise_level_axis_scale_rad", 0.01)
        )
        yaw_free_continuity_weight = float(
            execution.get(
                "descent_samplewise_yaw_free_continuity_weight", 0.003
            )
        )
        fixed_yaw_source_q = _q(
            trajectory["verified_preclose_q_physical_rad"],
            "samplewise fixed-yaw source q",
        )
        fixed_yaw_rotation = fk.pose(fixed_yaw_source_q).as_matrix()[:3, :3]
        fixed_yaw_approach = fixed_yaw_rotation[:, 0] - up * float(
            fixed_yaw_rotation[:, 0] @ up
        )
        fixed_yaw_approach /= np.linalg.norm(fixed_yaw_approach)
        fixed_yaw_axis_scale = float(
            execution.get("descent_samplewise_fixed_yaw_axis_scale_rad", 0.02)
        )
        maximum_fixed_yaw_error = 0.0
        final_fixed_yaw_error = math.inf
        maximum_position_error = 0.0
        maximum_rotation_error = 0.0
        maximum_level_axis_error = 0.0
        solver_nonconverged_samples: list[dict] = []
        for sample_index in range(1, descend_count + 1):
            fraction = _minimum_jerk(sample_index / descend_count)
            desired_pose = oriented_start_pose.copy()
            desired_pose[:4] = _nlerp_wxyz(
                oriented_start_pose[:4], target_pose[:4], fraction
            )
            desired_pose[4:] = oriented_start_pose[4:] + fraction * (
                target_pose[4:] - oriented_start_pose[4:]
            )
            desired_rotation = mink.SE3(desired_pose).as_matrix()[:3, :3]
            previous_q = q_path[-1]
            lower, upper = _joint_bounds(
                profile,
                fk,
                center_q=previous_q,
                maximum_delta_rad=maximum_step_bound,
            )

            def residual(q_physical):
                pose = fk.pose(q_physical)
                parameters = np.asarray(pose.parameters(), dtype=float)
                rotation = pose.as_matrix()[:3, :3]
                if orientation_mode in {"level_yaw_free", "level_fixed_yaw"}:
                    orientation_terms = [
                        (rotation[:, 1] - up)
                        / max(level_axis_scale, 1e-6)
                    ]
                    if orientation_mode == "level_fixed_yaw":
                        orientation_terms.append(
                            (rotation[:, 0] - fixed_yaw_approach)
                            / max(fixed_yaw_axis_scale, 1e-6)
                        )
                    return np.concatenate(
                        [
                            (parameters[4:] - desired_pose[4:])
                            / position_scale,
                            *orientation_terms,
                            yaw_free_continuity_weight
                            * (q_physical - previous_q),
                        ]
                    )
                return np.concatenate(
                    [
                        (parameters[4:] - desired_pose[4:])
                        / position_scale,
                        _rotation_error_vector(desired_rotation, rotation)
                        / rotation_scale,
                        continuity_weight * (q_physical - previous_q),
                    ]
                )

            solved = least_squares(
                residual,
                np.clip(previous_q, lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper),
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
                max_nfev=int(execution.get("descent_ik_max_nfev", 150)),
            )
            if not solved.success:
                # ``least_squares`` can exhaust its evaluation budget after it
                # has already found a physically valid sample (especially for
                # the redundant level + fixed-yaw constraint).  The complete
                # path is audited below from FK before *any* command is sent,
                # so retain the candidate and let those geometric gates decide
                # instead of treating an optimiser status code as a safety
                # property.
                solver_nonconverged_samples.append(
                    {
                        "sample_index": sample_index,
                        "message": str(solved.message),
                        "function_evaluations": int(solved.nfev),
                    }
                )
            q_physical = np.asarray(solved.x, dtype=float)
            pose = fk.pose(q_physical)
            solved_pose = np.asarray(pose.parameters(), dtype=float)
            maximum_position_error = max(
                maximum_position_error,
                float(np.linalg.norm(solved_pose[4:] - desired_pose[4:])),
            )
            maximum_rotation_error = max(
                maximum_rotation_error,
                float(
                    np.linalg.norm(
                        _rotation_error_vector(
                            desired_rotation, pose.as_matrix()[:3, :3]
                        )
                    )
                ),
            )
            solved_rotation = pose.as_matrix()[:3, :3]
            maximum_level_axis_error = max(
                maximum_level_axis_error,
                float(
                    math.acos(
                        float(
                            np.clip(solved_rotation[:, 1] @ up, -1.0, 1.0)
                        )
                    )
                ),
            )
            fixed_yaw_error = float(
                math.acos(
                    float(
                        np.clip(
                            solved_rotation[:, 0] @ fixed_yaw_approach,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            maximum_fixed_yaw_error = max(
                maximum_fixed_yaw_error, fixed_yaw_error
            )
            final_fixed_yaw_error = fixed_yaw_error
            q_path.append(q_physical)
            pose_path.append(solved_pose)
            samples.append(
                JointTrajectorySample(
                    t_s=prelevel_duration + sample_index / CONTROL_HZ,
                    stage="level_vertical_cartesian_orientation_descent",
                    right_q_physical_rad=q_physical.copy(),
                    right_gripper_open_ratio=1.0,
                )
            )
        samplewise_audit = {
            "accepted": bool(
                maximum_position_error
                <= float(
                    execution.get(
                        "descent_maximum_ik_position_error_m", 0.0003
                    )
                )
                and (
                    orientation_mode in {"level_yaw_free", "level_fixed_yaw"}
                    or maximum_rotation_error
                    <= float(
                        execution.get(
                            "descent_maximum_ik_rotation_error_rad", 0.003
                        )
                    )
                )
                and (
                    orientation_mode != "level_fixed_yaw"
                    or (
                        maximum_fixed_yaw_error
                        <= float(
                            execution.get(
                                "descent_samplewise_maximum_fixed_yaw_error_rad",
                                0.1,
                            )
                        )
                        and final_fixed_yaw_error
                        <= float(
                            execution.get(
                                "descent_samplewise_maximum_final_fixed_yaw_error_rad",
                                0.01,
                            )
                        )
                    )
                )
            ),
            "method": (
                "samplewise_cartesian_position_and_level_fixed_yaw"
                if orientation_mode == "level_fixed_yaw"
                else (
                    "samplewise_cartesian_position_and_level_yaw_free"
                    if orientation_mode == "level_yaw_free"
                    else "samplewise_cartesian_position_and_orientation"
                )
            ),
            "orientation_mode": orientation_mode,
            "maximum_ik_position_error_m": maximum_position_error,
            "maximum_ik_rotation_error_rad": maximum_rotation_error,
            "maximum_level_axis_error_rad": maximum_level_axis_error,
            "maximum_fixed_yaw_error_rad": maximum_fixed_yaw_error,
            "final_fixed_yaw_error_rad": final_fixed_yaw_error,
            "per_sample_joint_bound_rad": maximum_step_bound,
            "solver_nonconverged_sample_count": len(
                solver_nonconverged_samples
            ),
            "solver_nonconverged_samples": solver_nonconverged_samples,
        }
        dynamic_low_q = np.asarray(q_path[-1], dtype=float)
        dynamic_low_pose = np.asarray(pose_path[-1], dtype=float)
        dynamic_low_level = assess_jaw_level(
            dynamic_low_pose, level_reference, planned=True
        )
        dynamic_low_plan["samplewise_endpoint_q_physical_rad"] = (
            dynamic_low_q.tolist()
        )
    else:
        for sample_index in range(1, descend_count + 1):
            fraction = _minimum_jerk(sample_index / descend_count)
            q_physical = oriented_start_q + fraction * (
                dynamic_low_q - oriented_start_q
            )
            solved_pose = np.asarray(
                fk.pose(q_physical).parameters(), dtype=float
            )
            q_path.append(q_physical)
            pose_path.append(solved_pose)
            samples.append(
                JointTrajectorySample(
                    t_s=prelevel_duration + sample_index / CONTROL_HZ,
                    stage="level_vertical_descent_branch_replay",
                    right_q_physical_rad=q_physical.copy(),
                    right_gripper_open_ratio=1.0,
                )
            )
    q_path_array = np.asarray(q_path, dtype=float)
    pose_path_array = np.asarray(pose_path, dtype=float)
    level_assessments = [
        assess_jaw_level(pose, level_reference, planned=True)
        for pose in pose_path_array
    ]
    translation = pose_path_array[:, 4:7] - pose_path_array[0, 4:7]
    signed_height = translation @ up
    planar = translation - np.outer(signed_height, up)
    maximum_planar_motion = float(np.max(np.linalg.norm(planar, axis=1)))
    downward_progress = -signed_height
    minimum_downward_step = float(np.min(np.diff(downward_progress)))
    maximum_joint_step = float(np.max(np.abs(np.diff(q_path_array, axis=0))))
    final_reference_height_error = float(
        abs(
            (pose_path_array[-1, 4:7] - reference_low_pose[4:7])
            @ up
        )
    )
    contact = _right_joint_path_contact_audit(profile, q_path_array)
    maximum_joint_step_limit = float(
        execution.get("descent_maximum_joint_step_rad", 0.06)
    )
    maximum_reference_height_error = float(
        execution.get("descent_maximum_reference_low_height_error_m", 0.0002)
    )
    accepted = bool(
        maximum_planar_motion
        <= float(trajectory["maximum_descent_planar_motion_m"])
        and minimum_downward_step >= -0.0001
        and downward_progress[-1]
        >= clearance - maximum_reference_height_error
        and maximum_joint_step <= maximum_joint_step_limit
        and final_reference_height_error <= maximum_reference_height_error
        and all(item.accepted for item in level_assessments)
        and dynamic_low_level.accepted
        and (
            samplewise_audit is None
            or samplewise_audit["accepted"]
        )
        and contact["accepted"]
    )
    audit = {
        "accepted": accepted,
        "method": "dense_joint_branch_replay_vertical_descent",
        "command_path": "set_right_joint_target",
        "control_hz": CONTROL_HZ,
        "level_sample_count": len(prelevel_samples),
        "descent_sample_count": descend_count,
        "planned_translation_m": (
            dynamic_low_pose[4:] - oriented_start_pose[4:]
        ).tolist(),
        "nominal_translation_m": (-nominal_clearance * up).tolist(),
        "clearance_adjustment_m": clearance - nominal_clearance,
        "low_height_bias_m": low_height_bias,
        "fresh_cartesian_ik_used": False,
        "endpoint_cartesian_ik_used": True,
        "maximum_planar_motion_m": maximum_planar_motion,
        "minimum_downward_step_m": minimum_downward_step,
        "achieved_downward_progress_m": float(downward_progress[-1]),
        "maximum_joint_step_rad": maximum_joint_step,
        "final_reference_low_height_error_m": final_reference_height_error,
        "maximum_combined_tilt_deg": max(
            item.combined_tilt_deg for item in level_assessments
        ),
        "maximum_tip_height_difference_m": max(
            item.tip_height_difference_m for item in level_assessments
        ),
        "orientation_pairing": (
            "level_measured_hover_to_previous_low_orientation_endpoint"
        ),
        "prelevel": prelevel_audit,
        "verified_low_level": verified_low_level.to_dict(),
        "dynamic_low_level": dynamic_low_level.to_dict(),
        "dynamic_low_plan": dynamic_low_plan,
        "samplewise_cartesian_audit": samplewise_audit,
        "planned_low_q_physical_rad": q_path_array[-1].tolist(),
        "contact": contact,
        "limits": {
            "maximum_planar_motion_m": float(
                trajectory["maximum_descent_planar_motion_m"]
            ),
            "maximum_joint_step_rad": maximum_joint_step_limit,
            "maximum_reference_low_height_error_m": (
                maximum_reference_height_error
            ),
        },
    }
    if not accepted:
        raise RuntimeError(
            f"branch-locked vertical joint descent audit rejected: {audit}"
        )
    return samples, audit


def _plan_straight_level_joint_lift(
    profile,
    fk,
    *,
    start_q,
    aperture: float,
    distance_m: float | None = None,
    duration_s: float | None = None,
    stage: str = "verification_lift_straight_level",
    maximum_planar_motion_m: float | None = None,
    maximum_ik_position_error_m: float | None = None,
    minimum_progress_fraction: float = 0.99,
) -> tuple[list[JointTrajectorySample], dict]:
    """Plan a fixed-orientation vertical lift on one continuous IK branch."""

    start_q = _q(start_q, "straight lift start q")
    aperture = float(aperture)
    if not 0.0 <= aperture <= 1.0:
        raise ValueError("straight lift aperture must be within [0, 1]")
    trajectory = profile["trajectory"]
    execution = profile["execution"]
    reference = _load_level_reference(profile["level_config"])
    up = np.asarray(reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    distance = float(
        trajectory["verification_lift_m"]
        if distance_m is None
        else distance_m
    )
    duration = float(
        trajectory["verification_lift_duration_s"]
        if duration_s is None
        else duration_s
    )
    if distance == 0.0 or duration <= 0.0:
        raise ValueError(
            "straight vertical move distance must be non-zero and duration positive"
        )
    start_pose = np.asarray(fk.pose(start_q).parameters(), dtype=float)
    start_rotation = fk.pose(start_q).as_matrix()[:3, :3]
    start_level = assess_jaw_level(start_pose, reference, planned=True)
    if not start_level.accepted:
        raise RuntimeError(
            f"straight lift start is not level: {start_level.to_dict()}"
        )
    lower, upper = _joint_bounds(
        profile,
        fk,
        center_q=start_q,
        maximum_delta_rad=float(
            execution.get("descent_maximum_local_joint_delta_rad", 1.5)
        ),
    )
    position_scale = float(execution.get("descent_ik_position_scale_m", 0.0001))
    rotation_scale = float(
        execution.get("descent_ik_rotation_scale_rad", 0.001)
    )
    continuity_weight = float(
        execution.get("descent_ik_continuity_weight", 0.01)
    )
    count = max(1, int(math.ceil(duration * CONTROL_HZ)))
    q_path = [start_q.copy()]
    pose_path = [start_pose.copy()]
    samples = []
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0
    for index in range(1, count + 1):
        fraction = _minimum_jerk(index / count)
        target_position = start_pose[4:] + distance * fraction * up
        previous_q = q_path[-1]

        def residual(q_physical):
            pose = fk.pose(q_physical)
            parameters = np.asarray(pose.parameters(), dtype=float)
            return np.concatenate(
                [
                    (parameters[4:] - target_position) / position_scale,
                    _rotation_error_vector(
                        start_rotation, pose.as_matrix()[:3, :3]
                    )
                    / rotation_scale,
                    continuity_weight * (q_physical - previous_q),
                ]
            )

        solved = least_squares(
            residual,
            np.clip(previous_q, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            max_nfev=int(execution.get("descent_ik_max_nfev", 150)),
        )
        if not solved.success:
            raise RuntimeError(
                f"straight lift IK failed at sample {index}: {solved.message}"
            )
        solved_q = np.asarray(solved.x, dtype=float)
        pose = fk.pose(solved_q)
        solved_pose = np.asarray(pose.parameters(), dtype=float)
        maximum_position_error = max(
            maximum_position_error,
            float(np.linalg.norm(solved_pose[4:] - target_position)),
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            float(
                np.linalg.norm(
                    _rotation_error_vector(
                        start_rotation, pose.as_matrix()[:3, :3]
                    )
                )
            ),
        )
        q_path.append(solved_q)
        pose_path.append(solved_pose)
        samples.append(
            JointTrajectorySample(
                t_s=index / CONTROL_HZ,
                stage=str(stage),
                right_q_physical_rad=solved_q.copy(),
                # Zero is the original closure command.  If the object is
                # present, measured obstruction keeps the aperture non-zero.
                right_gripper_open_ratio=aperture,
            )
        )
    q_path_array = np.asarray(q_path, dtype=float)
    pose_path_array = np.asarray(pose_path, dtype=float)
    delta = pose_path_array[:, 4:7] - pose_path_array[0, 4:7]
    heights = delta @ up
    planar = delta - np.outer(heights, up)
    maximum_planar_motion = float(np.max(np.linalg.norm(planar, axis=1)))
    direction = math.copysign(1.0, distance)
    progress = direction * heights
    minimum_height_step = float(np.min(np.diff(progress)))
    maximum_joint_step = float(np.max(np.abs(np.diff(q_path_array, axis=0))))
    levels = [
        assess_jaw_level(pose, reference, planned=True)
        for pose in pose_path_array
    ]
    contact = _right_joint_path_contact_audit(profile, q_path_array)
    planar_limit = float(
        trajectory["maximum_descent_planar_motion_m"]
        if maximum_planar_motion_m is None
        else maximum_planar_motion_m
    )
    position_error_limit = float(
        execution.get("descent_maximum_ik_position_error_m", 0.0003)
        if maximum_ik_position_error_m is None
        else maximum_ik_position_error_m
    )
    minimum_progress_fraction = float(minimum_progress_fraction)
    if not 0.0 < minimum_progress_fraction <= 1.0:
        raise ValueError("minimum straight-move progress fraction must be in (0, 1]")
    accepted = bool(
        maximum_position_error
        <= position_error_limit
        and maximum_rotation_error
        <= float(
            execution.get("descent_maximum_ik_rotation_error_rad", 0.003)
        )
        and maximum_planar_motion
        <= planar_limit
        and minimum_height_step >= -0.0001
        and progress[-1] >= minimum_progress_fraction * abs(distance)
        and maximum_joint_step
        <= float(execution.get("descent_maximum_joint_step_rad", 0.06))
        and all(item.accepted for item in levels)
        and contact["accepted"]
    )
    audit = {
        "accepted": accepted,
        "method": "dense_fixed_orientation_joint_vertical_lift",
        "command_path": "set_right_joint_target",
        "sample_count": count,
        "planned_translation_m": (distance * up).tolist(),
        "planned_rotation_change_rad": 0.0,
        "maximum_ik_position_error_m": maximum_position_error,
        "maximum_ik_rotation_error_rad": maximum_rotation_error,
        "maximum_planar_motion_m": maximum_planar_motion,
        "minimum_height_step_m": minimum_height_step,
        "achieved_height_m": float(heights[-1]),
        "achieved_progress_m": float(progress[-1]),
        "maximum_joint_step_rad": maximum_joint_step,
        "maximum_combined_tilt_deg": max(
            item.combined_tilt_deg for item in levels
        ),
        "maximum_tip_height_difference_m": max(
            item.tip_height_difference_m for item in levels
        ),
        "planned_lift_q_physical_rad": q_path_array[-1].tolist(),
        "contact": contact,
        "limits": {
            "maximum_ik_position_error_m": position_error_limit,
            "maximum_planar_motion_m": planar_limit,
            "minimum_progress_fraction": minimum_progress_fraction,
        },
    }
    if not accepted:
        raise RuntimeError(f"straight level joint lift audit rejected: {audit}")
    return samples, audit


def _fixed_pose_gripper_ramp(
    profile, rpc, fk, *, finish_ratio: float, duration_s: float, stage: str
) -> dict:
    current_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    current_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    start_ratio = float(rpc.get_right_gripper_exact())
    count = max(1, int(math.ceil(float(duration_s) * CONTROL_HZ)))
    samples = []
    for index in range(1, count + 1):
        fraction = index / count
        samples.append(
            JointTrajectorySample(
                t_s=fraction * float(duration_s),
                stage=stage,
                right_q_physical_rad=current_q.copy(),
                right_gripper_open_ratio=float(
                    start_ratio
                    + _minimum_jerk(fraction) * (finish_ratio - start_ratio)
                ),
            )
        )
    return _streamer(profile, rpc, fk).execute(
        samples,
        pose_transformer=lambda _stage, _pose: mink.SE3(current_pose),
    )


def _fixed_pose_gripper_hold(
    profile,
    rpc,
    fk,
    *,
    aperture: float,
    duration_s: float,
    stage: str,
) -> dict:
    """Continuously command one aperture while holding the measured pose."""

    current_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    current_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    count = max(1, int(math.ceil(float(duration_s) * CONTROL_HZ)))
    samples = [
        JointTrajectorySample(
            t_s=index / CONTROL_HZ,
            stage=stage,
            right_q_physical_rad=current_q.copy(),
            right_gripper_open_ratio=float(aperture),
        )
        for index in range(1, count + 1)
    ]
    return _streamer(profile, rpc, fk).execute(
        samples,
        pose_transformer=lambda _stage, _pose: mink.SE3(current_pose),
    )


def _cartesian_move(
    profile,
    rpc,
    fk,
    *,
    target_pose: np.ndarray,
    duration_s: float,
    aperture: float,
    stage: str,
    settle_s: float = 0.0,
    maximum_start_fk_position_error_m: float | None = None,
    maximum_start_fk_rotation_error_rad: float | None = None,
) -> dict:
    start_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    start_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    target_pose = _pose(target_pose, "Cartesian target")
    count = max(1, int(math.ceil(float(duration_s) * CONTROL_HZ)))
    settle_count = max(0, int(math.ceil(float(settle_s) * CONTROL_HZ)))
    samples = []
    poses = []
    for index in range(1, count + 1):
        fraction = index / count
        blend = _minimum_jerk(fraction)
        quaternion = _nlerp_wxyz(start_pose[:4], target_pose[:4], blend)
        position = start_pose[4:] + blend * (target_pose[4:] - start_pose[4:])
        samples.append(
            JointTrajectorySample(
                t_s=fraction * float(duration_s),
                stage=stage,
                right_q_physical_rad=start_q.copy(),
                right_gripper_open_ratio=float(aperture),
            )
        )
        poses.append(mink.SE3(np.r_[quaternion, position]))
    for index in range(1, settle_count + 1):
        samples.append(
            JointTrajectorySample(
                t_s=float(duration_s) + index / CONTROL_HZ,
                stage=f"{stage}_settle",
                right_q_physical_rad=start_q.copy(),
                right_gripper_open_ratio=float(aperture),
            )
        )
        poses.append(mink.SE3(target_pose))
    cursor = {"index": 0}

    def transform(_stage, _pose):
        result = poses[cursor["index"]]
        cursor["index"] += 1
        return result

    return _streamer(
        profile,
        rpc,
        fk,
        maximum_start_fk_position_error_m=(
            maximum_start_fk_position_error_m
        ),
        maximum_start_fk_rotation_error_rad=(
            maximum_start_fk_rotation_error_rad
        ),
    ).execute(samples, pose_transformer=transform)


def _select_wrist_roll_level_endpoint(
    profile,
    fk,
    *,
    measured_q,
    planned_level_q,
) -> tuple[np.ndarray, object, dict]:
    """Choose a J6-only endpoint in the measured IK branch.

    The Cartesian controller can settle on a different joint branch from the
    offline level IK solution.  Copying J6 from that offline solution while
    retaining the other five measured joints is therefore not guaranteed to
    be level.  Search the deliberately small audited J6 interval in the
    *measured* branch instead.  Execution is still followed by the strict
    measured jaw checkpoint; the model score only selects the least-tilted
    bounded endpoint and never authorizes closure.
    """

    current_q = _q(measured_q, "measured post-Cartesian level q")
    planned_q = _q(planned_level_q, "planned Cartesian level q")
    maximum_delta = float(
        profile["execution"].get(
            "preclose_level_maximum_wrist_roll_endpoint_rad", 0.03
        )
    )
    preferred_delta = float(
        np.clip(planned_q[5] - current_q[5], -maximum_delta, maximum_delta)
    )
    sample_count = int(
        profile["execution"].get(
            "preclose_level_wrist_roll_search_samples", 121
        )
    )
    if sample_count < 3:
        raise ValueError("wrist-roll endpoint search requires at least 3 samples")
    deltas = np.unique(
        np.concatenate(
            [
                np.linspace(-maximum_delta, maximum_delta, sample_count),
                np.asarray([0.0, preferred_delta], dtype=float),
            ]
        )
    )
    strict_reference = replace(
        _load_level_reference(profile["level_config"]),
        maximum_tip_height_difference_m=float(
            profile["perception"]["maximum_preclose_tip_height_difference_m"]
        ),
    )
    candidates = []
    for delta in deltas:
        candidate_q = current_q.copy()
        candidate_q[5] += float(delta)
        pose = np.asarray(fk.pose(candidate_q).parameters(), dtype=float)
        level = assess_jaw_level(pose, strict_reference, planned=True)
        candidates.append(
            (
                (
                    0 if level.accepted else 1,
                    abs(float(level.tip_height_difference_m)),
                    float(level.combined_tilt_deg),
                    abs(float(delta) - preferred_delta),
                    abs(float(delta)),
                ),
                candidate_q,
                level,
                float(delta),
            )
        )
    _, selected_q, selected_level, selected_delta = min(
        candidates, key=lambda item: item[0]
    )
    return selected_q, selected_level, {
        "method": "bounded_measured_branch_j6_search",
        "sample_count": int(len(deltas)),
        "maximum_delta_rad": maximum_delta,
        "preferred_offline_ik_delta_rad": preferred_delta,
        "selected_delta_rad": selected_delta,
        "prediction_accepted": bool(selected_level.accepted),
    }


def _execute_same_position_level_cartesian(
    profile,
    rpc,
    fk,
    *,
    planned_level_q,
    aperture: float,
    stage: str,
) -> dict:
    """Apply an audited level orientation while holding measured XYZ.

    The model-space joint path produced by
    ``_plan_same_position_level_joint_samples`` is still the collision and
    orientation witness.  Sending those joints directly, however, exposed a
    physical/model offset that accumulated centimetres of measured XY while
    repeatedly correcting roll.  Human teleoperation already closes that
    offset in Cartesian space, so use its 30 Hz pose path: measured XYZ is
    fixed and only the audited target quaternion changes.  A fresh measured
    jaw checkpoint remains the authority after execution.
    """

    planned_q = _q(planned_level_q, "planned Cartesian level q")
    measured_pose = np.asarray(
        rpc.get_right_ee_pose().parameters(), dtype=float
    )
    planned_pose = np.asarray(fk.pose(planned_q).parameters(), dtype=float)
    target_pose = measured_pose.copy()
    target_pose[:4] = planned_pose[:4]
    cartesian_motion = _cartesian_move(
        profile,
        rpc,
        fk,
        target_pose=target_pose,
        duration_s=float(
            profile["execution"].get(
                "preclose_level_correction_cartesian_duration_s", 0.8
            )
        ),
        aperture=float(aperture),
        stage=stage,
        settle_s=float(
            profile["execution"].get(
                "preclose_level_correction_cartesian_settle_s", 0.5
            )
        ),
    )
    after_cartesian_q = _q(
        rpc.get_right_joint_positions(), "post-Cartesian level q"
    )
    wrist_roll_q, wrist_roll_level, wrist_roll_selection = (
        _select_wrist_roll_level_endpoint(
            profile,
            fk,
            measured_q=after_cartesian_q,
            planned_level_q=planned_q,
        )
    )
    wrist_roll_delta = float(wrist_roll_q[5] - after_cartesian_q[5])
    maximum_wrist_roll = float(
        profile["execution"].get(
            "preclose_level_maximum_wrist_roll_endpoint_rad", 0.03
        )
    )
    if abs(wrist_roll_delta) > maximum_wrist_roll:
        raise RuntimeError(
            "Cartesian level residual exceeds the audited wrist-roll endpoint "
            f"limit: {wrist_roll_delta:.6f}rad"
        )
    after_cartesian_fk_pose = np.asarray(
        fk.pose(after_cartesian_q).parameters(), dtype=float
    )
    wrist_roll_fk_pose = np.asarray(
        fk.pose(wrist_roll_q).parameters(), dtype=float
    )
    wrist_roll_position_shift = float(
        np.linalg.norm(
            wrist_roll_fk_pose[4:7] - after_cartesian_fk_pose[4:7]
        )
    )
    maximum_wrist_roll_position_shift = float(
        profile["execution"].get(
            "preclose_level_maximum_wrist_roll_position_shift_m", 0.0001
        )
    )
    if wrist_roll_position_shift > maximum_wrist_roll_position_shift:
        raise RuntimeError(
            "wrist-roll level endpoint would translate the gripper: "
            f"{wrist_roll_position_shift:.6f}m"
        )
    wrist_roll_motion = None
    wrist_roll_audit = None
    if abs(wrist_roll_delta) > float(
        profile["execution"].get(
            "preclose_level_minimum_wrist_roll_endpoint_rad", 0.0002
        )
    ):
        wrist_roll_samples = sample_joint_knots(
            [
                {
                    "stage": f"{stage}_wrist_roll_start",
                    "right_q_physical_rad": after_cartesian_q.tolist(),
                    "right_gripper_open_ratio": float(aperture),
                    "minimum_duration_s": 0.1,
                },
                {
                    "stage": f"{stage}_wrist_roll_endpoint",
                    "right_q_physical_rad": wrist_roll_q.tolist(),
                    "right_gripper_open_ratio": float(aperture),
                    "minimum_duration_s": float(
                        profile["execution"].get(
                            "preclose_level_wrist_roll_endpoint_duration_s",
                            0.35,
                        )
                    ),
                },
            ]
        )
        wrist_roll_path = np.vstack(
            [
                after_cartesian_q,
                *[
                    np.asarray(sample.right_q_physical_rad, dtype=float)
                    for sample in wrist_roll_samples
                ],
            ]
        )
        wrist_roll_audit = _right_joint_path_contact_audit(
            profile, wrist_roll_path
        )
        if not wrist_roll_audit["accepted"]:
            raise RuntimeError(
                "wrist-roll level endpoint predicts a collision: "
                f"{wrist_roll_audit}"
            )
        wrist_roll_motion = _execute_direct_joint_samples(
            profile,
            rpc,
            fk,
            wrist_roll_samples,
            final_tolerance_rad=float(
                profile["execution"].get(
                    "preclose_level_wrist_roll_final_tolerance_rad", 0.003
                )
            ),
            endpoint_correction_gain=float(
                profile["execution"].get(
                    "preclose_level_wrist_roll_endpoint_correction_gain", 1.0
                )
            ),
            maximum_endpoint_correction_rad=float(
                profile["execution"].get(
                    "preclose_level_wrist_roll_maximum_endpoint_correction_rad",
                    0.02,
                )
            ),
            accumulate_endpoint_correction=True,
            require_final_convergence=False,
        )
    final_q = _q(rpc.get_right_joint_positions(), "final hybrid level q")
    final_pose = np.asarray(
        rpc.get_right_ee_pose().parameters(), dtype=float
    )
    motion = dict(cartesian_motion)
    motion["method"] = (
        "teleop_cartesian_measured_xyz_plus_wrist_roll_level"
    )
    motion["measured_start_xyz_m"] = measured_pose[4:7].tolist()
    motion["measured_final_xyz_m"] = final_pose[4:7].tolist()
    motion["measured_position_drift_m"] = float(
        np.linalg.norm(final_pose[4:7] - measured_pose[4:7])
    )
    motion["wrist_roll_endpoint"] = {
        "commanded_delta_rad": wrist_roll_delta,
        "planned_position_shift_m": wrist_roll_position_shift,
        "planned_level": wrist_roll_level.to_dict(),
        "selection": wrist_roll_selection,
        "contact_audit": wrist_roll_audit,
        "motion": wrist_roll_motion,
    }
    motion["final_right_q_physical_rad"] = final_q.tolist()
    motion["final_right_ee_wxyz_xyz"] = final_pose.tolist()
    return motion


def _execute_preclose_vertical_height_settle(
    profile,
    rpc,
    fk,
    *,
    requested_down_m: float,
    support_up,
    settle_index: int,
) -> dict:
    """Execute the already-audited vertical preclose path.

    The Cartesian RPC is useful for teleoperation, but repeated identical
    3 mm endpoint requests settled by only roughly 0.2--0.5 mm on this Piper.
    The collision-audited joint path was already being generated and then
    discarded.  Stream that path directly with a bounded endpoint correction
    so a single settle actually reaches the intended support height.  A fresh
    image, measured jaw-level check, and measured height gate still run before
    closure.
    """

    requested_down_m = float(requested_down_m)
    if not np.isfinite(requested_down_m) or requested_down_m <= 0.0:
        raise ValueError("preclose height settle distance must be positive")
    support_up = np.asarray(support_up, dtype=float)
    support_up /= np.linalg.norm(support_up)
    duration_s = float(
        profile["execution"].get("preclose_height_settle_duration_s", 0.8)
    )
    start_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    stage = f"preclose_vertical_height_settle_{settle_index:02d}"
    samples, audit = _plan_straight_level_joint_lift(
        profile,
        fk,
        start_q=start_q,
        aperture=1.0,
        distance_m=-requested_down_m,
        duration_s=duration_s,
        stage=stage,
        maximum_planar_motion_m=float(
            profile["execution"].get(
                "preclose_height_settle_maximum_planar_motion_m", 0.0004
            )
        ),
        maximum_ik_position_error_m=float(
            profile["execution"].get(
                "preclose_height_settle_maximum_ik_position_error_m", 0.0004
            )
        ),
        minimum_progress_fraction=float(
            profile["execution"].get(
                "preclose_height_settle_minimum_progress_fraction", 0.95
            )
        ),
    )
    control = str(
        profile["execution"].get(
            "preclose_height_settle_control", "direct_joint"
        )
    )
    if control == "direct_joint":
        motion = _execute_direct_joint_samples(
            profile,
            rpc,
            fk,
            samples,
            final_tolerance_rad=float(
                profile["execution"].get(
                    "preclose_height_settle_joint_final_tolerance_rad", 0.002
                )
            ),
            endpoint_correction_gain=float(
                profile["execution"].get(
                    "preclose_height_settle_endpoint_correction_gain", 0.5
                )
            ),
            maximum_endpoint_correction_rad=float(
                profile["execution"].get(
                    "preclose_height_settle_maximum_endpoint_correction_rad",
                    0.03,
                )
            ),
            accumulate_endpoint_correction=True,
            require_final_convergence=True,
        )
    elif control == "cartesian":
        start_pose = np.asarray(
            rpc.get_right_ee_pose().parameters(), dtype=float
        )
        target_pose = start_pose.copy()
        target_pose[4:7] -= requested_down_m * support_up
        motion = _cartesian_move(
            profile,
            rpc,
            fk,
            target_pose=target_pose,
            duration_s=duration_s,
            aperture=1.0,
            stage=f"preclose_cartesian_height_settle_{settle_index:02d}",
            settle_s=float(
                profile["execution"].get(
                    "preclose_height_settle_cartesian_settle_s", 0.6
                )
            ),
        )
    else:
        raise ValueError(f"unsupported preclose height settle control: {control}")
    motion["height_settle_control"] = control
    motion["offline_vertical_audit"] = audit
    return motion


def _straight_lift(
    profile,
    rpc,
    fk,
    *,
    distance_m: float | None = None,
    duration_s: float | None = None,
    aperture: float = 0.0,
    stage: str = "verification_lift_straight_level",
    final_tolerance_rad: float | None = None,
    endpoint_correction_gain: float | None = None,
    maximum_endpoint_correction_rad: float | None = None,
    accumulate_endpoint_correction: bool = False,
    require_final_convergence: bool = True,
    maximum_planar_motion_m: float | None = None,
    maximum_ik_position_error_m: float | None = None,
    minimum_progress_fraction: float = 0.99,
) -> dict:
    start_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    samples, audit = _plan_straight_level_joint_lift(
        profile,
        fk,
        start_q=start_q,
        aperture=float(aperture),
        distance_m=distance_m,
        duration_s=duration_s,
        stage=stage,
        maximum_planar_motion_m=maximum_planar_motion_m,
        maximum_ik_position_error_m=maximum_ik_position_error_m,
        minimum_progress_fraction=minimum_progress_fraction,
    )
    result = _execute_direct_joint_samples(
        profile,
        rpc,
        fk,
        samples,
        final_tolerance_rad=(
            float(
                profile["execution"].get(
                    "lift_direct_joint_final_tolerance_rad", 0.015
                )
            )

            if final_tolerance_rad is None
            else float(final_tolerance_rad)
        ),
        endpoint_correction_gain=(
            float(
                profile["execution"].get(
                    "descent_direct_joint_endpoint_correction_gain", 1.0
                )
            )
            if endpoint_correction_gain is None
            else float(endpoint_correction_gain)
        ),
        maximum_endpoint_correction_rad=(
            float(
                profile["execution"].get(
                    "descent_direct_joint_maximum_endpoint_correction_rad", 0.05
                )
            )
            if maximum_endpoint_correction_rad is None
            else float(maximum_endpoint_correction_rad)
        ),
        accumulate_endpoint_correction=bool(accumulate_endpoint_correction),
        require_final_convergence=bool(require_final_convergence),
    )
    result["offline_vertical_audit"] = audit
    return result


def _staged_straight_lift(
    profile,
    rpc,
    fk,
    closure_calibration: ClosureCalibration,
    *,
    holding_aperture: float = 0.0,
) -> dict:
    """Lift a thin object in two vertical stages with a mechanical checkpoint.

    A full 20 mm lift hid where a shallow edge grasp slipped.  The first short
    stage keeps the same fixed orientation and close command, then reads only
    the calibrated gripper obstruction.  If the obstruction has disappeared,
    stop lifting immediately; the caller will retreat/open.  No image decision
    or Codex intervention is required.
    """

    trajectory = profile["trajectory"]
    requested_holding_aperture = float(holding_aperture)
    if not 0.0 <= requested_holding_aperture <= 1.0:
        raise ValueError("holding aperture must lie within [0, 1]")
    minimum_obstruction_gap = float(
        profile["closure"].get(
            "minimum_preloaded_obstruction_gap_ratio", 0.015
        )
    )
    total = float(trajectory["verification_lift_m"])
    initial = float(trajectory.get("initial_hold_check_lift_m", 0.007))
    if not 0.0 < initial < total:
        raise ValueError("initial hold-check lift must lie inside total lift")
    initial_duration = float(
        trajectory.get("initial_hold_check_lift_duration_s", 1.0)
    )
    total_duration = float(trajectory["verification_lift_duration_s"])
    remaining = total - initial
    remaining_duration = max(
        1.0 / CONTROL_HZ,
        total_duration - initial_duration,
    )
    first = _straight_lift(
        profile,
        rpc,
        fk,
        distance_m=initial,
        duration_s=initial_duration,
        # Keep the original closure command only for the prompt pickup.
        # Relaxing preload while the object still rests on support released it
        # before motion in physical trials.
        aperture=0.0,
        stage="verification_lift_initial_hold_check",
    )
    checkpoint = closure_calibration.classify(
        float(rpc.get_right_gripper_exact())
    )
    report = {
        "method": "prompt_full_close_pickup_then_finite_preload_lift",
        "initial_distance_m": initial,
        "remaining_distance_m": remaining,
        "initial_closure_commanded_open_ratio": 0.0,
        "requested_holding_aperture": requested_holding_aperture,
        "holding_aperture": 0.0,
        "minimum_obstruction_gap_ratio": minimum_obstruction_gap,
        "initial": first,
        "closure_after_initial_lift": checkpoint,
        "remaining": None,
        "completed_full_distance": False,
    }
    if not checkpoint["nonempty"]:
        return report
    measured_after_pickup = float(checkpoint["measured_open_ratio"])
    preload_delta = float(
        profile["closure"].get("holding_preload_ratio_delta", 0.06)
    )
    holding_aperture = max(0.0, measured_after_pickup - preload_delta)
    rpc.set_right_joint_target(
        np.asarray(rpc.get_right_joint_positions(), dtype=float),
        gripper_target=holding_aperture,
        preview_time=0.05,
    )
    report["holding_aperture"] = holding_aperture
    report["post_pickup_preload"] = {
        "commanded_open_ratio": holding_aperture,
        "source_measured_open_ratio": measured_after_pickup,
        "preload_ratio_delta": measured_after_pickup - holding_aperture,
    }
    report["remaining"] = _straight_lift(
        profile,
        rpc,
        fk,
        distance_m=remaining,
        duration_s=remaining_duration,
        aperture=holding_aperture,
        stage="verification_lift_remaining_vertical",
    )
    report["completed_full_distance"] = True
    return report


def _classify_preloaded_obstruction(
    measured_open_ratio: float,
    *,
    commanded_open_ratio: float,
    minimum_obstruction_gap_ratio: float,
) -> dict:
    """Detect an object while commanding finite, non-ejecting preload."""

    measured = float(measured_open_ratio)
    commanded = float(commanded_open_ratio)
    minimum_gap = float(minimum_obstruction_gap_ratio)
    if not 0.0 <= measured <= 1.0 or not 0.0 <= commanded <= 1.0:
        raise ValueError("preloaded gripper ratios must lie within [0, 1]")
    if minimum_gap <= 0.0:
        raise ValueError("minimum preloaded obstruction gap must be positive")
    gap = measured - commanded
    return {
        "measured_open_ratio": measured,
        "commanded_open_ratio": commanded,
        "obstruction_gap_ratio": gap,
        "minimum_obstruction_gap_ratio": minimum_gap,
        "nonempty": bool(gap >= minimum_gap),
        "method": "measured_minus_commanded_preload_gap",
    }


def _retreat_open(profile, rpc, fk) -> dict:
    current = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    safe = _q(profile["trajectory"]["safe_high_q_physical_rad"], "safe q")
    samples = sample_joint_knots(
        [
            {
                "stage": "retreat_start",
                "right_q_physical_rad": current.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "retreat_open",
                "right_q_physical_rad": safe.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": float(profile["trajectory"]["retreat_duration_s"]),
            },
        ]
    )
    return _execute_direct_joint_samples(profile, rpc, fk, samples)


def _wait_for_right_joint_settle(
    rpc,
    *,
    timeout_s: float = 5.0,
    poll_s: float = 0.10,
    maximum_delta_rad: float = 0.015,
    required_consecutive: int = 3,
    clock=time.monotonic,
    sleep=time.sleep,
) -> dict:
    """Wait read-only for an already accepted arm command to finish.

    A Cartesian command can move partway before a later incremental IK update
    is rejected.  Planning a joint-space recovery from the first readback in
    that state is unsafe: the controller may still be converging to the last
    accepted target.  This checkpoint sends no command and returns only after
    successive measured joint samples agree.
    """

    if timeout_s <= 0.0 or poll_s <= 0.0:
        raise ValueError("joint-settle timeout and poll period must be positive")
    if maximum_delta_rad <= 0.0 or required_consecutive < 1:
        raise ValueError("joint-settle tolerance and sample count must be positive")
    started = clock()
    previous = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    consecutive = 0
    sample_count = 1
    maximum_observed_delta = 0.0
    latest_delta = math.inf
    while clock() - started < timeout_s:
        sleep(poll_s)
        current = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        sample_count += 1
        latest_delta = float(np.max(np.abs(current - previous)))
        maximum_observed_delta = max(maximum_observed_delta, latest_delta)
        consecutive = consecutive + 1 if latest_delta <= maximum_delta_rad else 0
        previous = current
        if consecutive >= required_consecutive:
            return {
                "accepted": True,
                "read_only": True,
                "elapsed_s": float(clock() - started),
                "sample_count": sample_count,
                "maximum_observed_delta_rad": maximum_observed_delta,
                "final_sample_delta_rad": latest_delta,
                "q_physical_rad": current.tolist(),
            }
    raise TrajectoryStreamError(
        "right arm did not settle after a partially accepted Cartesian "
        f"command: final_sample_delta={latest_delta:.3f}rad"
    )


def _require_hover_level_after_endpoint_convergence(
    profile,
    rpc,
    fk,
    planned_hover_q,
    checkpoint: RightJawLevelCheckpoint,
) -> tuple[object, dict]:
    """Authorize descent from one measured level checkpoint.

    The descent planner constructs its low endpoint directly below the fresh
    measured hover, so equality with a nominal hover joint vector is no longer
    required.  Only an unlevel measured hover triggers one free-space endpoint
    correction, followed if necessary by a same-position leveling solve.
    """

    planned_hover_q = _q(planned_hover_q, "planned hover level endpoint")
    execution = profile["execution"]
    endpoint_tolerance = float(
        execution.get("hover_level_final_tolerance_rad", 0.012)
    )
    level_reference = _load_level_reference(profile["level_config"])
    up = np.asarray(level_reference.support_up_robot, dtype=float)
    up /= np.linalg.norm(up)
    planned_pose = np.asarray(fk.pose(planned_hover_q).parameters(), dtype=float)

    def branch_error(q_physical):
        q_physical = _q(q_physical, "measured hover branch q")
        measured_pose = np.asarray(fk.pose(q_physical).parameters(), dtype=float)
        delta = measured_pose[4:] - planned_pose[4:]
        height_error = float(delta @ up)
        planar_error = float(
            np.linalg.norm(delta - height_error * up)
        )
        return {
            "joint_error_rad": float(
                np.max(np.abs(q_physical - planned_hover_q))
            ),
            "planar_error_m": planar_error,
            "height_error_m": height_error,
            "q_physical_rad": q_physical.tolist(),
        }

    current = _q(rpc.get_right_joint_positions(), "measured hover endpoint q")
    initial_branch_error = branch_error(current)
    initial_level_error = None
    try:
        initial_assessment = checkpoint.require("before_descend")
    except RuntimeError as error:
        initial_assessment = None
        initial_level_error = str(error)
    if initial_assessment is not None:
        return initial_assessment, {
            "accepted": True,
            "endpoint_correction_required": False,
            "initial_branch_error": initial_branch_error,
            "endpoint_tolerance_rad": endpoint_tolerance,
            "descent_origin": "fresh_measured_level_hover",
        }

    duration = float(execution.get("hover_level_endpoint_duration_s", 0.8))
    samples = sample_joint_knots(
        [
            {
                "stage": "measured_unlevel_hover",
                "right_q_physical_rad": current.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "converge_audited_level_hover",
                "right_q_physical_rad": planned_hover_q.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": duration,
            },
        ]
    )
    motion_error = None
    try:
        motion = _execute_direct_joint_samples(
            profile,
            rpc,
            fk,
            samples,
            final_tolerance_rad=endpoint_tolerance,
            endpoint_correction_gain=float(
                execution.get("hover_level_endpoint_correction_gain", 1.0)
            ),
            maximum_endpoint_correction_rad=float(
                execution.get("hover_level_maximum_endpoint_correction_rad", 0.04)
            ),
            settle_timeout_s=float(
                execution.get("hover_level_endpoint_settle_timeout_s", 4.0)
            ),
            accumulate_endpoint_correction=True,
        )
    except TrajectoryStreamError as error:
        motion = None
        motion_error = f"{type(error).__name__}: {error}"

    joint_level_motion = None
    joint_level_audit = None
    endpoint_level_error = None
    try:
        assessment = checkpoint.require(
            "before_descend_after_endpoint_correction"
        )
    except RuntimeError as error:
        endpoint_level_error = str(error)
        level_samples, joint_level_audit = (
            _plan_same_position_level_joint_samples(
                profile,
                fk,
                start_q=rpc.get_right_joint_positions(),
            )
        )
        joint_level_motion = _execute_direct_joint_samples(
            profile,
            rpc,
            fk,
            level_samples,
            final_tolerance_rad=endpoint_tolerance,
            endpoint_correction_gain=float(
                profile["execution"].get(
                    "hover_level_endpoint_correction_gain", 1.0
                )
            ),
        )
        assessment = checkpoint.require(
            "before_descend_after_dense_joint_level_correction"
        )
    final_q = _q(rpc.get_right_joint_positions(), "final hover endpoint q")
    final_branch_error = branch_error(final_q)
    return assessment, {
        "accepted": True,
        "endpoint_correction_required": True,
        "initial_error": initial_level_error,
        "initial_branch_error": initial_branch_error,
        "final_branch_error": final_branch_error,
        "endpoint_tolerance_rad": endpoint_tolerance,
        "motion": motion,
        "motion_error": motion_error,
        "descent_origin": "fresh_measured_level_hover",
        "endpoint_level_error": endpoint_level_error,
        "joint_level_audit": joint_level_audit,
        "joint_level_motion": joint_level_motion,
    }


def _recover_vertical_then_open(profile, rpc, fk) -> dict:
    """Recover from low pose, allowing yaw only if exact Cartesian IK fails.

    The RPC's fixed-quaternion Cartesian recovery is preferred.  Some level
    wrist branches cannot move upward without changing yaw; rejecting that
    setpoint previously stranded an open gripper low over the support.  The
    fallback solves a 5 mm, fixed-XY, level pose and audits the complete FK
    interpolation for monotonic height and sub-millimetre planar drift before
    sending it through the same 30 Hz teleop path.
    """

    level_reference = _load_level_reference(profile["level_config"])
    requested = float(profile["trajectory"]["verification_lift_m"])
    try:
        return _streamer(profile, rpc, fk).recover_vertical_then_open(
            clearance_m=requested,
            support_up_robot=level_reference.support_up_robot,
        )
    except TrajectoryStreamError as direct_error:
        # The failed incremental Cartesian stream may already have accepted
        # several upward setpoints.  Let that safe upward motion finish, then
        # retry from the fresh measured pose.  The common failure mode observed
        # on hardware succeeds on this retry; the slower joint-space fallback
        # is reserved for two consecutive Cartesian rejections.
        settle_before_retry = _wait_for_right_joint_settle(rpc)
        try:
            retry = _streamer(profile, rpc, fk).recover_vertical_then_open(
                clearance_m=requested,
                support_up_robot=level_reference.support_up_robot,
            )
        except TrajectoryStreamError as retry_error:
            retry_error_text = f"{type(retry_error).__name__}: {retry_error}"
            settle_before_fallback = _wait_for_right_joint_settle(rpc)
        else:
            retry["method"] = "cartesian_retry_after_partial_rejection"
            retry["initial_error"] = (
                f"{type(direct_error).__name__}: {direct_error}"
            )
            retry["settle_before_retry"] = settle_before_retry
            return retry
        q0 = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        pose0 = fk.pose(q0)
        parameters0 = np.asarray(pose0.parameters(), dtype=float)
        up = np.asarray(level_reference.support_up_robot, dtype=float)
        up /= np.linalg.norm(up)
        clearance = min(requested, 0.005)
        target_position = parameters0[4:] + clearance * up
        angle_scale = math.sin(
            math.radians(level_reference.maximum_checkpoint_tilt_deg)
        )
        joint_ids = fk.solver.dof_ids
        lower = np.asarray(fk.solver.model.jnt_range[joint_ids, 0], dtype=float) + 1e-4
        upper = np.asarray(fk.solver.model.jnt_range[joint_ids, 1], dtype=float) - 1e-4
        upper[5] = min(
            upper[5],
            float(profile["head_localization"].get("maximum_joint6_rad", 2.98)),
        )

        def residual(q_physical):
            pose = fk.pose(q_physical)
            parameters = np.asarray(pose.parameters(), dtype=float)
            rotation = pose.as_matrix()[:3, :3]
            return np.concatenate(
                [
                    (parameters[4:] - target_position) / 0.002,
                    (rotation[:, 1] - up) / max(angle_scale, 1e-6),
                    0.003 * (q_physical - q0),
                ]
            )

        solved = least_squares(
            residual,
            q0,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=500,
        )
        lifted_parameters = np.asarray(fk.pose(solved.x).parameters(), dtype=float)
        position_error = float(
            np.linalg.norm(lifted_parameters[4:] - target_position)
        )
        level = assess_jaw_level(
            lifted_parameters,
            level_reference,
            planned=True,
        )
        planar_drifts = []
        heights = []
        for fraction in np.linspace(0.0, 1.0, 121):
            blend = _minimum_jerk(float(fraction))
            parameters = np.asarray(
                fk.pose(q0 + blend * (solved.x - q0)).parameters(), dtype=float
            )
            delta = parameters[4:] - parameters0[4:]
            planar = delta - up * float(delta @ up)
            planar_drifts.append(float(np.linalg.norm(planar)))
            heights.append(float(delta @ up))
        maximum_planar_drift = max(planar_drifts)
        minimum_height_step = float(np.min(np.diff(heights)))
        accepted = bool(
            solved.success
            and position_error <= 0.001
            and level.accepted
            and maximum_planar_drift <= 0.001
            and minimum_height_step >= -0.0002
            and heights[-1] >= 0.8 * clearance
        )
        if not accepted:
            raise TrajectoryStreamError(
                "level-yaw vertical recovery fallback failed offline audit: "
                f"position_error={position_error}, level={level.to_dict()}, "
                f"maximum_planar_drift={maximum_planar_drift}, "
                f"minimum_height_step={minimum_height_step}"
            ) from direct_error
        aperture = float(rpc.get_right_gripper_exact())
        samples = sample_joint_knots(
            [
                {
                    "stage": "low_measured",
                    "right_q_physical_rad": q0.tolist(),
                    "right_gripper_open_ratio": aperture,
                    "minimum_duration_s": 0.1,
                },
                {
                    "stage": "vertical_recovery_level_yaw_free",
                    "right_q_physical_rad": solved.x.tolist(),
                    "right_gripper_open_ratio": aperture,
                    "minimum_duration_s": 0.8,
                },
            ]
        )
        lift = _streamer(profile, rpc, fk).execute(samples)
        opening = _fixed_pose_gripper_ramp(
            profile,
            rpc,
            fk,
            finish_ratio=1.0,
            duration_s=0.5,
            stage="open_after_level_yaw_free_recovery",
        )
        return {
            "completed": True,
            "method": "level_yaw_free_joint_bridge_after_cartesian_rejection",
            "direct_error": f"{type(direct_error).__name__}: {direct_error}",
            "retry_error": retry_error_text,
            "settle_before_retry": settle_before_retry,
            "settle_before_fallback": settle_before_fallback,
            "clearance_m": clearance,
            "position_error_m": position_error,
            "maximum_planar_drift_m": maximum_planar_drift,
            "minimum_height_step_m": minimum_height_step,
            "level": level.to_dict(),
            "lift": lift,
            "opening": opening,
        }


def _place_open_retreat(
    profile,
    rpc,
    fk,
    support_pose: np.ndarray,
    *,
    holding_aperture: float = 0.0,
) -> dict:
    placement = _cartesian_move(
        profile,
        rpc,
        fk,
        target_pose=support_pose,
        duration_s=float(profile["trajectory"]["place_duration_s"]),
        # Maintain closure pressure until the object is back on support.
        aperture=float(holding_aperture),
        stage="return_object_to_support",
    )
    time.sleep(float(profile["trajectory"]["support_settle_s"]))
    opening = _fixed_pose_gripper_ramp(
        profile,
        rpc,
        fk,
        finish_ratio=1.0,
        duration_s=float(profile["trajectory"]["open_duration_s"]),
        stage="open_on_support",
    )
    retreat = _retreat_open(profile, rpc, fk)
    return {"placement": placement, "opening": opening, "retreat": retreat}


def _nearest_marker(image: np.ndarray, reference_px, maximum_fraction: float):
    centers = detect_blue_cross_centers(image)
    if not centers:
        raise RuntimeError("selected target marker disappeared")
    reference = np.asarray(reference_px, dtype=float)
    distances = np.asarray([np.linalg.norm(center - reference) for center in centers])
    selected = int(np.argmin(distances))
    normalized = float(distances[selected] / np.linalg.norm(image.shape[:2]))
    if normalized > float(maximum_fraction):
        raise RuntimeError("no target marker is near its previous camera location")
    return np.asarray(centers[selected], dtype=float), normalized


def _reacquire_direct_preclose_cross(
    image: np.ndarray,
    predicted_anchor,
    identity_anchor,
    template: GraspWindowTemplate,
    *,
    maximum_displacement_diagonal_fraction: float,
    minimum_component_area_scale: float,
    maximum_component_area_scale: float,
):
    """Reacquire the verified mark after a large camera-stage parallax.

    Ranking all blue components by area can confidently jump from the printed
    cross to a reflection in the neighbouring dish. First require a direct
    cross-shaped detection and choose the one nearest the calibrated parallax
    prediction. Then recover its complete permissive-blue component locally
    and audit it against the pre-descent identity anchor.
    """

    image = np.asarray(image)
    height, width = image.shape[:2]
    diagonal = math.hypot(width, height)
    centers = detect_blue_cross_centers(image)
    predicted_center = np.asarray(predicted_anchor.center_px, dtype=float)
    identity_center = np.asarray(identity_anchor.center_px, dtype=float)
    identity_pixels = max(1, int(identity_anchor.component_pixels))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    permissive = cv2.inRange(
        hsv,
        np.asarray([95, 20, 40], dtype=np.uint8),
        np.asarray([130, 255, 255], dtype=np.uint8),
    )
    permissive = cv2.morphologyEx(
        permissive, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, component_centers = cv2.connectedComponentsWithStats(
        permissive
    )
    verified_direct_centers = []
    for center in centers:
        cx = int(np.clip(round(float(center[0])), 0, width - 1))
        cy = int(np.clip(round(float(center[1])), 0, height - 1))
        label = int(labels[cy, cx])
        if label <= 0:
            continue
        pixels = int(stats[label, cv2.CC_STAT_AREA])
        area_scale = pixels / float(identity_pixels)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = bool(
            x <= 1
            or y <= 1
            or x + component_width >= width - 1
            or y + component_height >= height - 1
        )
        identity_displacement = float(
            np.linalg.norm(np.asarray(center, dtype=float) - identity_center)
            / diagonal
        )
        if (
            not touches_border
            and minimum_component_area_scale
            <= area_scale
            <= maximum_component_area_scale
            and identity_displacement
            <= float(maximum_displacement_diagonal_fraction)
        ):
            verified_direct_centers.append(np.asarray(center, dtype=float))
    direct_cross_detected = bool(verified_direct_centers)
    detection_method = "direct_cross_shape"
    if verified_direct_centers:
        distances = np.asarray(
            [
                np.linalg.norm(np.asarray(center, dtype=float) - predicted_center)
                for center in verified_direct_centers
            ]
        )
        selected_center = np.asarray(
            verified_direct_centers[int(np.argmin(distances))], dtype=float
        )
        prediction_displacement = float(np.min(distances) / diagonal)
    else:
        prediction_displacement = math.inf
    if (
        not direct_cross_detected
        or prediction_displacement
        > float(maximum_displacement_diagonal_fraction)
    ):
        # Brightness/exposure changes can merge the printed arms of the cross
        # into a non-cross-shaped blue component.  Recover only a component
        # whose area matches the last verified mark and whose centroid remains
        # inside both the predicted and identity continuity envelopes.  This
        # explicitly rejects the much larger cyan gripper body.
        component_candidates = []
        for label in range(1, count):
            pixels = int(stats[label, cv2.CC_STAT_AREA])
            area_scale = pixels / float(identity_pixels)
            center = np.asarray(component_centers[label], dtype=float)
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            touches_border = bool(
                x <= 1
                or y <= 1
                or x + component_width >= width - 1
                or y + component_height >= height - 1
            )
            predicted_displacement = float(
                np.linalg.norm(center - predicted_center) / diagonal
            )
            identity_displacement = float(
                np.linalg.norm(center - identity_center) / diagonal
            )
            if (
                not touches_border
                and minimum_component_area_scale
                <= area_scale
                <= maximum_component_area_scale
                and predicted_displacement
                <= float(maximum_displacement_diagonal_fraction)
                and identity_displacement
                <= float(maximum_displacement_diagonal_fraction)
            ):
                component_candidates.append(
                    (
                        abs(math.log(max(area_scale, 1e-12)))
                        + predicted_displacement,
                        center,
                        predicted_displacement,
                    )
                )
        if not component_candidates:
            if direct_cross_detected:
                raise ValueError(
                    "nearest direct preclose cross is outside the parallax "
                    f"envelope: displacement={prediction_displacement:.3f}"
                )
            raise ValueError(
                "no area-and-continuity verified blue marker component is "
                "visible for preclose reacquisition"
            )
        _, selected_center, prediction_displacement = min(
            component_candidates, key=lambda item: item[0]
        )
        direct_cross_detected = False
        detection_method = "permissive_blue_area_continuity"
    selected_uv = predicted_anchor.tool_frame.image_to_tool([selected_center])[0]
    direct_anchor = replace(
        predicted_anchor,
        center_px=(float(selected_center[0]), float(selected_center[1])),
        center_uv=(float(selected_uv[0]), float(selected_uv[1])),
        source=(
            "blue_marker"
            if direct_cross_detected
            else "local_blue_evidence_continuity"
        ),
        marker_cross_shaped=direct_cross_detected,
    )
    candidate = observe_local_blue_evidence_target(
        image,
        direct_anchor,
        template,
        maximum_target_displacement_diagonal_fraction=float(
            min(0.03, maximum_displacement_diagonal_fraction)
        ),
        minimum_component_area_scale=minimum_component_area_scale,
        maximum_component_area_scale=maximum_component_area_scale,
    )
    identity_displacement = float(
        np.linalg.norm(
            np.asarray(candidate.center_px, dtype=float)
            - np.asarray(identity_anchor.center_px, dtype=float)
        )
        / diagonal
    )
    area_scale = float(
        candidate.component_pixels / max(1, identity_anchor.component_pixels)
    )
    accepted = bool(
        not candidate.component_touches_border
        and identity_displacement <= float(maximum_displacement_diagonal_fraction)
        and minimum_component_area_scale <= area_scale <= maximum_component_area_scale
    )
    if not accepted:
        raise ValueError(
            "direct preclose cross failed identity-envelope audit: "
            f"displacement={identity_displacement:.3f}, "
            f"area_scale={area_scale:.3f}, border={candidate.component_touches_border}"
        )
    return replace(
        candidate,
        source=(
            "blue_marker"
            if direct_cross_detected
            else "local_blue_evidence_continuity"
        ),
        marker_cross_shaped=direct_cross_detected,
    ), {
        "detection_method": detection_method,
        "prediction_displacement_diagonal_fraction": prediction_displacement,
        "identity_displacement_diagonal_fraction": identity_displacement,
        "component_area_scale": area_scale,
        "candidate_count": len(centers),
    }


def _reacquire_persistent_hover_identity(
    image: np.ndarray,
    global_observation,
    identity_anchor: dict,
    template: GraspWindowTemplate,
    *,
    maximum_displacement_diagonal_fraction: float,
):
    """Choose the direct cross continuous with the last verified hover.

    The neighbouring dish can contain a blue reflection that independently
    passes the cross-shape test. Since every grasp attempt returns to the same
    rigid wrist-camera stage, the tapped object's image location is a stronger
    identity invariant than global component score.
    """

    image = np.asarray(image)
    height, width = image.shape[:2]
    diagonal = math.hypot(width, height)
    saved_center = np.asarray(identity_anchor.get("center_px"), dtype=float)
    saved_pixels = int(identity_anchor.get("component_pixels", 0))
    if (
        saved_center.shape != (2,)
        or not np.all(np.isfinite(saved_center))
        or saved_pixels <= 0
    ):
        raise ValueError("persistent hover identity anchor is invalid")
    saved_uv = global_observation.tool_frame.image_to_tool([saved_center])[0]
    continuity_anchor = replace(
        global_observation,
        center_px=(float(saved_center[0]), float(saved_center[1])),
        center_uv=(float(saved_uv[0]), float(saved_uv[1])),
        component_pixels=saved_pixels,
        component_area_per_tool_scale_sq=float(
            saved_pixels / global_observation.tool_frame.scale_px**2
        ),
        source="persisted_direct_cross_identity_anchor",
        marker_cross_shaped=True,
    )
    candidate = observe_local_blue_evidence_target(
        image,
        continuity_anchor,
        template,
        maximum_target_displacement_diagonal_fraction=float(
            maximum_displacement_diagonal_fraction
        ),
    )
    displacement = float(
        np.linalg.norm(np.asarray(candidate.center_px) - saved_center) / diagonal
    )
    # If the cross is fully visible, renew direct proof. At the image border
    # one arm of the printed cross may be clipped; the tightly bounded fresh
    # colour component is then explicit continuity from the persisted direct
    # proof, not a new global semantic initialization.
    centers = detect_blue_cross_centers(image)
    direct = False
    if centers:
        direct_distance = min(
            np.linalg.norm(
                np.asarray(center, dtype=float)
                - np.asarray(candidate.center_px, dtype=float)
            )
            for center in centers
        )
        direct = bool(direct_distance / diagonal <= 0.03)
    source = (
        "blue_marker"
        if direct
        else "persisted_direct_cross_identity_continuity"
    )
    return replace(candidate, source=source, marker_cross_shaped=direct), {
        "saved_center_px": saved_center.tolist(),
        "selected_center_px": list(candidate.center_px),
        "displacement_diagonal_fraction": displacement,
        "maximum_displacement_diagonal_fraction": float(
            maximum_displacement_diagonal_fraction
        ),
        "direct_candidate_count": len(centers),
        "direct_cross_renewed": direct,
    }


def _metric_replan(
    profile,
    observation,
    current_q,
    *,
    fk=None,
    reference_center_uv=None,
    servo_state=None,
    measured_current_xy_m=None,
    fixed_orientation_q=None,
    level_yaw_free: bool = False,
) -> tuple[np.ndarray, dict]:
    """Plan one trust-region correction in physical support-plane coordinates.

    The Jacobian maps measured physical EE translation to *relative* image
    error.  It is initialized from safe hover-only probes and refined online
    with Broyden updates.  No semantic-scene axis convention is used here.
    """
    target = profile["target_identity"]
    perception = profile["perception"]
    template = GraspWindowTemplate.from_dict(
        _load(target["grasp_window_selection"])["template"]
    )
    reference_center = np.asarray(
        target.get("canonical_preclose_goal_uv", template.reference_center_uv)
        if reference_center_uv is None
        else reference_center_uv,
        dtype=float,
    )
    error = np.asarray(observation.center_uv) - reference_center
    fk = ProductionRightFK(profile["production_model"]) if fk is None else fk
    current_q = _q(current_q, "visual-servo low q")
    current_position = np.asarray(fk.pose(current_q).parameters(), dtype=float)[4:]
    current_xy = current_position[:2].copy()
    if measured_current_xy_m is not None:
        measured_xy = np.asarray(measured_current_xy_m, dtype=float)
        if measured_xy.shape != (2,) or not np.all(np.isfinite(measured_xy)):
            raise ValueError("measured visual-servo XY must contain two finite values")
        current_xy = measured_xy.copy()
        current_position[:2] = measured_xy
    state = {} if servo_state is None else servo_state
    configured_jacobian = np.asarray(
        perception["hover_error_jacobian_uv_per_physical_m"], dtype=float
    )
    if configured_jacobian.shape != (2, 2) or not np.all(
        np.isfinite(configured_jacobian)
    ):
        raise ValueError("hover error Jacobian must be a finite 2x2 matrix")
    jacobian = np.asarray(state.get("jacobian", configured_jacobian), dtype=float)
    frame_size = np.asarray(
        perception.get("right_camera_frame_size_px", [640, 480]), dtype=float
    )
    if frame_size.shape != (2,) or np.any(frame_size <= 0.0):
        raise ValueError("right camera frame size must contain positive width and height")
    center_px = np.asarray(getattr(observation, "center_px", [math.inf, math.inf]))
    border_clearance_px = math.inf
    if center_px.shape == (2,) and np.all(np.isfinite(center_px)):
        border_clearance_px = float(
            min(
                center_px[0],
                center_px[1],
                frame_size[0] - center_px[0],
                frame_size[1] - center_px[1],
            )
        )
    component_interior_extent_px = getattr(
        observation, "component_interior_extent_px", None
    )
    if (
        getattr(observation, "component_touches_border", False)
        and component_interior_extent_px is not None
        and np.isfinite(float(component_interior_extent_px))
        and float(component_interior_extent_px) > 0.0
    ):
        # A clipped component's centroid is a censored and reflection-sensitive
        # statistic.  Its visible extent from the touched border grows as the
        # object enters the frame and is the correct progress signal here.
        border_clearance_px = float(component_interior_extent_px)
    border_progress_method = (
        "component_interior_extent"
        if component_interior_extent_px is not None
        and getattr(observation, "component_touches_border", False)
        else "component_center_clearance"
    )
    minimum_border_clearance_px = float(
        perception.get("minimum_runtime_axis_border_clearance_px", 0.0)
    )
    near_image_border = border_clearance_px < minimum_border_clearance_px
    border_search = bool(
        getattr(observation, "component_touches_border", False)
        or near_image_border
    )
    current_component_area = getattr(
        observation, "component_area_per_tool_scale_sq", None
    )
    if current_component_area is not None:
        current_component_area = float(current_component_area)
        if not np.isfinite(current_component_area) or current_component_area <= 0.0:
            current_component_area = None
    calibration_invalidation = None
    prior_calibration = state.get("runtime_axis_calibration")
    freeze_calibrated_jacobian = bool(
        perception.get("freeze_runtime_axis_calibration_jacobian", False)
    )
    if (
        bool(state.get("enable_runtime_axis_calibration", False))
        and isinstance(prior_calibration, dict)
        and prior_calibration.get("completed") is True
    ):
        calibration_origin = np.asarray(
            prior_calibration.get("baseline_xy_m", []), dtype=float
        )
        maximum_calibration_radius = float(
            perception.get("maximum_runtime_axis_calibration_radius_m", 0.015)
        )
        calibration_distance = (
            float(np.linalg.norm(current_xy - calibration_origin))
            if calibration_origin.shape == (2,)
            and np.all(np.isfinite(calibration_origin))
            else math.inf
        )
        if calibration_distance > maximum_calibration_radius:
            calibration_invalidation = {
                "reason": "camera_pose_left_local_axis_calibration_region",
                "distance_m": calibration_distance,
                "maximum_radius_m": maximum_calibration_radius,
            }
            state["runtime_axis_calibration"] = {}
            for key in (
                "jacobian",
                "last_xy_m",
                "last_error_uv",
                "last_component_area_per_tool_scale_sq",
            ):
                state.pop(key, None)
            jacobian = configured_jacobian.copy()
        elif freeze_calibrated_jacobian:
            # The two deliberate cardinal probes have much higher signal to
            # noise than a single small servo step.  Preserve that measured
            # local mapping inside its audited radius; undamped rank-one
            # Broyden updates otherwise learn centroid jitter and can make the
            # correction oscillate even while the wrist orientation is fixed.
            frozen_jacobian = np.asarray(
                prior_calibration.get("jacobian_uv_per_physical_m", []),
                dtype=float,
            )
            if frozen_jacobian.shape == (2, 2) and np.all(
                np.isfinite(frozen_jacobian)
            ):
                jacobian = frozen_jacobian
                state["jacobian"] = frozen_jacobian.tolist()
    update = None
    calibration_probe_delta = None
    calibration_probe_stage = None
    calibration_completed_now = False
    if bool(state.get("enable_runtime_axis_calibration", False)):
        calibration = state.setdefault("runtime_axis_calibration", {})
        if (
            calibration.get("completed") is not True
            and not border_search
            and calibration.get("border_probe_axis") is not None
        ):
            # Border escape can contain tens of millimetres spread across both
            # physical axes.  It is not one of the two local Jacobian probes.
            # Once the complete component has adequate margin, establish a
            # fresh local origin and calibrate +X/+Y from scratch.
            calibration.clear()
        if calibration.get("completed") is not True:
            probe_m = float(
                perception.get("runtime_hover_axis_probe_m", 0.006)
            )
            near_goal_probe_m = float(
                perception.get(
                    "runtime_hover_axis_near_goal_probe_m", probe_m
                )
            )
            near_goal_error_norm = float(
                perception.get(
                    "runtime_hover_axis_near_goal_error_norm", 0.0
                )
            )
            if (
                not border_search
                and near_goal_error_norm > 0.0
                and float(np.linalg.norm(error)) <= near_goal_error_norm
            ):
                probe_m = min(probe_m, near_goal_probe_m)
            minimum_response_m = float(
                perception.get("minimum_runtime_axis_probe_response_m", 0.001)
            )
            if probe_m <= 0.0 or minimum_response_m <= 0.0:
                raise ValueError("runtime hover-axis calibration distances must be positive")
            if calibration.pop("retry_y_opposite_from_boundary", False):
                if "x_delta_xy_m" not in calibration:
                    raise RuntimeError(
                        "cannot reverse runtime Y probe before X response exists"
                    )
                # The failed +Y request may have moved slightly before the
                # controller latched.  Rebase the second independent sample
                # at that measured state and probe the reachable -Y side.
                calibration["x_pose_xy_m"] = current_xy.tolist()
                calibration["x_pose_error_uv"] = error.tolist()
                calibration["stage"] = "y_opposite_probe_commanded"
                calibration_probe_delta = np.asarray([0.0, -probe_m])
                calibration_probe_stage = "runtime_axis_y_probe_opposite"
            elif "baseline_xy_m" not in calibration:
                calibration.update(
                    {
                        "baseline_xy_m": current_xy.tolist(),
                        "baseline_error_uv": error.tolist(),
                        "baseline_center_px": center_px.tolist(),
                        "baseline_border_clearance_px": border_clearance_px,
                        "last_border_progress_method": border_progress_method,
                        "baseline_low_q_physical_rad": current_q.tolist(),
                        "stage": "x_probe_commanded",
                        "x_probe_direction": 1.0,
                    }
                )
                calibration_probe_delta = np.asarray([probe_m, 0.0])
                calibration_probe_stage = "runtime_axis_x_probe"
            elif "x_delta_xy_m" not in calibration:
                baseline_xy = np.asarray(calibration["baseline_xy_m"], dtype=float)
                baseline_error = np.asarray(
                    calibration["baseline_error_uv"], dtype=float
                )
                x_delta = current_xy - baseline_xy
                if float(np.linalg.norm(x_delta)) < minimum_response_m:
                    raise RuntimeError(
                        "runtime X-axis calibration probe did not produce enough measured motion"
                    )
                baseline_clearance = float(
                    calibration.get("baseline_border_clearance_px", -math.inf)
                )
                minimum_border_progress_px = float(
                    perception.get("minimum_runtime_border_progress_px", 0.5)
                )
                border_nonprogress_patience = int(
                    perception.get("runtime_border_nonprogress_patience", 1)
                )
                if border_search:
                    border_axis = str(
                        calibration.get("border_probe_axis", "x")
                    )
                    if border_axis == "x":
                        direction = float(
                            calibration.get("x_probe_direction", 1.0)
                        )
                        comparison_clearance = float(
                            calibration.get(
                                "last_border_clearance_px", baseline_clearance
                            )
                        )
                        if calibration.get("last_border_progress_method") != (
                            border_progress_method
                        ):
                            comparison_clearance = -math.inf
                        improvement = (
                            border_clearance_px - comparison_clearance
                        )
                        if improvement > minimum_border_progress_px:
                            calibration["border_x_nonprogress_count"] = 0
                            calibration["stage"] = "x_border_clearance_commanded"
                            calibration["last_border_clearance_px"] = border_clearance_px
                            calibration["last_border_progress_method"] = (
                                border_progress_method
                            )
                            calibration_probe_delta = np.asarray(
                                [direction * probe_m, 0.0]
                            )
                            calibration_probe_stage = "runtime_axis_x_border_clearance"
                        elif int(
                            calibration.get("border_x_nonprogress_count", 0)
                        ) < border_nonprogress_patience:
                            calibration["border_x_nonprogress_count"] = int(
                                calibration.get("border_x_nonprogress_count", 0)
                            ) + 1
                            calibration["stage"] = "x_border_hysteresis_probe"
                            calibration_probe_delta = np.asarray(
                                [direction * probe_m, 0.0]
                            )
                            calibration_probe_stage = "runtime_axis_x_border_hysteresis"
                        elif direction > 0.0:
                            # Cross the baseline once and test the other X side.
                            calibration["x_probe_direction"] = -1.0
                            calibration["border_x_nonprogress_count"] = 0
                            calibration["stage"] = "x_probe_direction_reversed"
                            calibration_probe_delta = np.asarray(
                                [-2.0 * probe_m, 0.0]
                            )
                            calibration_probe_stage = "runtime_axis_x_probe_reverse"
                        else:
                            # X is nearly tangent to this image boundary.  A
                            # camera-frame assumption would fail here, so
                            # switch to physical Y and test both signs before
                            # declaring the target unreachable.
                            calibration["border_probe_axis"] = "y"
                            calibration["y_probe_direction"] = 1.0
                            calibration["border_y_nonprogress_count"] = 0
                            calibration["border_y_baseline_clearance_px"] = (
                                border_clearance_px
                            )
                            calibration["last_border_clearance_px"] = (
                                border_clearance_px
                            )
                            calibration["last_border_progress_method"] = (
                                border_progress_method
                            )
                            calibration["stage"] = "y_border_probe_commanded"
                            calibration_probe_delta = np.asarray([0.0, probe_m])
                            calibration_probe_stage = "runtime_axis_y_border_probe"
                    else:
                        direction = float(
                            calibration.get("y_probe_direction", 1.0)
                        )
                        y_baseline_clearance = float(
                            calibration.get(
                                "border_y_baseline_clearance_px",
                                border_clearance_px,
                            )
                        )
                        comparison_clearance = float(
                            calibration.get(
                                "last_border_clearance_px",
                                y_baseline_clearance,
                            )
                        )
                        if calibration.get("last_border_progress_method") != (
                            border_progress_method
                        ):
                            comparison_clearance = -math.inf
                        improvement = (
                            border_clearance_px - comparison_clearance
                        )
                        if improvement > minimum_border_progress_px:
                            calibration["border_y_nonprogress_count"] = 0
                            calibration["stage"] = "y_border_clearance_commanded"
                            calibration["last_border_clearance_px"] = border_clearance_px
                            calibration["last_border_progress_method"] = (
                                border_progress_method
                            )
                            calibration_probe_delta = np.asarray(
                                [0.0, direction * probe_m]
                            )
                            calibration_probe_stage = "runtime_axis_y_border_clearance"
                        elif int(
                            calibration.get("border_y_nonprogress_count", 0)
                        ) < border_nonprogress_patience:
                            calibration["border_y_nonprogress_count"] = int(
                                calibration.get("border_y_nonprogress_count", 0)
                            ) + 1
                            calibration["stage"] = "y_border_hysteresis_probe"
                            calibration_probe_delta = np.asarray(
                                [0.0, direction * probe_m]
                            )
                            calibration_probe_stage = "runtime_axis_y_border_hysteresis"
                        elif direction > 0.0:
                            calibration["y_probe_direction"] = -1.0
                            calibration["border_y_nonprogress_count"] = 0
                            calibration["stage"] = "y_probe_direction_reversed"
                            calibration_probe_delta = np.asarray(
                                [0.0, -2.0 * probe_m]
                            )
                            calibration_probe_stage = "runtime_axis_y_probe_reverse"
                        else:
                            raise RuntimeError(
                                "runtime XY probes did not move the target away from the image border"
                            )
                    # Do not estimate an image Jacobian from a censored rim.
                    # Continue the cardinal clearance search first.
                    x_delta = None
                if calibration_probe_delta is not None:
                    pass
                else:
                    calibration["x_delta_xy_m"] = x_delta.tolist()
                    calibration["x_delta_error_uv"] = (
                        error - baseline_error
                    ).tolist()
                    calibration["x_pose_xy_m"] = current_xy.tolist()
                    calibration["x_pose_error_uv"] = error.tolist()
                    # Continue from the inward X probe instead of returning to
                    # the baseline before probing Y.
                    calibration_probe_delta = np.asarray([0.0, probe_m])
                    calibration_probe_stage = "runtime_axis_y_probe"
                    calibration["stage"] = "y_probe_commanded"
            else:
                x_pose_xy = np.asarray(calibration["x_pose_xy_m"], dtype=float)
                x_pose_error = np.asarray(
                    calibration["x_pose_error_uv"], dtype=float
                )
                y_delta = current_xy - x_pose_xy
                if float(np.linalg.norm(y_delta)) < minimum_response_m:
                    raise RuntimeError(
                        "runtime Y-axis calibration probe did not produce enough measured motion"
                    )
                motion = np.column_stack(
                    [
                        np.asarray(calibration["x_delta_xy_m"], dtype=float),
                        y_delta,
                    ]
                )
                response = np.column_stack(
                    [
                        np.asarray(
                            calibration["x_delta_error_uv"], dtype=float
                        ),
                        error - x_pose_error,
                    ]
                )
                motion_condition = float(np.linalg.cond(motion))
                maximum_motion_condition = float(
                    perception.get(
                        "maximum_runtime_axis_motion_condition", 12.0
                    )
                )
                if (
                    not np.all(np.isfinite(motion))
                    or motion_condition > maximum_motion_condition
                ):
                    # Height settling and same-position roll correction can
                    # add a few millimetres of planar tracking residual between
                    # process-replayed probes.  If that makes the two measured
                    # probe columns nearly collinear, do not fit an explosive
                    # Jacobian.  A previously measured, well-conditioned local
                    # Jacobian may originate one bounded visual step instead;
                    # closure still requires a fresh low image afterwards.
                    prior_jacobian = np.asarray(
                        state.get("jacobian", []), dtype=float
                    )
                    prior_xy = np.asarray(
                        state.get("last_xy_m", []), dtype=float
                    )
                    maximum_local_radius = float(
                        perception.get(
                            "maximum_runtime_axis_calibration_radius_m", 0.015
                        )
                    )
                    prior_distance = (
                        float(np.linalg.norm(current_xy - prior_xy))
                        if prior_xy.shape == (2,)
                        and np.all(np.isfinite(prior_xy))
                        else math.inf
                    )
                    prior_condition = (
                        float(np.linalg.cond(prior_jacobian))
                        if prior_jacobian.shape == (2, 2)
                        and np.all(np.isfinite(prior_jacobian))
                        else math.inf
                    )
                    maximum_jacobian_condition = float(
                        perception.get("maximum_jacobian_condition", 100.0)
                    )
                    maximum_gain = float(
                        perception.get(
                            "maximum_hover_jacobian_gain_uv_per_m", 100.0
                        )
                    )
                    prior_gain = (
                        float(np.linalg.norm(prior_jacobian, ord=2))
                        if prior_jacobian.shape == (2, 2)
                        and np.all(np.isfinite(prior_jacobian))
                        else math.inf
                    )
                    if not (
                        prior_distance <= maximum_local_radius
                        and prior_condition <= maximum_jacobian_condition
                        and prior_gain <= maximum_gain
                    ):
                        raise RuntimeError(
                            "runtime hover-axis calibration probes were not independent"
                        )
                    jacobian = prior_jacobian
                    calibration.update(
                        {
                            "completed": True,
                            "stage": "completed_with_prior_local_jacobian",
                            "baseline_xy_m": current_xy.tolist(),
                            "y_delta_xy_m": y_delta.tolist(),
                            "y_delta_error_uv": (
                                error - x_pose_error
                            ).tolist(),
                            "motion_condition": motion_condition,
                            "jacobian_condition": prior_condition,
                            "jacobian_uv_per_physical_m": jacobian.tolist(),
                            "prior_sample_distance_m": prior_distance,
                            "fresh_low_image_required": True,
                        }
                    )
                    state["jacobian"] = jacobian.tolist()
                    calibration_completed_now = True
                    update = {
                        "accepted": True,
                        "reason": (
                            "prior_local_jacobian_after_degenerate_runtime_probes"
                        ),
                        "motion_condition": motion_condition,
                        "condition": prior_condition,
                        "prior_sample_distance_m": prior_distance,
                    }
                else:
                    calibrated = response @ np.linalg.inv(motion)
                    calibrated_condition = float(np.linalg.cond(calibrated))
                    maximum_gain = float(
                        perception.get(
                            "maximum_hover_jacobian_gain_uv_per_m", 100.0
                        )
                    )
                    if (
                        not np.all(np.isfinite(calibrated))
                        or calibrated_condition
                        > float(
                            perception.get(
                                "maximum_jacobian_condition", 100.0
                            )
                        )
                        or float(np.linalg.norm(calibrated, ord=2)) > maximum_gain
                    ):
                        raise RuntimeError(
                            "runtime hover-axis image Jacobian failed conditioning gates"
                        )
                    jacobian = calibrated
                    calibration.update(
                        {
                            "completed": True,
                            "stage": "completed",
                            "y_delta_xy_m": y_delta.tolist(),
                            "y_delta_error_uv": (
                                error - x_pose_error
                            ).tolist(),
                            "motion_condition": motion_condition,
                            "jacobian_condition": calibrated_condition,
                            "jacobian_uv_per_physical_m": calibrated.tolist(),
                        }
                    )
                    state["jacobian"] = calibrated.tolist()
                    calibration_completed_now = True
                    update = {
                        "accepted": True,
                        "reason": "runtime_cartesian_axis_calibration",
                        "motion_condition": motion_condition,
                        "condition": calibrated_condition,
                    }
                # The two deliberately separated calibration observations
                # supersede any cross-process last-sample Broyden state.
                state.pop("last_xy_m", None)
                state.pop("last_error_uv", None)
                state.pop("last_component_area_per_tool_scale_sq", None)
    completed_calibration = state.get("runtime_axis_calibration")
    frozen_runtime_jacobian = bool(
        freeze_calibrated_jacobian
        and isinstance(completed_calibration, dict)
        and completed_calibration.get("completed") is True
    )
    skip_broyden = bool(
        calibration_probe_delta is not None
        or calibration_completed_now
        or frozen_runtime_jacobian
    )
    if (
        frozen_runtime_jacobian
        and not calibration_completed_now
        and calibration_probe_delta is None
    ):
        update = {
            "accepted": False,
            "reason": "frozen_high_signal_runtime_axis_calibration",
        }
    if (
        not skip_broyden
        and not border_search
        and "last_xy_m" in state
        and "last_error_uv" in state
    ):
        dx = current_xy - np.asarray(state["last_xy_m"], dtype=float)
        de = error - np.asarray(state["last_error_uv"], dtype=float)
        denominator = float(dx @ dx)
        if denominator >= float(perception.get("minimum_broyden_motion_m", 0.002)) ** 2:
            previous_component_area = state.get("last_component_area_per_tool_scale_sq")
            area_ratio = None
            if current_component_area is not None and previous_component_area is not None:
                previous_component_area = float(previous_component_area)
                area_ratio = max(
                    current_component_area / previous_component_area,
                    previous_component_area / current_component_area,
                )
            maximum_area_ratio = float(
                perception.get("maximum_broyden_component_area_ratio", 2.0)
            )
            if area_ratio is not None and area_ratio > maximum_area_ratio:
                update = {
                    "accepted": False,
                    "reason": "semantic_component_area_changed",
                    "delta_xy_m": dx.tolist(),
                    "delta_error_uv": de.tolist(),
                    "component_area_ratio": area_ratio,
                    "maximum_component_area_ratio": maximum_area_ratio,
                }
            else:
                candidate = jacobian + np.outer(de - jacobian @ dx, dx) / denominator
                condition = float(np.linalg.cond(candidate))
                maximum_gain = float(
                    perception.get("maximum_hover_jacobian_gain_uv_per_m", 100.0)
                )
                if (
                    np.all(np.isfinite(candidate))
                    and condition
                    <= float(perception.get("maximum_jacobian_condition", 100.0))
                    and float(np.linalg.norm(candidate, ord=2)) <= maximum_gain
                ):
                    jacobian = candidate
                    update = {
                        "accepted": True,
                        "delta_xy_m": dx.tolist(),
                        "delta_error_uv": de.tolist(),
                        "condition": condition,
                        "component_area_ratio": area_ratio,
                    }
                else:
                    update = {
                        "accepted": False,
                        "delta_xy_m": dx.tolist(),
                        "delta_error_uv": de.tolist(),
                        "condition": condition,
                        "component_area_ratio": area_ratio,
                    }

    raw_delta = (
        calibration_probe_delta
        if calibration_probe_delta is not None
        else -np.linalg.pinv(jacobian, rcond=1e-4) @ error
    )
    configured_maximum = float(perception["maximum_planar_correction_m"])
    current_error = float(np.linalg.norm(error))
    full_trust_error = float(
        perception.get("full_trust_hover_error_norm", 0.8)
    )
    minimum_trust = float(
        perception.get("minimum_planar_correction_m", 0.002)
    )
    maximum = configured_maximum
    if calibration_probe_delta is not None:
        maximum = configured_maximum
    elif not border_search and (
        "best_error_norm" in state
        or bool(state.get("adaptive_trust_from_first_observation", False))
    ):
        maximum = min(
            configured_maximum,
            max(
                minimum_trust,
                configured_maximum
                * current_error
                / max(full_trust_error, 1e-9),
            ),
        )
    raw_norm = float(np.linalg.norm(raw_delta))
    trust_scale = 1.0 if raw_norm <= maximum else maximum / raw_norm
    delta_xy = raw_delta * trust_scale
    if border_search and calibration_probe_delta is None:
        maximum = configured_maximum
        direction = state.get("border_search_direction_xy")
        if direction is None:
            if raw_norm <= 1e-9:
                raise RuntimeError("border search has no observable motion direction")
            direction = (raw_delta / raw_norm).tolist()
            state["border_search_direction_xy"] = direction
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        delta_xy = maximum * direction
        trust_scale = maximum / max(raw_norm, 1e-9)
        update = {
            "accepted": False,
            "reason": "border_censored_candidate_not_used_for_broyden",
        }
    elif border_search:
        # Runtime axis calibration deliberately chooses the signed probe.  In
        # particular, a failed +X border probe requests -X here.  The generic
        # border-search direction used to overwrite that request back to +X,
        # so both "opposite" probes moved the same way and the target remained
        # clipped forever.  Preserve the explicit signed probe, only applying
        # the ordinary trust-radius clipping computed above.
        update = {
            "accepted": False,
            "reason": "signed_runtime_axis_probe_at_censored_border",
        }
    else:
        state.pop("border_search_direction_xy", None)
    if not np.all(np.isfinite(delta_xy)):
        raise RuntimeError("visual servo produced a non-finite planar correction")

    best_error = float(state.get("best_error_norm", math.inf))
    regressed_from_best = False
    should_backtrack = False
    if border_search:
        # A clipped component is a lower-bound observation and can also be an
        # arm-attached distractor.  Do not make it the best-state anchor, but
        # do return to an earlier fully visible best pose when the error has
        # clearly grown.  Suppressing both updates *and* regression here used
        # to compound one bad Jacobian step until the target left the image.
        should_backtrack = bool(
            calibration_probe_delta is None
            and current_error
            > best_error * float(perception.get("servo_regression_ratio", 1.10))
            and "best_low_q_physical_rad" in state
        )
    elif current_error < best_error:
        state["best_error_norm"] = current_error
        state["best_low_q_physical_rad"] = current_q.tolist()
        state["best_xy_m"] = current_xy.tolist()
    else:
        should_backtrack = bool(
            calibration_probe_delta is None
            and current_error
            > best_error * float(perception.get("servo_regression_ratio", 1.10))
            and "best_low_q_physical_rad" in state
        )
    if should_backtrack:
        # A regressed probe is never compounded.  Return to the best known
        # hover; the accepted Broyden update will choose a new direction there.
        # Once that hover has actually been reached, however, camera/component
        # jitter can make the new measurement slightly worse than the stored
        # scalar.  Backtracking to the pose we already occupy would deadlock
        # the servo forever, so refresh the anchor and take a new probe.
        best_q = _q(state["best_low_q_physical_rad"], "best visual-servo q")
        best_xy = np.asarray(
            state.get(
                "best_xy_m",
                np.asarray(fk.pose(best_q).parameters(), dtype=float)[4:6],
            ),
            dtype=float,
        )
        at_best_joint_target = float(
            np.max(np.abs(best_q - current_q))
        ) <= 1e-5
        at_best = at_best_joint_target or (
            float(np.linalg.norm(best_xy - current_xy))
            <= float(perception.get("minimum_broyden_motion_m", 0.002)) / 2.0
        )
        if not at_best:
            # Return toward the best measured Cartesian pose, but obey the
            # same trust radius as every forward probe.  A historical best can
            # be several corrections away; jumping there both violated the
            # motion contract and paired a partial physical move with the
            # wrong semantic low-pose witness.
            backtrack_delta = best_xy - current_xy
            backtrack_norm = float(np.linalg.norm(backtrack_delta))
            if backtrack_norm > maximum:
                backtrack_delta *= maximum / backtrack_norm
            delta_xy = backtrack_delta
            raw_delta = best_xy - current_xy
            trust_scale = float(
                np.linalg.norm(delta_xy) / max(np.linalg.norm(raw_delta), 1e-9)
            )
            regressed_from_best = True
        else:
            # We are already back at the best joint target.  A noisier
            # centroid at that same pose must not erase the historically best
            # observation; keep the anchor and take a fresh bounded probe.
            state["backtrack_anchor_refreshes"] = int(
                state.get("backtrack_anchor_refreshes", 0)
            ) + 1

    target_position = current_position.copy()
    target_position[:2] += delta_xy
    reference_low_q = _q(
        (
            profile["trajectory"]["verified_preclose_q_physical_rad"]
            if fixed_orientation_q is None
            else fixed_orientation_q
        ),
        "fixed low orientation q",
    )
    if level_yaw_free:
        corrected_physical, ik = _plan_level_yaw_free_pose(
            profile,
            fk,
            target_position=target_position,
            seed_q=current_q,
            role="level_yaw_free_low_visual_servo",
        )
    else:
        try:
            corrected_physical, ik = _plan_fixed_orientation_pose(
                profile,
                fk,
                target_position=target_position,
                orientation_q=reference_low_q,
                seed_q=current_q,
                role="fixed_orientation_low_visual_servo",
            )
        except RuntimeError as fixed_error:
            # Visual servo assumes a rigid wrist camera. If the exact
            # orientation is locally infeasible, relax roll/pitch only enough
            # to restore jaw level while preserving horizontal yaw. A fully
            # yaw-free fallback can rotate the view by tens of degrees and
            # turn the next image error into a different camera geometry.
            fixed_yaw_error = None
            try:
                corrected_physical, ik = _plan_level_fixed_yaw_pose(
                    profile,
                    fk,
                    target_position=target_position,
                    seed_q=current_q,
                    role="level_fixed_yaw_low_visual_servo",
                )
            except RuntimeError as error_at_full_step:
                fixed_yaw_error = error_at_full_step
                corrected_physical = None
                ik = None
                original_delta_xy = delta_xy.copy()
                for line_search_scale in (0.5, 0.25, 0.125):
                    candidate_delta_xy = original_delta_xy * line_search_scale
                    candidate_position = current_position.copy()
                    candidate_position[:2] += candidate_delta_xy
                    try:
                        candidate_q, candidate_ik = _plan_level_fixed_yaw_pose(
                            profile,
                            fk,
                            target_position=candidate_position,
                            seed_q=current_q,
                            role="level_fixed_yaw_low_visual_servo",
                        )
                    except RuntimeError:
                        continue
                    corrected_physical = candidate_q
                    ik = candidate_ik
                    delta_xy = candidate_delta_xy
                    target_position = candidate_position
                    ik["trust_region_line_search"] = {
                        "accepted": True,
                        "scale": line_search_scale,
                        "full_step_error": str(error_at_full_step),
                    }
                    break
                if corrected_physical is None or ik is None:
                    raise fixed_yaw_error
            ik["fixed_orientation_error"] = str(fixed_error)
    state["jacobian"] = jacobian.tolist()
    state["last_xy_m"] = current_xy.tolist()
    state["last_error_uv"] = error.tolist()
    state["last_low_q_physical_rad"] = current_q.tolist()
    if current_component_area is not None:
        state["last_component_area_per_tool_scale_sq"] = current_component_area
    return corrected_physical, {
        "method": (
            "fixed_camera_physical_xy_broyden_backtrack"
            if regressed_from_best
            else (
                "fixed_camera_runtime_cartesian_axis_probe"
                if calibration_probe_stage is not None
                else "fixed_camera_physical_xy_trust_region_broyden"
            )
        ),
        "runtime_axis_probe_stage": calibration_probe_stage,
        "reference_center_uv": reference_center.tolist(),
        "error_uv": error.tolist(),
        "current_low_xy_m": current_xy.tolist(),
        "jacobian_uv_per_physical_m": jacobian.tolist(),
        "broyden_update": update,
        "runtime_axis_calibration_invalidation": calibration_invalidation,
        "border_search": border_search,
        "border_progress_px": border_clearance_px,
        "border_progress_method": border_progress_method,
        "raw_delta_xy_m": raw_delta.tolist(),
        "trust_scale": trust_scale,
        "trust_radius_m": maximum,
        "selected_delta_xy_m": delta_xy.tolist(),
        "target_low_xy_m": target_position[:2].tolist(),
        "regressed_from_best": regressed_from_best,
        "ik": ik,
        "corrected_q_physical_rad": corrected_physical.tolist(),
    }


def _compensate_hover_tracking_bias(
    profile: dict,
    fk,
    corrected_q,
    correction: dict,
    *,
    hover: dict,
    hover_plan: dict,
) -> tuple[np.ndarray, dict]:
    """Feed forward the measured high-hover XY tracking bias.

    The dynamic descent intentionally preserves the measured hover XY.  If
    the motor settles a few millimetres away from the planned hover, applying
    a low-pose visual correction without this feed-forward term reproduces the
    same bias on the next attempt.  Compensate only bounded, freshly measured
    bias and audit the resulting level low pose before persisting it.
    """

    corrected_q = _q(corrected_q, "uncompensated corrected low q")
    desired_xy = np.asarray(correction.get("target_low_xy_m"), dtype=float)
    measured_hover_xy = np.asarray(
        hover.get("measured_hover_xy_m"), dtype=float
    )
    planned_hover_position = np.asarray(
        hover_plan.get("target_position_m"), dtype=float
    )
    if (
        desired_xy.shape != (2,)
        or measured_hover_xy.shape != (2,)
        or planned_hover_position.shape != (3,)
        or not np.all(
            np.isfinite(
                np.r_[desired_xy, measured_hover_xy, planned_hover_position]
            )
        )
    ):
        return corrected_q, correction
    bias_xy = measured_hover_xy - planned_hover_position[:2]
    maximum_bias = float(
        profile["execution"].get(
            "maximum_hover_tracking_bias_compensation_m", 0.006
        )
    )
    bias_norm = float(np.linalg.norm(bias_xy))
    if bias_norm > maximum_bias:
        updated = dict(correction)
        updated["hover_tracking_bias_compensation"] = {
            "accepted": False,
            "measured_bias_xy_m": bias_xy.tolist(),
            "bias_norm_m": bias_norm,
            "maximum_bias_m": maximum_bias,
        }
        return corrected_q, updated
    corrected_pose = np.asarray(
        fk.pose(corrected_q).parameters(), dtype=float
    )
    commanded_xy = desired_xy - bias_xy
    target_position = corrected_pose[4:].copy()
    target_position[:2] = commanded_xy
    compensated_q, compensated_plan = _plan_level_fixed_yaw_pose(
        profile,
        fk,
        target_position=target_position,
        seed_q=corrected_q,
        role="tracking_bias_compensated_fixed_yaw_low_visual_servo",
    )
    updated = dict(correction)
    updated["uncompensated_corrected_q_physical_rad"] = corrected_q.tolist()
    updated["commanded_target_low_xy_m"] = commanded_xy.tolist()
    updated["hover_tracking_bias_compensation"] = {
        "accepted": True,
        "measured_bias_xy_m": bias_xy.tolist(),
        "bias_norm_m": bias_norm,
        "maximum_bias_m": maximum_bias,
        "semantic_target_low_xy_m": desired_xy.tolist(),
        "commanded_target_low_xy_m": commanded_xy.tolist(),
        "ik": compensated_plan,
    }
    updated["corrected_q_physical_rad"] = compensated_q.tolist()
    return compensated_q, updated


def _move_between_hovers(
    profile,
    rpc,
    fk,
    target_hover_q,
    *,
    selected_delta_xy_m,
    fixed_orientation_wxyz=None,
    allow_branch_escape: bool = True,
) -> dict:
    """Apply one audited visual correction through teleop's Cartesian path.

    The IK hover remains the MuJoCo/collision witness, but commanding its joint
    solution changed wrist branches and made the eye-in-hand image Jacobian
    non-stationary.  Command measured EE XY directly while preserving one
    measured quaternion for the complete hover-servo episode, exactly as the
    responsive teleop path does.  Re-latching the quaternion after every
    correction compounds small tracking residuals and rotates an eye-in-hand
    camera until its image Jacobian is no longer stationary.
    """

    target_hover_q = _q(target_hover_q, "audited corrected hover q")
    delta = np.asarray(selected_delta_xy_m, dtype=float)
    if delta.shape != (2,) or not np.all(np.isfinite(delta)):
        raise ValueError("hover Cartesian correction must contain finite XY")
    maximum = float(profile["perception"]["maximum_planar_correction_m"])
    if float(np.linalg.norm(delta)) > maximum + 1e-9:
        raise ValueError("hover Cartesian correction exceeds the audited trust region")
    start_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    target_pose = start_pose.copy()
    if fixed_orientation_wxyz is not None:
        orientation = np.asarray(fixed_orientation_wxyz, dtype=float)
        if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
            raise ValueError("fixed hover orientation must contain four finite values")
        orientation_norm = float(np.linalg.norm(orientation))
        if orientation_norm <= 1e-9:
            raise ValueError("fixed hover orientation quaternion has zero norm")
        target_pose[:4] = orientation / orientation_norm
    target_pose[4:6] += delta

    def workspace_boundary_hold(reason: str) -> dict:
        """Keep the current camera branch when target identity is established."""

        measured_q = _q(
            rpc.get_right_joint_positions(), "measured workspace-boundary hover q"
        )
        measured_pose = np.asarray(
            rpc.get_right_ee_pose().parameters(), dtype=float
        )
        actual_delta_xy = measured_pose[4:6] - start_pose[4:6]
        return {
            "commands_sent": True,
            "method": "fixed_camera_workspace_boundary_hold",
            "cartesian_error": reason,
            "selected_delta_xy_m": delta.tolist(),
            "actual_delta_xy_m": actual_delta_xy.tolist(),
            "measured_post_cartesian_q_physical_rad": measured_q.tolist(),
            "measured_post_cartesian_pose_wxyz_xyz": measured_pose.tolist(),
            "camera_branch_preserved": True,
            "closure_authorized": False,
            "fresh_hover_observation_required": True,
        }

    def execute_branch_escape(reason: str) -> dict:
        # The fixed camera yaw has reached a Cartesian workspace boundary.
        # Move slowly to the already collision-audited alternate hover
        # witness, then reacquire the target in a fresh image.  This free-space
        # view change can never authorize descent or closure.
        measured_q = _q(
            rpc.get_right_joint_positions(), "measured IK-boundary hover q"
        )
        branch_duration = float(
            profile["trajectory"].get("hover_branch_escape_duration_s", 2.5)
        )
        branch_samples = sample_joint_knots(
            [
                {
                    "stage": "measured_cartesian_ik_boundary_hover",
                    "right_q_physical_rad": measured_q.tolist(),
                    "right_gripper_open_ratio": 1.0,
                    "minimum_duration_s": 0.1,
                },
                {
                    "stage": "audited_hover_joint_branch_escape",
                    "right_q_physical_rad": target_hover_q.tolist(),
                    "right_gripper_open_ratio": 1.0,
                    "minimum_duration_s": branch_duration,
                },
            ]
        )
        branch_path = np.vstack(
            [
                measured_q,
                *[
                    np.asarray(sample.right_q_physical_rad, dtype=float)
                    for sample in branch_samples
                ],
            ]
        )
        branch_audit = _right_joint_path_contact_audit(profile, branch_path)
        if not branch_audit["accepted"]:
            raise RuntimeError(
                "hover joint-branch escape predicts a collision: "
                f"{branch_audit}"
            )
        branch_motion = _execute_direct_joint_samples(
            profile,
            rpc,
            fk,
            branch_samples,
            final_tolerance_rad=float(
                profile["execution"].get(
                    "hover_branch_escape_final_tolerance_rad", 0.05
                )
            ),
            endpoint_correction_gain=0.0,
            require_final_convergence=False,
        )
        return {
            "commands_sent": True,
            "method": "audited_joint_branch_escape_after_cartesian_ik_limit",
            "cartesian_error": reason,
            "selected_delta_xy_m": delta.tolist(),
            "audited_target_hover_q_physical_rad": target_hover_q.tolist(),
            "branch_collision_audit": branch_audit,
            "branch_motion": branch_motion,
            "closure_authorized": False,
            "fresh_hover_observation_required": True,
        }

    try:
        result = _cartesian_move(
            profile,
            rpc,
            fk,
            target_pose=target_pose,
            duration_s=float(
                profile["trajectory"].get("hover_replan_duration_s", 0.9)
            ),
            aperture=1.0,
            stage="fixed_quaternion_hover_visual_correction",
            settle_s=float(
                profile["trajectory"].get(
                    "hover_cartesian_correction_settle_s", 0.25
                )
            ),
            maximum_start_fk_position_error_m=float(
                profile["execution"].get(
                    "hover_cartesian_maximum_start_fk_position_error_m", 0.002
                )
            ),
            maximum_start_fk_rotation_error_rad=float(
                profile["execution"].get(
                    "hover_cartesian_maximum_start_fk_rotation_error_rad", 0.04
                )
            ),
        )
    except TrajectoryStreamError as cartesian_error:
        if "teleop setpoint" not in str(cartesian_error):
            raise
        if not allow_branch_escape:
            return workspace_boundary_hold(str(cartesian_error))
        return execute_branch_escape(str(cartesian_error))
    if not bool(
        profile["execution"].get(
            "hover_probe_converge_to_audited_joint_endpoint", False
        )
    ):
        measured_q = _q(
            rpc.get_right_joint_positions(), "measured post-Cartesian hover q"
        )
        measured_pose = np.asarray(
            rpc.get_right_ee_pose().parameters(), dtype=float
        )
        commanded_norm = float(np.linalg.norm(delta))
        actual_delta_xy = measured_pose[4:6] - start_pose[4:6]
        directional_progress = float(
            actual_delta_xy @ delta / max(commanded_norm**2, 1e-12)
        )
        minimum_stall_command = float(
            profile["perception"].get(
                "minimum_hover_stall_detection_command_m", 0.002
            )
        )
        minimum_progress = float(
            profile["perception"].get(
                "minimum_hover_directional_progress_fraction", 0.20
            )
        )
        if (
            commanded_norm >= minimum_stall_command
            and directional_progress < minimum_progress
        ):
            reason = (
                "Cartesian hover correction stalled at workspace boundary: "
                f"commanded_delta_xy_m={delta.tolist()}, "
                f"actual_delta_xy_m={actual_delta_xy.tolist()}, "
                f"directional_progress_fraction={directional_progress:.3f}"
            )
            if not allow_branch_escape:
                return workspace_boundary_hold(reason)
            return execute_branch_escape(
                reason
            )
        result["method"] = "measured_pose_fixed_quaternion_cartesian_xy"
        result["selected_delta_xy_m"] = delta.tolist()
        result["audited_target_hover_q_physical_rad"] = target_hover_q.tolist()
        result["measured_post_cartesian_q_physical_rad"] = measured_q.tolist()
        result["measured_post_cartesian_pose_wxyz_xyz"] = measured_pose.tolist()
        result["actual_delta_xy_m"] = actual_delta_xy.tolist()
        result["directional_progress_fraction"] = directional_progress
        result["fixed_hover_orientation_wxyz"] = target_pose[:4].tolist()
        measured_pose = _pose(measured_pose, "measured post-Cartesian hover pose")
        quaternion_dot = float(
            np.clip(abs(target_pose[:4] @ measured_pose[:4]), 0.0, 1.0)
        )
        result["final_orientation_error_rad"] = float(
            2.0 * math.acos(quaternion_dot)
        )
        result["orientation_endpoint_convergence"] = {
            "enabled": False,
            "reason": (
                "teleop Cartesian pose and fresh wrist image are authoritative; "
                "alternate IK witness branch is not commanded"
            ),
        }
        return result
    # The Cartesian teleop-equivalent command is deliberately permissive so it
    # remains responsive, but that also means the physical wrist can stop a
    # few milliradians away from the audited IK witness.  For an eye-in-hand
    # camera those small orientation changes invalidate the image Jacobian: a
    # repeated XY probe no longer produces a repeatable pixel displacement.
    # At free-space hover, finish each correction on the audited joint witness
    # before taking the next image.  The short path is collision checked and
    # keeps the gripper fully open; it never authorizes descent or closure.
    measured_q = _q(
        rpc.get_right_joint_positions(), "measured post-Cartesian hover q"
    )
    endpoint_duration = float(
        profile["execution"].get("hover_visual_endpoint_duration_s", 0.65)
    )
    endpoint_samples = sample_joint_knots(
        [
            {
                "stage": "measured_post_cartesian_hover",
                "right_q_physical_rad": measured_q.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "converge_audited_camera_hover",
                "right_q_physical_rad": target_hover_q.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": endpoint_duration,
            },
        ]
    )
    endpoint_path = np.vstack(
        [
            measured_q,
            *[
                np.asarray(sample.right_q_physical_rad, dtype=float)
                for sample in endpoint_samples
            ],
        ]
    )
    endpoint_audit = _right_joint_path_contact_audit(profile, endpoint_path)
    if not endpoint_audit["accepted"]:
        raise RuntimeError(
            "audited camera-hover endpoint path predicts a collision: "
            f"{endpoint_audit}"
        )
    endpoint_motion = _execute_direct_joint_samples(
        profile,
        rpc,
        fk,
        endpoint_samples,
        final_tolerance_rad=float(
            profile["execution"].get(
                "hover_probe_endpoint_final_tolerance_rad", 0.008
            )
        ),
        endpoint_correction_gain=float(
            profile["execution"].get(
                "hover_probe_endpoint_correction_gain", 0.5
            )
        ),
        maximum_endpoint_correction_rad=float(
            profile["execution"].get(
                "hover_probe_maximum_endpoint_correction_rad", 0.03
            )
        ),
        accumulate_endpoint_correction=bool(
            profile["execution"].get(
                "hover_probe_integral_endpoint_correction", True
            )
        ),
        # A fresh rigid-tool-frame image follows immediately.  Joint equality
        # is a motor proxy and must not discard a useful physical probe.
        require_final_convergence=bool(
            profile["execution"].get(
                "hover_probe_require_final_joint_convergence", False
            )
        ),
    )
    result["method"] = (
        "measured_pose_fixed_quaternion_cartesian_xy_then_audited_joint_endpoint"
    )
    result["selected_delta_xy_m"] = delta.tolist()
    result["audited_target_hover_q_physical_rad"] = target_hover_q.tolist()
    result["orientation_endpoint_convergence"] = {
        "collision_audit": endpoint_audit,
        "motion": endpoint_motion,
    }
    return result


def _trajectory_contact_audit(
    profile: dict, low: np.ndarray, *, left_q_physical: np.ndarray | None = None
) -> dict:
    """Audit safe-to-low motion in the canonical scene model."""

    import mujoco

    safe = _q(profile["trajectory"]["safe_high_q_physical_rad"], "safe q")
    low = _q(low, "dynamic low q")
    model = mujoco.MjModel.from_xml_path(str(Path(profile["planning_model"]).resolve()))
    data = mujoco.MjData(model)
    offset = physical_to_semantic_model_q_offset("right")
    ids = np.asarray(
        [model.joint(f"right/joint{index}").qposadr[0] for index in range(1, 7)],
        dtype=int,
    )
    left_ids = np.asarray(
        [model.joint(f"left/joint{index}").qposadr[0] for index in range(1, 7)],
        dtype=int,
    )
    left_q = (
        physical_home_q("left")
        if left_q_physical is None
        else _q(left_q_physical, "measured left q")
    )
    data.qpos[left_ids] = left_q + physical_to_semantic_model_q_offset("left")
    disallowed = set()
    expected_support = set()
    for fraction in np.linspace(0.0, 1.0, 121):
        blend = _minimum_jerk(float(fraction))
        data.qpos[ids] = safe + blend * (low - safe) + offset
        mujoco.mj_forward(model, data)
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1 = model.geom(int(contact.geom1))
            geom2 = model.geom(int(contact.geom2))
            body1 = model.body(int(geom1.bodyid[0])).name
            body2 = model.body(int(geom2.bodyid[0])).name
            if not (body1.startswith("right/") or body2.startswith("right/")):
                continue
            pair = tuple(sorted((geom1.name, geom2.name)))
            text = " ".join((*pair, body1, body2))
            if "right/nyu_gripper_collision" in text and "support-platform" in text:
                expected_support.add(pair)
            else:
                disallowed.add(pair)
    return {
        "accepted": not disallowed,
        "sample_count": 121,
        "expected_target_support_contacts": [
            list(pair) for pair in sorted(expected_support)
        ],
        "non_support_contacts": [list(pair) for pair in sorted(disallowed)],
    }


def _plan_relocated_level_rim(
    profile: dict, current_center_scene: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Map a support-plane scene *delta* from one empirical right-arm anchor.

    Absolute camera-to-arm fits were too noisy and once sent the gripper toward
    the incubator.  A fixed tag gives a stable scene displacement, while one
    physically aligned low pose supplies the translation anchor.  Only the
    planar delta is rotated into the physical-right production frame.  The
    resulting level pose is collision-audited and remains merely a wrist-camera
    seed; it never authorizes closure.
    """
    settings = profile["head_localization"]
    target = profile["target_identity"]
    relocation = target.get("relocation") or {}
    calibration_path = Path(target["runtime_alignment_calibration"]).resolve()
    calibration = _load(calibration_path) if calibration_path.is_file() else {}
    saved_hover_goal = np.asarray(
        calibration.get("canonical_hover_goal_uv"), dtype=float
    )
    saved_preclose_goal = np.asarray(
        calibration.get("canonical_preclose_goal_uv"), dtype=float
    )
    configured_hover_goal = np.asarray(
        target.get("canonical_hover_goal_uv"), dtype=float
    )
    configured_preclose_goal = np.asarray(
        target.get("canonical_preclose_goal_uv"), dtype=float
    )
    runtime_anchor_valid = bool(
        calibration.get("schema") == RUNTIME_ALIGNMENT_SCHEMA
        and saved_hover_goal.shape == (2,)
        and configured_hover_goal.shape == (2,)
        and np.allclose(saved_hover_goal, configured_hover_goal, atol=1e-9)
        and saved_preclose_goal.shape == (2,)
        and configured_preclose_goal.shape == (2,)
        and np.allclose(
            saved_preclose_goal, configured_preclose_goal, atol=1e-9
        )
    )
    runtime_anchor_geometry = None
    if runtime_anchor_valid:
        runtime_anchor_geometry = _runtime_alignment_geometry_audit(
            profile,
            calibration.get("target_center_scene_xyz_m"),
            calibration.get("low_q_physical_rad"),
        )
        runtime_anchor_valid = bool(runtime_anchor_geometry["accepted"])
    reference_q_physical = _q(
        (
            calibration["low_q_physical_rad"]
            if runtime_anchor_valid
            else profile["trajectory"]["verified_preclose_q_physical_rad"]
        ),
        "reference preclose q",
    )
    reference_center = np.asarray(
        (
            calibration["target_center_scene_xyz_m"]
            if runtime_anchor_valid
            else settings["reference_target_center_scene_xyz_m"]
        ),
        dtype=float,
    )
    current_center = np.asarray(current_center_scene, dtype=float)
    scene_delta = current_center - reference_center
    fk = ProductionRightFK(profile["production_model"])
    reference_pose = np.asarray(fk.pose(reference_q_physical).parameters(), dtype=float)
    local_limit = float(
        target.get("maximum_runtime_target_scene_displacement_m", 0.008)
    )
    scene_distance = float(np.linalg.norm(scene_delta[:2]))
    relocation_yaw = float(
        relocation.get("scene_delta_to_production_xy_yaw_rad", math.nan)
    )
    relocation_valid = bool(
        relocation.get("accepted") is True
        and np.isfinite(relocation_yaw)
        and float(relocation.get("maximum_scene_displacement_m", 0.0)) > 0.0
    )
    applied = False
    transform_report = None
    selected_q = reference_q_physical.copy()
    if relocation_valid and scene_distance > local_limit:
        maximum = float(relocation.get("maximum_scene_displacement_m", 0.1))
        if scene_distance > maximum:
            raise RuntimeError(
                "tapped target exceeds the calibrated scene-delta relocation envelope: "
                f"{scene_distance:.4f}m > {maximum:.4f}m"
            )
        yaw = relocation_yaw
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=float,
        )
        production_delta = rotation @ scene_delta[:2]
        target_position = reference_pose[4:].copy()
        target_position[:2] += production_delta
        selected_q, transform_report = _plan_level_yaw_free_pose(
            profile,
            fk,
            target_position=target_position,
            seed_q=reference_q_physical,
            role="fixed_tag_scene_delta_relocation_seed",
        )
        applied = True
    level_reference = _load_level_reference(profile["level_config"])
    level = assess_jaw_level(
        fk.pose(selected_q).parameters(),
        level_reference,
        planned=True,
    )
    collision = _trajectory_contact_audit(profile, selected_q)
    accepted = bool(level.accepted and collision["accepted"])
    selected = {
        "accepted": accepted,
        "q_physical_rad": selected_q.tolist(),
        "level": level.to_dict(),
        "collision": collision,
    }
    if not accepted:
        raise RuntimeError(f"wrist visual-servo seed failed level/collision audit: {selected}")
    return selected_q.copy(), {
        "method": (
            relocation.get("method")
            if applied
            else "fixed_head_identity_plus_proven_wrist_visual_servo_seed"
        ),
        "reference_target_center_scene_xyz_m": reference_center.tolist(),
        "current_target_center_scene_xyz_m": current_center.tolist(),
        "target_scene_delta_m": scene_delta.tolist(),
        "target_scene_displacement_norm_m": scene_distance,
        "head_scene_to_production_translation_applied": applied,
        "relocation_calibration": {
            "accepted": relocation_valid,
            "path": str(calibration_path),
            "anchor_source": (
                "runtime_alignment_with_matching_stage_goals"
                if runtime_anchor_valid
                else "configured_reference_scene_and_verified_preclose_pose"
            ),
            "runtime_anchor_accepted": runtime_anchor_valid,
            "runtime_anchor_rejection_reason": (
                None
                if runtime_anchor_valid
                else (
                    (runtime_anchor_geometry or {}).get("reason")
                    or "stage-specific image goal calibration changed"
                )
            ),
            "runtime_anchor_cross_modal_geometry": runtime_anchor_geometry,
            "method": relocation.get("method"),
            "scene_delta_to_production_xy_yaw_rad": relocation.get(
                "scene_delta_to_production_xy_yaw_rad"
            ),
            "maximum_scene_displacement_m": relocation.get(
                "maximum_scene_displacement_m"
            ),
        },
        "transform_plan": transform_report,
        "closure_authorized": False,
        "fresh_wrist_alignment_required": True,
        "selected": selected,
    }


def audit_profile(profile: dict) -> dict:
    import mujoco

    fk = ProductionRightFK(profile["production_model"])
    safe = _q(profile["trajectory"]["safe_high_q_physical_rad"], "safe q")
    low = _q(profile["trajectory"]["verified_preclose_q_physical_rad"], "low q")
    level = _load_level_reference(profile["level_config"])
    safe_level = assess_jaw_level(fk.pose(safe).parameters(), level, planned=True)
    low_level = assess_jaw_level(fk.pose(low).parameters(), level, planned=True)
    if not safe_level.accepted or not low_level.accepted:
        raise ValueError("configured trajectory is not jaw-level")
    model = mujoco.MjModel.from_xml_path(str(Path(profile["planning_model"]).resolve()))
    data = mujoco.MjData(model)
    offset = physical_to_semantic_model_q_offset("right")
    ids = np.asarray(
        [model.joint(f"right/joint{index}").qposadr[0] for index in range(1, 7)],
        dtype=int,
    )
    left_ids = np.asarray(
        [model.joint(f"left/joint{index}").qposadr[0] for index in range(1, 7)],
        dtype=int,
    )
    data.qpos[left_ids] = physical_home_q("left") + physical_to_semantic_model_q_offset("left")
    disallowed = set()
    expected_support = set()
    for fraction in np.linspace(0.0, 1.0, 121):
        blend = _minimum_jerk(float(fraction))
        data.qpos[ids] = safe + blend * (low - safe) + offset
        mujoco.mj_forward(model, data)
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1 = model.geom(int(contact.geom1))
            geom2 = model.geom(int(contact.geom2))
            body1 = model.body(int(geom1.bodyid[0])).name
            body2 = model.body(int(geom2.bodyid[0])).name
            if not (body1.startswith("right/") or body2.startswith("right/")):
                continue
            pair = tuple(sorted((geom1.name, geom2.name)))
            text = " ".join((*pair, body1, body2))
            if "right/nyu_gripper_collision" in text and "support-platform" in text:
                expected_support.add(pair)
            else:
                disallowed.add(pair)
    if disallowed:
        raise ValueError(f"MuJoCo predicts non-support right-arm contacts: {sorted(disallowed)}")
    return {
        "accepted": True,
        "production_fk": "validated",
        "safe_level": safe_level.to_dict(),
        "preclose_level": low_level.to_dict(),
        "planning_model": str(Path(profile["planning_model"]).resolve()),
        "sample_count": 121,
        "expected_target_support_contacts": [list(pair) for pair in sorted(expected_support)],
        "non_support_contacts": [],
    }


def preflight_head_registration(
    profile: dict, output_dir: Path
) -> tuple[dict, np.ndarray]:
    image, timestamp = capture_named_camera("head")
    output = output_dir / "preflight_head.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    reference = cv2.imread(profile["target_identity"]["head_reference_image"])
    if reference is None:
        raise RuntimeError("head reference image is missing")
    registration = register_fixed_head(reference, image)
    if not registration.accepted:
        raise RuntimeError(f"fixed-head registration rejected: {registration}")
    settings = profile["head_localization"]
    marker = select_relocated_target_marker(
        image,
        homography_reference_to_current=registration.homography,
        reference_target_center_px=settings["reference_target_marker_center_px"],
        reference_target_component_pixels=int(
            settings["reference_target_component_pixels"]
        ),
        stationary_anchor_centers_px=settings.get(
            "stationary_anchor_reference_centers_px", []
        ),
        maximum_anchor_displacement_diagonal_fraction=float(
            settings["maximum_anchor_displacement_diagonal_fraction"]
        ),
        maximum_target_displacement_diagonal_fraction=float(
            settings["maximum_target_displacement_diagonal_fraction"]
        ),
        minimum_component_area_scale=float(settings["minimum_component_area_scale"]),
        maximum_component_area_scale=float(settings["maximum_component_area_scale"]),
    )
    continuity_marker, continuity_report = _runtime_head_continuity_marker(
        profile, image
    )
    if continuity_marker is not None:
        marker = continuity_marker
    camera_matrix = np.asarray(settings["camera_matrix_landscape"], dtype=float)
    camera_from_tag, tag_rms = tag_pose_camera(
        image,
        camera_matrix,
        tag_id=int(settings["tag_id"]),
        tag_size_m=float(settings["tag_size_m"]),
    )
    if tag_rms > float(settings["maximum_tag_reprojection_rms_px"]):
        raise RuntimeError(f"fixed-tag reprojection rejected: {tag_rms:.3f}px")
    scene_from_camera = bridge_camera_from_fixed_tag(
        np.asarray(settings["scene_from_tag"], dtype=float),
        camera_from_tag,
    )
    center_scene = intersect_pixel_with_horizontal_plane(
        marker.center_px,
        camera_matrix,
        scene_from_camera,
        plane_z_m=float(settings["support_plane_z_m"]),
    )
    center_scene[2] += float(settings["target_height_m"]) / 2.0
    target_q, rim_plan = _plan_relocated_level_rim(profile, center_scene)
    overlay = image.copy()
    center_int = tuple(int(round(value)) for value in marker.center_px)
    cv2.circle(overlay, center_int, 28, (0, 255, 0), 4)
    cv2.putText(
        overlay,
        "REACQUIRED TAPPED TARGET",
        (max(0, center_int[0] - 210), max(35, center_int[1] - 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    overlay_path = output_dir / "head_target_reacquired.png"
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "timestamp": timestamp,
        "image": str(output),
        "overlay": str(overlay_path),
        "registration": {
            "accepted": registration.accepted,
            "matches": registration.matches,
            "inlier_fraction": registration.inlier_fraction,
            "median_residual_diagonal_fraction": registration.median_residual_diagonal_fraction,
        },
        "marker": marker.to_dict(),
        "runtime_head_continuity": continuity_report,
        "tag_reprojection_rms_px": tag_rms,
        "target_center_scene_xyz_m": center_scene.tolist(),
        "rim_plan": rim_plan,
    }, target_q


def _preclose_marker_center_error_scale(profile: dict, observation) -> float:
    """Return blue-marker error to the successful-demo goal in tool units."""

    template = GraspWindowTemplate.from_dict(
        _load(profile["target_identity"]["grasp_window_selection"])["template"]
    )
    reference_center = np.asarray(
        profile["target_identity"].get(
            "canonical_preclose_goal_uv", template.reference_center_uv
        ),
        dtype=float,
    )
    return float(
        np.linalg.norm(
            np.asarray(observation.center_uv, dtype=float) - reference_center
        )
        / float(template.square_side_u)
    )


def _preclose_visual_alignment_allowed(profile: dict, observation) -> bool:
    """Authorize leveling from resolution-independent tool-frame geometry.

    A thin transparent target changes apparent mask shape with illumination and
    viewpoint, so its quantile-shape residual is diagnostic only.  What matters
    before closing is that the semantic target evidence is contained by the
    gripper-relative grasp window with sufficient margin and remains near the
    demonstrated center.  Both quantities are normalized by the observed tool
    scale rather than expressed in camera pixels.
    """

    assessment = observation.grasp_window
    marker_center_error_scale = _preclose_marker_center_error_scale(
        profile, observation
    )
    return bool(
        observation.tool_frame_source
        in (
            "light_pad_nominal",
            "audited_expected_tool_frame",
            "rigid_expected_tool_frame",
        )
        and assessment.target_inside_fraction
        >= float(
            profile["perception"].get(
                "minimum_preclose_target_inside_fraction", 0.98
            )
        )
        and marker_center_error_scale
        <= float(profile["perception"]["maximum_preclose_center_error_scale"])
    )


def _preclose_height_report(
    profile: dict,
    *,
    low_pose,
    verified_preclose_pose,
    support_up,
) -> dict:
    low_pose = _pose(low_pose, "measured low pose for height gate")
    verified = _pose(verified_preclose_pose, "verified preclose height pose")
    up = np.asarray(support_up, dtype=float)
    up /= np.linalg.norm(up)
    height = float((low_pose[4:7] - verified[4:7]) @ up)
    maximum = float(
        profile["perception"].get(
            "maximum_preclose_height_above_verified_m", 0.0002
        )
    )
    maximum_below = float(
        profile["perception"].get(
            "maximum_preclose_height_below_verified_m", 0.001
        )
    )
    return {
        "accepted": -maximum_below <= height <= maximum,
        "height_above_verified_m": height,
        "maximum_height_above_verified_m": maximum,
        "maximum_height_below_verified_m": maximum_below,
        "verified_support_height_m": float(verified[4:7] @ up),
        "measured_support_height_m": float(low_pose[4:7] @ up),
    }


def _defer_contact_settle_until_visual_alignment(
    profile: dict, *, visually_aligned: bool
) -> bool:
    return bool(
        not visually_aligned
        and profile["execution"].get(
            "defer_contact_settle_until_visually_aligned", False
        )
    )


def run_attempt(
    profile,
    rpc,
    fk,
    camera,
    output_dir: Path,
    target_q,
    *,
    servo_state=None,
    allow_descent: bool = True,
    cached_hover_alignment: dict | None = None,
    initial_hover_q_physical_rad=None,
    initial_expected_tool_frame: dict | None = None,
    initial_target_observation: dict | None = None,
    initial_direct_cross_verified: bool = False,
) -> tuple[bool, dict, np.ndarray]:
    machine = FastLidGraspMachine()
    machine.tap_accepted()
    level_reference = _load_level_reference(profile["level_config"])
    preclose_reference = replace(
        level_reference,
        maximum_tip_height_difference_m=float(
            profile["perception"]["maximum_preclose_tip_height_difference_m"]
        ),
        source=f"{level_reference.source}:measured_preclose",
    )
    # Descent is the first moment when fingertip mismatch can push a thin
    # object laterally.  Use the strict calibrated level reference here; the
    # looser free-space hover tolerance is not a descent authorization.
    descent_checkpoint = RightJawLevelCheckpoint(rpc, level_reference)
    preclose_checkpoint = RightJawLevelCheckpoint(rpc, preclose_reference)
    servo_state = {} if servo_state is None else servo_state
    if initial_direct_cross_verified and initial_target_observation:
        servo_state["direct_cross_verified"] = True
    else:
        # Identity continuity is attached to an explicit saved target
        # observation, not to the process.  Carrying this bare boolean across
        # a failed coarse approach let a different blue patch inherit proof
        # obtained from the real cross during the previous attempt.
        servo_state.pop("direct_cross_verified", None)
    incomplete_axis_calibration = servo_state.get("runtime_axis_calibration")
    if (
        isinstance(incomplete_axis_calibration, dict)
        and incomplete_axis_calibration.get("completed") is not True
    ):
        # A failed attempt retreats and approaches again, so an unfinished
        # two-axis probe no longer shares its physical origin or camera pose.
        # Carrying it into the next attempt made the first image look like the
        # missing Y response and aborted immediately.
        for key in (
            "runtime_axis_calibration",
            "jacobian",
            "last_xy_m",
            "last_error_uv",
            "last_component_area_per_tool_scale_sq",
            "best_error_norm",
            "best_low_q_physical_rad",
            "best_xy_m",
            "best_hover_q_physical_rad",
            "best_tool_frame",
            "best_normalized_center_error",
            "best_target_observation",
            "best_direct_cross_verified",
            "best_observation_error_norm",
            "best_observation_low_q_physical_rad",
            "best_observation_hover_q_physical_rad",
            "best_observation_tool_frame",
            "best_observation_normalized_center_error",
            "best_observation_target",
            "best_observation_direct_cross_verified",
        ):
            servo_state.pop(key, None)
    preclose_servo_state = {}
    if cached_hover_alignment is not None:
        preclose_servo_state.update(
            cached_hover_alignment.get("preclose_servo_state") or {}
        )
    if bool(
        profile["perception"].get(
            "require_runtime_hover_axis_calibration", False
        )
    ):
        # The closure checkpoint lives at a different eye-in-hand pose from
        # the hover checkpoint.  Its pixel Jacobian must be calibrated at that
        # low, fixed-orientation pose rather than inherited from hover.
        preclose_servo_state["enable_runtime_axis_calibration"] = True
    # Do not seed low-pose correction from a process-old hover Jacobian here.
    # If there is no persisted low-pose sample, the freshly measured hover
    # calibration is copied immediately before the preclose correction.
    result = {
        "success": False,
        "stages": {},
        "state_history": machine.history,
        "hover_iterations": [],
        "visual_replans": [],
    }

    def persist() -> None:
        result["state_history"] = list(machine.history)
        result["left_arm_commands"] = 0
        _atomic_json(output_dir / "attempt.json", result)

    persist()
    fresh_low_correction_replay = bool(
        cached_hover_alignment is not None
        and cached_hover_alignment.get("fresh_preclose_required") is True
        and cached_hover_alignment.get("method")
        in (
            "semantically_verified_preclose_level_q_correction",
            "semantically_verified_same_xy_level_retry",
            "semantically_verified_preclose_axis_probe",
            "semantically_detected_regression_backtrack",
        )
    )
    skip_high_view_for_fresh_low_replay = (
        _may_skip_high_view_for_fresh_low_replay(profile, cached_hover_alignment)
    )
    initial_hover_seed = None
    precise_camera_replay = False
    corrected_hover_orientation_seed = None
    if (
        initial_hover_q_physical_rad is not None
        and not fresh_low_correction_replay
    ):
        initial_hover_seed = _q(
            initial_hover_q_physical_rad, "pending-progress hover seed"
        )
    if cached_hover_alignment is not None:
        cached_orientation_seed = cached_hover_alignment.get(
            "source_hover_orientation_seed_q_physical_rad"
        )
        if cached_orientation_seed is not None:
            corrected_hover_orientation_seed = _q(
                cached_orientation_seed,
                "cached corrected-hover orientation seed",
            )
        cached_hover_seed = cached_hover_alignment.get(
            "source_hover_q_physical_rad"
        )
        if cached_hover_seed is not None:
            initial_hover_seed = _q(
                cached_hover_seed, "cached fixed-yaw hover seed"
            )
    if initial_hover_seed is not None:
        hover_q = initial_hover_seed.copy()
        try:
            hover_plan = _audit_camera_visible_hover_seed(
                profile, fk, low_q=target_q, hover_q=hover_q
            )
            precise_camera_replay = True
        except RuntimeError as persisted_hover_error:
            # A low-pose correction can make the prior camera viewpoint no
            # longer vertically above the target.  The stale view must never
            # block a fresh autonomous attempt: rebuild a level hover above
            # the corrected low pose, then require fresh visual alignment.
            hover_q, hover_plan = _plan_level_vertical_offset(
                profile,
                fk,
                target_q,
                float(profile["trajectory"]["verification_lift_m"]),
                seed_hover_q=initial_hover_seed,
            )
            hover_plan["persisted_hover_rejected"] = str(
                persisted_hover_error
            )
            hover_plan["method"] = (
                "replanned_vertical_hover_after_stale_camera_seed"
            )
            precise_camera_replay = False
    else:
        hover_q, hover_plan = _plan_level_vertical_offset(
            profile,
            fk,
            target_q,
            float(profile["trajectory"]["verification_lift_m"]),
            seed_hover_q=corrected_hover_orientation_seed,
        )
    result["hover_plan"] = hover_plan
    approach = _joint_approach(
        profile,
        rpc,
        fk,
        hover_q,
        precise_camera_replay=precise_camera_replay,
    )
    approach["precise_camera_replay"] = precise_camera_replay
    result["stages"]["approach"] = approach
    measured_hover_pose = _pose(
        rpc.get_right_ee_pose().parameters(), "measured initial hover pose"
    )
    fixed_hover_orientation_wxyz = measured_hover_pose[:4].copy()
    result["hover_orientation_lock"] = {
        "source": "first_measured_pose_after_audited_hover_approach",
        "wxyz": fixed_hover_orientation_wxyz.tolist(),
        "re_latched_each_correction": False,
    }

    def move_hover_with_orientation_lock(next_hover_q, selected_delta_xy_m):
        nonlocal fixed_hover_orientation_wxyz
        nonlocal previous_hover_image, previous_hover_observation
        nonlocal last_observed_low_q, last_observed_hover_xy
        nonlocal pending_branch_identity_image, pending_branch_identity_observation
        motion = _move_between_hovers(
            profile,
            rpc,
            fk,
            next_hover_q,
            selected_delta_xy_m=selected_delta_xy_m,
            fixed_orientation_wxyz=fixed_hover_orientation_wxyz,
            allow_branch_escape=not bool(
                servo_state.get("branch_identity_verified", False)
            ),
        )
        if motion.get("method") == "fixed_camera_workspace_boundary_hold":
            calibration = servo_state.get("runtime_axis_calibration")
            if isinstance(calibration, dict) and str(
                calibration.get("stage", "")
            ).startswith("y_"):
                calibration["retry_y_opposite_from_boundary"] = True
        if motion.get("method") == (
            "audited_joint_branch_escape_after_cartesian_ik_limit"
        ):
            if (
                bool(servo_state.get("direct_cross_verified", False))
                and previous_hover_image is not None
                and previous_hover_observation is not None
            ):
                # Save the last identity-verified plane point.  The next frame
                # uses a background homography to carry that exact point
                # across the deliberate wrist-yaw discontinuity before any
                # global blue candidate is allowed to compete.
                pending_branch_identity_image = previous_hover_image.copy()
                pending_branch_identity_observation = previous_hover_observation
            # A deliberate free-space branch escape creates a new camera
            # orientation.  Re-latch once at that explicit discontinuity and
            # discard all image-Jacobian history from the old view.
            branch_pose = _pose(
                rpc.get_right_ee_pose().parameters(),
                "measured post-branch hover pose",
            )
            fixed_hover_orientation_wxyz = branch_pose[:4].copy()
            for key in (
                "jacobian",
                "last_xy_m",
                "last_error_uv",
                "last_component_area_per_tool_scale_sq",
                "best_error_norm",
                "best_low_q_physical_rad",
                "best_xy_m",
                "best_hover_q_physical_rad",
                "best_tool_frame",
                "best_normalized_center_error",
                "best_target_observation",
                "best_direct_cross_verified",
                "runtime_axis_calibration",
            ):
                servo_state.pop(key, None)
            servo_state["enable_runtime_axis_calibration"] = True
            previous_hover_image = None
            previous_hover_observation = None
            last_observed_low_q = None
            last_observed_hover_xy = None
            result.setdefault("hover_orientation_relocks", []).append(
                {
                    "reason": "audited_joint_branch_escape",
                    "wxyz": fixed_hover_orientation_wxyz.tolist(),
                }
            )
        return motion

    persist()
    machine.transit_complete()
    template = GraspWindowTemplate.from_dict(
        _load(profile["target_identity"]["grasp_window_selection"])["template"]
    )
    expected_tool_frame = None
    fixed_tool_frame = profile["target_identity"].get(
        "fixed_right_tool_frame"
    )
    if fixed_tool_frame is not None:
        expected_tool_frame = ToolImageFrame(**fixed_tool_frame)
    elif initial_expected_tool_frame is not None:
        expected_tool_frame = ToolImageFrame(**initial_expected_tool_frame)
    elif (
        cached_hover_alignment is not None
        and cached_hover_alignment.get("source_tool_frame") is not None
    ):
        expected_tool_frame = ToolImageFrame(
            **cached_hover_alignment["source_tool_frame"]
        )
    if (
        fixed_tool_frame is None
        and hover_plan.get("orientation_mode_change") is not None
    ):
        # A large horizontal-yaw change invalidates pixel-frame geometry from
        # the prior attempt.  The fresh target observation remains mandatory.
        expected_tool_frame = None

    def observe(
        image,
        *,
        proximity_score_weight=0.03,
        maximum_candidate_distance_tool_units=None,
    ):
        nonlocal expected_tool_frame
        observation = observe_marked_target(
            image,
            template,
            maximum_candidate_distance_tool_units=float(
                profile["perception"]["maximum_candidate_distance_tool_units"]
                if maximum_candidate_distance_tool_units is None
                else maximum_candidate_distance_tool_units
            ),
            reference_component_area_per_tool_scale_sq=float(
                profile["target_identity"][
                    "right_reference_component_area_per_tool_scale_sq"
                ]
            ),
            minimum_reference_area_fraction=float(
                profile["target_identity"].get(
                    "minimum_right_reference_area_fraction", 0.03
                )
            ),
            maximum_reference_area_fraction=float(
                profile["target_identity"].get(
                    "maximum_right_reference_area_fraction", 3.0
                )
            ),
            proximity_score_weight=float(proximity_score_weight),
            fallback_minimum_light_value=profile["perception"].get(
                "fallback_minimum_light_value"
            ),
            expected_tool_frame=expected_tool_frame,
            # Wrist-camera extrinsics are rigid. The persisted frame was
            # measured at this audited horizontal orientation, so lighting
            # changes must not redefine its pixel scale.
            prefer_expected_tool_frame=expected_tool_frame is not None,
            # At distant hover, the printed cross shape rejects translucent
            # rim mergers. Near preclose, exposure can split that same mark;
            # calibrated whole-component area is then more stable than one
            # surviving cross-shaped fragment.
            prefer_cross_shape=float(proximity_score_weight) < 1.0,
        )
        if observation.tool_frame_source == "light_pad_nominal":
            expected_tool_frame = observation.tool_frame
        return observation

    def enforce_rigid_tool_frame(observation):
        """Re-express fallback tracking in the calibrated rigid image frame."""

        if observation is None or expected_tool_frame is None:
            return observation
        center_uv = expected_tool_frame.image_to_tool(
            [observation.center_px]
        )[0]
        center_error = float(
            np.linalg.norm(
                center_uv - np.asarray(template.reference_center_uv, dtype=float)
            )
            / template.square_side_u
        )
        assessment = replace(
            observation.grasp_window,
            normalized_center_error=center_error,
            normalized_quantile_error=center_error,
            target_center_uv=(float(center_uv[0]), float(center_uv[1])),
        )
        return replace(
            observation,
            center_uv=(float(center_uv[0]), float(center_uv[1])),
            tool_frame=expected_tool_frame,
            grasp_window=assessment,
            tool_frame_source="rigid_expected_tool_frame",
        )

    plane_reference = cv2.imread(
        profile["target_identity"]["right_plane_registration_reference_image"]
    )
    canonical_hover_goal_uv = profile["target_identity"].get(
        "canonical_hover_goal_uv"
    )
    if plane_reference is None and canonical_hover_goal_uv is None:
        raise RuntimeError("right plane-registration reference image is missing")
    reference_goal_px = np.asarray(
        profile["target_identity"]["right_reference_target_center_px"],
        dtype=np.float32,
    ).reshape(1, 1, 2)
    maximum_hover_replans = int(profile["perception"]["maximum_hover_replans"])
    hover_q_seed = hover_q.copy()
    previous_hover_image = None
    previous_hover_observation = None
    pending_branch_identity_image = None
    pending_branch_identity_observation = None
    last_observed_low_q = None
    last_observed_hover_xy = None
    level_hover_transition_done = False
    try:
        for hover_index in range(maximum_hover_replans + 1):
            if (
                hover_index == 0
                and cached_hover_alignment is not None
                and (
                    skip_high_view_for_fresh_low_replay
                    or not bool(
                        profile["perception"].get(
                            "require_fresh_cached_hover_observation", False
                        )
                    )
                )
            ):
                hover_at = time.time()
                hover_image, hover_ts = camera.frame(fresh_after_s=hover_at)
                hover_path = output_dir / "hover_00.png"
                cv2.imwrite(str(hover_path), hover_image)
                measured_hover_q = _q(
                    rpc.get_right_joint_positions(), "measured cached hover q"
                )
                measured_hover_xy = np.asarray(
                    fk.pose(measured_hover_q).parameters(), dtype=float
                )[4:6]
                normalized = float(
                    cached_hover_alignment["prior_normalized_hover_error"]
                )
                hover_record = {
                    "index": 0,
                    "timestamp": hover_ts,
                    "image": str(hover_path),
                    "low_q_physical_rad": np.asarray(target_q, dtype=float).tolist(),
                    "hover_q_physical_rad": np.asarray(
                        hover_q_seed, dtype=float
                    ).tolist(),
                    "measured_hover_q_physical_rad": measured_hover_q.tolist(),
                    "measured_hover_xy_m": measured_hover_xy.tolist(),
                    "observation": {
                        "source": str(
                            cached_hover_alignment.get(
                                "method", "validated_cached_hover_alignment"
                            )
                        ),
                        "closure_authorized": False,
                    },
                    "error_uv": [
                        normalized * float(template.square_side_u),
                        0.0,
                    ],
                    "normalized_center_error": normalized,
                    "aligned_for_descent": True,
                    "runtime_replay": cached_hover_alignment,
                    "high_view_skipped_for_fresh_low_replay": bool(
                        skip_high_view_for_fresh_low_replay
                    ),
                }
                result["hover_iterations"].append(hover_record)
                result["hover"] = hover_record
                persist()
                machine.hover_assessment(target_visible=True, aligned=True)
                break
            hover_at = time.time()
            hover_image, hover_ts = camera.frame(fresh_after_s=hover_at)
            hover_path = output_dir / f"hover_{hover_index:02d}.png"
            cv2.imwrite(str(hover_path), hover_image)
            semantic_observation = None
            semantic_error = None
            try:
                semantic_observation = observe(
                    hover_image,
                    # Immediately after a deliberate wrist-yaw branch change,
                    # the true target and every distractor can be far from the
                    # old grasp window.  A wide global observation is used
                    # only to recover the new rigid tool image frame; target
                    # identity is then selected exclusively near the
                    # homography-projected old target below.
                    maximum_candidate_distance_tool_units=(
                        float(
                            profile["perception"].get(
                                "branch_identity_tool_frame_search_radius_units",
                                6.0,
                            )
                        )
                        if pending_branch_identity_image is not None
                        else None
                    ),
                )
                persistent_identity_anchor = servo_state.get(
                    "hover_identity_anchor"
                )
                if (
                    persistent_identity_anchor is not None
                    and pending_branch_identity_image is None
                ):
                    semantic_observation, continuity = (
                        _reacquire_persistent_hover_identity(
                            hover_image,
                            semantic_observation,
                            persistent_identity_anchor,
                            template,
                            maximum_displacement_diagonal_fraction=float(
                                profile["perception"].get(
                                    "maximum_persistent_hover_identity_displacement_diagonal_fraction",
                                    0.08,
                                )
                            ),
                        )
                    )
                    result.setdefault("persistent_hover_identity_checks", []).append(
                        {"hover_index": hover_index, **continuity}
                    )
                if hover_index == 0 and precise_camera_replay:
                    saved_observation = dict(initial_target_observation or {})
                    saved_center = np.asarray(
                        saved_observation.get("center_px"), dtype=float
                    )
                    saved_pixels = int(
                        saved_observation.get("component_pixels", 0)
                    )
                    if (
                        not initial_direct_cross_verified
                        or saved_center.shape != (2,)
                        or not np.all(np.isfinite(saved_center))
                        or saved_pixels <= 0
                    ):
                        raise ValueError(
                            "saved camera replay does not directly observe the blue cross"
                        )
                    saved_frame = semantic_observation.tool_frame
                    saved_uv = saved_frame.image_to_tool([saved_center])[0]
                    # Exact camera replay is specifically a target-identity
                    # checkpoint.  Always reacquire within its saved local
                    # envelope, even when a distant distractor also looks like
                    # a blue cross.  Global cross ranking is illumination
                    # dependent and previously selected the neighbouring dish
                    # at the same physical camera pose.
                    continuity_anchor = replace(
                        semantic_observation,
                        center_px=(float(saved_center[0]), float(saved_center[1])),
                        center_uv=(float(saved_uv[0]), float(saved_uv[1])),
                        component_pixels=saved_pixels,
                        component_area_per_tool_scale_sq=float(
                            saved_pixels / saved_frame.scale_px**2
                        ),
                        source="persisted_direct_cross_identity_anchor",
                        marker_cross_shaped=True,
                    )
                    semantic_observation = observe_local_blue_evidence_target(
                        hover_image,
                        continuity_anchor,
                        template,
                        maximum_target_displacement_diagonal_fraction=float(
                            profile["perception"].get(
                                "maximum_hover_local_continuity_diagonal_fraction",
                                0.025,
                            )
                        ),
                    )
                if pending_branch_identity_image is not None:
                    if pending_branch_identity_observation is None:
                        raise ValueError(
                            "wrist-branch identity bridge has no source observation"
                        )
                    bridge = register_fixed_head(
                        pending_branch_identity_image,
                        hover_image,
                        minimum_matches=int(
                            profile["perception"].get(
                                "branch_identity_minimum_homography_matches", 20
                            )
                        ),
                        minimum_inlier_fraction=float(
                            profile["perception"].get(
                                "branch_identity_minimum_inlier_fraction", 0.45
                            )
                        ),
                        maximum_median_residual_diagonal_fraction=float(
                            profile["perception"].get(
                                "branch_identity_maximum_residual_diagonal_fraction",
                                0.004,
                            )
                        ),
                    )
                    if not bridge.accepted or bridge.homography is None:
                        raise ValueError(
                            "wrist-branch target-identity homography rejected: "
                            f"{bridge}"
                        )
                    old_center = np.asarray(
                        pending_branch_identity_observation.center_px,
                        dtype=np.float32,
                    ).reshape(1, 1, 2)
                    projected = cv2.perspectiveTransform(
                        old_center, bridge.homography
                    )[0, 0]
                    projected_uv = semantic_observation.tool_frame.image_to_tool(
                        [projected]
                    )[0]
                    bridge_anchor = replace(
                        semantic_observation,
                        center_px=(float(projected[0]), float(projected[1])),
                        center_uv=(float(projected_uv[0]), float(projected_uv[1])),
                        component_pixels=int(
                            pending_branch_identity_observation.component_pixels
                        ),
                        component_area_per_tool_scale_sq=float(
                            pending_branch_identity_observation.component_pixels
                            / semantic_observation.tool_frame.scale_px**2
                        ),
                        source="wrist_branch_plane_homography_identity_anchor",
                        marker_cross_shaped=True,
                    )
                    semantic_observation = observe_local_blue_evidence_target(
                        hover_image,
                        bridge_anchor,
                        template,
                        maximum_target_displacement_diagonal_fraction=float(
                            profile["perception"].get(
                                "maximum_branch_identity_local_displacement_diagonal_fraction",
                                0.025,
                            )
                        ),
                    )
                    result.setdefault("branch_identity_bridges", []).append(
                        {
                            "hover_index": hover_index,
                            "matches": bridge.matches,
                            "inlier_fraction": bridge.inlier_fraction,
                            "median_residual_diagonal_fraction": (
                                bridge.median_residual_diagonal_fraction
                            ),
                            "projected_center_px": projected.tolist(),
                            "reacquired_center_px": list(
                                semantic_observation.center_px
                            ),
                            "source": semantic_observation.source,
                        }
                    )
                    servo_state["branch_identity_verified"] = True
                    pending_branch_identity_image = None
                    pending_branch_identity_observation = None
            except ValueError as error:
                semantic_error = error
                semantic_observation = None
            tracked_observation = None
            tracking_error = None
            local_observation = None
            local_error = None
            if previous_hover_image is not None and previous_hover_observation is not None:
                try:
                    tracked_observation = track_target_center_lk(
                        previous_hover_image,
                        hover_image,
                        previous_hover_observation,
                        template,
                    )
                    tracked_observation = enforce_rigid_tool_frame(
                        tracked_observation
                    )
                except ValueError as error:
                    tracking_error = error
                try:
                    local_observation = observe_local_blue_evidence_target(
                        hover_image,
                        previous_hover_observation,
                        template,
                        maximum_target_displacement_diagonal_fraction=float(
                            profile["perception"].get(
                                "maximum_hover_local_continuity_diagonal_fraction",
                                0.025,
                            )
                        ),
                    )
                except ValueError as error:
                    local_error = error
            hover_observation = semantic_observation
            disagreement = None
            maximum_disagreement = float(
                profile["perception"].get(
                    "maximum_semantic_flow_disagreement_tool_units", 0.6
                )
            )
            if semantic_observation is not None and tracked_observation is not None:
                disagreement = float(
                    np.linalg.norm(
                        np.asarray(semantic_observation.center_px)
                        - np.asarray(tracked_observation.center_px)
                    )
                    / tracked_observation.tool_frame.scale_px
                )
                if disagreement > maximum_disagreement:
                    hover_observation = tracked_observation
            elif semantic_observation is None:
                hover_observation = tracked_observation or local_observation
            elif previous_hover_observation is not None:
                semantic_step = float(
                    np.linalg.norm(
                        np.asarray(semantic_observation.center_px)
                        - np.asarray(previous_hover_observation.center_px)
                    )
                    / semantic_observation.tool_frame.scale_px
                )
                if semantic_step > maximum_disagreement:
                    disagreement = semantic_step
                    hover_observation = local_observation
            if hover_observation is tracked_observation and tracked_observation is not None:
                result.setdefault("perception_fallbacks", []).append(
                    {
                        "hover_index": hover_index,
                        "method": hover_observation.source,
                        "semantic_error": (
                            None if semantic_error is None else str(semantic_error)
                        ),
                        "semantic_flow_disagreement_tool_units": disagreement,
                        "tracking_inlier_fraction": (
                            hover_observation.tracking_inlier_fraction
                        ),
                    }
                )
            elif hover_observation is local_observation and local_observation is not None:
                result.setdefault("perception_fallbacks", []).append(
                    {
                        "hover_index": hover_index,
                        "method": hover_observation.source,
                        "semantic_error": (
                            None if semantic_error is None else str(semantic_error)
                        ),
                        "semantic_flow_disagreement_tool_units": disagreement,
                        "tracking_error": (
                            None if tracking_error is None else str(tracking_error)
                        ),
                    }
                )
            if hover_observation is None:
                if last_observed_low_q is None:
                    raise semantic_error or tracking_error or ValueError(
                        "target observation unavailable"
                    )
                measured_occluded_q = _q(
                    rpc.get_right_joint_positions(), "measured occluded hover q"
                )
                measured_occluded_pose = np.asarray(
                    fk.pose(measured_occluded_q).parameters(), dtype=float
                )
                current_position = measured_occluded_pose[4:].copy()
                if last_observed_hover_xy is None:
                    observed_position = np.asarray(
                        fk.pose(last_observed_low_q).parameters(), dtype=float
                    )[4:].copy()
                else:
                    observed_position = current_position.copy()
                    observed_position[:2] = np.asarray(
                        last_observed_hover_xy, dtype=float
                    )
                bracket_distance = float(
                    np.linalg.norm(current_position[:2] - observed_position[:2])
                )
                if bracket_distance <= float(
                    profile["perception"].get("minimum_planar_correction_m", 0.002)
                ):
                    raise RuntimeError(
                        "target remains occluded at the minimum bisection radius"
                    )
                midpoint = 0.5 * (current_position + observed_position)
                corrected_q, midpoint_plan = _plan_level_yaw_free_pose(
                    profile,
                    fk,
                    target_position=midpoint,
                    seed_q=last_observed_low_q,
                    role="semantic_occlusion_bisection_low",
                )
                next_hover_q, next_hover_plan = _plan_level_vertical_offset(
                    profile,
                    fk,
                    corrected_q,
                    float(profile["trajectory"]["verification_lift_m"]),
                    seed_hover_q=hover_q_seed,
                )
                # Execution below preserves the measured Cartesian quaternion;
                # an alternate IK witness branch does not change the camera.
                correction = {
                    "method": "last_visible_to_occluded_pose_bisection",
                    "image": str(hover_path),
                    "semantic_error": (
                        None if semantic_error is None else str(semantic_error)
                    ),
                    "tracking_error": (
                        None if tracking_error is None else str(tracking_error)
                    ),
                    "bracket_planar_distance_m": bracket_distance,
                    "current_measured_hover_xy_m": current_position[:2].tolist(),
                    "last_visible_measured_hover_xy_m": observed_position[:2].tolist(),
                    "selected_delta_xy_m": (
                        midpoint[:2] - current_position[:2]
                    ).tolist(),
                    "ik": midpoint_plan,
                    "next_hover_plan": next_hover_plan,
                }
                correction["motion"] = move_hover_with_orientation_lock(
                    next_hover_q,
                    correction["selected_delta_xy_m"],
                )
                result["visual_replans"].append(correction)
                result["visual_replan"] = correction
                result.setdefault("perception_fallbacks", []).append(
                    {
                        "hover_index": hover_index,
                        "method": correction["method"],
                        "bracket_planar_distance_m": bracket_distance,
                    }
                )
                target_q = corrected_q
                hover_q_seed = next_hover_q
                persist()
                if correction["motion"].get("method") == (
                    "audited_joint_branch_escape_after_cartesian_ik_limit"
                ):
                    # This branch deliberately changes the wrist-camera
                    # orientation and therefore destroys the only local pixel
                    # identity available during semantic occlusion.  A global
                    # reinitialization here can select the neighbouring dish.
                    # Preserve the audit evidence, retreat, and let the next
                    # attempt reacquire the tapped object from fixed-head 3D.
                    raise RuntimeError(_BRANCH_ESCAPE_IDENTITY_ERROR)
                continue
            if canonical_hover_goal_uv is not None:
                hover_goal_uv = np.asarray(canonical_hover_goal_uv, dtype=float)
                if hover_goal_uv.shape != (2,) or not np.all(
                    np.isfinite(hover_goal_uv)
                ):
                    raise ValueError("canonical hover goal must be finite uv")
                hover_goal_px = hover_observation.tool_frame.tool_to_image(
                    [hover_goal_uv]
                )[0]
                plane_registration_record = {
                    "method": "calibrated_canonical_tool_frame_goal",
                    "accepted": True,
                }
            else:
                plane_registration = register_fixed_head(
                    plane_reference, hover_image
                )
                if not plane_registration.accepted:
                    raise RuntimeError(
                        "right support-plane registration rejected: "
                        f"{plane_registration}"
                    )
                hover_goal_px = cv2.perspectiveTransform(
                    reference_goal_px,
                    plane_registration.homography,
                )[0, 0]
                hover_goal_uv = hover_observation.tool_frame.image_to_tool(
                    [hover_goal_px]
                )[0]
                plane_registration_record = {
                    "method": "per_frame_support_plane_homography",
                    "accepted": True,
                    "matches": plane_registration.matches,
                    "inlier_fraction": plane_registration.inlier_fraction,
                    "median_residual_diagonal_fraction": plane_registration.median_residual_diagonal_fraction,
                }
            hover_error_uv = (
                np.asarray(hover_observation.center_uv, dtype=float) - hover_goal_uv
            )
            hover_center_error = float(
                np.linalg.norm(hover_error_uv) / template.square_side_u
            )
            hover_aligned_by_canonical_goal = bool(
                hover_center_error
                <= float(profile["perception"]["maximum_hover_center_error_scale"])
            )
            # A rejected low-pose image is the most direct observation of the
            # rim relative to the jaws. Its bounded XY correction defines a
            # new low candidate. At that candidate's freshly planned hover,
            # require semantic target identity but do not pull the arm back to
            # the old hover pixel goal: doing so erased every low correction
            # and repeated the same miss indefinitely. Closure remains
            # impossible until a new strict low image passes all gates.
            hover_aligned, hover_alignment_mode = _hover_alignment_policy(
                profile,
                hover_observation,
                canonical_goal_aligned=hover_aligned_by_canonical_goal,
                direct_cross_verified=bool(
                    servo_state.get("direct_cross_verified", False)
                    or (
                        hover_observation is semantic_observation
                        and semantic_observation is not None
                        and semantic_observation.marker_cross_shaped is True
                    )
                ),
                replaying_low_pose_correction=bool(
                    fresh_low_correction_replay and hover_index == 0
                ),
            )
            hover_record = {
                "index": hover_index,
                "timestamp": hover_ts,
                "image": str(hover_path),
                "low_q_physical_rad": np.asarray(target_q, dtype=float).tolist(),
                "hover_q_physical_rad": np.asarray(hover_q_seed, dtype=float).tolist(),
                "observation": hover_observation.to_dict(),
                "dynamic_goal_px": hover_goal_px.tolist(),
                "dynamic_goal_uv": hover_goal_uv.tolist(),
                "error_uv": hover_error_uv.tolist(),
                "normalized_center_error": hover_center_error,
                "aligned_by_canonical_goal": hover_aligned_by_canonical_goal,
                "hover_alignment_mode": hover_alignment_mode,
                "low_pose_correction_replay": bool(
                    fresh_low_correction_replay and hover_index == 0
                ),
                "plane_registration": plane_registration_record,
                "aligned_for_descent": hover_aligned,
            }
            measured_hover_q = _q(
                rpc.get_right_joint_positions(), "measured hover q"
            )
            measured_hover_xy = np.asarray(
                fk.pose(measured_hover_q).parameters(), dtype=float
            )[4:6]
            hover_record["measured_hover_q_physical_rad"] = measured_hover_q.tolist()
            hover_record["measured_hover_xy_m"] = measured_hover_xy.tolist()
            current_error_norm = float(np.linalg.norm(hover_error_uv))
            if (
                hover_observation is semantic_observation
                and semantic_observation is not None
                and (
                    semantic_observation.marker_cross_shaped is True
                    or semantic_observation.source
                    == "persisted_direct_cross_identity_continuity"
                )
            ):
                servo_state["direct_cross_verified"] = True
                servo_state["hover_identity_anchor"] = {
                    "center_px": list(hover_observation.center_px),
                    "component_pixels": int(hover_observation.component_pixels),
                }
            if current_error_norm < float(
                servo_state.get("best_observation_error_norm", math.inf)
            ):
                # This checkpoint is independent of whether the target is
                # clipped by an image border.  Border samples are censored for
                # Jacobian estimation, but a closer identity-consistent camera
                # view is exactly the state we want after a recoverable error.
                servo_state["best_observation_error_norm"] = current_error_norm
                servo_state["best_observation_low_q_physical_rad"] = (
                    np.asarray(target_q, dtype=float).tolist()
                )
                servo_state["best_observation_hover_q_physical_rad"] = (
                    measured_hover_q.tolist()
                )
                servo_state["best_observation_tool_frame"] = dict(
                    hover_record["observation"]["tool_frame"]
                )
                servo_state["best_observation_normalized_center_error"] = (
                    hover_center_error
                )
                servo_state["best_observation_target"] = {
                    "center_px": list(hover_observation.center_px),
                    "component_pixels": int(hover_observation.component_pixels),
                    "source": str(hover_observation.source),
                    "marker_cross_shaped": hover_observation.marker_cross_shaped,
                }
                servo_state["best_observation_direct_cross_verified"] = bool(
                    servo_state.get("direct_cross_verified", False)
                )
            if current_error_norm < float(
                servo_state.get("best_error_norm", math.inf)
            ):
                servo_state["best_hover_q_physical_rad"] = (
                    measured_hover_q.tolist()
                )
                servo_state["best_tool_frame"] = dict(
                    hover_record["observation"]["tool_frame"]
                )
                servo_state["best_normalized_center_error"] = (
                    hover_center_error
                )
                servo_state["best_target_observation"] = {
                    "center_px": list(hover_observation.center_px),
                    "component_pixels": int(hover_observation.component_pixels),
                    "source": str(hover_observation.source),
                    "marker_cross_shaped": hover_observation.marker_cross_shaped,
                }
                servo_state["best_direct_cross_verified"] = bool(
                    servo_state.get("direct_cross_verified", False)
                )
            result["hover_iterations"].append(hover_record)
            result["hover"] = hover_record
            if semantic_observation is not None and hover_observation is semantic_observation:
                servo_state["last_semantic_low_q_physical_rad"] = np.asarray(
                    target_q, dtype=float
                ).tolist()
            previous_hover_image = hover_image.copy()
            previous_hover_observation = hover_observation
            last_observed_low_q = np.asarray(target_q, dtype=float).copy()
            last_observed_hover_xy = measured_hover_xy.copy()
            persist()
            if (
                hover_alignment_mode
                == "identity_only_before_fresh_preclose"
                and not hover_aligned
            ):
                # This stage has no image-space motion objective.  If the
                # direct marked identity is not visible at the independently
                # planned vertical hover, moving toward a demonstration pixel
                # can only select a distractor.  Retreat and reacquire from the
                # fixed head instead.
                raise RuntimeError(
                    "right vertical-hover frame did not directly verify the "
                    "fixed-head target identity"
                )
            if hover_aligned and not level_hover_transition_done:
                measured_camera_pose = _pose(
                    rpc.get_right_ee_pose().parameters(),
                    "measured aligned camera-view pose",
                )
                camera_view_level = assess_jaw_level(
                    measured_camera_pose, level_reference, planned=False
                )
                if not camera_view_level.accepted:
                    transition = _transition_to_level_hover(
                        profile, rpc, fk, target_q
                    )
                    result["level_hover_transition"] = transition
                    result["stages"]["level_hover_transition"] = transition[
                        "execution"
                    ]
                    level_hover_transition_done = True
                    transitioned_pose = _pose(
                        transition["measured_pose_wxyz_xyz"],
                        "transitioned level-hover pose",
                    )
                    fixed_hover_orientation_wxyz = transitioned_pose[:4].copy()
                    hover_q_seed = _q(
                        transition["level_hover_q_physical_rad"],
                        "transitioned level-hover q",
                    )
                    for key in (
                        "jacobian",
                        "last_xy_m",
                        "last_error_uv",
                        "last_component_area_per_tool_scale_sq",
                        "best_error_norm",
                        "best_low_q_physical_rad",
                        "best_xy_m",
                        "best_hover_q_physical_rad",
                        "best_tool_frame",
                        "best_normalized_center_error",
                        "best_target_observation",
                        "best_direct_cross_verified",
                        "best_observation_error_norm",
                        "best_observation_low_q_physical_rad",
                        "best_observation_hover_q_physical_rad",
                        "best_observation_tool_frame",
                        "best_observation_normalized_center_error",
                        "best_observation_target",
                        "best_observation_direct_cross_verified",
                        "runtime_axis_calibration",
                    ):
                        servo_state.pop(key, None)
                    servo_state["enable_runtime_axis_calibration"] = True
                    previous_hover_image = None
                    previous_hover_observation = None
                    last_observed_low_q = None
                    last_observed_hover_xy = None
                    result.setdefault("hover_orientation_relocks", []).append(
                        {
                            "reason": "descent_ready_level_hover_transition",
                            "wxyz": fixed_hover_orientation_wxyz.tolist(),
                        }
                    )
                    persist()
                    continue
            action = machine.hover_assessment(
                target_visible=True, aligned=hover_aligned
            )
            if action.name != "correct_xy_at_hover":
                break
            if hover_index >= maximum_hover_replans:
                raise RuntimeError(
                    "hover visual servo exhausted its configured trust-region steps"
                )
            corrected_q, correction = _metric_replan(
                profile,
                hover_observation,
                target_q,
                fk=fk,
                reference_center_uv=hover_goal_uv,
                servo_state=servo_state,
                measured_current_xy_m=measured_hover_xy,
            )
            next_hover_q, next_hover_plan = _plan_level_vertical_offset(
                profile,
                fk,
                corrected_q,
                float(profile["trajectory"]["verification_lift_m"]),
                seed_hover_q=hover_q_seed,
            )
            # Execution below preserves the measured Cartesian quaternion;
            # an alternate IK witness branch does not change the camera.
            correction["next_hover_plan"] = next_hover_plan
            correction["motion"] = move_hover_with_orientation_lock(
                next_hover_q,
                correction["selected_delta_xy_m"],
            )
            result["visual_replans"].append(correction)
            result["visual_replan"] = correction
            target_q = corrected_q
            hover_q_seed = next_hover_q
            persist()
    except BaseException as error:
        result["hover_perception_or_servo_error"] = (
            f"{type(error).__name__}: {error}"
        )
        result["stages"]["hover_retreat"] = _retreat_open(profile, rpc, fk)
        exhausted = (
            isinstance(error, RuntimeError)
            and str(error)
            == "hover visual servo exhausted its configured trust-region steps"
        )
        best_q = servo_state.get(
            "best_observation_low_q_physical_rad",
            servo_state.get("best_low_q_physical_rad"),
        )
        best_hover_q = servo_state.get(
            "best_observation_hover_q_physical_rad",
            servo_state.get("best_hover_q_physical_rad"),
        )
        best_tool_frame = servo_state.get(
            "best_observation_tool_frame", servo_state.get("best_tool_frame")
        )
        if (
            best_q is not None
            and best_hover_q is not None
            and isinstance(best_tool_frame, dict)
        ):
            target_q = _q(best_q, "best recoverable-error hover low q")
            result["next_attempt_seed"] = {
                "method": (
                    "best_observed_hover_after_trust_region_exhaustion"
                    if exhausted
                    else "best_observed_hover_after_recoverable_error"
                ),
                "source_error": result["hover_perception_or_servo_error"],
                "low_q_physical_rad": target_q.tolist(),
                "hover_q_physical_rad": _q(
                    best_hover_q, "best measured recoverable-error hover q"
                ).tolist(),
                "tool_frame": dict(best_tool_frame),
                "normalized_center_error": float(
                    servo_state.get(
                        "best_observation_normalized_center_error",
                        servo_state.get("best_normalized_center_error", math.inf),
                    )
                ),
                "target_observation": dict(
                    servo_state.get(
                        "best_observation_target",
                        servo_state.get("best_target_observation"),
                    )
                    or {}
                ),
                "direct_cross_verified": bool(
                    servo_state.get(
                        "best_observation_direct_cross_verified",
                        servo_state.get("best_direct_cross_verified", False),
                    )
                ),
                "closure_authorized": False,
            }
            result["hover_progress_preserved_after_error"] = True
        if exhausted:
            result["hover_search_exhausted"] = True
        persist()
        if exhausted or not isinstance(error, KeyboardInterrupt):
            return False, result, target_q
        raise

    if not allow_descent:
        result["hover_only_complete"] = True
        result["stages"]["hover_only_retreat"] = _retreat_open(
            profile, rpc, fk
        )
        persist()
        return False, result, target_q

    try:
        level_endpoint_q = (
            result["hover"]["measured_hover_q_physical_rad"]
            if result["visual_replans"]
            else result["hover"]["hover_q_physical_rad"]
        )
        measured_hover_level, hover_level_report = (
            _require_hover_level_after_endpoint_convergence(
                profile,
                rpc,
                fk,
                level_endpoint_q,
                descent_checkpoint,
            )
        )
    except BaseException as error:
        result["hover_level_error"] = f"{type(error).__name__}: {error}"
        result["stages"]["hover_level_retreat"] = _retreat_open(
            profile, rpc, fk
        )
        persist()
        return False, result, target_q
    result["hover_level_checkpoint"] = hover_level_report
    machine.level_checkpoint(accepted=measured_hover_level.accepted)
    result["hover"]["level"] = measured_hover_level.to_dict()
    try:
        descent_result = _descend_from_hover(
            profile, rpc, fk, target_q
        )
        result["stages"]["descent"] = descent_result
        # Subsequent image correction must stay on the exact, audited wrist
        # branch used by the level descent rather than returning to the old
        # Cartesian-IK witness branch.
        target_q = _q(
            descent_result["planned_low_q_physical_rad"],
            "executed level-descent low q",
        )
    except DescentPlanRejected as error:
        # No low setpoint has been sent.  The gripper is still open at hover,
        # so a normal retreat is sufficient and a vertical "recovery" would
        # only add unplanned motion.
        result["descent_planning_error"] = f"{type(error).__name__}: {error}"
        result["stages"]["retreat"] = _retreat_open(profile, rpc, fk)
        persist()
        return False, result, target_q
    except BaseException as error:
        result["descent_execution_error"] = f"{type(error).__name__}: {error}"
        result["recovery"] = _recover_vertical_then_open(profile, rpc, fk)
        result["stages"]["retreat"] = _retreat_open(profile, rpc, fk)
        persist()
        return False, result, target_q
    machine.descent_complete()
    low_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    low_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    captured_at = time.time()
    preclose_image, preclose_ts = camera.frame(fresh_after_s=captured_at)
    cv2.imwrite(str(output_dir / "preclose.png"), preclose_image)
    persist()
    preclose_identity_observation = None
    cached_preclose_identity_observation = None
    if fresh_low_correction_replay and cached_hover_alignment is not None:
        cached_preclose_value = cached_hover_alignment.get(
            "source_preclose_identity_observation"
        )
        if isinstance(cached_preclose_value, dict):
            cached_preclose_identity_observation = _target_observation_from_dict(
                cached_preclose_value
            )

    def observe_preclose_image(image, *, recovery: bool = False):
        """Select the same hover cross after the calibrated vertical parallax."""

        nonlocal preclose_identity_observation
        if str(
            profile["perception"].get(
                "hover_alignment_mode", "canonical_goal"
            )
        ) != "identity_only_before_fresh_preclose":
            return observe(
                image,
                proximity_score_weight=float(
                    profile["perception"].get(
                        "preclose_proximity_score_weight", 2.0
                    )
                ),
                maximum_candidate_distance_tool_units=(
                    float(
                        profile["perception"].get(
                            "maximum_recovery_candidate_distance_tool_units",
                            profile["perception"][
                                "maximum_candidate_distance_tool_units"
                            ],
                        )
                    )
                    if recovery
                    else None
                ),
            )
        anchor = (
            preclose_identity_observation
            or previous_hover_observation
            or cached_preclose_identity_observation
        )
        if anchor is None:
            raise ValueError("preclose identity has no verified hover anchor")
        identity_anchor = anchor
        if (
            preclose_identity_observation is None
            and anchor is cached_preclose_identity_observation
        ):
            # This anchor was captured at the same low camera stage. Applying
            # hover-to-low parallax a second time would point at a distractor.
            maximum_fraction = float(
                profile["perception"].get(
                    "maximum_initial_preclose_identity_displacement_diagonal_fraction",
                    0.06,
                )
            )
        elif preclose_identity_observation is None:
            delta = np.asarray(
                servo_state.get(
                    "vertical_descent_target_pixel_delta_px",
                    profile["perception"].get(
                        "vertical_descent_target_pixel_delta_px", [0.0, 0.0]
                    ),
                ),
                dtype=float,
            )
            if delta.shape != (2,) or not np.all(np.isfinite(delta)):
                raise ValueError(
                    "vertical descent target pixel delta must contain two finite values"
                )
            predicted_center = np.asarray(anchor.center_px, dtype=float) + delta
            predicted_uv = anchor.tool_frame.image_to_tool(
                [predicted_center]
            )[0]
            anchor = replace(
                anchor,
                center_px=(
                    float(predicted_center[0]),
                    float(predicted_center[1]),
                ),
                center_uv=(float(predicted_uv[0]), float(predicted_uv[1])),
                source="vertical_descent_parallax_identity_anchor",
                marker_cross_shaped=True,
            )
            maximum_fraction = float(
                profile["perception"].get(
                    "maximum_initial_preclose_identity_displacement_diagonal_fraction",
                    0.06,
                )
            )
        else:
            maximum_fraction = float(
                profile["perception"].get(
                    "maximum_hover_local_continuity_diagonal_fraction", 0.025
                )
            )
        if recovery:
            maximum_fraction *= 1.5
        try:
            selected = observe_local_blue_evidence_target(
                image,
                anchor,
                template,
                maximum_target_displacement_diagonal_fraction=maximum_fraction,
            )
        except ValueError as local_error:
            if preclose_identity_observation is not None:
                raise
            # Camera-stage changes can produce a larger deterministic
            # parallax than an old calibration predicts.  Reacquire globally
            # only inside a broad continuity envelope from the verified hover
            # identity; this rejects the distant blue reflection in the upper
            # dish while allowing the same cross to move across the frame.
            maximum_reacquisition = float(
                profile["perception"].get(
                    "maximum_preclose_parallax_reacquisition_diagonal_fraction",
                    0.20,
                )
            )
            minimum_area_scale = float(
                profile["perception"].get(
                    "minimum_preclose_parallax_component_area_scale", 0.25
                )
            )
            maximum_area_scale = float(
                profile["perception"].get(
                    "maximum_preclose_parallax_component_area_scale", 4.0
                )
            )
            candidate, direct_reacquisition = _reacquire_direct_preclose_cross(
                image,
                anchor,
                identity_anchor,
                template,
                maximum_displacement_diagonal_fraction=maximum_reacquisition,
                minimum_component_area_scale=minimum_area_scale,
                maximum_component_area_scale=maximum_area_scale,
            )
            candidate = enforce_rigid_tool_frame(candidate)
            learned_delta = (
                np.asarray(candidate.center_px, dtype=float)
                - np.asarray(identity_anchor.center_px, dtype=float)
            )
            servo_state["vertical_descent_target_pixel_delta_px"] = (
                learned_delta.tolist()
            )
            result["preclose_parallax_reacquisition"] = {
                "accepted": True,
                "local_error": str(local_error),
                "learned_delta_px": learned_delta.tolist(),
                **direct_reacquisition,
                "maximum_displacement_diagonal_fraction": maximum_reacquisition,
            }
            selected = candidate
        selected = enforce_rigid_tool_frame(selected)
        preclose_identity_observation = selected
        return selected

    try:
        observation = observe_preclose_image(preclose_image)
    except BaseException as error:
        # The failure state is known here: lowest pose, jaws still fully open,
        # and no close command has been sent.  Recover vertically before any
        # joint-space retreat so a perception exception cannot strand a tip
        # beside the support.
        result["preclose_perception_error"] = f"{type(error).__name__}: {error}"
        persist()
        recovery_observation = None
        try:
            recovery_observation = observe_preclose_image(
                preclose_image, recovery=True
            )
        except BaseException as recovery_error:
            result["preclose_recovery_perception_error"] = (
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
        if recovery_observation is not None:
            try:
                recovery_level = preclose_checkpoint.require(
                    "before_recovery_only_replan"
                )
                result["preclose"] = {
                    "timestamp": preclose_ts,
                    "observation": recovery_observation.to_dict(),
                    "level": recovery_level.to_dict(),
                    "allowed_to_close": False,
                    "recovery_only": True,
                }
                if (
                    "last_xy_m" not in preclose_servo_state
                    and "jacobian" in servo_state
                ):
                    preclose_servo_state["jacobian"] = np.asarray(
                        servo_state["jacobian"], dtype=float
                    ).tolist()
                corrected_q, correction = _metric_replan(
                    profile,
                    recovery_observation,
                    target_q,
                    fk=fk,
                    servo_state=preclose_servo_state,
                    measured_current_xy_m=low_pose[4:6],
                    fixed_orientation_q=low_q,
                    level_yaw_free=False,
                )
                # A low-confidence/dark-scene observation may prove that the
                # last probe regressed, but it may not originate another probe
                # or authorize closure.  Return only to the previously seen
                # best pose and demand a fresh nominal preclose image there.
                if (
                    correction.get("method")
                    == "fixed_camera_physical_xy_broyden_backtrack"
                ):
                    result["visual_replan"] = correction
                    result["preclose_recovery_decision"] = (
                        "backtrack_to_previously_observed_best"
                    )
                else:
                    corrected_q = target_q
                    result["preclose_recovery_decision"] = (
                        "hold_current_alignment_no_new_probe"
                    )
                result["preclose_servo_state"] = dict(preclose_servo_state)
            except BaseException as recovery_replan_error:
                corrected_q = target_q
                result["preclose_recovery_replan_error"] = (
                    f"{type(recovery_replan_error).__name__}: "
                    f"{recovery_replan_error}"
                )
        else:
            corrected_q = target_q
        result["recovery"] = _recover_vertical_then_open(profile, rpc, fk)
        result["stages"]["retreat"] = _retreat_open(profile, rpc, fk)
        persist()
        return False, result, corrected_q
    assessment = observation.grasp_window
    quantile_limit = profile["perception"].get(
        "maximum_preclose_quantile_error_scale"
    )
    allowed = _preclose_visual_alignment_allowed(profile, observation) and (
        quantile_limit is None
        or assessment.normalized_quantile_error <= float(quantile_limit)
    )
    canonical_marker_center_error_scale = (
        _preclose_marker_center_error_scale(profile, observation)
    )
    if _defer_contact_settle_until_visual_alignment(
        profile, visually_aligned=allowed
    ):
        # Calibration/exploration needs one consistent eye-in-hand image, not
        # repeated contact-height and sub-mm level corrections.  Replan from
        # this audited low observation and reserve the final support settle,
        # strict level checkpoint, and closure for an image that is already
        # inside the grasp window.
        coarse_level = assess_jaw_level(
            low_pose, preclose_reference, planned=False
        )
        result["preclose"] = {
            "timestamp": preclose_ts,
            "observation": observation.to_dict(),
            "level": coarse_level.to_dict(),
            "height": None,
            "canonical_marker_center_error_scale": (
                canonical_marker_center_error_scale
            ),
            "allowed_to_close": False,
            "rejection_reason": "visual_alignment_before_contact_settle",
            "contact_settle_deferred": True,
        }
        corrected_q = target_q
        try:
            if (
                "last_xy_m" not in preclose_servo_state
                and "jacobian" in servo_state
            ):
                preclose_servo_state["jacobian"] = np.asarray(
                    servo_state["jacobian"], dtype=float
                ).tolist()
            corrected_q, correction = _metric_replan(
                profile,
                observation,
                target_q,
                fk=fk,
                servo_state=preclose_servo_state,
                measured_current_xy_m=low_pose[4:6],
                fixed_orientation_q=low_q,
                level_yaw_free=False,
            )
            corrected_q, correction = _compensate_hover_tracking_bias(
                profile,
                fk,
                corrected_q,
                correction,
                hover=result["hover"],
                hover_plan=result["hover_plan"],
            )
            result["visual_replan"] = correction
            result["preclose_servo_state"] = dict(preclose_servo_state)
        except BaseException as replan_error:
            result["preclose_coarse_replan_error"] = (
                f"{type(replan_error).__name__}: {replan_error}"
            )
        persist()
        result["recovery"] = _recover_vertical_then_open(profile, rpc, fk)
        result["stages"]["retreat_after_coarse_visual_replan"] = (
            _retreat_open(profile, rpc, fk)
        )
        persist()
        return False, result, corrected_q
    try:
        measured_level = preclose_checkpoint.require("before_close")
    except BaseException as error:
        initial_level_error = f"{type(error).__name__}: {error}"
        level_correction_enabled = bool(
            profile["execution"].get(
                "preclose_same_position_level_correction", False
            )
        )
        if level_correction_enabled:
            # Jaw level is a prerequisite for both vertical height settling
            # and planar visual correction. It must not depend on whether the
            # current image is already centered; otherwise tracking-induced
            # fingertip mismatch blocks the very motion needed to obtain the
            # canonical lowest-pose image.
            try:
                maximum_level_corrections = int(
                    profile["execution"].get(
                        "preclose_maximum_same_position_level_corrections", 2
                    )
                )
                if maximum_level_corrections <= 0:
                    raise ValueError(
                        "preclose level correction count must be positive"
                )
                commanded_tip_bias = 0.0
                correction_records = []
                for correction_index in range(maximum_level_corrections):
                    level_start_pose = np.asarray(
                        rpc.get_right_ee_pose().parameters(), dtype=float
                    )
                    signed_tip_error = _signed_tip_height_difference_m(
                        level_start_pose, preclose_reference
                    )
                    # Cartesian pose streaming tracks the requested
                    # quaternion directly while holding measured XYZ.  The
                    # residual/secant bias was only needed by the old direct
                    # joint endpoint and deliberately tilts a Cartesian move.
                    # Request the geometric zero; the fresh measured
                    # checkpoint below remains authoritative.
                    next_tip_bias = 0.0
                    bias_update_method = "cartesian_geometric_zero"
                    commanded_tip_bias = next_tip_bias
                    level_samples, level_audit = (
                        _plan_same_position_level_joint_samples(
                            profile,
                            fk,
                            start_q=rpc.get_right_joint_positions(),
                            target_signed_tip_height_difference_m=(
                                commanded_tip_bias
                            ),
                        )
                    )
                    level_motion = _execute_same_position_level_cartesian(
                        profile,
                        rpc,
                        fk,
                        planned_level_q=level_audit[
                            "planned_level_q_physical_rad"
                        ],
                        aperture=1.0,
                        stage="preclose_cartesian_same_position_level",
                    )
                    level_motion["offline_level_audit"] = level_audit
                    corrected_at = time.time()
                    preclose_image, preclose_ts = camera.frame(
                        fresh_after_s=corrected_at
                    )
                    cv2.imwrite(
                        str(
                            output_dir
                            / (
                                "preclose_after_level_correction_"
                                f"{correction_index + 1:02d}.png"
                            )
                        ),
                        preclose_image,
                    )
                    observation = observe_preclose_image(preclose_image)
                    assessment = observation.grasp_window
                    allowed = _preclose_visual_alignment_allowed(
                        profile, observation
                    ) and (
                        quantile_limit is None
                        or assessment.normalized_quantile_error
                        <= float(quantile_limit)
                    )
                    low_q = np.asarray(
                        rpc.get_right_joint_positions(), dtype=float
                    )
                    low_pose = np.asarray(
                        rpc.get_right_ee_pose().parameters(), dtype=float
                    )
                    record = {
                        "index": correction_index + 1,
                        "measured_signed_tip_height_difference_m": (
                            signed_tip_error
                        ),
                        "commanded_signed_tip_height_difference_m": (
                            commanded_tip_bias
                        ),
                        "bias_update_method": bias_update_method,
                        "audit": level_audit,
                        "motion": level_motion,
                        "fresh_observation": observation.to_dict(),
                        "fresh_alignment_allowed": allowed,
                    }
                    correction_records.append(record)
                    result["preclose_level_correction"] = {
                        "initial_error": initial_level_error,
                        "corrections": correction_records,
                    }
                    persist()
                    try:
                        measured_level = preclose_checkpoint.require(
                            "before_close_after_same_position_level_correction_"
                            f"{correction_index + 1:02d}"
                        )
                    except RuntimeError as residual_level_error:
                        record["fresh_level_error"] = str(residual_level_error)
                        persist()
                        if correction_index + 1 >= maximum_level_corrections:
                            raise
                    else:
                        record["fresh_level"] = measured_level.to_dict()
                        persist()
                        break
            except BaseException as correction_error:
                result["preclose"] = {
                    "timestamp": preclose_ts,
                    "observation": observation.to_dict(),
                    "level_error": initial_level_error,
                    "level_correction_error": (
                        f"{type(correction_error).__name__}: "
                        f"{correction_error}"
                    ),
                    "allowed_to_close": False,
                }
                corrected_q = target_q
                try:
                    if allowed:
                        corrected_q, ik = _plan_level_fixed_yaw_pose(
                            profile,
                            fk,
                            target_position=low_pose[4:7],
                            seed_q=low_q,
                            role="level_fixed_yaw_low_level_retry",
                        )
                        correction = {
                            "method": "preclose_same_xy_level_retry",
                            "reference_center_uv": np.asarray(
                                GraspWindowTemplate.from_dict(
                                    _load(
                                        profile["target_identity"][
                                            "grasp_window_selection"
                                        ]
                                    )["template"]
                                ).reference_center_uv,
                                dtype=float,
                            ).tolist(),
                            "error_uv": (
                                np.asarray(observation.center_uv, dtype=float)
                                - np.asarray(
                                    GraspWindowTemplate.from_dict(
                                        _load(
                                            profile["target_identity"][
                                                "grasp_window_selection"
                                            ]
                                        )["template"]
                                    ).reference_center_uv,
                                    dtype=float,
                                )
                            ).tolist(),
                            "selected_delta_xy_m": [0.0, 0.0],
                            "target_low_xy_m": low_pose[4:6].tolist(),
                            "corrected_q_physical_rad": corrected_q.tolist(),
                            "ik": ik,
                        }
                    else:
                        if (
                            "last_xy_m" not in preclose_servo_state
                            and "jacobian" in servo_state
                        ):
                            preclose_servo_state["jacobian"] = np.asarray(
                                servo_state["jacobian"], dtype=float
                            ).tolist()
                        corrected_q, correction = _metric_replan(
                            profile,
                            observation,
                            target_q,
                            fk=fk,
                            servo_state=preclose_servo_state,
                            measured_current_xy_m=low_pose[4:6],
                            fixed_orientation_q=low_q,
                            level_yaw_free=False,
                        )
                        corrected_q, correction = _compensate_hover_tracking_bias(
                            profile,
                            fk,
                            corrected_q,
                            correction,
                            hover=result["hover"],
                            hover_plan=result["hover_plan"],
                        )
                    result["visual_replan"] = correction
                    result["preclose_servo_state"] = dict(
                        preclose_servo_state
                    )
                except BaseException as replan_error:
                    result["preclose_level_failure_replan_error"] = (
                        f"{type(replan_error).__name__}: {replan_error}"
                    )
                persist()
                result["recovery"] = _recover_vertical_then_open(
                    profile, rpc, fk
                )
                result["stages"]["retreat"] = _retreat_open(
                    profile, rpc, fk
                )
                persist()
                return False, result, corrected_q
        else:
            result["preclose"] = {
                "timestamp": preclose_ts,
                "observation": observation.to_dict(),
                "level_error": initial_level_error,
                "level_correction_skipped": (
                    "disabled"
                ),
                "allowed_to_close": False,
            }
            corrected_q = target_q
            if not allowed:
                try:
                    if (
                        "last_xy_m" not in preclose_servo_state
                        and "jacobian" in servo_state
                    ):
                        preclose_servo_state["jacobian"] = np.asarray(
                            servo_state["jacobian"], dtype=float
                        ).tolist()
                    corrected_q, correction = _metric_replan(
                        profile,
                        observation,
                        target_q,
                        fk=fk,
                        servo_state=preclose_servo_state,
                        measured_current_xy_m=low_pose[4:6],
                        fixed_orientation_q=low_q,
                        level_yaw_free=False,
                    )
                    corrected_q, correction = _compensate_hover_tracking_bias(
                        profile,
                        fk,
                        corrected_q,
                        correction,
                        hover=result["hover"],
                        hover_plan=result["hover_plan"],
                    )
                    result["visual_replan"] = correction
                    result["preclose_servo_state"] = dict(
                        preclose_servo_state
                    )
                except BaseException as replan_error:
                    result["preclose_unlevel_replan_error"] = (
                        f"{type(replan_error).__name__}: {replan_error}"
                    )
            # Preserve the fresh visual observation and exact failed checkpoint
            # even if the subsequent recovery itself raises.
            persist()
            result["recovery"] = _recover_vertical_then_open(profile, rpc, fk)
            result["stages"]["retreat"] = _retreat_open(profile, rpc, fk)
            persist()
            return False, result, corrected_q
    verified_preclose_pose = _pose(
        profile["trajectory"].get(
            "verified_support_contact_pose_wxyz_xyz",
            profile["trajectory"]["verified_preclose_pose_wxyz_xyz"],
        ),
        "verified preclose pose for height gate",
    )
    support_up = np.asarray(level_reference.support_up_robot, dtype=float)
    support_up /= np.linalg.norm(support_up)
    maximum_height_settles = int(
        profile["execution"].get(
            "preclose_maximum_height_settle_corrections", 2
        )
    )
    height_report = _preclose_height_report(
        profile,
        low_pose=low_pose,
        verified_preclose_pose=verified_preclose_pose,
        support_up=support_up,
    )
    if not height_report["accepted"]:
        # Establish the lowest audited open-jaw camera stage before deciding
        # XY. A 0.1--0.5 mm height residual changes both parallax and apparent
        # rim position; mixing that residual into a planar IK correction made
        # the solver fail and repeated the same high image on the next run.
        # The settle remains straight, open, level checked, and freshly imaged.
        height_settle_records = []
        maximum_height_settle_m = float(
            profile["execution"].get(
                "preclose_maximum_height_settle_m", 0.003
            )
        )
        height_settle_duration_s = float(
            profile["execution"].get(
                "preclose_height_settle_duration_s", 0.8
            )
        )
        # Cartesian leveling requests geometric zero while holding measured
        # XYZ.  Keep the explicit state in the record so older direct-joint
        # runs remain comparable; for this controller it stays zero.
        height_settle_commanded_tip_bias = 0.0
        for settle_index in range(maximum_height_settles):
            requested_down_m = min(
                float(height_report["height_above_verified_m"]),
                maximum_height_settle_m,
            )
            if requested_down_m <= 0.0:
                break
            settle_started_at = time.time()
            inherited_tip_bias = height_settle_commanded_tip_bias
            post_settle_level_corrections = []
            try:
                settle_motion = _execute_preclose_vertical_height_settle(
                    profile,
                    rpc,
                    fk,
                    requested_down_m=requested_down_m,
                    support_up=support_up,
                    settle_index=settle_index + 1,
                )
                low_q = np.asarray(
                    rpc.get_right_joint_positions(), dtype=float
                )
                low_pose = np.asarray(
                    rpc.get_right_ee_pose().parameters(), dtype=float
                )
                commanded_tip_bias = height_settle_commanded_tip_bias
                maximum_post_settle_level_corrections = int(
                    profile["execution"].get(
                        "preclose_maximum_same_position_level_corrections", 2
                    )
                )
                for level_index in range(
                    maximum_post_settle_level_corrections + 1
                ):
                    try:
                        measured_level = preclose_checkpoint.require(
                            "after_cartesian_height_settle_level_"
                            f"{settle_index + 1:02d}_"
                            f"{level_index:02d}"
                        )
                        break
                    except RuntimeError as post_settle_level_error:
                        if level_index >= maximum_post_settle_level_corrections:
                            post_settle_level_corrections.append(
                                {
                                    "index": level_index + 1,
                                    "source_error": str(
                                        post_settle_level_error
                                    ),
                                    "measured_signed_tip_height_difference_m": (
                                        _signed_tip_height_difference_m(
                                            low_pose, preclose_reference
                                        )
                                    ),
                                    "commanded_signed_tip_height_difference_m": (
                                        commanded_tip_bias
                                    ),
                                    "bias_update_method": "correction_budget_exhausted",
                                    "motion": None,
                                }
                            )
                            raise
                        signed_tip_error = _signed_tip_height_difference_m(
                            low_pose, preclose_reference
                        )
                        next_tip_bias = 0.0
                        bias_update_method = "cartesian_geometric_zero"
                        commanded_tip_bias = next_tip_bias
                        level_samples, level_audit = (
                            _plan_same_position_level_joint_samples(
                                profile,
                                fk,
                                start_q=low_q,
                                target_signed_tip_height_difference_m=(
                                    commanded_tip_bias
                                ),
                            )
                        )
                        level_motion = _execute_same_position_level_cartesian(
                            profile,
                            rpc,
                            fk,
                            planned_level_q=level_audit[
                                "planned_level_q_physical_rad"
                            ],
                            aperture=1.0,
                            stage=(
                                "preclose_cartesian_post_height_level_"
                                f"{settle_index + 1:02d}_"
                                f"{level_index + 1:02d}"
                            ),
                        )
                        level_motion["offline_level_audit"] = level_audit
                        post_settle_level_corrections.append(
                            {
                                "index": level_index + 1,
                                "source_error": str(post_settle_level_error),
                                "measured_signed_tip_height_difference_m": (
                                    signed_tip_error
                                ),
                                "commanded_signed_tip_height_difference_m": (
                                    commanded_tip_bias
                                ),
                                "bias_update_method": bias_update_method,
                                "audit": level_audit,
                                "motion": level_motion,
                            }
                        )
                        low_q = np.asarray(
                            rpc.get_right_joint_positions(), dtype=float
                        )
                        low_pose = np.asarray(
                            rpc.get_right_ee_pose().parameters(), dtype=float
                        )
                height_settle_commanded_tip_bias = commanded_tip_bias
                preclose_image, preclose_ts = camera.frame(
                    fresh_after_s=settle_started_at
                )
                cv2.imwrite(
                    str(
                        output_dir
                        / (
                            "preclose_after_height_settle_"
                            f"{settle_index + 1:02d}.png"
                        )
                    ),
                    preclose_image,
                )
                observation = observe_preclose_image(preclose_image)
                assessment = observation.grasp_window
                allowed = _preclose_visual_alignment_allowed(
                    profile, observation
                ) and (
                    quantile_limit is None
                    or assessment.normalized_quantile_error
                    <= float(quantile_limit)
                )
                height_report = _preclose_height_report(
                    profile,
                    low_pose=low_pose,
                    verified_preclose_pose=verified_preclose_pose,
                    support_up=support_up,
                )
                record = {
                    "index": settle_index + 1,
                    "requested_down_m": requested_down_m,
                    "motion": settle_motion,
                    "inherited_commanded_tip_height_bias_m": (
                        inherited_tip_bias
                    ),
                    "final_commanded_tip_height_bias_m": (
                        height_settle_commanded_tip_bias
                    ),
                    "post_settle_level_corrections": (
                        post_settle_level_corrections
                    ),
                    "fresh_observation": observation.to_dict(),
                    "fresh_alignment_allowed": allowed,
                    "fresh_level": measured_level.to_dict(),
                    "fresh_height": height_report,
                }
                height_settle_records.append(record)
                result["preclose_height_settle"] = {
                    "corrections": height_settle_records
                }
                persist()
                if height_report["accepted"]:
                    break
            except BaseException as height_settle_error:
                height_settle_records.append(
                    {
                        "index": settle_index + 1,
                        "requested_down_m": requested_down_m,
                        "inherited_commanded_tip_height_bias_m": (
                            inherited_tip_bias
                        ),
                        "post_settle_level_corrections": (
                            post_settle_level_corrections
                        ),
                        "error": (
                            f"{type(height_settle_error).__name__}: "
                            f"{height_settle_error}"
                        ),
                    }
                )
                result["preclose_height_settle"] = {
                    "corrections": height_settle_records
                }
                persist()
                break
    if not height_report["accepted"]:
        result["preclose"] = {
            "timestamp": preclose_ts,
            "observation": observation.to_dict(),
            "level": measured_level.to_dict(),
            "height": height_report,
            "allowed_to_close": False,
            "rejection_reason": "gripper_not_low_enough",
        }
        corrected_q = target_q
        if not allowed:
            try:
                if (
                    "last_xy_m" not in preclose_servo_state
                    and "jacobian" in servo_state
                ):
                    preclose_servo_state["jacobian"] = np.asarray(
                        servo_state["jacobian"], dtype=float
                    ).tolist()
                corrected_q, correction = _metric_replan(
                    profile,
                    observation,
                    target_q,
                    fk=fk,
                    servo_state=preclose_servo_state,
                    measured_current_xy_m=low_pose[4:6],
                    fixed_orientation_q=low_q,
                    level_yaw_free=False,
                )
                corrected_q, correction = _compensate_hover_tracking_bias(
                    profile,
                    fk,
                    corrected_q,
                    correction,
                    hover=result["hover"],
                    hover_plan=result["hover_plan"],
                )
                result["visual_replan"] = correction
                result["preclose_servo_state"] = dict(
                    preclose_servo_state
                )
            except BaseException as replan_error:
                result["preclose_height_failure_replan_error"] = (
                    f"{type(replan_error).__name__}: {replan_error}"
                )
        persist()
        result["recovery"] = _recover_vertical_then_open(profile, rpc, fk)
        result["stages"]["retreat"] = _retreat_open(profile, rpc, fk)
        persist()
        return False, result, corrected_q

    # The operator-established contact policy is intentionally split into two
    # semantic primitives: first complete the normal descent/height settle,
    # then issue exactly one additional 2 mm seating command.  Keeping this
    # outside the correction loop prevents retries from accumulating multiple
    # 2 mm pushes.  A fresh image and measured level check are mandatory after
    # the single-shot command; the command itself never authorizes closure.
    final_seating_distance_m = float(
        profile["execution"].get("final_seating_extra_down_m", 0.0)
    )
    if final_seating_distance_m > 0.0:
        if final_seating_distance_m > 0.002 + 1e-12:
            raise ValueError("final seating command may not exceed 2 mm")
        seating_started_at = time.time()
        try:
            final_seating_motion = _execute_preclose_vertical_height_settle(
                profile,
                rpc,
                fk,
                requested_down_m=final_seating_distance_m,
                support_up=support_up,
                settle_index=maximum_height_settles + 1,
            )
            low_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
            target_q = low_q.copy()
            low_pose = np.asarray(
                rpc.get_right_ee_pose().parameters(), dtype=float
            )
            measured_level = preclose_checkpoint.require(
                "after_single_shot_final_2mm_seating"
            )
            preclose_image, preclose_ts = camera.frame(
                fresh_after_s=seating_started_at
            )
            cv2.imwrite(
                str(output_dir / "preclose_after_final_seating.png"),
                preclose_image,
            )
            observation = observe_preclose_image(preclose_image)
            assessment = observation.grasp_window
            allowed = _preclose_visual_alignment_allowed(
                profile, observation
            ) and (
                quantile_limit is None
                or assessment.normalized_quantile_error
                <= float(quantile_limit)
            )
            result["final_seating"] = {
                "primitive": "seat-2mm",
                "single_shot": True,
                "requested_down_m": final_seating_distance_m,
                "motion": final_seating_motion,
                "fresh_observation": observation.to_dict(),
                "fresh_level": measured_level.to_dict(),
                "fresh_alignment_allowed": allowed,
                "closure_authorized": bool(allowed and measured_level.accepted),
            }
            persist()
        except BaseException as final_seating_error:
            result["final_seating"] = {
                "primitive": "seat-2mm",
                "single_shot": True,
                "requested_down_m": final_seating_distance_m,
                "closure_authorized": False,
                "error": (
                    f"{type(final_seating_error).__name__}: "
                    f"{final_seating_error}"
                ),
            }
            persist()
            result["recovery"] = _recover_vertical_then_open(
                profile, rpc, fk
            )
            result["stages"]["retreat"] = _retreat_open(
                profile, rpc, fk
            )
            persist()
            return False, result, target_q
    result["preclose"] = {
        "timestamp": preclose_ts,
        "observation": observation.to_dict(),
        "level": measured_level.to_dict(),
        "height": height_report,
        "allowed_to_close": allowed,
    }
    persist()
    action = machine.preclose_assessment(
        rim_between_fingers=allowed,
        level_accepted=measured_level.accepted,
    )
    if action.name != "close_once_ramped":
        if (
            "last_xy_m" not in preclose_servo_state
            and "jacobian" in servo_state
        ):
            preclose_servo_state["jacobian"] = np.asarray(
                servo_state["jacobian"], dtype=float
            ).tolist()
        corrected_q, correction = _metric_replan(
            profile,
            observation,
            target_q,
            fk=fk,
            servo_state=preclose_servo_state,
            measured_current_xy_m=low_pose[4:6],
            # Cartesian descent can converge to a different joint-space IK
            # branch even though its measured EE pose is correct.  Preserve
            # the already validated wrist-camera orientation and only correct
            # measured support-plane XY; otherwise the next attempt may yaw
            # the camera and make image feedback incomparable.
            fixed_orientation_q=low_q,
            # Keep the validated wrist-camera yaw so successive image errors
            # share one coordinate frame.  _metric_replan still falls back to
            # a level yaw-free solve if this exact orientation is unreachable.
            level_yaw_free=False,
        )
        corrected_q, correction = _compensate_hover_tracking_bias(
            profile,
            fk,
            corrected_q,
            correction,
            hover=result["hover"],
            hover_plan=result["hover_plan"],
        )
        result["preclose_servo_state"] = dict(preclose_servo_state)
        try:
            recovery = _recover_vertical_then_open(profile, rpc, fk)
        except BaseException as error:
            recovery = {
                "completed": False,
                "error": f"{type(error).__name__}: {error}",
                "fallback": "direct_open_joint_retreat",
            }
        retreat = _retreat_open(profile, rpc, fk)
        machine.recovery_lift_complete()
        result["recovery"] = recovery
        result["stages"]["retreat_after_preclose_rejection"] = retreat
        result["visual_replan"] = correction
        result["state_history"] = list(machine.history)
        persist()
        return False, result, corrected_q

    close = _fixed_pose_gripper_ramp(
        profile,
        rpc,
        fk,
        finish_ratio=0.0,
        duration_s=float(profile["trajectory"]["close_duration_s"]),
        stage="close_once_continuous",
    )
    result["stages"]["close"] = close
    persist()
    closure_calibration = ClosureCalibration(
        tuple(profile["closure"]["empty_reference_ratios"]),
        tuple(profile["closure"]["nonempty_reference_ratios"]),
    )
    closure_after_close = closure_calibration.classify(
        float(rpc.get_right_gripper_exact())
    )
    result["closure_after_close"] = closure_after_close
    holding_aperture = float(
        max(
            0.0,
            closure_after_close["measured_open_ratio"]
            - float(profile["closure"].get("holding_preload_ratio_delta", 0.03)),
        )
    )
    closure_before = closure_after_close
    result["closure_before_lift"] = closure_before
    persist()
    if not closure_before["nonempty"]:
        machine.closure_measured(
            ClosureEvidence(
                closure_before["measured_open_ratio"],
                tuple(profile["closure"]["nonempty_reference_ratios"]),
                tuple(profile["closure"]["empty_reference_ratios"])[0],
            )
        )
        result["recovery"] = _recover_vertical_then_open(profile, rpc, fk)
        result["stages"]["retreat"] = _retreat_open(profile, rpc, fk)
        result["state_history"] = list(machine.history)
        persist()
        return False, result, target_q
    # Advance the strict state machine using the same calibrated populations.
    machine.closure_measured(
        ClosureEvidence(
            closure_before["measured_open_ratio"],
            tuple(profile["closure"]["nonempty_reference_ratios"]),
            tuple(profile["closure"]["empty_reference_ratios"])[0],
        )
    )
    closed_at = time.time()
    closed_image, closed_ts = camera.frame(fresh_after_s=closed_at)
    cv2.imwrite(str(output_dir / "closed.png"), closed_image)
    closed_center = None
    closed_reacquisition = None
    closed_marker_error = None
    try:
        closed_center, closed_reacquisition = _nearest_marker(
            closed_image,
            observation.center_px,
            maximum_fraction=4.0
            * float(
                profile["perception"][
                    "maximum_follow_displacement_diagonal_fraction"
                ]
            ),
        )
    except RuntimeError as error:
        # The jaw commonly occludes the small cross after closure.  Persistent
        # calibrated obstruction after a vertical lift is still independent
        # mechanical evidence; marker loss must not strand the closed arm.
        closed_marker_error = str(error)
    lift = _staged_straight_lift(
        profile,
        rpc,
        fk,
        closure_calibration,
        holding_aperture=holding_aperture,
    )
    result["stages"]["verification_lift"] = lift
    holding_aperture = float(lift["holding_aperture"])
    result["grip_preload"] = lift.get("post_pickup_preload")
    persist()
    if lift["completed_full_distance"]:
        closure_after = _classify_preloaded_obstruction(
            float(rpc.get_right_gripper_exact()),
            commanded_open_ratio=holding_aperture,
            minimum_obstruction_gap_ratio=float(
                profile["closure"].get(
                    "minimum_preloaded_obstruction_gap_ratio", 0.015
                )
            ),
        )
    else:
        closure_after = dict(lift["closure_after_initial_lift"])
    lifted_at = time.time()
    lifted_image, lifted_ts = camera.frame(fresh_after_s=lifted_at)
    cv2.imwrite(str(output_dir / "lifted.png"), lifted_image)
    lifted_center = None
    lifted_reacquisition = None
    lifted_marker_error = None
    try:
        lifted_center, lifted_reacquisition = _nearest_marker(
            lifted_image,
            observation.center_px if closed_center is None else closed_center,
            maximum_fraction=4.0
            * float(
                profile["perception"][
                    "maximum_follow_displacement_diagonal_fraction"
                ]
            ),
        )
    except RuntimeError as error:
        lifted_marker_error = str(error)
    if closed_center is not None and lifted_center is not None:
        follow = target_follow_evidence(
            closed_center,
            lifted_center,
            lifted_image.shape[:2],
            maximum_displacement_diagonal_fraction=float(
                profile["perception"][
                    "maximum_follow_displacement_diagonal_fraction"
                ]
            ),
            closure_before=closure_before,
            closure_after=closure_after,
        )
    else:
        persistent_obstruction = bool(
            closure_before["nonempty"] and closure_after["nonempty"]
        )
        follow = {
            "accepted": persistent_obstruction,
            "method": "persistent_calibrated_obstruction_marker_occluded",
            "marker_follow_available": False,
            "failure_reasons": (
                []
                if persistent_obstruction
                else ["obstruction_disappeared_during_vertical_lift"]
            ),
        }
    result["follow"] = {
        **follow,
        "closed_timestamp": closed_ts,
        "lifted_timestamp": lifted_ts,
        "closed_center_px": (
            None if closed_center is None else closed_center.tolist()
        ),
        "lifted_center_px": (
            None if lifted_center is None else lifted_center.tolist()
        ),
        "closed_reacquisition_fraction": closed_reacquisition,
        "lifted_reacquisition_fraction": lifted_reacquisition,
        "closed_marker_error": closed_marker_error,
        "lifted_marker_error": lifted_marker_error,
    }
    result["closure_after_lift"] = closure_after
    persist()
    machine.lift_complete(still_nonempty=follow["accepted"])
    if follow["accepted"]:
        result["stages"]["place_open_retreat"] = _place_open_retreat(
            profile,
            rpc,
            fk,
            low_pose,
            holding_aperture=holding_aperture,
        )
        result["success"] = True
    elif closure_after["nonempty"]:
        result["stages"]["failed_follow_place"] = _place_open_retreat(
            profile,
            rpc,
            fk,
            low_pose,
            holding_aperture=holding_aperture,
        )
    else:
        result["stages"]["failed_follow_retreat"] = _retreat_open(profile, rpc, fk)
    result["state_history"] = list(machine.history)
    result["left_arm_commands"] = 0
    persist()
    return bool(result["success"]), result, target_q


def _close_rpc(rpc) -> None:
    rpc.socket.close(linger=0)
    rpc.context.term()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/configs/pasteur_codexless_thin_object_grasp.json",
    )
    parser.add_argument("--output-dir", default="data/runs/pasteur/codexless_thin_object_latest")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--skip-head-registration", action="store_true")
    parser.add_argument("--hover-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--controller-lock",
        default="/tmp/piper_robot_right_arm_controller.lock",
        help="exclusive physical-controller lease (used only with --execute)",
    )
    args = parser.parse_args(argv)
    profile = load_profile(args.config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_profile(profile)
    requested = int(
        args.cycles
        if args.cycles is not None
        else profile["repeat"]["required_consecutive_successes"]
    )
    maximum_attempts = int(
        args.max_attempts
        if args.max_attempts is not None
        else profile["repeat"]["maximum_attempts"]
    )
    report = {
        "schema": RUN_SCHEMA,
        "profile": str(Path(args.config).resolve()),
        "commands_sent": False,
        "required_consecutive_successes": requested,
        "maximum_attempts": maximum_attempts,
        "audit": audit,
        "camera_mapping": None,
        "head_registration": None,
        "attempts": [],
        "complete": False,
    }
    _atomic_json(output_dir / "run.json", report)
    if not args.execute:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    controller_lease = ControllerLease(
        args.controller_lock,
        owner={"entrypoint": __file__, "output_dir": str(output_dir)},
    ).acquire()
    atexit.register(controller_lease.release)

    mapping, live = configure_camera_map_by_udid(profile["camera_udids"])
    report["camera_mapping"] = {"mapping": mapping, "live_udids": live}
    target_q = _q(
        profile["trajectory"]["verified_preclose_q_physical_rad"], "preclose q"
    )
    head_relocated_target_q = target_q.copy()
    runtime_servo_seed = {}
    runtime_fast_replay = None
    pending_hover_seed = None
    pending_expected_tool_frame = None
    pending_target_observation = None
    pending_direct_cross_verified = False
    if not args.skip_head_registration:
        head, target_q = preflight_head_registration(profile, output_dir)
        # Preserve the independent head-depth geometric proposal. Runtime
        # camera progress may supersede it, but a stale/occluded wrist view
        # must be able to fall back here without restarting the process.
        head_relocated_target_q = target_q.copy()
        report["head_registration"] = head["registration"]
        report["target_localization"] = head
        runtime_q, runtime_report = _load_runtime_alignment(profile, head)
        report["runtime_alignment_seed"] = runtime_report
        runtime_servo_seed = _accepted_runtime_servo_seed(
            runtime_q, runtime_report
        )
        if runtime_q is not None:
            target_q = runtime_q
            runtime_fast_replay = _runtime_fast_replay_alignment(
                profile, runtime_report, head
            )
        elif (
            float(runtime_report.get("target_scene_displacement_m", 0.0))
            > float(
                profile["target_identity"].get(
                    "maximum_runtime_target_scene_displacement_m", 0.008
                )
            )
            and not bool(
                (head.get("rim_plan") or {}).get(
                    "head_scene_to_production_translation_applied", False
                )
            )
        ):
            report["abort_reason"] = (
                "target moved outside the locally validated alignment and the "
                "head-scene-to-right-arm relocation transform is not accepted; "
                "no arm command was sent"
            )
            report["requires_relocation_calibration"] = True
            _atomic_json(output_dir / "run.json", report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 2
        # A semantically verified low-pose checkpoint has its own current
        # head-identity audit and does not depend on accepting the older
        # runtime joint seed. Previously this indentation silently discarded
        # the best right-camera checkpoint whenever only the FK/head bridge
        # was stale.
        pending_alignment, pending_report = _load_pending_preclose_alignment(
            profile, head
        )
        report["pending_alignment_seed"] = pending_report
        if pending_alignment is not None:
            runtime_fast_replay = pending_alignment
            identity_anchor = pending_alignment.get(
                "source_hover_identity_anchor"
            )
            if identity_anchor is not None:
                runtime_servo_seed["hover_identity_anchor"] = dict(
                    identity_anchor
                )
        if runtime_fast_replay is not None:
            target_q = _q(
                runtime_fast_replay["aligned_low_q_physical_rad"],
                "runtime fast-replay target q",
            )
            if runtime_report.get("hover_q_physical_rad") is not None:
                pending_hover_seed = _q(
                    runtime_report["hover_q_physical_rad"],
                    "runtime camera-visible hover q",
                )
                pending_expected_tool_frame = dict(
                    runtime_report["tool_frame"]
                )
        pending_hover, pending_hover_report = _load_pending_hover_progress(
            profile, head
        )
        if str(
            profile["perception"].get(
                "hover_alignment_mode", "canonical_goal"
            )
        ) == "identity_only_before_fresh_preclose":
            # This policy never optimizes a high-view pixel goal, so a saved
            # high-view servo endpoint cannot be meaningful input.  Low-pose
            # correction has its own separately audited pending artifact.
            pending_hover = None
            pending_hover_report = {
                **pending_hover_report,
                "accepted": False,
                "reason": (
                    "pending high-view servo is disabled by the stage-separated "
                    "preclose policy"
                ),
            }
        report["pending_hover_seed"] = pending_hover_report
        if pending_hover is not None:
            pending_low_q = _q(
                pending_hover["low_q_physical_rad"],
                "pending hover-progress target q",
            )
            maximum_seed_mismatch = float(
                profile["perception"].get(
                    "maximum_pending_hover_low_q_mismatch_rad", 0.05
                )
            )
            seed_compatible = bool(
                runtime_fast_replay is None
                or float(np.max(np.abs(pending_low_q - target_q)))
                <= maximum_seed_mismatch
            )
            pending_hover_report["low_q_seed_compatible"] = seed_compatible
            pending_hover_report["maximum_low_q_mismatch_rad"] = (
                maximum_seed_mismatch
            )
            if seed_compatible:
                if runtime_fast_replay is None:
                    target_q = pending_low_q
                pending_hover_seed = _q(
                    pending_hover["hover_q_physical_rad"],
                    "pending hover-progress hover q",
                )
                pending_expected_tool_frame = dict(
                    pending_hover["tool_frame"]
                )
                pending_target_observation = dict(
                    pending_hover.get("target_observation") or {}
                )
                pending_direct_cross_verified = bool(
                    pending_hover.get("direct_cross_verified", False)
                )
                runtime_servo_seed.update(
                    pending_hover.get("servo_seed", {})
                )
        report["runtime_fast_replay"] = runtime_fast_replay
    _atomic_json(output_dir / "run.json", report)
    rpc = RPCClient("localhost", 8081, timeout_ms=10000)
    fk = ProductionRightFK(profile["production_model"])
    ledger = ConsecutiveSuccessLedger(requested)
    servo_state = runtime_servo_seed
    if bool(
        profile["perception"].get(
            "require_runtime_hover_axis_calibration", False
        )
    ):
        servo_state["enable_runtime_axis_calibration"] = True
    next_attempt_hover_alignment = runtime_fast_replay
    next_attempt_progress_hover_q = pending_hover_seed
    next_attempt_expected_tool_frame = pending_expected_tool_frame
    next_attempt_target_observation = pending_target_observation
    next_attempt_direct_cross_verified = pending_direct_cross_verified
    left_start = np.asarray(rpc.get_left_joint_positions(), dtype=float)
    report["measured_left_fixed_audit"] = _trajectory_contact_audit(
        profile, target_q, left_q_physical=left_start
    )
    if not report["measured_left_fixed_audit"]["accepted"]:
        _close_rpc(rpc)
        raise RuntimeError("measured fixed-left pose makes the right trajectory unsafe")
    _atomic_json(output_dir / "run.json", report)
    try:
        with LiveCamera("right") as camera:
            while ledger.attempts < maximum_attempts and ledger.consecutive < requested:
                attempt_dir = output_dir / f"attempt_{ledger.attempts + 1:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                started = time.time()
                cached_alignment = next_attempt_hover_alignment
                next_attempt_hover_alignment = None
                progress_hover_q = next_attempt_progress_hover_q
                next_attempt_progress_hover_q = None
                expected_tool_frame = next_attempt_expected_tool_frame
                next_attempt_expected_tool_frame = None
                target_observation = next_attempt_target_observation
                next_attempt_target_observation = None
                direct_cross_verified = next_attempt_direct_cross_verified
                next_attempt_direct_cross_verified = False
                stop_after_attempt = False
                try:
                    success, attempt, target_q = run_attempt(
                        profile,
                        rpc,
                        fk,
                        camera,
                        attempt_dir,
                        target_q,
                        servo_state=servo_state,
                        allow_descent=not args.hover_only,
                        cached_hover_alignment=(
                            cached_alignment
                        ),
                        initial_hover_q_physical_rad=progress_hover_q,
                        initial_expected_tool_frame=expected_tool_frame,
                        initial_target_observation=target_observation,
                        initial_direct_cross_verified=direct_cross_verified,
                    )
                    if not success:
                        next_attempt_hover_alignment = (
                            _preclose_correction_replay_alignment(
                                profile, attempt, target_q
                            )
                        )
                        if next_attempt_hover_alignment is not None:
                            attempt["next_attempt_hover_alignment"] = (
                                next_attempt_hover_alignment
                            )
                            if report.get("target_localization") is not None:
                                attempt["pending_alignment"] = (
                                    _save_pending_preclose_alignment(
                                        profile,
                                        report["target_localization"],
                                        next_attempt_hover_alignment,
                                        source_run=str(output_dir),
                                        source_attempt=attempt_dir.name,
                                    )
                                )
                        elif attempt.get("preclose_perception_error") is not None:
                            attempt["pending_alignment_invalidated"] = (
                                _invalidate_pending_preclose_alignment(
                                    profile,
                                    source_run=str(output_dir),
                                    reason=str(
                                        attempt["preclose_perception_error"]
                                    ),
                                )
                            )
                        hover_progress = attempt.get("next_attempt_seed")
                        if hover_progress is not None:
                            target_q = _q(
                                hover_progress["low_q_physical_rad"],
                                "next-attempt best hover low q",
                            )
                            next_attempt_progress_hover_q = _q(
                                hover_progress["hover_q_physical_rad"],
                                "next-attempt best hover q",
                            )
                            next_attempt_expected_tool_frame = dict(
                                hover_progress["tool_frame"]
                            )
                            next_attempt_target_observation = dict(
                                hover_progress.get("target_observation") or {}
                            )
                            next_attempt_direct_cross_verified = bool(
                                hover_progress.get("direct_cross_verified", False)
                            )
                            if report.get("target_localization") is not None:
                                attempt["pending_hover_progress"] = (
                                    _save_pending_hover_progress(
                                        profile,
                                        report["target_localization"],
                                        hover_progress,
                                        servo_state,
                                        source_run=str(output_dir),
                                        source_attempt=attempt_dir.name,
                                    )
                                )
                        stale_precise_camera_replay_reason = (
                            _stale_precise_camera_replay_reason(attempt)
                        )
                        if stale_precise_camera_replay_reason is not None:
                            reason = (
                                "fresh right-camera image did not contain the tapped "
                                "marker near the saved camera-visible grasp window: "
                                f"{stale_precise_camera_replay_reason}"
                            )
                            attempt["pending_hover_progress_invalidated"] = (
                                _invalidate_pending_hover_progress(
                                    profile,
                                    source_run=str(output_dir),
                                    reason=reason,
                                )
                            )
                            attempt["retry_from_head_geometric_seed"] = True
                            target_q = head_relocated_target_q.copy()
                            next_attempt_hover_alignment = None
                            next_attempt_progress_hover_q = None
                            next_attempt_expected_tool_frame = None
                            next_attempt_target_observation = None
                            next_attempt_direct_cross_verified = False
                            servo_state.clear()
                            if bool(
                                profile["perception"].get(
                                    "require_runtime_hover_axis_calibration", False
                                )
                            ):
                                servo_state["enable_runtime_axis_calibration"] = True
                        branch_escape = _BRANCH_ESCAPE_IDENTITY_ERROR in str(
                            attempt.get("hover_perception_or_servo_error", "")
                        )
                        if branch_escape:
                            if _identity_consistent_hover_progress(hover_progress):
                                attempt["retry_from_identity_checkpoint"] = True
                            else:
                                reason = (
                                    "wrist-camera branch changed without an "
                                    "identity-consistent best observation"
                                )
                                attempt["pending_hover_progress_invalidated"] = (
                                    _invalidate_pending_hover_progress(
                                        profile,
                                        source_run=str(output_dir),
                                        reason=reason,
                                    )
                                )
                                attempt["retry_from_head_geometric_seed"] = True
                                target_q = head_relocated_target_q.copy()
                                next_attempt_hover_alignment = None
                                next_attempt_progress_hover_q = None
                                next_attempt_expected_tool_frame = None
                                next_attempt_target_observation = None
                                next_attempt_direct_cross_verified = False
                                servo_state.clear()
                                if bool(
                                    profile["perception"].get(
                                        "require_runtime_hover_axis_calibration", False
                                    )
                                ):
                                    servo_state[
                                        "enable_runtime_axis_calibration"
                                    ] = True
                    if (
                        report.get("target_localization") is not None
                        and attempt.get("hover", {}).get("aligned_for_descent") is True
                        and (
                            success
                            or not attempt.get("hover", {}).get("runtime_replay")
                        )
                    ):
                        attempt["runtime_alignment"] = _save_runtime_alignment(
                            profile,
                            report["target_localization"],
                            attempt,
                            target_q,
                            source_run=str(output_dir),
                            servo_state=servo_state,
                        )
                        if success:
                            attempt["pending_alignment_consumed"] = (
                                _deactivate_pending_preclose_alignment(
                                    profile, source_run=str(output_dir)
                                )
                            )
                            attempt["pending_hover_progress_consumed"] = (
                                _deactivate_pending_hover_progress(
                                    profile, source_run=str(output_dir)
                                )
                            )
                except BaseException as error:
                    success = False
                    partial = attempt_dir / "attempt.json"
                    attempt = _load(partial) if partial.is_file() else {"success": False}
                    attempt["error"] = f"{type(error).__name__}: {error}"
                    attempt["left_arm_commands"] = 0
                    correction_q = (
                        (attempt.get("visual_replan") or {}).get(
                            "corrected_q_physical_rad"
                        )
                    )
                    if correction_q is not None:
                        target_q = _q(
                            correction_q,
                            "partial-attempt corrected resume q",
                        )
                        attempt["exception_resume_target_q_physical_rad"] = (
                            target_q.tolist()
                        )
                    # The streamer latches the measured pose on errors.  Do not
                    # invent a recovery when the exact failure state is unknown.
                    attempt.setdefault("recovery", "measured_pose_hold")
                    stop_after_attempt = isinstance(error, KeyboardInterrupt)
                attempt["started_at_s"] = started
                attempt["finished_at_s"] = time.time()
                attempt["ledger"] = ledger.record(success)
                report["attempts"].append(attempt)
                report["commands_sent"] = True
                report["complete"] = ledger.consecutive >= requested
                report["consecutive_successes"] = ledger.consecutive
                report["visual_servo_state"] = servo_state
                report["left_arm_max_abs_delta_rad"] = float(
                    np.max(
                        np.abs(
                            np.asarray(rpc.get_left_joint_positions(), dtype=float)
                            - left_start
                        )
                    )
                )
                _atomic_json(output_dir / "run.json", report)
                if stop_after_attempt:
                    break
                if (
                    not success
                    and "visual_replan" not in attempt
                    and not attempt.get("retry_from_head_geometric_seed", False)
                ):
                    break
    finally:
        _close_rpc(rpc)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Observe and execute a demo-relative incubator-door opening.

Hardware execution is deliberately staged.  ``retreat-orient`` cannot close
the gripper or pull the door; it only backs away along the measured tool
approach axis and adopts the orientation learned from verified successes.
Later contact stages consume a fresh image-derived live contact pose rather
than the historical absolute contact position.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.rpc import RPCClient
from robot.arm.ik_solver import SingleArmIK
from rollout.contact_visibility import (
    aperture_is_stably_nonempty,
    proof_pull_holds,
)
from rollout.incubator_door_demo import (
    quaternion_distance_rad,
    retarget_relative_pose,
    retarget_relative_trajectory,
)
from rollout.incubator_door_close import (
    load_open_jaw_close_trajectory,
    nearest_pose_index,
    register_close_trajectory,
    reverse_opening_from_live_pose,
)
from rollout.incubator_door_visual import extract_feature, predict_local_delta
from rollout.appliance_frame import matrix4, matrix_to_pose7
from rollout.teleop_trajectory_stream import (
    CONTROL_HZ,
    JointTrajectorySample,
    ProductionRightFK,
    TeleopTrajectoryStreamer,
    TrajectoryStreamError,
    sample_joint_knots,
)
from src.run_demo_relative_servo import LiveSource


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _load_appliance_registration(path: Path | None) -> np.ndarray:
    """Load a bounded, accepted reference-lab to live-lab registration."""

    if path is None:
        return np.eye(4)
    payload = _load(path)
    if not bool(payload.get("accepted", False)):
        raise RuntimeError("appliance registration is not accepted")
    transform = payload.get("T_registration")
    if transform is None:
        raise RuntimeError("appliance registration lacks T_registration")
    return matrix4(transform, "T_registration")


def _minimum_jerk(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def _nlerp(first, second, fraction: float) -> np.ndarray:
    first = np.asarray(first, dtype=float).copy()
    second = np.asarray(second, dtype=float).copy()
    if first @ second < 0.0:
        second *= -1.0
    result = first + float(fraction) * (second - first)
    return result / np.linalg.norm(result)


def world_yaw_pose(pose_wxyz_xyz, yaw_deg: float) -> np.ndarray:
    """Rotate a tool pose about world Z without moving its origin.

    This is intentionally a world-frame correction: an incubator yaw change
    should rotate both the grasp orientation and the subsequent relative pull
    while preserving the visually aligned preclose position.
    """

    start = mink.SE3(np.asarray(pose_wxyz_xyz, dtype=float))
    yaw = mink.SO3.from_z_radians(math.radians(float(yaw_deg)))
    target = mink.SE3.from_rotation_and_translation(
        yaw @ start.rotation(), start.translation()
    )
    return np.asarray(target.parameters(), dtype=float)


def _streamer(profile: dict, rpc, fk) -> TeleopTrajectoryStreamer:
    torque = _load(profile["torque_config"])
    execution = profile["execution"]
    return TeleopTrajectoryStreamer(
        rpc,
        fk,
        torque_limit_nm=torque["thresholds"]["right"],
        consecutive_torque_samples=int(torque.get("consecutive_samples", 5)),
        enforce_torque_stop=bool(execution["torque_stop_enforced"]),
        maximum_start_fk_position_error_m=0.001,
        maximum_start_fk_rotation_error_rad=0.01,
        maximum_tracking_position_error_m=float(
            execution["maximum_tracking_position_error_m"]
        ),
        maximum_tracking_rotation_error_rad=float(
            execution["maximum_tracking_rotation_error_rad"]
        ),
        tracking_check_interval=int(execution["tracking_check_interval"]),
    )


def _cartesian_segment(
    profile: dict,
    rpc,
    fk,
    *,
    target_pose: np.ndarray,
    duration_s: float,
    aperture: float,
    stage: str,
    final_dwell_s: float = 0.0,
) -> dict:
    start_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    start_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    count = max(1, int(math.ceil(float(duration_s) * CONTROL_HZ)))
    samples = []
    poses = []
    for index in range(1, count + 1):
        fraction = index / count
        blend = _minimum_jerk(fraction)
        pose = np.r_[
            _nlerp(start_pose[:4], target_pose[:4], blend),
            start_pose[4:] + blend * (target_pose[4:] - start_pose[4:]),
        ]
        samples.append(
            JointTrajectorySample(
                t_s=fraction * float(duration_s),
                stage=stage,
                right_q_physical_rad=start_q.copy(),
                right_gripper_open_ratio=float(aperture),
            )
        )
        poses.append(mink.SE3(pose))
    dwell_count = max(0, int(math.ceil(float(final_dwell_s) * CONTROL_HZ)))
    for index in range(1, dwell_count + 1):
        samples.append(
            JointTrajectorySample(
                t_s=float(duration_s) + index / CONTROL_HZ,
                stage=f"{stage}_final_dwell",
                right_q_physical_rad=start_q.copy(),
                right_gripper_open_ratio=float(aperture),
            )
        )
        poses.append(mink.SE3(np.asarray(target_pose, dtype=float)))
    cursor = {"index": 0}

    def transform(_stage, _planned):
        result = poses[cursor["index"]]
        cursor["index"] += 1
        return result

    return _streamer(profile, rpc, fk).execute(samples, pose_transformer=transform)


def _settle_cartesian_target(
    profile: dict,
    rpc,
    fk,
    *,
    target_pose: np.ndarray,
    duration_s: float,
    aperture: float,
    stage: str,
) -> dict:
    """Resend an absolute Cartesian target until measured FK has settled."""

    attempts = []
    maximum_attempts = int(profile.get("cartesian_settle_maximum_attempts", 3))
    position_tolerance = float(
        profile.get("cartesian_settle_position_tolerance_m", 0.0015)
    )
    rotation_tolerance = math.radians(
        float(profile.get("cartesian_settle_rotation_tolerance_deg", 0.75))
    )
    for attempt in range(maximum_attempts):
        motion = _cartesian_segment(
            profile,
            rpc,
            fk,
            target_pose=target_pose,
            duration_s=(
                float(duration_s)
                if attempt == 0
                else float(profile.get("cartesian_settle_retry_duration_s", 1.2))
            ),
            aperture=aperture,
            stage=f"{stage}_settle_{attempt + 1}",
            final_dwell_s=float(profile.get("cartesian_settle_final_dwell_s", 1.0)),
        )
        measured = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
        position_error = float(np.linalg.norm(measured[4:] - target_pose[4:]))
        rotation_error = quaternion_distance_rad(measured[:4], target_pose[:4])
        attempts.append(
            {
                "attempt": attempt + 1,
                "motion": motion,
                "measured_wxyz_xyz": measured.tolist(),
                "position_error_m": position_error,
                "rotation_error_deg": math.degrees(rotation_error),
            }
        )
        if (
            position_error <= position_tolerance
            and rotation_error <= rotation_tolerance
        ):
            break
    accepted = bool(
        attempts[-1]["position_error_m"] <= position_tolerance
        and math.radians(attempts[-1]["rotation_error_deg"]) <= rotation_tolerance
    )
    maximum_position_error = float(
        profile.get("cartesian_motion_eligible_position_error_m", 0.008)
    )
    maximum_rotation_error = float(
        profile.get("cartesian_motion_eligible_rotation_error_deg", 4.0)
    )
    motion_eligible = bool(
        attempts[-1]["position_error_m"] <= maximum_position_error
        and attempts[-1]["rotation_error_deg"] <= maximum_rotation_error
    )
    if not motion_eligible:
        raise RuntimeError(
            "absolute Cartesian target did not settle: "
            f"position={attempts[-1]['position_error_m']:.4f}m, "
            f"rotation={attempts[-1]['rotation_error_deg']:.2f}deg"
        )
    return {
        "accepted": accepted,
        "motion_eligible": True,
        "attempts": attempts,
    }


def _capture(profile: dict, rpc, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = LiveSource(rpc, output_dir)
    try:
        source.start(timeout_s=20.0)
        time.sleep(0.5)
        observation = source.observe()
        for name, image in observation.images.items():
            cv2.imwrite(str(output_dir / f"{name}.png"), image)
        if "head" in observation.depths:
            np.save(output_dir / "head_depth.npy", observation.depths["head"])
        result = {
            "timestamp_s": float(observation.timestamp),
            "right_ee_wxyz_xyz": observation.ee_pose.tolist(),
            "right_q_rad": observation.joint_positions.tolist(),
            "right_gripper": float(observation.gripper_ratio),
            "image_paths": {
                name: str((output_dir / f"{name}.png").resolve())
                for name in observation.images
            },
        }
        (output_dir / "observation.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result
    finally:
        source.stop()


def _fixed_pose_gripper_ramp(
    profile: dict,
    rpc,
    fk,
    *,
    finish_ratio: float,
    duration_s: float,
    stage: str,
) -> dict:
    current_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    current_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    start_ratio = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
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


def _sample_aperture(rpc, *, duration_s: float, hz: float) -> list[float]:
    count = max(1, int(math.ceil(float(duration_s) * float(hz))))
    started = time.monotonic()
    result = []
    for index in range(count):
        result.append(
            float(
                np.asarray(rpc.get_right_gripper_exact(), dtype=float)
                .reshape(-1)[0]
            )
        )
        deadline = started + (index + 1) / float(hz)
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
    return result


def _stream_retargeted_segment(
    profile: dict,
    rpc,
    fk,
    segment: list[dict],
    *,
    stage: str,
    time_scale: float = 1.0,
    checkpoint_frames: int | None = None,
    minimum_aperture: float | None = None,
) -> dict:
    if not math.isfinite(time_scale) or time_scale < 1.0:
        raise ValueError("time_scale must be finite and at least 1")
    if checkpoint_frames is not None and checkpoint_frames <= 0:
        raise ValueError("checkpoint_frames must be positive")
    current_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    samples = [
        JointTrajectorySample(
            t_s=float(item["t_s"]) * float(time_scale),
            stage=(
                f"{stage}_checkpoint_{index // checkpoint_frames:02d}"
                if checkpoint_frames is not None
                else stage
            ),
            right_q_physical_rad=current_q.copy(),
            right_gripper_open_ratio=float(item["gripper"]),
        )
        for index, item in enumerate(segment)
    ]
    poses = [mink.SE3(np.asarray(item["pose_wxyz_xyz"], dtype=float)) for item in segment]
    cursor = {"index": 0}

    def transform(_stage, _pose):
        result = poses[cursor["index"]]
        cursor["index"] += 1
        return result

    aperture_checks: list[dict] = []

    def gate(gated_stage: str) -> None:
        if minimum_aperture is None:
            return
        aperture = float(
            np.asarray(rpc.get_right_gripper_exact(), dtype=float)
            .reshape(-1)[0]
        )
        accepted = aperture >= float(minimum_aperture)
        aperture_checks.append(
            {
                "stage": gated_stage,
                "aperture": aperture,
                "minimum_aperture": float(minimum_aperture),
                "accepted": accepted,
            }
        )
        if not accepted:
            raise TrajectoryStreamError(
                "incubator grasp lost before checkpoint: "
                f"aperture={aperture:.4f} < {minimum_aperture:.4f}"
            )

    result = _streamer(profile, rpc, fk).execute(
        samples,
        pose_transformer=transform,
        stage_gate=gate if checkpoint_frames is not None else None,
    )
    if minimum_aperture is not None:
        final_aperture = float(
            np.asarray(rpc.get_right_gripper_exact(), dtype=float)
            .reshape(-1)[0]
        )
        aperture_checks.append(
            {
                "stage": f"{stage}_final",
                "aperture": final_aperture,
                "minimum_aperture": float(minimum_aperture),
                "accepted": final_aperture >= float(minimum_aperture),
            }
        )
    result["time_scale"] = float(time_scale)
    result["aperture_checkpoints"] = aperture_checks
    return result


def close_and_verify(
    profile: dict,
    rpc,
    output_dir: Path,
    contact_reference_path: Path | None = None,
) -> dict:
    settings = profile["contact"]
    compiled = _load(profile["compiled_demo"])
    fk = ProductionRightFK(profile["production_model"])
    before = _capture(profile, rpc, output_dir / "before")
    open_ratio = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if open_ratio < 0.95:
        raise RuntimeError("close-verify requires a fully open right gripper")
    contact_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    orientation_reference = "compiled_demo_contact"
    canonical_quaternion = np.asarray(
        compiled["medoid"]["contact_pose_wxyz_xyz"][:4], dtype=float
    )
    if contact_reference_path is not None:
        reference = _load(contact_reference_path)
        measured_reference = reference.get("after", {}).get(
            "right_ee_wxyz_xyz"
        )
        if measured_reference is None:
            measured_reference = reference.get("target_wxyz_xyz")
        if measured_reference is None:
            raise RuntimeError(
                "contact reference has no measured or target contact pose"
            )
        canonical_quaternion = np.asarray(measured_reference[:4], dtype=float)
        orientation_reference = str(contact_reference_path.resolve())
    orientation_error_deg = math.degrees(
        quaternion_distance_rad(contact_pose[:4], canonical_quaternion)
    )
    if orientation_error_deg > float(settings["maximum_orientation_error_deg"]):
        raise RuntimeError(
            "close denied: wrist orientation drifted from the fresh "
            f"contact reference by {orientation_error_deg:.2f}deg"
        )
    motion = _fixed_pose_gripper_ramp(
        profile,
        rpc,
        fk,
        finish_ratio=0.0,
        duration_s=float(settings["close_duration_s"]),
        stage="incubator_close_once",
    )
    samples = _sample_aperture(
        rpc,
        duration_s=float(settings["close_observation_s"]),
        hz=float(settings["aperture_sample_hz"]),
    )
    nonempty = aperture_is_stably_nonempty(
        samples,
        empty_upper_bound=float(settings["empty_aperture_upper_bound"]),
        minimum_samples=int(settings["aperture_minimum_samples"]),
        maximum_range=float(settings["maximum_settled_aperture_range"]),
    )
    after = _capture(profile, rpc, output_dir / "after")
    state = {
        "schema": "piper_robot.incubator_door_contact/v1",
        "contact_pose_wxyz_xyz": contact_pose.tolist(),
        "closed_aperture": float(np.median(samples[-int(settings["aperture_minimum_samples"]):])),
        "aperture_samples": samples,
        "stable_nonempty": bool(nonempty),
        "orientation_reference": orientation_reference,
        "orientation_error_deg": orientation_error_deg,
        "before": before,
        "after": after,
    }
    state_path = output_dir / "contact_state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    result = {
        "commands_sent": True,
        "stage": "close-verify",
        "motion": motion,
        "contact_state": state,
        "contact_state_path": str(state_path.resolve()),
    }
    (output_dir / "close_verify.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def proof_pull(
    profile: dict,
    rpc,
    output_dir: Path,
    contact_state_path: Path,
) -> dict:
    settings = profile["contact"]
    compiled = _load(profile["compiled_demo"])
    state = _load(contact_state_path)
    if not state.get("stable_nonempty"):
        raise RuntimeError("proof pull denied: close was not stably nonempty")
    before = _capture(profile, rpc, output_dir / "before")
    live_aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if not proof_pull_holds(
        closed_aperture=float(state["closed_aperture"]),
        live_aperture=live_aperture,
        empty_upper_bound=float(settings["empty_aperture_upper_bound"]),
        minimum_retention_fraction=float(settings["minimum_proof_retention_fraction"]),
    ):
        raise RuntimeError("proof pull denied: closure was not retained before motion")
    proof_frame = int(compiled["medoid"]["proof_frame"])
    close_frame = int(compiled["medoid"]["close_frame"])
    segment = retarget_relative_trajectory(
        state["contact_pose_wxyz_xyz"],
        compiled["relative_pull_trajectory"],
        first_frame=close_frame + 1,
        last_frame=proof_frame,
    )
    motion = _stream_retargeted_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        segment,
        stage="incubator_proof_pull_5mm",
    )
    samples = _sample_aperture(
        rpc,
        duration_s=float(settings["minimum_close_settle_s"]),
        hz=float(settings["aperture_sample_hz"]),
    )
    retained = all(
        proof_pull_holds(
            closed_aperture=float(state["closed_aperture"]),
            live_aperture=value,
            empty_upper_bound=float(settings["empty_aperture_upper_bound"]),
            minimum_retention_fraction=float(settings["minimum_proof_retention_fraction"]),
        )
        for value in samples[-int(settings["aperture_minimum_samples"]):]
    )
    after = _capture(profile, rpc, output_dir / "after")
    proof_state = {
        **state,
        "proof_frame": proof_frame,
        "proof_aperture_samples": samples,
        "proof_retained": bool(retained),
        "proof_before": before,
        "proof_after": after,
    }
    proof_path = output_dir / "proof_state.json"
    proof_path.write_text(json.dumps(proof_state, indent=2) + "\n")
    result = {
        "commands_sent": True,
        "stage": "proof-pull",
        "motion": motion,
        "proof_state": proof_state,
        "proof_state_path": str(proof_path.resolve()),
    }
    (output_dir / "proof_pull.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def reverify_proof_hold(
    profile: dict,
    rpc,
    output_dir: Path,
    proof_state_path: Path,
) -> dict:
    """Re-evaluate a stationary proof grasp without replaying the pull."""

    settings = profile["contact"]
    state = _load(proof_state_path)
    observation = _capture(profile, rpc, output_dir / "observation")
    samples = _sample_aperture(
        rpc,
        duration_s=float(settings["minimum_close_settle_s"]),
        hz=float(settings["aperture_sample_hz"]),
    )
    tail = samples[-int(settings["aperture_minimum_samples"]):]
    stable_nonempty = aperture_is_stably_nonempty(
        tail,
        empty_upper_bound=float(settings["empty_aperture_upper_bound"]),
        minimum_samples=int(settings["aperture_minimum_samples"]),
        maximum_range=float(settings["maximum_settled_aperture_range"]),
    )
    retained = bool(
        stable_nonempty
        and all(
            proof_pull_holds(
                closed_aperture=float(state["closed_aperture"]),
                live_aperture=value,
                empty_upper_bound=float(settings["empty_aperture_upper_bound"]),
                minimum_retention_fraction=float(
                    settings["minimum_proof_retention_fraction"]
                ),
            )
            for value in tail
        )
    )
    updated = {
        **state,
        "proof_aperture_samples": samples,
        "proof_retained": retained,
        "proof_reverification": {
            "source": str(proof_state_path.resolve()),
            "stationary": True,
            "stable_nonempty": bool(stable_nonempty),
            "minimum_retention_fraction": float(
                settings["minimum_proof_retention_fraction"]
            ),
            "observation": observation,
        },
    }
    path = output_dir / "proof_state.json"
    path.write_text(json.dumps(updated, indent=2) + "\n")
    result = {
        "commands_sent": False,
        "stage": "reverify-proof",
        "proof_retained": retained,
        "proof_state_path": str(path.resolve()),
        "proof_state": updated,
    }
    (output_dir / "reverify_proof.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def open_door(
    profile: dict,
    rpc,
    output_dir: Path,
    proof_state_path: Path,
) -> dict:
    compiled = _load(profile["compiled_demo"])
    state = _load(proof_state_path)
    if not state.get("proof_retained"):
        raise RuntimeError("door pull denied: 5mm proof pull was not retained")
    before = _capture(profile, rpc, output_dir / "before")
    proof_frame = int(state["proof_frame"])
    release_frame = int(compiled["medoid"]["release_frame"])
    segment = retarget_relative_trajectory(
        state["contact_pose_wxyz_xyz"],
        compiled["relative_pull_trajectory"],
        first_frame=proof_frame + 1,
        last_frame=release_frame,
    )
    proof_aperture = float(
        np.median(
            np.asarray(state["proof_aperture_samples"], dtype=float)
        )
    )
    minimum_aperture = max(
        float(profile["contact"]["empty_aperture_upper_bound"]),
        proof_aperture
        * float(profile["contact"]["full_pull_minimum_retention_fraction"]),
    )
    motion = _stream_retargeted_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        segment,
        stage="incubator_demo_relative_open",
        time_scale=float(profile["contact"]["full_pull_time_scale"]),
        checkpoint_frames=int(
            profile["contact"]["full_pull_checkpoint_frames"]
        ),
        minimum_aperture=minimum_aperture,
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "open-door",
        "motion": motion,
        "before": before,
        "after": after,
        "gripper_left_closed": True,
        "minimum_retained_aperture": minimum_aperture,
        "grip_retained_at_end": bool(
            motion["final_right_gripper_open_ratio"] >= minimum_aperture
        ),
    }
    (output_dir / "open_door.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def close_door_by_reversing_opening(
    profile: dict,
    rpc,
    output_dir: Path,
) -> dict:
    """Push the open door closed with open jaws on the executed IK branch."""

    settings = profile["close_door"]
    compiled = _load(profile["compiled_demo"])
    proof_state_path = Path(settings["reference_proof_state"])
    state = _load(proof_state_path)
    before = _capture(profile, rpc, output_dir / "before")
    live_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    live_aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    required_aperture = float(settings["open_gripper_ratio"])
    if live_aperture < required_aperture - 0.05:
        raise RuntimeError("door close requires the right gripper fully open")
    opening = retarget_relative_trajectory(
        state["contact_pose_wxyz_xyz"],
        compiled["relative_pull_trajectory"],
        first_frame=int(state["proof_frame"]) + 1,
        last_frame=int(compiled["medoid"]["release_frame"]),
    )
    nearest = nearest_pose_index(
        live_pose,
        opening,
        rotation_weight_m_per_rad=float(
            settings["rotation_score_weight_m_per_rad"]
        ),
    )
    if nearest["position_error_m"] > float(
        settings["maximum_anchor_position_error_m"]
    ) or nearest["rotation_error_deg"] > float(
        settings["maximum_anchor_rotation_error_deg"]
    ):
        raise RuntimeError(
            "live wrist is not close to the actually executed opening path: "
            f"position={nearest['position_error_m']:.4f}m, "
            f"rotation={nearest['rotation_error_deg']:.2f}deg"
        )
    exact_anchor_pose = opening[int(nearest["index"])]["pose_wxyz_xyz"]
    approach = _settle_cartesian_target(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=np.asarray(exact_anchor_pose, dtype=float),
        duration_s=float(settings["anchor_approach_duration_s"]),
        aperture=required_aperture,
        stage="incubator_close_contact_approach",
    )
    measured_anchor = np.asarray(
        rpc.get_right_ee_pose().parameters(), dtype=float
    )
    anchor_position_error = float(
        np.linalg.norm(measured_anchor[4:] - np.asarray(exact_anchor_pose)[4:])
    )
    anchor_rotation_error = math.degrees(
        quaternion_distance_rad(
            measured_anchor[:4], np.asarray(exact_anchor_pose)[:4]
        )
    )
    closing = reverse_opening_from_live_pose(
        exact_anchor_pose,
        opening,
        nearest_index=int(nearest["index"]),
        control_hz=CONTROL_HZ,
        aperture=required_aperture,
    )
    motion = _stream_retargeted_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        closing,
        stage="incubator_reverse_successful_opening_to_close",
        time_scale=float(settings["time_scale"]),
        checkpoint_frames=int(settings["checkpoint_frames"]),
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "close-door",
        "strategy": (
            "open jaws from Peacock close demo; geometry from reversed "
            "current successful opening"
        ),
        "proof_state_source": str(proof_state_path.resolve()),
        "peacock_strategy_reference": str(
            Path(settings["peacock_strategy_reference"]).resolve()
        ),
        "nearest_opening_sample": nearest,
        "contact_approach": approach,
        "measured_anchor_error_m": anchor_position_error,
        "measured_anchor_rotation_error_deg": anchor_rotation_error,
        "closing_sample_count": len(closing),
        "motion": motion,
        "before": before,
        "after": after,
        "extra_push_sent": False,
    }
    (output_dir / "close_door.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def restore_after_missed_close(
    profile: dict,
    rpc,
    output_dir: Path,
    close_state_path: Path,
) -> dict:
    """Reverse a non-contact close attempt back to its observed open start."""

    previous = _load(close_state_path)
    if previous.get("extra_push_sent") is not False:
        raise RuntimeError("restore requires a close run with no extra push")
    settings = profile["close_door"]
    compiled = _load(profile["compiled_demo"])
    state = _load(settings["reference_proof_state"])
    opening = retarget_relative_trajectory(
        state["contact_pose_wxyz_xyz"],
        compiled["relative_pull_trajectory"],
        first_frame=int(state["proof_frame"]) + 1,
        last_frame=int(compiled["medoid"]["release_frame"]),
    )
    index = int(previous["nearest_opening_sample"]["index"])
    sent_close = reverse_opening_from_live_pose(
        previous["before"]["right_ee_wxyz_xyz"],
        opening,
        nearest_index=index,
        control_hz=CONTROL_HZ,
        aperture=float(settings["open_gripper_ratio"]),
    )
    live = mink.SE3(
        np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    )
    endpoint = mink.SE3(np.asarray(sent_close[-1]["pose_wxyz_xyz"]))
    correction = live @ endpoint.inverse()
    restore = []
    for output_index, sample in enumerate(reversed(sent_close), start=1):
        source = mink.SE3(np.asarray(sample["pose_wxyz_xyz"], dtype=float))
        restore.append(
            {
                **sample,
                "t_s": output_index / CONTROL_HZ,
                "pose_wxyz_xyz": np.asarray(
                    (correction @ source).parameters(), dtype=float
                ).tolist(),
                "gripper": float(settings["open_gripper_ratio"]),
            }
        )
    before = _capture(profile, rpc, output_dir / "before")
    motion = _stream_retargeted_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        restore,
        stage="incubator_restore_after_noncontact_close",
        time_scale=float(settings["time_scale"]),
        checkpoint_frames=int(settings["checkpoint_frames"]),
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "restore-open-start",
        "source": str(close_state_path.resolve()),
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "restore_open_start.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def close_door_from_peacock_demo(
    profile: dict,
    rpc,
    output_dir: Path,
    appliance_registration_path: Path | None = None,
) -> dict:
    """Use the dedicated low push point from a verified close demonstration."""

    settings = profile["close_door"]
    closing = load_open_jaw_close_trajectory(
        settings["peacock_strategy_reference"]
    )
    registration = _load_appliance_registration(appliance_registration_path)
    closing = register_close_trajectory(
        closing, matrix_to_pose7(registration)
    )
    target = np.asarray(closing[0]["pose_wxyz_xyz"], dtype=float)
    target_rotation = mink.SE3(target).as_matrix()[:3, :3]
    precontact = target.copy()
    precontact[4:] -= float(settings["demo_precontact_retreat_m"]) * target_rotation[:, 0]
    before = _capture(profile, rpc, output_dir / "before")
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if aperture < float(settings["open_gripper_ratio"]) - 0.05:
        raise RuntimeError("Peacock close demo requires fully open jaws")
    fk = ProductionRightFK(profile["production_model"])
    precontact_motion = _settle_cartesian_target(
        profile,
        rpc,
        fk,
        target_pose=precontact,
        duration_s=float(settings["demo_precontact_duration_s"]),
        aperture=1.0,
        stage="incubator_close_demo_precontact",
    )
    contact_motion = _settle_cartesian_target(
        profile,
        rpc,
        fk,
        target_pose=target,
        duration_s=float(settings["demo_contact_duration_s"]),
        aperture=1.0,
        stage="incubator_close_demo_contact",
    )
    motion = _stream_retargeted_segment(
        profile,
        rpc,
        fk,
        closing[1:],
        stage="incubator_peacock_open_jaw_close",
        time_scale=float(settings["time_scale"]),
        checkpoint_frames=int(settings["checkpoint_frames"]),
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "close-door-demo",
        "strategy": "raw Peacock low open-jaw pushing trajectory",
        "scene_registration": (
            "identity; verified Pasteur robot coordinates"
            if appliance_registration_path is None
            else str(appliance_registration_path.resolve())
        ),
        "source": str(Path(settings["peacock_strategy_reference"]).resolve()),
        "precontact_motion": precontact_motion,
        "contact_motion": contact_motion,
        "motion": motion,
        "before": before,
        "after": after,
        "extra_push_sent": False,
    }
    (output_dir / "close_door_demo.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def restore_peacock_close_start(
    profile: dict,
    rpc,
    output_dir: Path,
) -> dict:
    """Reverse the registered open-jaw close demo to its low start pose."""

    settings = profile["close_door"]
    compiled = _load(profile["compiled_demo"])
    proof_state = _load(settings["reference_proof_state"])
    registration = mink.SE3(
        np.asarray(proof_state["contact_pose_wxyz_xyz"], dtype=float)
    ) @ mink.SE3(
        np.asarray(compiled["medoid"]["contact_pose_wxyz_xyz"], dtype=float)
    ).inverse()
    closing = register_close_trajectory(
        load_open_jaw_close_trajectory(settings["peacock_strategy_reference"]),
        registration.parameters(),
    )
    live = mink.SE3(np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float))
    endpoint = mink.SE3(np.asarray(closing[-1]["pose_wxyz_xyz"], dtype=float))
    correction = live @ endpoint.inverse()
    restore = []
    for index, sample in enumerate(reversed(closing), start=1):
        source = mink.SE3(np.asarray(sample["pose_wxyz_xyz"], dtype=float))
        restore.append(
            {
                **sample,
                "t_s": index / CONTROL_HZ,
                "pose_wxyz_xyz": np.asarray(
                    (correction @ source).parameters(), dtype=float
                ).tolist(),
                "gripper": 1.0,
            }
        )
    before = _capture(profile, rpc, output_dir / "before")
    motion = _stream_retargeted_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        restore,
        stage="incubator_restore_peacock_close_start",
        time_scale=float(settings["time_scale"]),
        checkpoint_frames=int(settings["checkpoint_frames"]),
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "restore-close-demo-start",
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "restore_close_demo_start.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def recover_empty_close(profile: dict, rpc, output_dir: Path) -> dict:
    """Retreat along the measured tool approach axis, then reopen."""

    settings = profile["contact"]
    fk = ProductionRightFK(profile["production_model"])
    before = _capture(profile, rpc, output_dir / "before")
    start = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    rotation = mink.SE3(start).as_matrix()[:3, :3]
    target = start.copy()
    target[4:] -= float(settings["recovery_retreat_m"]) * rotation[:, 0]
    retreat = _cartesian_segment(
        profile,
        rpc,
        fk,
        target_pose=target,
        duration_s=float(settings["recovery_duration_s"]),
        aperture=aperture,
        stage="incubator_empty_close_retreat",
    )
    opened = _fixed_pose_gripper_ramp(
        profile,
        rpc,
        fk,
        finish_ratio=1.0,
        duration_s=float(settings["close_duration_s"]),
        stage="incubator_reopen_at_clearance",
    )
    time.sleep(float(settings["open_observation_s"]))
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "recover-empty-close",
        "retreat": retreat,
        "open": opened,
        "before": before,
        "after": after,
    }
    (output_dir / "recover_empty_close.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def open_in_place(profile: dict, rpc, output_dir: Path) -> dict:
    """Open the right gripper without translating or rotating the wrist."""

    fk = ProductionRightFK(profile["production_model"])
    before_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    before_aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    motion = _fixed_pose_gripper_ramp(
        profile,
        rpc,
        fk,
        finish_ratio=1.0,
        duration_s=float(profile["contact"]["close_duration_s"]),
        stage="incubator_open_in_place",
    )
    after_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    after_aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    result = {
        "commands_sent": True,
        "stage": "open-in-place",
        "before_pose_wxyz_xyz": before_pose.tolist(),
        "after_pose_wxyz_xyz": after_pose.tolist(),
        "before_aperture": before_aperture,
        "after_aperture": after_aperture,
        "motion": motion,
    }
    (output_dir / "open_in_place.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def retreat_and_orient(profile: dict, rpc, output_dir: Path) -> dict:
    compiled = _load(profile["compiled_demo"])
    fk = ProductionRightFK(profile["production_model"])
    start = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    aperture = float(np.asarray(rpc.get_right_gripper_exact()).reshape(-1)[0])
    if aperture < 0.95:
        raise RuntimeError("retreat-orient requires the right gripper fully open")
    rotation = mink.SE3(start).as_matrix()[:3, :3]
    retreat = start.copy()
    retreat[4:] -= float(profile["retreat"]["distance_m"]) * rotation[:, 0]
    translation = _cartesian_segment(
        profile,
        rpc,
        fk,
        target_pose=retreat,
        duration_s=float(profile["retreat"]["translation_duration_s"]),
        aperture=aperture,
        stage="incubator_retreat_open",
    )
    target_pose = retreat.copy()
    target_pose[:4] = np.asarray(
        compiled["medoid"]["contact_pose_wxyz_xyz"][:4], dtype=float
    )
    measured_retreat_q = np.asarray(
        rpc.get_right_joint_positions(), dtype=float
    )
    solver = SingleArmIK(
        profile["production_model"],
        joint_names=[f"left_arm_joint{index}" for index in range(1, 7)],
        ee_frame="left_arm_ee",
    )
    solver.init(
        np.asarray(profile["retreat"]["orientation_ik_seed_q_rad"], dtype=float)
    )
    target_q, _ = solver.solve_ik(mink.SE3(target_pose), max_iter=300)
    target_q = np.asarray(target_q, dtype=float)
    predicted_target = np.asarray(fk.pose(target_q).parameters(), dtype=float)
    endpoint_position_error = float(
        np.linalg.norm(predicted_target[4:] - target_pose[4:])
    )
    endpoint_rotation_error = quaternion_distance_rad(
        predicted_target[:4], target_pose[:4]
    )
    if (
        not np.all(np.isfinite(target_q))
        or np.max(np.abs(target_q)) >= 2.95
        or endpoint_position_error > 0.002
        or endpoint_rotation_error > 0.02
    ):
        raise RuntimeError(
            "demo orientation IK endpoint is not motion eligible: "
            f"position_error={endpoint_position_error:.6f}m, "
            f"rotation_error={endpoint_rotation_error:.6f}rad, "
            f"max_abs_q={np.max(np.abs(target_q)):.3f}"
        )
    orientation_samples = sample_joint_knots(
        [
            {
                "stage": "incubator_demo_orientation_start",
                "right_q_physical_rad": measured_retreat_q.tolist(),
                "right_gripper_open_ratio": aperture,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "incubator_demo_orientation_open",
                "right_q_physical_rad": target_q.tolist(),
                "right_gripper_open_ratio": aperture,
                "minimum_duration_s": float(
                    profile["retreat"]["orientation_duration_s"]
                ),
            },
        ]
    )
    approach_axis = rotation[:, 0]
    path_positions = np.asarray(
        [
            fk.pose(sample.right_q_physical_rad).translation()
            for sample in orientation_samples
        ],
        dtype=float,
    )
    clearance = -np.max((path_positions - start[4:]) @ approach_axis)
    minimum_clearance = float(profile["retreat"]["minimum_door_clearance_m"])
    if clearance < minimum_clearance:
        raise RuntimeError(
            "orientation path approaches the observed door plane: "
            f"clearance={clearance:.6f}m < {minimum_clearance:.6f}m"
        )
    orientation = _streamer(profile, rpc, fk).execute(orientation_samples)
    time.sleep(float(profile["retreat"]["settle_s"]))
    final_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    result = {
        "commands_sent": True,
        "stage": "retreat-orient",
        "start_pose_wxyz_xyz": start.tolist(),
        "retreat_target_wxyz_xyz": retreat.tolist(),
        "orientation_target_wxyz_xyz": target_pose.tolist(),
        "orientation_target_q_rad": target_q.tolist(),
        "minimum_observed_door_clearance_m": float(clearance),
        "orientation_endpoint_position_error_m": endpoint_position_error,
        "orientation_endpoint_rotation_error_rad": endpoint_rotation_error,
        "final_pose_wxyz_xyz": final_pose.tolist(),
        "final_orientation_error_deg": math.degrees(
            quaternion_distance_rad(final_pose[:4], target_pose[:4])
        ),
        "translation_motion": translation,
        "orientation_motion": orientation,
    }
    (output_dir / "motion.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def move_to_demo_hover(profile: dict, rpc, output_dir: Path) -> dict:
    """Move at clearance to the verified demo's absolute contact neighborhood."""

    compiled = _load(profile["compiled_demo"])
    settings = profile["reanchor"]
    contact = np.asarray(
        compiled["medoid"]["contact_pose_wxyz_xyz"], dtype=float
    )
    canonical_rotation = mink.SE3(contact).as_matrix()[:3, :3]
    target = contact.copy()
    target[4:] -= (
        float(settings["demo_hover_distance_m"])
        * canonical_rotation[:, 0]
    )
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if aperture < 0.95:
        raise RuntimeError("demo hover requires the right gripper fully open")
    before = _capture(profile, rpc, output_dir / "before")
    motion = _cartesian_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=target,
        duration_s=float(settings["hover_duration_s"]),
        aperture=aperture,
        stage="incubator_demo_absolute_hover",
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "demo-hover",
        "target_wxyz_xyz": target.tolist(),
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "demo_hover.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def move_to_demo_preclose(profile: dict, rpc, output_dir: Path) -> dict:
    """Move from hover to the medoid's verified open-gripper preclose pose."""

    compiled = _load(profile["compiled_demo"])
    target = np.asarray(
        compiled["medoid"]["preclose_pose_wxyz_xyz"], dtype=float
    )
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if aperture < 0.95:
        raise RuntimeError("demo preclose requires the right gripper fully open")
    before = _capture(profile, rpc, output_dir / "before")
    motion = _cartesian_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=target,
        duration_s=float(profile["reanchor"]["preclose_duration_s"]),
        aperture=aperture,
        stage="incubator_demo_absolute_preclose",
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "demo-preclose",
        "target_wxyz_xyz": target.tolist(),
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "demo_preclose.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def move_to_demo_contact(profile: dict, rpc, output_dir: Path) -> dict:
    """Complete the verified preclose-to-contact segment with jaws open."""

    compiled = _load(profile["compiled_demo"])
    target = np.asarray(
        compiled["medoid"]["contact_pose_wxyz_xyz"], dtype=float
    )
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if aperture < 0.95:
        raise RuntimeError("demo contact requires the right gripper fully open")
    before = _capture(profile, rpc, output_dir / "before")
    motion = _cartesian_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=target,
        duration_s=float(profile["reanchor"]["contact_duration_s"]),
        aperture=aperture,
        stage="incubator_demo_absolute_contact_open",
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "demo-contact",
        "target_wxyz_xyz": target.tolist(),
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "demo_contact.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def move_from_aligned_preclose_to_contact(
    profile: dict, rpc, output_dir: Path
) -> dict:
    """Preserve live visual alignment while applying demo approach geometry."""

    compiled = _load(profile["compiled_demo"])
    demo_preclose = mink.SE3(
        np.asarray(
            compiled["medoid"]["preclose_pose_wxyz_xyz"], dtype=float
        )
    )
    demo_contact = mink.SE3(
        np.asarray(
            compiled["medoid"]["contact_pose_wxyz_xyz"], dtype=float
        )
    )
    relative = demo_preclose.inverse() @ demo_contact
    live_preclose = np.asarray(
        rpc.get_right_ee_pose().parameters(), dtype=float
    )
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if aperture < 0.95:
        raise RuntimeError(
            "aligned contact approach requires the right gripper fully open"
        )
    target = retarget_relative_pose(
        live_preclose,
        np.asarray(relative.parameters(), dtype=float),
    )
    before = _capture(profile, rpc, output_dir / "before")
    motion = _cartesian_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=target,
        duration_s=float(profile["reanchor"]["contact_duration_s"]),
        aperture=aperture,
        stage="incubator_visual_aligned_contact_open",
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "aligned-contact",
        "live_preclose_wxyz_xyz": live_preclose.tolist(),
        "demo_preclose_to_contact_wxyz_xyz": np.asarray(
            relative.parameters(), dtype=float
        ).tolist(),
        "target_wxyz_xyz": target.tolist(),
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "aligned_contact.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def restore_aligned_preclose(
    profile: dict,
    rpc,
    output_dir: Path,
    aligned_state_path: Path,
) -> dict:
    """Restore learned lateral alignment at the demonstrated table height."""

    state = _load(aligned_state_path)
    compiled = _load(profile["compiled_demo"])
    target = np.asarray(state["live_preclose_wxyz_xyz"], dtype=float)
    demo_preclose = np.asarray(
        compiled["medoid"]["preclose_pose_wxyz_xyz"], dtype=float
    )
    # The incubator may translate or yaw on its platform, but the platform
    # fixes its world height.  Image-fragmentation must not ratchet Z away
    # from the demonstrated grasp band.
    target[6] = demo_preclose[6]
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if aperture < 0.95:
        raise RuntimeError(
            "aligned preclose restore requires the right gripper fully open"
        )
    before = _capture(profile, rpc, output_dir / "before")
    motion = _cartesian_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=target,
        duration_s=float(profile["reanchor"]["hover_duration_s"]),
        aperture=aperture,
        stage="incubator_restore_aligned_preclose_at_demo_height",
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "restore-aligned-preclose",
        "source_state": str(aligned_state_path.resolve()),
        "target_wxyz_xyz": target.tolist(),
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "restore_aligned_preclose.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def orientation_probe(
    profile: dict,
    rpc,
    output_dir: Path,
    *,
    world_yaw_deg: float,
) -> dict:
    """Probe a small door-yaw correction at open-jaw preclose clearance."""

    maximum = float(profile.get("orientation_probe_maximum_deg", 7.0))
    if abs(float(world_yaw_deg)) > maximum:
        raise RuntimeError(
            f"orientation probe {world_yaw_deg:.3f} deg exceeds {maximum:.3f} deg"
        )
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if aperture < 0.95:
        raise RuntimeError(
            "orientation probe requires the right gripper fully open"
        )
    start = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    target = world_yaw_pose(start, world_yaw_deg)
    before = _capture(profile, rpc, output_dir / "before")
    motion = _cartesian_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=target,
        duration_s=float(profile.get("orientation_probe_duration_s", 1.5)),
        aperture=aperture,
        stage="incubator_open_jaw_world_yaw_probe",
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "orientation-probe",
        "world_yaw_deg": float(world_yaw_deg),
        "start_wxyz_xyz": start.tolist(),
        "target_wxyz_xyz": target.tolist(),
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "orientation_probe.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def aligned_yaw_preclose(
    profile: dict,
    rpc,
    output_dir: Path,
    aligned_state_path: Path,
    *,
    world_yaw_deg: float,
    appliance_registration_path: Path | None = None,
) -> dict:
    """Move to one absolute, visually aligned preclose with a door-yaw offset."""

    maximum = float(profile.get("aligned_yaw_maximum_deg", 15.0))
    if abs(float(world_yaw_deg)) > maximum:
        raise RuntimeError(
            f"aligned yaw {world_yaw_deg:.3f} deg exceeds {maximum:.3f} deg"
        )
    aperture = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    if aperture < 0.95:
        raise RuntimeError("aligned yaw preclose requires fully open jaws")
    state = _load(aligned_state_path)
    compiled = _load(profile["compiled_demo"])
    base = np.asarray(state["live_preclose_wxyz_xyz"], dtype=float)
    base[6] = float(compiled["medoid"]["preclose_pose_wxyz_xyz"][6])
    registration = _load_appliance_registration(appliance_registration_path)
    base = matrix_to_pose7(registration @ mink.SE3(base).as_matrix())
    target = world_yaw_pose(base, world_yaw_deg)
    before = _capture(profile, rpc, output_dir / "before")
    settle = _settle_cartesian_target(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=target,
        duration_s=float(profile["reanchor"]["hover_duration_s"]),
        aperture=aperture,
        stage="incubator_absolute_aligned_yaw_preclose",
    )
    after = _capture(profile, rpc, output_dir / "after")
    result = {
        "commands_sent": True,
        "stage": "aligned-yaw-preclose",
        "source_state": str(aligned_state_path.resolve()),
        "world_yaw_deg": float(world_yaw_deg),
        "appliance_registration": (
            None
            if appliance_registration_path is None
            else str(appliance_registration_path.resolve())
        ),
        "base_wxyz_xyz": base.tolist(),
        "target_wxyz_xyz": target.tolist(),
        "settle": settle,
        "before": before,
        "after": after,
    }
    (output_dir / "aligned_yaw_preclose.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def visual_align_step(profile: dict, rpc, output_dir: Path) -> dict:
    compiled = _load(profile["compiled_demo"])
    model = compiled["visual_servo"]
    settings = profile["visual_feature"]
    before = _capture(profile, rpc, output_dir / "before")
    before_image = cv2.imread(before["image_paths"]["right"])
    before_feature, before_report = extract_feature(before_image, settings)
    predicted = predict_local_delta(model, before_feature)
    # Translation along local X changes door clearance.  The first visual
    # stage is therefore lateral-only and always followed by a fresh frame.
    lateral = predicted.copy()
    lateral[0] = 0.0
    lateral *= float(settings["lateral_gain"])
    norm = float(np.linalg.norm(lateral))
    maximum = float(settings["maximum_lateral_step_m"])
    if norm > maximum:
        lateral *= maximum / norm
    start_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    aperture = float(np.asarray(rpc.get_right_gripper_exact()).reshape(-1)[0])
    if aperture < 0.95:
        raise RuntimeError("visual alignment requires the right gripper fully open")
    rotation = mink.SE3(start_pose).as_matrix()[:3, :3]
    world_delta = rotation @ lateral
    target_pose = start_pose.copy()
    target_pose[4:] += world_delta
    fk = ProductionRightFK(profile["production_model"])
    motion = _cartesian_segment(
        profile,
        rpc,
        fk,
        target_pose=target_pose,
        duration_s=float(settings["lateral_duration_s"]),
        aperture=aperture,
        stage="incubator_visual_lateral_open",
    )
    after = _capture(profile, rpc, output_dir / "after")
    after_image = cv2.imread(after["image_paths"]["right"])
    after_feature, after_report = extract_feature(after_image, settings)
    goal = np.asarray(model["goal_feature_mean"], dtype=float)
    # UV and log-area have different physical units.  Report both separately;
    # do not collapse them into one arbitrary scalar motion gate.
    result = {
        "commands_sent": True,
        "stage": "visual-align-step",
        "before_feature": before_report,
        "after_feature": after_report,
        "goal_feature_uv_log_area": goal.tolist(),
        "predicted_local_delta_m": predicted.tolist(),
        "applied_lateral_local_delta_m": lateral.tolist(),
        "applied_lateral_world_delta_m": world_delta.tolist(),
        "before_uv_error": (goal[:2] - before_feature[:2]).tolist(),
        "after_uv_error": (goal[:2] - after_feature[:2]).tolist(),
        "before_log_area_error": float(goal[2] - before_feature[2]),
        "after_log_area_error": float(goal[2] - after_feature[2]),
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "visual_alignment.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def measured_lateral_step(
    profile: dict,
    rpc,
    output_dir: Path,
    *,
    local_x_m: float,
    local_y_m: float,
    local_z_m: float,
) -> dict:
    settings = profile["visual_feature"]
    compiled = _load(profile["compiled_demo"])
    local = np.asarray(
        [float(local_x_m), float(local_y_m), float(local_z_m)]
    )
    maximum_lateral = float(settings["maximum_lateral_step_m"])
    maximum_approach = float(settings["maximum_approach_probe_m"])
    if (
        not np.all(np.isfinite(local))
        or np.linalg.norm(local[1:]) > maximum_lateral
        or abs(local[0]) > maximum_approach
    ):
        raise ValueError("measured local step exceeds configured magnitude")
    before = _capture(profile, rpc, output_dir / "before")
    before_image = cv2.imread(before["image_paths"]["right"])
    try:
        before_feature, before_report = extract_feature(before_image, settings)
    except RuntimeError as error:
        before_feature = None
        before_report = {"detected": False, "reason": str(error)}
    start_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    canonical_quaternion = np.asarray(
        compiled["medoid"]["contact_pose_wxyz_xyz"][:4], dtype=float
    )
    orientation_error_deg = math.degrees(
        quaternion_distance_rad(start_pose[:4], canonical_quaternion)
    )
    maximum_orientation_error_deg = float(
        profile["contact"]["maximum_orientation_error_deg"]
    )
    if orientation_error_deg > maximum_orientation_error_deg:
        raise RuntimeError(
            "measured step denied: run retreat-orient before correcting a "
            f"{orientation_error_deg:.2f}deg wrist drift"
        )
    aperture = float(np.asarray(rpc.get_right_gripper_exact()).reshape(-1)[0])
    if aperture < 0.95:
        raise RuntimeError("lateral step requires the right gripper fully open")
    rotation = mink.SE3(start_pose).as_matrix()[:3, :3]
    world = rotation @ local
    target = start_pose.copy()
    target[:4] = canonical_quaternion
    target[4:] += world
    motion = _cartesian_segment(
        profile,
        rpc,
        ProductionRightFK(profile["production_model"]),
        target_pose=target,
        duration_s=float(settings["lateral_duration_s"]),
        aperture=aperture,
        stage="incubator_measured_lateral_open",
    )
    after = _capture(profile, rpc, output_dir / "after")
    after_image = cv2.imread(after["image_paths"]["right"])
    try:
        after_feature, after_report = extract_feature(after_image, settings)
    except RuntimeError as error:
        after_feature = None
        after_report = {"detected": False, "reason": str(error)}
    observed_uv_delta = None
    if before_feature is not None and after_feature is not None:
        observed_uv_delta = (
            after_feature[:2] - before_feature[:2]
        ).tolist()
    result = {
        "commands_sent": True,
        "stage": "measured-lateral-step",
        "local_delta_m": local.tolist(),
        "world_delta_m": world.tolist(),
        "start_orientation_error_deg": orientation_error_deg,
        "before_feature": before_report,
        "after_feature": after_report,
        "observed_uv_delta": observed_uv_delta,
        "motion": motion,
        "before": before,
        "after": after,
    }
    (output_dir / "lateral_step.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="src/configs/pasteur_incubator_door_demo.json",
    )
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8081)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-y-m", type=float, default=0.0)
    parser.add_argument("--local-z-m", type=float, default=0.0)
    parser.add_argument("--local-x-m", type=float, default=0.0)
    parser.add_argument("--contact-state", type=Path)
    parser.add_argument("--proof-state", type=Path)
    parser.add_argument("--aligned-state", type=Path)
    parser.add_argument("--close-state", type=Path)
    parser.add_argument("--world-yaw-deg", type=float, default=0.0)
    parser.add_argument("--appliance-registration", type=Path)
    parser.add_argument(
        "stage",
        choices=(
            "observe",
            "retreat-orient",
            "retreat-orient-observe",
            "visual-align-step",
            "measured-lateral-step",
            "close-verify",
            "proof-pull",
            "reverify-proof",
            "open-door",
            "close-door",
            "restore-open-start",
            "close-door-demo",
            "restore-close-demo-start",
            "recover-empty-close",
            "open-in-place",
            "demo-hover",
            "demo-preclose",
            "demo-contact",
            "aligned-contact",
            "restore-aligned-preclose",
            "orientation-probe",
            "aligned-yaw-preclose",
        ),
    )
    args = parser.parse_args()
    profile = _load(args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rpc = RPCClient(args.rpc_host, args.rpc_port, timeout_ms=5000)
    result = {}
    if args.stage in {"retreat-orient", "retreat-orient-observe"}:
        result["motion"] = retreat_and_orient(profile, rpc, args.output_dir)
    if args.stage in {"observe", "retreat-orient-observe"}:
        result["observation"] = _capture(profile, rpc, args.output_dir)
    if args.stage == "visual-align-step":
        result["visual_alignment"] = visual_align_step(
            profile, rpc, args.output_dir
        )
    if args.stage == "measured-lateral-step":
        result["lateral_step"] = measured_lateral_step(
            profile,
            rpc,
            args.output_dir,
            local_x_m=args.local_x_m,
            local_y_m=args.local_y_m,
            local_z_m=args.local_z_m,
        )
    if args.stage == "close-verify":
        result["close_verify"] = close_and_verify(
            profile, rpc, args.output_dir, args.contact_state
        )
    if args.stage == "proof-pull":
        if args.contact_state is None:
            parser.error("proof-pull requires --contact-state")
        result["proof_pull"] = proof_pull(
            profile, rpc, args.output_dir, args.contact_state
        )
    if args.stage == "reverify-proof":
        if args.proof_state is None:
            parser.error("reverify-proof requires --proof-state")
        result["reverify_proof"] = reverify_proof_hold(
            profile, rpc, args.output_dir, args.proof_state
        )
    if args.stage == "open-door":
        if args.proof_state is None:
            parser.error("open-door requires --proof-state")
        result["open_door"] = open_door(
            profile, rpc, args.output_dir, args.proof_state
        )
    if args.stage == "close-door":
        result["close_door"] = close_door_by_reversing_opening(
            profile, rpc, args.output_dir
        )
    if args.stage == "restore-open-start":
        if args.close_state is None:
            parser.error("restore-open-start requires --close-state")
        result["restore_open_start"] = restore_after_missed_close(
            profile, rpc, args.output_dir, args.close_state
        )
    if args.stage == "close-door-demo":
        result["close_door_demo"] = close_door_from_peacock_demo(
            profile, rpc, args.output_dir, args.appliance_registration
        )
    if args.stage == "restore-close-demo-start":
        result["restore_close_demo_start"] = restore_peacock_close_start(
            profile, rpc, args.output_dir
        )
    if args.stage == "recover-empty-close":
        result["recover_empty_close"] = recover_empty_close(
            profile, rpc, args.output_dir
        )
    if args.stage == "open-in-place":
        result["open_in_place"] = open_in_place(
            profile, rpc, args.output_dir
        )
    if args.stage == "demo-hover":
        result["demo_hover"] = move_to_demo_hover(
            profile, rpc, args.output_dir
        )
    if args.stage == "demo-preclose":
        result["demo_preclose"] = move_to_demo_preclose(
            profile, rpc, args.output_dir
        )
    if args.stage == "demo-contact":
        result["demo_contact"] = move_to_demo_contact(
            profile, rpc, args.output_dir
        )
    if args.stage == "aligned-contact":
        result["aligned_contact"] = move_from_aligned_preclose_to_contact(
            profile, rpc, args.output_dir
        )
    if args.stage == "restore-aligned-preclose":
        if args.aligned_state is None:
            parser.error(
                "restore-aligned-preclose requires --aligned-state"
            )
        result["restore_aligned_preclose"] = restore_aligned_preclose(
            profile, rpc, args.output_dir, args.aligned_state
        )
    if args.stage == "orientation-probe":
        result["orientation_probe"] = orientation_probe(
            profile,
            rpc,
            args.output_dir,
            world_yaw_deg=args.world_yaw_deg,
        )
    if args.stage == "aligned-yaw-preclose":
        if args.aligned_state is None:
            parser.error("aligned-yaw-preclose requires --aligned-state")
        result["aligned_yaw_preclose"] = aligned_yaw_preclose(
            profile,
            rpc,
            args.output_dir,
            args.aligned_state,
            world_yaw_deg=args.world_yaw_deg,
            appliance_registration_path=args.appliance_registration,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SAM-only staged pregrasp: horizontal alignment before any descent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_realtime_sam_grasp import LiveSamGrasp
from robot.camera_id import load_camera_map
from rollout.camera import USBWristCameraFeedManager
from rollout.sam_segmentation import (
    choose_lid_candidate,
    enhance_low_light,
)


def _atomic_write_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


class MotionExecutionClaim:
    """One-shot, process-independent authorization for physical motion."""

    def __init__(self, path: Path, payload):
        self.path = Path(path)
        self.payload = dict(payload)
        self.finalized = False
        self.result = None

    def set_result(self, result):
        self.result = result

    def finalize(self, error: BaseException | None = None):
        if self.finalized:
            return
        self.finalized = True
        self.payload["finished_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        if error is None:
            self.payload["status"] = "completed"
            if self.result is not None:
                self.payload["result"] = self.result
        else:
            self.payload["status"] = "failed"
            self.payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        _atomic_write_json(self.path, self.payload)


def claim_motion_execution(
    motion_token: str,
    claim_dir: str | Path,
    output_dir: str | Path,
    intent,
):
    """Atomically consume a token and reserve a never-overwritten run dir."""

    token = str(motion_token)
    if not token or len(token) > 256 or any(char.isspace() for char in token):
        raise ValueError("motion token must be non-empty and contain no spaces")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    claim_root = Path(claim_dir)
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_path = claim_root / f"{token_hash}.json"
    payload = {
        "schema": "piper_motion_claim/v1",
        "token_sha256": token_hash,
        "status": "claimed",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "output_dir": str(Path(output_dir).resolve()),
        "intent": intent,
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    try:
        descriptor = os.open(
            claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "motion token was already consumed; refusing duplicate execution"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(claim_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

    claim = MotionExecutionClaim(claim_path, payload)
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=False)
    except BaseException as exc:
        claim.finalize(exc)
        raise RuntimeError(
            "physical-run output directory already exists; refusing overwrite"
        ) from exc
    return claim


def _right_state(rpc):
    pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    torque = np.asarray(rpc.get_right_joint_torque(), dtype=float)
    if (
        pose.shape != (7,)
        or q.shape != (6,)
        or torque.shape != (6,)
        or not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(torque))
    ):
        raise RuntimeError("invalid right-arm state in calibration journal")
    return {
        "monotonic_s": time.monotonic(),
        "pose_wxyz_xyz": pose.tolist(),
        "joint_position_rad": q.tolist(),
        "joint_torque_nm": torque.tolist(),
    }


def verify_right_stationary(
    rpc,
    duration_s=0.5,
    sample_period_s=0.05,
    torque_limit=None,
):
    duration_s = float(duration_s)
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("stationary verification duration must be non-negative")
    if torque_limit is not None:
        torque_limit = np.asarray(torque_limit, dtype=float)
        if (
            torque_limit.shape != (6,)
            or not np.all(np.isfinite(torque_limit))
            or np.any(torque_limit <= 0.0)
        ):
            raise ValueError("invalid stationary torque limit")
    samples = [_right_state(rpc)]
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        if torque_limit is not None and np.any(
            np.abs(samples[-1]["joint_torque_nm"]) > torque_limit
        ):
            break
        time.sleep(min(sample_period_s, max(0.0, deadline - time.monotonic())))
        samples.append(_right_state(rpc))
    poses = np.asarray([sample["pose_wxyz_xyz"] for sample in samples])
    joints = np.asarray(
        [sample["joint_position_rad"] for sample in samples]
    )
    torques = np.abs(
        np.asarray([sample["joint_torque_nm"] for sample in samples])
    )
    xyz_span = np.ptp(poses[:, 4:7], axis=0)
    joint_span = np.ptp(joints, axis=0)
    max_abs_torque = np.max(torques, axis=0)
    torque_ok = True
    if torque_limit is not None:
        torque_ok = bool(np.all(max_abs_torque <= torque_limit))
    return {
        "duration_s": duration_s,
        "sample_count": len(samples),
        "xyz_span_m": xyz_span.tolist(),
        "joint_span_rad": joint_span.tolist(),
        "max_abs_torque_nm": max_abs_torque.tolist(),
        "torque_within_limit": torque_ok,
        "verified": bool(
            np.max(xyz_span) <= 0.0005
            and np.max(joint_span) <= 0.01
            and torque_ok
        ),
        "final_state": samples[-1],
    }


def _image_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_horizontal_jacobian(robot_deltas, feature_deltas, rcond=0.12):
    """Return image XY change per robot XY metre from horizontal probes."""

    robot = np.asarray(robot_deltas, dtype=float)
    feature = np.asarray(feature_deltas, dtype=float)
    if (
        robot.ndim != 2
        or feature.ndim != 2
        or robot.shape != feature.shape
        or robot.shape[1] != 3
        or robot.shape[0] < 2
    ):
        raise ValueError("at least two paired XYZ observations are required")
    horizontal_robot = robot[:, :2]
    horizontal_feature = feature[:, :2]
    if np.linalg.matrix_rank(horizontal_robot, tol=5e-4) < 2:
        raise ValueError("live probes span fewer than two horizontal directions")
    jacobian = horizontal_feature.T @ np.linalg.pinv(
        horizontal_robot.T, rcond=rcond
    )
    return jacobian


def estimate_horizontal_displacement(jacobian, feature_error):
    """Map SAM pixel error into local robot XY without requesting descent."""

    jacobian = np.asarray(jacobian, dtype=float).reshape(2, 2)
    error = np.asarray(feature_error, dtype=float).reshape(3)[:2]
    horizontal = np.linalg.lstsq(jacobian, error, rcond=0.12)[0]
    return np.array([horizontal[0], horizontal[1], 0.0])


def bound_horizontal_step(displacement, max_norm_m, max_axis_m):
    step = np.zeros(3, dtype=float)
    step[:2] = np.asarray(displacement, dtype=float).reshape(3)[:2]
    peak = float(np.max(np.abs(step[:2])))
    if peak > max_axis_m:
        step[:2] *= max_axis_m / peak
    norm = float(np.linalg.norm(step[:2]))
    if norm > max_norm_m:
        step[:2] *= max_norm_m / norm
    return step


def execute_single_horizontal_probe(
    runner,
    right_observer,
    axis: str,
    distance_m: float,
    *,
    hold_window_s: float = 0.5,
):
    """Execute one horizontal probe and return an auditable atomic sample."""

    if axis not in ("x", "y"):
        raise ValueError("single probe axis must be x or y")
    distance_m = float(distance_m)
    if not np.isfinite(distance_m) or not 0.0 < distance_m <= 0.008:
        raise ValueError("single probe distance must be within (0, 0.008] m")
    request = np.zeros(3, dtype=float)
    request[0 if axis == "x" else 1] = distance_m

    before_feature, before_error, before_head_path, before_head_timestamp = (
        runner.observe(0.0)
    )
    (
        before_right_geometry,
        before_right_candidate,
        before_right_path,
        before_right_timestamp,
    ) = right_observer.observe(require_lid=False)
    before_state = _right_state(runner.rpc)
    actual_immediate = runner.move_cartesian_delta(
        request, minimum_progress=0.0
    )
    post_motion_state = _right_state(runner.rpc)
    initial_hold = verify_right_stationary(
        runner.rpc,
        duration_s=hold_window_s,
        torque_limit=getattr(runner, "torque_limit", None),
    )
    if not initial_hold["verified"]:
        runner.hold_measured()
        initial_hold = verify_right_stationary(
            runner.rpc,
            duration_s=hold_window_s,
            torque_limit=getattr(runner, "torque_limit", None),
        )
        if not initial_hold["verified"]:
            raise RuntimeError(
                "right arm did not become stationary after single probe"
            )
    after_feature, after_error, after_head_path, after_head_timestamp = (
        runner.observe(0.0)
    )
    (
        after_right_geometry,
        after_right_candidate,
        after_right_path,
        after_right_timestamp,
    ) = right_observer.observe(require_lid=False)
    hold = verify_right_stationary(
        runner.rpc,
        duration_s=hold_window_s,
        torque_limit=getattr(runner, "torque_limit", None),
    )
    if not hold["verified"]:
        runner.hold_measured()
        hold = verify_right_stationary(
            runner.rpc,
            duration_s=hold_window_s,
            torque_limit=getattr(runner, "torque_limit", None),
        )
        if not hold["verified"]:
            raise RuntimeError(
                "right arm moved during post-probe SAM observation"
            )
    initial_hold_pose = np.asarray(
        initial_hold["final_state"]["pose_wxyz_xyz"], dtype=float
    )
    initial_hold_q = np.asarray(
        initial_hold["final_state"]["joint_position_rad"], dtype=float
    )
    final_hold_pose = np.asarray(
        hold["final_state"]["pose_wxyz_xyz"], dtype=float
    )
    final_hold_q = np.asarray(
        hold["final_state"]["joint_position_rad"], dtype=float
    )
    observation_xyz_shift = (
        final_hold_pose[4:7] - initial_hold_pose[4:7]
    )
    observation_joint_shift = final_hold_q - initial_hold_q
    if (
        np.max(np.abs(observation_xyz_shift)) > 0.0005
        or np.max(np.abs(observation_joint_shift)) > 0.01
    ):
        raise RuntimeError(
            "right arm changed pose while post-probe SAM was running"
        )
    final_pose = np.asarray(
        hold["final_state"]["pose_wxyz_xyz"], dtype=float
    )
    before_pose = np.asarray(before_state["pose_wxyz_xyz"], dtype=float)
    actual_settled = final_pose[4:7] - before_pose[4:7]
    gripper_delta = (
        after_feature.gripper_feature - before_feature.gripper_feature
    )
    lid_delta = (
        after_feature.lid_grasp_feature - before_feature.lid_grasp_feature
    )
    pixel_signal = float(np.linalg.norm(gripper_delta[:2]))
    usable_for_fit = bool(pixel_signal >= 2.0)

    def observation(
        feature,
        error,
        head_path,
        head_timestamp,
        right_geometry,
        right_candidate,
        right_path,
        right_timestamp,
    ):
        return {
            "feature_error": np.asarray(error, dtype=float).tolist(),
            "gripper_feature": np.asarray(
                feature.gripper_feature, dtype=float
            ).tolist(),
            "lid_grasp_feature": np.asarray(
                feature.lid_grasp_feature, dtype=float
            ).tolist(),
            "head_image": head_path,
            "head_image_sha256": _image_sha256(head_path),
            "head_timestamp": float(head_timestamp),
            "right_lid_center_px": (
                right_geometry.center_px.tolist()
                if right_geometry is not None
                else None
            ),
            "right_lid_score": (
                float(right_candidate.score)
                if right_candidate is not None
                else None
            ),
            "right_image": right_path,
            "right_image_sha256": _image_sha256(right_path),
            "right_timestamp": float(right_timestamp),
        }

    return {
        "schema": "sam_horizontal_probe/v1",
        "status": f"SINGLE_{axis.upper()}_PROBE_COMMITTED",
        "motion": {
            "requested_xyz_m": request.tolist(),
            "actual_immediate_xyz_m": np.asarray(
                actual_immediate, dtype=float
            ).tolist(),
            "actual_settled_xyz_m": actual_settled.tolist(),
            "before_state": before_state,
            "post_motion_state": post_motion_state,
            "initial_hold": initial_hold,
            "hold": hold,
            "observation_xyz_shift_m": observation_xyz_shift.tolist(),
            "observation_joint_shift_rad": (
                observation_joint_shift.tolist()
            ),
        },
        "observation": {
            "before": observation(
                before_feature,
                before_error,
                before_head_path,
                before_head_timestamp,
                before_right_geometry,
                before_right_candidate,
                before_right_path,
                before_right_timestamp,
            ),
            "after": observation(
                after_feature,
                after_error,
                after_head_path,
                after_head_timestamp,
                after_right_geometry,
                after_right_candidate,
                after_right_path,
                after_right_timestamp,
            ),
            "gripper_feature_delta": gripper_delta.tolist(),
            "lid_feature_delta": lid_delta.tolist(),
        },
        "quality": {
            "pixel_signal_norm": pixel_signal,
            "usable_for_fit": usable_for_fit,
            "reasons": (
                []
                if usable_for_fit
                else [
                    reason
                    for reason, rejected in (
                        ("image motion was below 2 px", pixel_signal < 2.0),
                    )
                    if rejected
                ]
            ),
        },
    }


class RightLidObserver:
    """Independent live SAM check from the right wrist camera."""

    def __init__(self, runner: LiveSamGrasp, output_dir: Path):
        camera_map = load_camera_map()
        self.runner = runner
        self.camera = USBWristCameraFeedManager(
            runner.stop_event,
            device_index=camera_map.get("right", 2),
            label="right wrist",
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.previous_center = None
        self.sequence = 0

    def start(self):
        self.camera.start()
        deadline = time.time() + 12.0
        while time.time() < deadline:
            rgb, timestamp, _ = self.camera.get_latest_frame()
            if rgb is not None and timestamp is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("right wrist camera frame unavailable")

    def stop(self):
        self.camera.stop()

    def observe(self, *, require_lid=True):
        rgb, timestamp, _ = self.camera.get_latest_frame()
        if rgb is None or timestamp is None:
            raise RuntimeError("right wrist camera frame disappeared")
        if time.time() - float(timestamp) > 0.5:
            raise RuntimeError("stale right wrist camera frame")
        image = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sequence = self.sequence
        self.sequence += 1
        raw_path = self.output_dir / f"{sequence:03d}_raw.png"
        cv2.imwrite(str(raw_path), image)
        sam_image = (
            enhance_low_light(image) if float(image.mean()) < 35.0 else image
        )
        input_path = self.output_dir / f"{sequence:03d}_sam_input.png"
        cv2.imwrite(str(input_path), sam_image)
        selected = None
        attempts = []
        for prompt in (
            "transparent round petri dish lid with blue cross",
            "petri dish lid",
            "round transparent plastic dish",
        ):
            result = self.runner.sam.segment(
                sam_image,
                frame_id=self.runner.frame_id,
                timestamp=timestamp,
                prompt=prompt,
                confidence_threshold=0.05,
            )
            self.runner.frame_id += 1
            attempts.append((prompt, len(result.candidates)))
            selected = choose_lid_candidate(
                result.candidates,
                image_bgr=sam_image,
                previous_center_px=self.previous_center,
                require_blue_cross=True,
            )
            if selected is not None:
                break
        if selected is None:
            if require_lid:
                raise RuntimeError(
                    "right SAM did not identify the blue-cross lid; "
                    f"attempts={attempts}, raw={raw_path}, input={input_path}"
                )
            overlay = sam_image.copy()
            label = f"RIGHT lid outside view; attempts={attempts}"
            cv2.putText(
                overlay,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                overlay,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            path = self.output_dir / f"{sequence:03d}.png"
            cv2.imwrite(str(path), overlay)
            return None, None, str(path), float(timestamp)
        candidate, geometry = selected
        self.previous_center = geometry.center_px.copy()
        overlay = sam_image.copy()
        mask = np.asarray(candidate.mask, dtype=bool)
        tint = np.zeros_like(overlay)
        tint[:] = (0, 255, 0)
        overlay[mask] = cv2.addWeighted(
            overlay[mask], 0.55, tint[mask], 0.45, 0
        )
        center = tuple(np.rint(geometry.center_px).astype(int))
        cv2.drawMarker(
            overlay, center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3
        )
        label = (
            f"RIGHT live SAM score={candidate.score:.3f} "
            f"center=({center[0]},{center[1]})"
        )
        cv2.putText(
            overlay,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            overlay,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        path = self.output_dir / f"{sequence:03d}.png"
        cv2.imwrite(str(path), overlay)
        return geometry, candidate, str(path), float(timestamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:15563")
    parser.add_argument(
        "--torque-config", default="src/configs/pasteur_lid_torque.json"
    )
    parser.add_argument(
        "--scene-config",
        default="src/configs/pasteur_lid_scene3d.json",
    )
    parser.add_argument("--reference-points")
    parser.add_argument(
        "--output-dir", default="/tmp/realtime_sam_horizontal"
    )
    parser.add_argument("--depth-frames", type=int, default=15)
    parser.add_argument("--preview-time", type=float, default=2.0)
    parser.add_argument("--probe-m", type=float, default=0.006)
    parser.add_argument("--max-step-m", type=float, default=0.008)
    parser.add_argument("--max-axis-m", type=float, default=0.006)
    parser.add_argument("--horizontal-tolerance-m", type=float, default=0.006)
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument(
        "--single-probe-axis",
        choices=("x", "y"),
        help="execute exactly one horizontal probe, capture both views, stop",
    )
    parser.add_argument("--single-probe-m", type=float, default=0.006)
    parser.add_argument(
        "--motion-token",
        help="one-shot user-approval token required for any physical motion",
    )
    parser.add_argument(
        "--motion-claim-dir",
        default="/var/tmp/piper_robot_motion_claims",
    )
    parser.add_argument(
        "--execute-horizontal",
        action="store_true",
        help="allow right-arm horizontal motion; default is camera-only dry-run",
    )
    args = parser.parse_args()
    if args.single_probe_axis and not args.execute_horizontal:
        parser.error("--single-probe-axis requires --execute-horizontal")
    if args.execute_horizontal and not args.motion_token:
        parser.error("--execute-horizontal requires --motion-token")

    execution_claim = None
    if args.execute_horizontal:
        execution_claim = claim_motion_execution(
            args.motion_token,
            args.motion_claim_dir,
            args.output_dir,
            {
                "mode": (
                    f"single_probe_{args.single_probe_axis}"
                    if args.single_probe_axis
                    else "horizontal_alignment"
                ),
                "axis": args.single_probe_axis,
                "distance_m": (
                    args.single_probe_m
                    if args.single_probe_axis
                    else None
                ),
                "left_arm": False,
                "gripper": False,
                "descent": False,
                "home": False,
            },
        )

    # LiveSamGrasp owns camera, SAM, torque monitoring, and right-only RPC.
    args.control_space = "cartesian"
    args.minimum_progress = 0.0
    args.joint_minimum_progress = 0.0
    try:
        runner = LiveSamGrasp(args)
        right = RightLidObserver(
            runner, Path(args.output_dir) / "right"
        )
    except BaseException as exc:
        if execution_claim is not None:
            execution_claim.finalize(exc)
        raise
    samples_robot = []
    samples_feature = []
    moved = False
    execution_error = None
    try:
        runner.start()
        right.start()
        runner.observe(0.0)  # warm Record3D temporal depth
        feature, raw_error, head_path, timestamp = runner.observe(0.0)
        registration = runner.check_scene_registration(
            args.reference_points
        )
        right_geometry, right_candidate, right_path, right_timestamp = (
            right.observe(require_lid=False)
        )
        print(
            json.dumps(
                {
                    "mode": (
                        f"SINGLE_PROBE_{args.single_probe_axis.upper()}"
                        if args.single_probe_axis
                        else (
                            "HORIZONTAL_ONLY"
                            if args.execute_horizontal
                            else "DRY_RUN_NO_MOTION"
                        )
                    ),
                    "feature_error": raw_error.round(2).tolist(),
                    "target_camera_xyz_m": (
                        runner.last_target_3d.point_camera_xyz_m.round(4).tolist()
                    ),
                    "support_plane_confidence": (
                        runner.last_target_3d.confidence
                    ),
                    "geometry_quality": {
                        "accepted": runner.last_geometry_quality.accepted,
                        "view_angle_deg": (
                            runner.last_geometry_quality.view_angle_deg
                        ),
                        "native_depth_pixel_footprint_mm": [
                            runner.last_geometry_quality.native_pixel_footprint_x_m
                            * 1000.0,
                            runner.last_geometry_quality.native_pixel_footprint_y_m
                            * 1000.0,
                        ],
                        "reasons": list(
                            runner.last_geometry_quality.reasons
                        ),
                    },
                    "proximity_warning_only_m": (
                        runner.last_proximity_m
                    ),
                    "registration": registration,
                    "head_image": head_path,
                    "head_timestamp": timestamp,
                    "right_lid_center_px": (
                        right_geometry.center_px.round(2).tolist()
                        if right_geometry is not None
                        else None
                    ),
                    "right_lid_score": (
                        float(right_candidate.score)
                        if right_candidate is not None
                        else None
                    ),
                    "right_image": right_path,
                    "right_timestamp": right_timestamp,
                }
            ),
            flush=True,
        )
        if not args.execute_horizontal:
            return
        if not runner.last_geometry_quality.accepted:
            raise RuntimeError(
                "camera geometry is too oblique for horizontal execution: "
                + "; ".join(runner.last_geometry_quality.reasons)
            )
        if args.single_probe_axis:
            moved = True
            report = execute_single_horizontal_probe(
                runner,
                right,
                args.single_probe_axis,
                args.single_probe_m,
            )
            record_path = Path(args.output_dir) / "probe_record.json"
            report["record_path"] = str(record_path)
            _atomic_write_json(record_path, report)
            execution_claim.set_result(
                {
                    "status": report["status"],
                    "record_path": str(record_path),
                    "usable_for_fit": report["quality"]["usable_for_fit"],
                }
            )
            print(json.dumps(report), flush=True)
            return
        origin = np.asarray(
            runner.rpc.get_right_ee_pose().parameters(), dtype=float
        )

        # Horizontal calibration only.  The old version probed Z here, which
        # violated the required align-first/descend-later state ordering.
        for axis in range(2):
            before, _, _, _ = runner.observe(0.0)
            current = np.asarray(
                runner.rpc.get_right_ee_pose().parameters(), dtype=float
            )
            request = np.zeros(3, dtype=float)
            request[axis] = args.probe_m
            if axis < 2:
                # Horizontal probes may experience a little IK sag. Ask the
                # controller to restore, never lower, the starting height.
                request[2] = max(0.0, float(origin[6] - current[6]))
            moved = True
            actual = runner.move_cartesian_delta(
                request, minimum_progress=0.0
            )
            after, _, path, _ = runner.observe(0.0)
            delta_feature = (
                after.gripper_feature - before.gripper_feature
            )
            if np.linalg.norm(actual) < 5e-4:
                raise RuntimeError(
                    f"Cartesian probe {axis} produced no motion"
                )
            samples_robot.append(actual)
            samples_feature.append(delta_feature)
            print(
                json.dumps(
                    {
                        "probe_axis": axis,
                        "actual_xyz_m": actual.round(5).tolist(),
                        "feature_delta": delta_feature.round(2).tolist(),
                        "image": path,
                    }
                ),
                flush=True,
            )

        worse = 0
        previous_horizontal = None
        final_path = None
        for iteration in range(args.max_iters):
            feature, raw_error, path, timestamp = runner.observe(0.0)
            (
                right_geometry,
                right_candidate,
                right_path,
                right_timestamp,
            ) = right.observe(require_lid=False)
            jacobian = fit_horizontal_jacobian(
                samples_robot, samples_feature
            )
            displacement = estimate_horizontal_displacement(
                jacobian, raw_error
            )
            horizontal_norm = float(
                np.linalg.norm(displacement[:2])
            )
            report = {
                "stage": "HORIZONTAL_ONLY",
                "iteration": iteration,
                "feature_error": raw_error.round(2).tolist(),
                "estimated_robot_displacement_m": (
                    displacement.round(4).tolist()
                ),
                "horizontal_norm_m": horizontal_norm,
                "descent_not_executed_m": float(displacement[2]),
                "image": path,
                "timestamp": timestamp,
                "right_lid_center_px": (
                    right_geometry.center_px.round(2).tolist()
                    if right_geometry is not None
                    else None
                ),
                "right_lid_score": (
                    float(right_candidate.score)
                    if right_candidate is not None
                    else None
                ),
                "right_image": right_path,
                "right_timestamp": right_timestamp,
            }
            print(json.dumps(report), flush=True)
            final_path = path
            if horizontal_norm <= args.horizontal_tolerance_m:
                runner.hold_measured()
                if right_candidate is None:
                    print(
                        json.dumps(
                            {
                                "status": (
                                    "HORIZONTAL_ALIGNED_"
                                    "RIGHT_CONFIRMATION_MISSING"
                                ),
                                "image": path,
                                "right_image": right_path,
                                "estimated_descent_m": float(
                                    displacement[2]
                                ),
                            }
                        ),
                        flush=True,
                    )
                    return
                print(
                    json.dumps(
                        {
                            "status": "HORIZONTAL_ALIGNED_DESCENT_PAUSED",
                            "image": path,
                            "estimated_descent_m": float(
                                displacement[2]
                            ),
                        }
                    ),
                    flush=True,
                )
                return

            if (
                previous_horizontal is not None
                and horizontal_norm > 1.25 * previous_horizontal
            ):
                worse += 1
                if worse >= 2:
                    runner.hold_measured()
                    raise RuntimeError(
                        "horizontal-only SAM estimate diverged"
                    )
            else:
                worse = 0

            step = bound_horizontal_step(
                displacement,
                max_norm_m=args.max_step_m,
                max_axis_m=args.max_axis_m,
            )
            current = np.asarray(
                runner.rpc.get_right_ee_pose().parameters(), dtype=float
            )
            # Correct only upward for any IK sag; this stage never requests
            # descent below its starting height.
            step[2] = np.clip(
                max(0.0, float(origin[6] - current[6])),
                0.0,
                0.002,
            )
            before_gripper = feature.gripper_feature.copy()
            moved = True
            actual = runner.move_cartesian_delta(
                step, minimum_progress=0.0
            )
            after, _, _, _ = runner.observe(0.0)
            if np.linalg.norm(actual) >= 5e-4:
                samples_robot.append(actual)
                samples_feature.append(
                    after.gripper_feature - before_gripper
                )
                samples_robot[:] = samples_robot[-8:]
                samples_feature[:] = samples_feature[-8:]
            previous_horizontal = horizontal_norm

        runner.hold_measured()
        raise RuntimeError(
            f"horizontal-only alignment did not converge; image={final_path}"
        )
    except BaseException as exc:
        execution_error = exc
        if moved:
            try:
                runner.hold_measured()
            except Exception:
                pass
        raise
    finally:
        try:
            right.stop()
        except Exception:
            pass
        try:
            runner.stop()
        finally:
            if execution_claim is not None:
                execution_claim.finalize(execution_error)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real-time SAM+depth grasp servo with no demo or endpoint calibration."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.rpc import RPCClient
from rollout.camera import CameraFeedManager
from rollout.realtime_sam_servo import (
    bounded_reachable_servo_step,
    estimate_reachable_feature_model,
    gripper_mask_center,
    lid_left_grasp_px,
    render_scene,
    scene_feature,
)
from rollout.sam_segmentation import SamSegmentationClient
from rollout.sam_segmentation import (
    choose_lid_candidate,
    enhance_low_light,
)
from rollout.scene_3d import (
    assess_target_geometry,
    backproject,
    estimate_target_on_support_plane,
    nearest_scene_distance,
    register_point_clouds,
    scaled_camera_matrix,
    temporal_median_depth,
)


class TorqueStop(RuntimeError):
    pass


PIPER_MIT_MODE_CAN_ID = 0x151
PIPER_MIT_MODE_PAYLOAD = bytes.fromhex("010400AD00000000")
DEFAULT_HOLDING_KP = np.full(6, 2.5)
DEFAULT_HOLDING_KD = np.full(6, 0.2)
DEFAULT_MOTION_KP = np.array([7.0, 7.0, 7.0, 5.0, 5.0, 5.0])
DEFAULT_MOTION_KD = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3])


def refresh_right_mit_mode(
    can_interface: str = "can_right", socket_factory=socket.socket
):
    """Reassert Piper's right-arm MIT mode without reset, enable, or homing."""

    if can_interface != "can_right":
        raise ValueError("MIT mode refresh is restricted to can_right")
    frame = struct.pack(
        "=IB3x8s",
        PIPER_MIT_MODE_CAN_ID,
        len(PIPER_MIT_MODE_PAYLOAD),
        PIPER_MIT_MODE_PAYLOAD,
    )
    can_socket = socket_factory(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        can_socket.bind((can_interface,))
        sent = can_socket.send(frame)
    finally:
        can_socket.close()
    if sent != len(frame):
        raise RuntimeError(
            f"short right MIT mode CAN write: {sent}/{len(frame)} bytes"
        )


def _gain_vector(values, name: str, maximum) -> np.ndarray:
    gain = np.asarray(values, dtype=float)
    if gain.shape != (6,) or not np.all(np.isfinite(gain)):
        raise ValueError(f"{name} must contain six finite values")
    if np.any(gain < 0.0) or np.any(gain > np.asarray(maximum, dtype=float)):
        raise ValueError(f"{name} is outside the tested safe range")
    return gain


def _note_exception(error: BaseException, note: str):
    """Attach cleanup context without replacing the primary failure."""

    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:
        print(f"{error!r}; NOTE: {note}", file=sys.stderr)


class LiveSamGrasp:
    def __init__(self, args):
        self.args = args
        self.rpc = RPCClient("localhost", 8081, timeout_ms=10000)
        self.stop_event = threading.Event()
        self.camera = CameraFeedManager(
            self.stop_event, display=False, head_stream=False
        )
        self.sam = SamSegmentationClient(args.sam_endpoint, timeout_ms=20000)
        torque_cfg = json.loads(Path(args.torque_config).read_text())
        self.torque_limit = np.asarray(
            torque_cfg["thresholds"]["right"], dtype=float
        )
        self.torque_samples = int(torque_cfg["consecutive_samples"])
        self.holding_kp = _gain_vector(
            getattr(args, "holding_kp", DEFAULT_HOLDING_KP),
            "holding kp",
            DEFAULT_MOTION_KP,
        )
        self.holding_kd = _gain_vector(
            getattr(args, "holding_kd", DEFAULT_HOLDING_KD),
            "holding kd",
            DEFAULT_MOTION_KD,
        )
        self.motion_kp = _gain_vector(
            getattr(args, "motion_kp", DEFAULT_MOTION_KP),
            "motion kp",
            DEFAULT_MOTION_KP,
        )
        self.motion_kd = _gain_vector(
            getattr(args, "motion_kd", DEFAULT_MOTION_KD),
            "motion kd",
            DEFAULT_MOTION_KD,
        )
        self.gain_ramp_s = float(getattr(args, "gain_ramp_s", 1.0))
        self.mode_settle_s = float(getattr(args, "mode_settle_s", 0.5))
        self.hold_settle_s = float(getattr(args, "hold_settle_s", 0.25))
        preparation_times = np.array(
            [self.gain_ramp_s, self.mode_settle_s, self.hold_settle_s]
        )
        if (
            not np.all(np.isfinite(preparation_times))
            or np.any(preparation_times < 0.0)
            or self.gain_ramp_s > 5.0
            or max(self.mode_settle_s, self.hold_settle_s) > 2.0
        ):
            raise ValueError("motion preparation times are outside safe bounds")
        right_can_interface = str(
            getattr(args, "right_can_interface", "can_right")
        )
        self.motion_mode_refresher = lambda: refresh_right_mit_mode(
            right_can_interface
        )
        self.frame_id = 1000
        self.sequence = 0
        self.previous_lid_center = None
        self.previous_gripper_center = None
        self.orientation = None
        self.joint_command = None
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        profile = json.loads(Path(args.scene_config).read_text())
        self.head_camera_matrix = np.asarray(
            profile["head_camera_matrix_rotated"], dtype=float
        )
        reference_shape = profile.get("head_camera_reference_shape_hw")
        if (
            reference_shape is None
            or len(reference_shape) != 2
            or any(int(value) <= 0 for value in reference_shape)
        ):
            raise ValueError(
                "scene profile requires head_camera_reference_shape_hw"
            )
        self.head_camera_reference_shape = tuple(
            int(value) for value in reference_shape
        )
        self.support_plane_config = profile.get("support_plane", {})
        self.registration_config = profile.get("registration", {})
        self.geometry_quality_config = profile.get("geometry_quality", {})
        self.proximity_warning_m = float(
            profile.get("proximity_warning_m", 0.02)
        )
        self.last_target_3d = None
        self.last_geometry_quality = None
        self.last_depth = None
        self.last_camera_matrix = None
        self.last_proximity_m = None

    def start(self):
        self.camera.start()
        deadline = time.time() + 15.0
        while time.time() < deadline:
            rgb, _, depth = self.camera.get_latest_frame()
            if rgb is not None and depth is not None and np.asarray(depth).size:
                return
            time.sleep(0.05)
        raise RuntimeError("head RGB/depth stream unavailable")

    def stop(self):
        self.sam.close()
        self.stop_event.set()
        self.camera.stop()

    def observe(self, clearance_m: float):
        rgb, timestamp, depth_raw = self.camera.get_latest_frame()
        if rgb is None or depth_raw is None:
            raise RuntimeError("head RGB/depth frame disappeared")
        age = time.time() - float(timestamp)
        if age > 0.5:
            raise RuntimeError(f"stale head frame: {age:.2f}s")
        image = cv2.cvtColor(np.rot90(rgb, k=3), cv2.COLOR_RGB2BGR)
        raw_path = self.output_dir / f"{self.sequence:03d}_head_raw.png"
        cv2.imwrite(str(raw_path), image)
        sam_image = (
            enhance_low_light(image) if float(image.mean()) < 35.0 else image
        )
        enhanced_path = (
            self.output_dir / f"{self.sequence:03d}_head_sam_input.png"
        )
        cv2.imwrite(str(enhanced_path), sam_image)
        lid = None
        selected_lid = None
        lid_attempts = []
        for prompt in (
            "transparent round petri dish lid with blue cross",
            "petri dish lid",
            "round transparent plastic dish",
        ):
            lid = self.sam.segment(
                sam_image,
                frame_id=self.frame_id,
                timestamp=timestamp,
                prompt=prompt,
                confidence_threshold=0.05,
            )
            self.frame_id += 1
            lid_attempts.append((prompt, len(lid.candidates)))
            selected_lid = choose_lid_candidate(
                lid.candidates,
                image_bgr=sam_image,
                previous_center_px=self.previous_lid_center,
                require_blue_cross=True,
            )
            if selected_lid is not None:
                break
        if selected_lid is None:
            raise ValueError(
                "SAM did not identify the blue-cross lid; "
                f"attempts={lid_attempts}, raw={raw_path}, "
                f"input={enhanced_path}"
            )
        gripper = self.sam.segment(
            sam_image,
            frame_id=self.frame_id,
            timestamp=timestamp,
            prompt="blue clamp",
            confidence_threshold=0.10,
        )
        self.frame_id += 1
        # Record3D's per-pixel depth is visibly noisy on the transparent lid
        # and reflective gripper.  Aggregate a short live burst while the arm
        # is stationary; no previously collected scene data is used.
        depth_frames = [np.asarray(depth_raw)]
        last_timestamp = float(timestamp)
        deadline = time.time() + 0.5
        while (
            len(depth_frames) < self.args.depth_frames
            and time.time() < deadline
        ):
            _, depth_timestamp, next_depth = self.camera.get_latest_frame()
            if (
                next_depth is not None
                and float(depth_timestamp) > last_timestamp
            ):
                depth_frames.append(np.asarray(next_depth))
                last_timestamp = float(depth_timestamp)
            else:
                time.sleep(0.01)
        if len(depth_frames) < max(3, self.args.depth_frames // 2):
            raise RuntimeError("insufficient fresh depth frames")
        depth = temporal_median_depth(
            depth_frames, rotate_clockwise=True
        )
        native_depth_shape = depth.shape
        depth = cv2.resize(
            depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        lid_candidate, lid_geometry = selected_lid
        grasp_px = lid_left_grasp_px(lid_candidate, lid_geometry)
        camera_matrix = scaled_camera_matrix(
            self.head_camera_matrix,
            self.head_camera_reference_shape,
            depth.shape,
        )
        self.last_depth = depth
        self.last_camera_matrix = camera_matrix
        target_3d = estimate_target_on_support_plane(
            depth,
            camera_matrix,
            lid_candidate.mask,
            grasp_px,
            ring_margin_px=int(
                self.support_plane_config.get("ring_margin_px", 70)
            ),
            plane_threshold_m=float(
                self.support_plane_config.get(
                    "ransac_threshold_m", 0.006
                )
            ),
        )
        minimum_plane_confidence = float(
            self.support_plane_config.get("minimum_confidence", 0.20)
        )
        if target_3d.confidence < minimum_plane_confidence:
            raise RuntimeError(
                "low-confidence local support plane: "
                f"{target_3d.confidence:.3f}"
            )
        self.last_target_3d = target_3d
        self.last_geometry_quality = assess_target_geometry(
            target_3d,
            camera_matrix,
            native_pixel_stride_xy=(
                image.shape[1] / native_depth_shape[1],
                image.shape[0] / native_depth_shape[0],
            ),
            maximum_view_angle_deg=float(
                self.geometry_quality_config.get(
                    "maximum_view_angle_deg", 40.0
                )
            ),
            maximum_native_footprint_m=float(
                self.geometry_quality_config.get(
                    "maximum_native_depth_pixel_footprint_m", 0.007
                )
            ),
        )
        feature = scene_feature(
            lid_candidates=lid.candidates,
            gripper_candidates=gripper.candidates,
            depth_m=depth,
            previous_lid_center_px=self.previous_lid_center,
            previous_gripper_center_px=self.previous_gripper_center,
            clearance_m=clearance_m,
            lid_support_depth_m=float(
                target_3d.point_camera_xyz_m[2]
            ),
            selected_lid=selected_lid,
        )
        self.previous_lid_center = feature.lid_geometry.center_px.copy()
        self.previous_gripper_center = gripper_mask_center(
            feature.gripper_candidate
        )
        xyz = backproject(depth, camera_matrix)
        exclusion = (
            np.asarray(feature.lid_candidate.mask, dtype=np.uint8)
            | np.asarray(feature.gripper_candidate.mask, dtype=np.uint8)
        )
        exclusion = cv2.dilate(
            exclusion,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        ).astype(bool)
        scene_valid = (
            ~exclusion
            & np.isfinite(depth)
            & (depth > 0.05)
            & (depth < 5.0)
        )
        tip = np.rint(feature.gripper_feature[:2]).astype(int)
        tip[0] = np.clip(tip[0], 0, depth.shape[1] - 1)
        tip[1] = np.clip(tip[1], 0, depth.shape[0] - 1)
        tool_point = xyz[tip[1], tip[0]]
        self.last_proximity_m = nearest_scene_distance(
            tool_point, xyz[scene_valid]
        )
        error = feature.lid_grasp_feature - feature.gripper_feature
        proximity_label = (
            "WARN"
            if self.last_proximity_m < self.proximity_warning_m
            else "ok"
        )
        geometry_label = (
            "ok" if self.last_geometry_quality.accepted else "BAD"
        )
        label = (
            f"live SAM seq={self.sequence} error="
            f"({error[0]:.1f}px,{error[1]:.1f}px,{error[2]:.1f}mm) "
            f"plane={target_3d.confidence:.2f} "
            f"view={self.last_geometry_quality.view_angle_deg:.0f}deg"
            f"({geometry_label}) "
            f"near={self.last_proximity_m*1000:.0f}mm({proximity_label})"
        )
        overlay = render_scene(image, feature, label)
        path = self.output_dir / f"{self.sequence:03d}.png"
        cv2.imwrite(str(path), overlay)
        self.sequence += 1
        return feature, error, str(path), float(timestamp)

    def check_scene_registration(self, reference_points_path: str | None):
        """Validate head-camera placement without detecting an AprilTag."""

        if not reference_points_path:
            return None
        if self.last_depth is None or self.last_camera_matrix is None:
            raise RuntimeError("observe a depth frame before registration")
        xyz = backproject(self.last_depth, self.last_camera_matrix)
        valid = (
            np.all(np.isfinite(xyz), axis=2)
            & (self.last_depth >= 0.20)
            & (self.last_depth <= 2.00)
        )
        config = self.registration_config
        result = register_point_clouds(
            xyz[valid],
            np.load(reference_points_path),
            max_correspondence_m=float(
                config.get("max_correspondence_m", 0.035)
            ),
            acceptance_rmse_m=float(
                config.get("maximum_rmse_m", 0.012)
            ),
            acceptance_inlier_fraction=float(
                config.get("minimum_inlier_fraction", 0.55)
            ),
        )
        report = {
            "accepted": result.accepted,
            "rmse_m": result.rmse_m,
            "inlier_fraction": result.inlier_fraction,
            "iterations": result.iterations,
            "live_to_reference": result.live_to_reference.tolist(),
        }
        (self.output_dir / "scene_registration.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if not result.accepted:
            raise RuntimeError(
                "head scene registration rejected; do not use old alignment"
            )
        return report

    def hold_measured(self):
        q = np.asarray(self.rpc.get_right_joint_positions(), dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise RuntimeError(
                "refusing to hold invalid measured right joint state"
            )
        self.rpc.set_right_joint_target(
            q, gripper_target=None, preview_time=0.2
        )
        self.joint_command = None

    def monitor_settle(self, duration_s: float):
        strikes = 0
        deadline = time.time() + duration_s
        while time.time() < deadline:
            torque = np.abs(
                np.asarray(self.rpc.get_right_joint_torque(), dtype=float)
            )
            strikes = strikes + 1 if np.any(torque > self.torque_limit) else 0
            if strikes >= self.torque_samples:
                self.hold_measured()
                raise TorqueStop(
                    f"right torque stop: {np.round(torque, 3).tolist()}"
                )
            time.sleep(0.05)

    @staticmethod
    def _wait_with_torque(
        duration_s, check_torque, stage, check_state=None
    ):
        if duration_s <= 0.0:
            return
        deadline = time.monotonic() + duration_s
        while True:
            check_torque(stage)
            if check_state is not None:
                check_state(stage)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(0.05, remaining))

    def _prepare_cartesian_motion(self, check_torque):
        # First latch the measured joints. Reasserting MIT mode can otherwise
        # activate a stale target left by an earlier client.
        def measured_state():
            q = np.asarray(
                self.rpc.get_right_joint_positions(), dtype=float
            )
            xyz = np.asarray(
                self.rpc.get_right_ee_pose().translation(), dtype=float
            )
            if (
                q.shape != (6,)
                or xyz.shape != (3,)
                or not np.all(np.isfinite(q))
                or not np.all(np.isfinite(xyz))
            ):
                raise RuntimeError("invalid measured right-arm state")
            return q, xyz

        start_q, start_xyz = measured_state()

        def check_preparation_drift(stage):
            q, xyz = measured_state()
            if (
                np.max(np.abs(q - start_q)) > 0.02
                or np.linalg.norm(xyz - start_xyz) > 0.002
            ):
                raise RuntimeError(
                    f"unexpected right-arm motion {stage}"
                )

        self.hold_measured()
        self.rpc.set_right_gain(self.holding_kp, self.holding_kd)
        self._wait_with_torque(
            self.hold_settle_s,
            check_torque,
            "while latching measured right pose",
            check_preparation_drift,
        )
        check_preparation_drift("while latching measured pose")
        self.motion_mode_refresher()
        self._wait_with_torque(
            self.mode_settle_s,
            check_torque,
            "after right MIT mode refresh",
            check_preparation_drift,
        )
        check_preparation_drift("after MIT mode refresh")

        if self.gain_ramp_s <= 0.0:
            self.rpc.set_right_gain(self.motion_kp, self.motion_kd)
            check_torque("after right motion gain")
            check_preparation_drift("after right motion gain")
        else:
            steps = max(1, int(np.ceil(self.gain_ramp_s / 0.1)))
            step_duration = self.gain_ramp_s / steps
            for index in range(steps):
                # Re-latch the measured pose before every stiffness increase.
                # If the arm yielded under low gain, this avoids pulling it
                # back toward a stale target as gain rises.
                self.hold_measured()
                alpha = (index + 1) / steps
                kp = (
                    self.holding_kp * (1.0 - alpha)
                    + self.motion_kp * alpha
                )
                kd = (
                    self.holding_kd * (1.0 - alpha)
                    + self.motion_kd * alpha
                )
                self.rpc.set_right_gain(kp, kd)
                self._wait_with_torque(
                    step_duration,
                    check_torque,
                    f"during right gain ramp {index + 1}/{steps}",
                    check_preparation_drift,
                )
                check_preparation_drift(
                    f"during gain ramp {index + 1}/{steps}"
                )

        # The Piper V2 MIT examples reassert 0x151 immediately before each
        # motion command. The first refresh above verifies a stable hold; this
        # second one removes any ambiguity before the first non-zero target.
        self.motion_mode_refresher()
        check_torque("after final right MIT mode refresh")
        check_preparation_drift("after final MIT mode refresh")

    def _finish_cartesian_motion(self):
        # Hold before lowering stiffness so a completed or interrupted move
        # cannot resume toward an old interpolator waypoint.
        hold_error = None
        try:
            self.hold_measured()
        except BaseException as exc:
            hold_error = exc
        try:
            self.rpc.set_right_gain(self.holding_kp, self.holding_kd)
        except BaseException as gain_error:
            if hold_error is not None:
                _note_exception(
                    hold_error,
                    f"holding-gain restoration also failed: {gain_error!r}"
                )
            else:
                raise
        if hold_error is not None:
            raise hold_error

    def move_cartesian_delta(
        self, delta_xyz, preview_time=None, minimum_progress=None
    ):
        if preview_time is None:
            preview_time = self.args.preview_time
        if minimum_progress is None:
            minimum_progress = self.args.minimum_progress
        preview_time = float(preview_time)
        if not np.isfinite(preview_time) or not 0.0 < preview_time <= 10.0:
            raise ValueError("preview_time must be within (0, 10] seconds")
        requested = np.asarray(delta_xyz, dtype=float).reshape(3)
        if not np.all(np.isfinite(requested)):
            raise ValueError("Cartesian delta must contain three finite values")

        # ConeE teleoperation succeeds by refreshing 0.05-second targets at
        # 30 Hz.  A single long-horizon target can stall under the deliberately
        # low position gains, so stream the same straight path with exactly
        # those proven timing parameters while checking torque every tick.
        command_rate_hz = 30.0
        steps = max(1, int(np.ceil(preview_time * command_rate_hz)))
        period_s = preview_time / steps
        command_preview_s = 0.05
        strikes = 0

        def check_torque(stage):
            nonlocal strikes
            torque = np.abs(
                np.asarray(self.rpc.get_right_joint_torque(), dtype=float)
            )
            if (
                torque.shape != self.torque_limit.shape
                or not np.all(np.isfinite(torque))
            ):
                raise TorqueStop(
                    f"invalid right torque sample {stage}: "
                    f"shape={torque.shape}, values={torque.tolist()}"
                )
            strikes = (
                strikes + 1
                if np.any(torque > self.torque_limit)
                else 0
            )
            if strikes >= self.torque_samples:
                raise TorqueStop(
                    f"right torque stop {stage}: "
                    f"{np.round(torque, 3).tolist()}"
                )

        motion_error = None
        try:
            self._prepare_cartesian_motion(check_torque)
            before = np.asarray(
                self.rpc.get_right_ee_pose().parameters(), dtype=float
            )
            if before.shape != (7,) or not np.all(np.isfinite(before)):
                raise RuntimeError(
                    "refusing to move from invalid measured right pose"
                )
            # Preparation deliberately takes longer than a short probe.  Start
            # the streaming deadline only after the arm is stably in MIT mode
            # with the motion gains applied.
            started = time.monotonic()
            for index in range(steps):
                check_torque("during streamed move")

                target = before.copy()
                target[4:7] += requested * ((index + 1) / steps)
                accepted = self.rpc.set_right_ee_target(
                    mink.SE3(target),
                    gripper_target=None,
                    preview_time=command_preview_s,
                )
                if accepted is not True:
                    raise RuntimeError(
                        f"right Cartesian setpoint {index + 1}/{steps} rejected"
                    )

                deadline = started + (index + 1) * period_s
                remaining = deadline - time.monotonic()
                if remaining < -period_s:
                    raise RuntimeError(
                        "right Cartesian streaming missed its control "
                        f"deadline at setpoint {index + 1}/{steps}"
                    )
                if remaining > 0.0:
                    time.sleep(remaining)

            settle_deadline = time.monotonic() + command_preview_s + 0.15
            while True:
                check_torque("while settling after streamed move")
                remaining = settle_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.05, remaining))
            check_torque("after streamed move settled")
        except BaseException as exc:
            motion_error = exc
            raise
        finally:
            try:
                self._finish_cartesian_motion()
            except BaseException as cleanup_error:
                if motion_error is None:
                    raise
                _note_exception(
                    motion_error,
                    f"right-arm cleanup also failed: {cleanup_error!r}"
                )

        after = np.asarray(
            self.rpc.get_right_ee_pose().parameters(), dtype=float
        )
        if after.shape != (7,) or not np.all(np.isfinite(after)):
            raise RuntimeError("invalid right pose after Cartesian move")
        actual = after[4:7] - before[4:7]
        if np.linalg.norm(requested) > 0.001:
            progress = float(
                np.dot(actual, requested)
                / np.dot(requested, requested)
            )
            if progress < minimum_progress:
                raise RuntimeError(
                    f"right arm did not follow request: {progress:.2f}"
                )
        return actual

    def move_joint_delta(
        self,
        delta_joints,
        preview_time=None,
        minimum_progress=None,
        *,
        accumulate=False,
    ):
        if preview_time is None:
            preview_time = self.args.preview_time
        if minimum_progress is None:
            minimum_progress = self.args.joint_minimum_progress
        before = np.asarray(
            self.rpc.get_right_joint_positions(), dtype=float
        )
        if accumulate:
            if self.joint_command is None:
                self.joint_command = before.copy()
            target = self.joint_command.copy()
            # The wrist is never accumulated or commanded away from its
            # measured state by this position-only visual servo.
            target[3:] = before[3:]
        else:
            target = before.copy()
        target[:3] += np.asarray(delta_joints, dtype=float)
        self.rpc.set_right_joint_target(
            target,
            gripper_target=None,
            preview_time=preview_time,
        )
        self.monitor_settle(preview_time + 0.15)
        if accumulate:
            self.joint_command = target.copy()
        after = np.asarray(
            self.rpc.get_right_joint_positions(), dtype=float
        )
        actual = after[:3] - before[:3]
        requested = np.asarray(delta_joints, dtype=float)
        if np.linalg.norm(requested) > 0.003:
            progress = np.linalg.norm(actual) / np.linalg.norm(requested)
            if progress < minimum_progress:
                self.hold_measured()
                raise RuntimeError(
                    f"right joints did not follow request: {progress:.2f}"
                )
        return actual

    def move_control_delta(
        self, delta, *, minimum_progress=None, accumulate=False
    ):
        if self.args.control_space == "joint":
            kwargs = {}
            if minimum_progress is not None:
                kwargs["minimum_progress"] = minimum_progress
            return self.move_joint_delta(
                delta, accumulate=accumulate, **kwargs
            )
        return self.move_cartesian_delta(
            delta, minimum_progress=minimum_progress
        )

    def calibrate(self, clearance_m: float):
        if self.args.control_space == "joint":
            origin = np.asarray(
                self.rpc.get_right_joint_positions(), dtype=float
            )
            probe_sizes = np.array(
                [
                    self.args.joint_probe_rad,
                    self.args.joint_probe2_rad,
                    self.args.joint_probe3_rad,
                ],
                dtype=float,
            )
            calibration_min_progress = (
                self.args.joint_calibration_min_progress
            )
        else:
            origin = np.asarray(
                self.rpc.get_right_ee_pose().parameters(), dtype=float
            )
            self.orientation = origin[:4].copy()
            probe_sizes = np.full(3, self.args.probe_m, dtype=float)
            calibration_min_progress = 0.25
        robot_deltas = []
        feature_deltas = []
        for axis in range(3):
            probe_size = float(probe_sizes[axis])
            baseline = observed = None
            actual = None
            path = None
            direction = None
            for sign in (1.0, -1.0):
                baseline, _, _, _ = self.observe(clearance_m)
                probe = np.zeros(3)
                probe[axis] = sign * probe_size
                actual = self.move_control_delta(
                    probe, minimum_progress=0.0
                )
                progress = float(np.linalg.norm(actual) / probe_size)
                if progress >= calibration_min_progress:
                    observed, _, path, _ = self.observe(clearance_m)
                    direction = sign
                    break
                self.hold_measured()
            if direction is None:
                raise RuntimeError(
                    f"right arm could not produce probe motion on axis {axis}"
                )
            robot_deltas.append(actual)
            feature_deltas.append(
                observed.gripper_feature - baseline.gripper_feature
            )
            print(
                json.dumps(
                    {
                        "control_space": self.args.control_space,
                        "probe_axis": axis,
                        "probe_sign": direction,
                        "actual_m": actual.round(5).tolist(),
                        "feature_delta": feature_deltas[-1].round(2).tolist(),
                        "image": path,
                    }
                ),
                flush=True,
            )
            # Do not force a return to the exact origin between probes. Near a
            # joint branch boundary the reverse IK can fail even though each
            # small forward probe is valid. Differential measurements from
            # consecutive fresh observations are sufficient for the Jacobian.
        model = estimate_reachable_feature_model(robot_deltas, feature_deltas)
        print(
            json.dumps(
                {
                    "reachable_rank": model.rank,
                    "reachable_basis": model.basis_xyz.round(4).tolist(),
                    "feature_matrix": model.feature_matrix.round(2).tolist(),
                    "condition": model.condition,
                }
            ),
            flush=True,
        )
        return robot_deltas, feature_deltas, origin

    def approach(
        self, robot_deltas, feature_deltas, origin, clearance_m: float
    ):
        last_error_norm = None
        worse = 0
        tolerances = np.array(
            [
                self.args.xy_tolerance_px,
                self.args.xy_tolerance_px,
                self.args.depth_tolerance_mm,
            ],
            dtype=float,
        )
        for iteration in range(self.args.max_iters):
            feature, error, path, timestamp = self.observe(clearance_m)
            xy_ok = float(np.linalg.norm(error[:2])) <= self.args.xy_tolerance_px
            depth_ok = abs(float(error[2])) <= self.args.depth_tolerance_mm
            report = {
                "iteration": iteration,
                "error": error.round(2).tolist(),
                "right_pose": np.asarray(
                    self.rpc.get_right_ee_pose().parameters()
                ).round(6).tolist(),
                "image": path,
                "timestamp": timestamp,
            }
            print(json.dumps(report), flush=True)
            if xy_ok and depth_ok:
                return feature, path
            error_norm = float(
                np.linalg.norm(error / tolerances)
            )
            if last_error_norm is not None and error_norm > 1.25 * last_error_norm:
                worse += 1
                if worse >= 2:
                    self.hold_measured()
                    raise RuntimeError("real-time SAM error diverged")
            else:
                worse = 0
            model = estimate_reachable_feature_model(
                robot_deltas, feature_deltas
            )
            step = bounded_reachable_servo_step(
                model,
                error,
                tolerances=tolerances,
                max_norm_m=(
                    self.args.joint_max_step_rad
                    if self.args.control_space == "joint"
                    else self.args.max_step_m
                ),
                max_axis_m=(
                    self.args.joint_max_axis_rad
                    if self.args.control_space == "joint"
                    else self.args.max_axis_m
                ),
            )
            report = {
                "reachable_rank": model.rank,
                "condition": model.condition,
                "step_m": step.round(5).tolist(),
            }
            print(json.dumps(report), flush=True)
            if self.args.control_space == "cartesian":
                current = np.asarray(
                    self.rpc.get_right_ee_pose().parameters(), dtype=float
                )
                if (
                    np.linalg.norm(
                        current[4:7] + step - origin[4:7]
                    )
                    > 0.20
                ):
                    self.hold_measured()
                    raise RuntimeError(
                        "right-arm SAM servo excursion exceeded 200mm"
                    )
            before_gripper = feature.gripper_feature.copy()
            actual = self.move_control_delta(step, accumulate=True)
            after, _, _, _ = self.observe(clearance_m)
            delta_feature = after.gripper_feature - before_gripper
            minimum_update = (
                5e-4
                if self.args.control_space == "joint"
                else 3e-4
            )
            if np.linalg.norm(actual) >= minimum_update:
                robot_deltas.append(actual)
                feature_deltas.append(delta_feature)
                # Refit from recent real-time observations.  This adapts to
                # local geometry while zero-motion depth noise cannot erase
                # the last useful excitation.
                robot_deltas[:] = robot_deltas[-6:]
                feature_deltas[:] = feature_deltas[-6:]
            last_error_norm = error_norm
        self.hold_measured()
        raise RuntimeError("real-time SAM pregrasp did not converge")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:15563")
    parser.add_argument("--output-dir", default="/tmp/realtime_sam_grasp")
    parser.add_argument(
        "--torque-config", default="src/configs/pasteur_lid_torque.json"
    )
    parser.add_argument(
        "--scene-config",
        default="src/configs/pasteur_lid_scene3d.json",
    )
    parser.add_argument("--reference-points")
    parser.add_argument("--probe-m", type=float, default=0.008)
    parser.add_argument(
        "--control-space",
        choices=("cartesian", "joint"),
        default="cartesian",
    )
    parser.add_argument("--joint-probe-rad", type=float, default=0.025)
    parser.add_argument("--joint-probe2-rad", type=float, default=0.025)
    parser.add_argument("--joint-probe3-rad", type=float, default=0.025)
    parser.add_argument("--joint-max-step-rad", type=float, default=0.035)
    parser.add_argument("--joint-max-axis-rad", type=float, default=0.025)
    parser.add_argument(
        "--joint-minimum-progress", type=float, default=0.25
    )
    parser.add_argument(
        "--joint-calibration-min-progress", type=float, default=0.05
    )
    parser.add_argument("--clearance-m", type=float, default=0.040)
    parser.add_argument("--max-step-m", type=float, default=0.012)
    parser.add_argument("--max-axis-m", type=float, default=0.008)
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument("--minimum-progress", type=float, default=0.15)
    parser.add_argument("--depth-frames", type=int, default=15)
    parser.add_argument("--preview-time", type=float, default=1.2)
    parser.add_argument("--xy-tolerance-px", type=float, default=6.0)
    parser.add_argument("--depth-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--right-can-interface", default="can_right")
    parser.add_argument(
        "--holding-kp", type=float, nargs=6, default=DEFAULT_HOLDING_KP
    )
    parser.add_argument(
        "--holding-kd", type=float, nargs=6, default=DEFAULT_HOLDING_KD
    )
    parser.add_argument(
        "--motion-kp", type=float, nargs=6, default=DEFAULT_MOTION_KP
    )
    parser.add_argument(
        "--motion-kd", type=float, nargs=6, default=DEFAULT_MOTION_KD
    )
    parser.add_argument("--gain-ramp-s", type=float, default=1.0)
    parser.add_argument("--mode-settle-s", type=float, default=0.5)
    parser.add_argument("--hold-settle-s", type=float, default=0.25)
    parser.add_argument("--execute-pregrasp", action="store_true")
    args = parser.parse_args()

    runner = LiveSamGrasp(args)
    try:
        runner.start()
        # The first Record3D depth burst after stream startup is often an
        # outlier; warm the live temporal filter before reporting or moving.
        runner.observe(args.clearance_m)
        feature, error, path, timestamp = runner.observe(args.clearance_m)
        registration = runner.check_scene_registration(
            args.reference_points
        )
        initial = {
            "mode": "execute" if args.execute_pregrasp else "dry_run",
            "error": error.round(2).tolist(),
            "image": path,
            "timestamp": timestamp,
            "lid_center": feature.lid_geometry.center_px.round(2).tolist(),
            "lid_score": feature.lid_candidate.score,
            "gripper_score": feature.gripper_candidate.score,
            "proximity_warning_only_m": runner.last_proximity_m,
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
                "reasons": list(runner.last_geometry_quality.reasons),
            },
            "registration": registration,
        }
        print(json.dumps(initial), flush=True)
        if not args.execute_pregrasp:
            return
        if not runner.last_geometry_quality.accepted:
            raise RuntimeError(
                "camera geometry is too oblique for autonomous motion: "
                + "; ".join(runner.last_geometry_quality.reasons)
            )
        robot_deltas, feature_deltas, origin = runner.calibrate(
            args.clearance_m
        )
        _, final_path = runner.approach(
            robot_deltas, feature_deltas, origin, args.clearance_m
        )
        print(
            json.dumps(
                {
                    "status": "CONTACT_CONFIRMATION_REQUIRED",
                    "image": final_path,
                }
            ),
            flush=True,
        )
    except Exception:
        try:
            runner.hold_measured()
        except Exception:
            pass
        raise
    finally:
        runner.stop()


if __name__ == "__main__":
    main()

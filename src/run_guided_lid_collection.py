#!/usr/bin/env python3
"""Run deterministic, operator-guided lid demonstration collection.

This entry point never opens inference sockets and never loads an ACT model.
It shares the proven ConeE Cartesian command path, cameras, controller lock,
and crash-tolerant 30 Hz episode recorder with the existing collector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import cv2
import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.camera_id import configure_camera_map_by_udid
from robot.rpc import RPCClient
from rollout.agent_collection import (
    AgentEpisodeRecorder,
    AgentRecordingSample,
    ControllerClaim,
)
from rollout.agent_home import _agent_pressure_guard, run_agent_auto_home
from rollout.camera import CameraFeedManager, USBWristCameraFeedManager
from rollout.guided_lid_collection import (
    BaselineTrajectory,
    CollectionPhase,
    GuidedLidCycle,
    PoseCommand,
    build_grasp_prefix,
    build_reposition_commands,
    rebase_post_grasp,
    sample_pose_segment,
)
from rollout.guided_lid_ui import GuidedLidUI
from rollout.gripper_level import (
    JawLevelReference,
    assess_jaw_level,
    signed_outward_tip_pitch_deg,
)
from rollout.pose_locked_descent import PhysicalRightPoseLockedPlanner
from rollout.recovery_teleop_safety import RecoveryTorqueGuard
from rollout.teleop_trajectory_stream import (
    DEFAULT_HOLDING_KD,
    DEFAULT_HOLDING_KP,
    DEFAULT_MOTION_KD,
    DEFAULT_MOTION_KP,
    refresh_right_mit_mode,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


class GuidedPoseExecutor:
    """Stream finite Cartesian commands at 30 Hz with pressure hold."""

    def __init__(self, rpc, guard, *, stop_requested, on_command,
                 pressure_sample_provider):
        self.rpc = rpc
        self.guard = guard
        self.stop_requested = stop_requested
        self.on_command = on_command
        self.pressure_sample_provider = pressure_sample_provider

    def _hold(self) -> None:
        measured = np.asarray(self.rpc.get_right_joint_positions(), dtype=float)
        self.rpc.set_right_joint_target(
            measured, gripper_target=None, preview_time=0.2
        )
        self.rpc.set_right_gain(DEFAULT_HOLDING_KP, DEFAULT_HOLDING_KD)

    @staticmethod
    def _validate(commands: Sequence[PoseCommand]) -> None:
        if not commands:
            raise ValueError("cannot execute an empty command sequence")
        previous = -1.0
        descent_xy = []
        for command in commands:
            if command.t_s < 0 or command.t_s <= previous:
                raise ValueError("command times must be strictly increasing")
            previous = command.t_s
            pose = command.pose()
            if not 0.0 <= command.gripper_open_ratio <= 1.0:
                raise ValueError("gripper command is outside [0, 1]")
            if command.stage == "continuous_descent":
                descent_xy.append(pose[4:6])
        if descent_xy and np.ptp(np.stack(descent_xy), axis=0).max() > 1e-9:
            raise ValueError("continuous descent contains a low planar move")

    def execute(self, commands: Sequence[PoseCommand]) -> dict[str, Any]:
        self._validate(commands)
        self.stop_requested.clear()
        self.guard.reset("right")
        measured = np.asarray(self.rpc.get_right_joint_positions(), dtype=float)
        self.rpc.set_right_joint_target(
            measured, gripper_target=None, preview_time=0.2
        )
        self.rpc.set_right_gain(DEFAULT_HOLDING_KP, DEFAULT_HOLDING_KD)
        refresh_right_mit_mode()
        self.rpc.set_right_gain(DEFAULT_MOTION_KP, DEFAULT_MOTION_KD)
        started = time.monotonic()
        sent = 0
        stages = []
        try:
            for command in commands:
                if self.stop_requested.is_set():
                    raise RuntimeError("operator stop")
                torque, age_s = self.pressure_sample_provider()
                if torque is None or age_s > 0.15:
                    raise RuntimeError(
                        f"fresh pressure sample unavailable (age={age_s:.3f}s)"
                    )
                if not self.guard.check("right", torque_sample=torque):
                    raise RuntimeError(
                        "pressure guard stopped motion: "
                        + json.dumps(self.guard.latched.get("right", {}))
                    )
                target_time = started + command.t_s
                remaining = target_time - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                pose = mink.SE3(command.pose())
                accepted = self.rpc.set_right_ee_target(
                    pose,
                    gripper_target=command.gripper_open_ratio,
                    preview_time=0.05,
                )
                if accepted is not True:
                    raise RuntimeError(
                        f"ConeE rejected deterministic stage {command.stage}"
                    )
                self.on_command(command)
                sent += 1
                if not stages or stages[-1] != command.stage:
                    stages.append(command.stage)
        finally:
            self._hold()
        return {"commands_sent": sent, "stages": stages}


class GuidedLidRuntime:
    def __init__(self, profile: Mapping[str, Any]):
        self.profile = dict(profile)
        if self.profile.get("policy", {}).get("inference_enabled") is not False:
            raise ValueError("guided collector requires inference_enabled=false")
        if int(self.profile.get("control_frequency_hz", 0)) != 30:
            raise ValueError("guided collection is fixed at 30 Hz")
        self.stop_event = threading.Event()
        self.motion_stop = threading.Event()
        self.command_queue: queue.Queue[tuple[int, str, dict]] = queue.Queue()
        self.command_id = 0
        self.lock = threading.RLock()
        self.metrics: dict[str, Any] = {"runner": "starting", "last_error": None}
        sweep = tuple(int(v) for v in self.profile["sweep"]["physical_right_mm"])
        self.cycle = GuidedLidCycle(sweep_right_mm=sweep)
        self.latest_state = None
        self.latest_right_torque = None
        self.latest_right_torque_at = None
        self.latest_frames: dict[str, tuple[np.ndarray | None, float | None]] = {}
        self.current_commanded = np.full(16, np.nan, dtype=float)
        self.safety_rejected_count = 0
        self.attempt_started_s = None
        self.review_started_s = None

        storage = self.profile["storage"]
        self.claim = ControllerClaim(storage["controller_lock"])
        self.claim.acquire()
        self.command_rpc = RPCClient("localhost", 8081, timeout_ms=10000)
        self.command_rpc.init()
        self.observation_rpc = RPCClient("localhost", 8081, timeout_ms=10000)
        self.observation_rpc.init()

        torque_path = _resolve(self.profile["motion"]["pressure_guard"]["config"])
        self.guard = _agent_pressure_guard(
            self.command_rpc,
            torque_path,
            "/var/tmp/piper-agent-collection/guided_lid_pressure.jsonl",
            {"primitive": "guided_lid_collection"},
        )
        if not bool(self.profile["motion"]["pressure_guard"].get("enforce", True)):
            raise ValueError("guided lid pressure guard must remain enforced")
        self.executor = GuidedPoseExecutor(
            self.command_rpc,
            self.guard,
            stop_requested=self.motion_stop,
            on_command=self._note_command,
            pressure_sample_provider=self._pressure_sample,
        )

        locked = self.profile["pose_locked_descent"]
        level = json.loads(_resolve(locked["level_config"]).read_text())
        self.level_reference = JawLevelReference(
            support_up_robot=tuple(level["support_up_robot"]),
            tip_baseline_ee=tuple(level["tip_baseline_ee"]),
            approach_axis_ee=tuple(level["approach_axis_ee"]),
            open_tip_span_m=float(level["open_tip_span_m"]),
            maximum_checkpoint_tilt_deg=float(
                level["maximum_checkpoint_tilt_deg"]
            ),
            maximum_planned_tilt_deg=float(level["maximum_planned_tilt_deg"]),
            maximum_tip_height_difference_m=float(
                level["maximum_tip_height_difference_m"]
            ),
        )
        self.pose_locked_planner = PhysicalRightPoseLockedPlanner(
            _resolve(locked["production_model"]), self.level_reference
        )

        self.baselines = {
            task: BaselineTrajectory.load(_resolve(path), task=task)
            for task, path in self.profile["baselines"].items()
        }
        if set(self.baselines) != {"lid_open", "lid_close"}:
            raise ValueError("both lid_open and lid_close baselines are required")

        cam_map, live = configure_camera_map_by_udid(
            self.profile["camera_udids"], preserve_unknown=False
        )
        self.metrics["camera_map"] = cam_map
        self.metrics["camera_udids_live"] = live
        self.head = CameraFeedManager(
            self.stop_event, display=False, head_stream=False
        )
        self.right = USBWristCameraFeedManager(
            self.stop_event, device_index=cam_map["right"], label="right wrist"
        )
        self.left = USBWristCameraFeedManager(
            self.stop_event, device_index=cam_map["left"], label="left wrist"
        )
        self.head.start(); self.right.start(); self.left.start()
        self.recorder = AgentEpisodeRecorder(
            _resolve(storage["root"]), self.stop_event, fps=30
        )

        ui_config = self.profile["ui"]
        self.ui = GuidedLidUI(
            ui_config["host"], ui_config["port"], ui_config.get("token", ""),
            self.snapshot, self.submit, self.frame_provider,
        )
        self.ui.start()
        self.observation_thread = threading.Thread(
            target=self._observation_loop, daemon=True
        )
        self.worker_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.observation_thread.start(); self.worker_thread.start()
        with self.lock:
            self.metrics["runner"] = "ready"
            self.metrics["ui_url"] = f"http://{ui_config['host']}:{self.ui.port}/"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"cycle": self.cycle.snapshot(), "metrics": dict(self.metrics)}

    def submit(self, command: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "home", "start", "adjust", "success", "failure",
            "enable_auto", "stop", "jog", "level_descend", "close_review",
            "open_review",
        }
        if command not in allowed:
            raise ValueError(f"unsupported command {command!r}")
        if command == "stop":
            # STOP must interrupt the active streamer instead of waiting
            # behind it in the command queue.
            self.motion_stop.set()
        with self.lock:
            self.command_id += 1
            request_id = self.command_id
        self.command_queue.put_nowait((request_id, command, dict(payload)))
        return {"accepted": True, "request_id": request_id, "command": command}

    def frame_provider(self, camera: str):
        with self.lock:
            frame, timestamp = self.latest_frames.get(camera, (None, None))
            if frame is None:
                return None, None
            shown = np.rot90(frame.copy(), k=3)
            if shown.shape[:2] != (480, 640):
                shown = cv2.resize(shown, (640, 480), interpolation=cv2.INTER_AREA)
            return shown, int(float(timestamp) * 1e9) if timestamp else 0

    def _note_command(self, command: PoseCommand) -> None:
        target = np.full(16, np.nan, dtype=float)
        target[7:14] = command.pose()
        target[15] = command.gripper_open_ratio
        with self.lock:
            self.current_commanded = target
            self.metrics["stage"] = command.stage

    def _pressure_sample(self):
        with self.lock:
            torque = (
                None
                if self.latest_right_torque is None
                else self.latest_right_torque.copy()
            )
            sampled_at = self.latest_right_torque_at
        if torque is None:
            # Rolling deployment fallback.  It is safe but may make the
            # episode cadence-ineligible; restarting ConeE loads the coherent
            # torque snapshot path and restores the intended 30 Hz behavior.
            torque = np.asarray(
                self.command_rpc.get_right_joint_torque(), dtype=float
            )
            with self.lock:
                self.metrics["pressure_source"] = "legacy_synchronous_fallback"
            return torque, 0.0
        age = float("inf") if sampled_at is None else time.monotonic() - sampled_at
        return torque, age

    def _camera_samples(self):
        head_frame, head_ts, _ = self.head.get_latest_frame()
        left_frame, left_ts, _ = self.left.get_latest_frame()
        right_frame, right_ts, _ = self.right.get_latest_frame()
        with self.lock:
            self.latest_frames = {
                "head": (head_frame, head_ts),
                "left": (left_frame, left_ts),
                "right": (right_frame, right_ts),
            }
        return (head_frame, left_frame, right_frame), (head_ts, left_ts, right_ts)

    @staticmethod
    def _finite_timestamp(value):
        return np.nan if value is None else float(value)

    @staticmethod
    def _frame_id(value):
        return -1 if value is None else int(float(value) * 1e9)

    def _observation_loop(self) -> None:
        period = 1.0 / 30.0
        deadline = time.monotonic()
        last_left_gripper = 1.0
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                state = self.observation_rpc.get_observation_state(active_arm="right")
                if state.get("left_gripper_exact") is None:
                    state["left_gripper_exact"] = last_left_gripper
                else:
                    last_left_gripper = float(state["left_gripper_exact"])
                frames, camera_ts = self._camera_samples()
                with self.lock:
                    self.latest_state = state
                    torque = state.get("right_joint_torque")
                    if torque is not None:
                        torque = np.asarray(torque, dtype=float)
                        if torque.shape == (6,) and np.all(np.isfinite(torque)):
                            self.latest_right_torque = torque.copy()
                            self.latest_right_torque_at = time.monotonic()
                    commanded = self.current_commanded.copy()
                    phase = self.cycle.phase
                if self.recorder.is_recording and phase in {
                    CollectionPhase.EXECUTING,
                    CollectionPhase.COMPLETING,
                }:
                    right_exact = float(state["right_gripper_exact"])
                    left_exact = float(state["left_gripper_exact"])
                    now = time.monotonic()
                    sample = AgentRecordingSample(
                        wall_timestamp=float(state.get("sampled_at", time.time())),
                        active_timestamp=self.recorder.active_timestamp(now=now),
                        left_ee_pose=state["left_ee_pose"],
                        right_ee_pose=state["right_ee_pose"],
                        left_gripper_exact=left_exact,
                        right_gripper_exact=right_exact,
                        left_gripper=float(left_exact >= 0.5),
                        right_gripper=float(right_exact >= 0.5),
                        left_joint_positions=np.asarray(state["left_joint_positions"]),
                        right_joint_positions=np.asarray(state["right_joint_positions"]),
                        head_rgb=frames[0], left_rgb=frames[1], right_rgb=frames[2],
                        camera_timestamps=tuple(
                            self._finite_timestamp(value) for value in camera_ts
                        ),
                        camera_frame_ids=tuple(
                            self._frame_id(value) for value in camera_ts
                        ),
                        policy_action_quat16=np.full(16, np.nan),
                        commanded_target_quat16=commanded,
                        xyz_bias_left_right=np.zeros(6),
                        chunk_index=-1,
                        action_generation=-1,
                        action_observation_timestamp=np.nan,
                        intervention_revision=self.command_id,
                        safety_rejected_count=self.safety_rejected_count,
                    )
                    self.recorder.record_sample(sample)
                with self.lock:
                    self.metrics["right_gripper_exact"] = float(
                        state["right_gripper_exact"]
                    )
                    self.metrics["observation_age_ms"] = round(
                        (time.monotonic() - started) * 1000.0, 1
                    )
                    if (
                        self.cycle.phase == CollectionPhase.REVIEW
                        and self.review_started_s is not None
                    ):
                        review_age = time.monotonic() - self.review_started_s
                        timeout = float(
                            self.profile["motion"]["grasp_review_timeout_s"]
                        )
                        self.metrics["review_age_s"] = round(review_age, 1)
                        self.metrics["review_timeout_reached"] = (
                            review_age >= timeout
                        )
            except Exception as error:
                with self.lock:
                    self.metrics["last_error"] = (
                        f"observation:{type(error).__name__}:{error}"
                    )
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                if self.recorder.is_recording:
                    self.recorder.note_deadline_miss()
                deadline = time.monotonic()

    def _measured_pose(self) -> np.ndarray:
        with self.lock:
            state = self.latest_state
        if state is None:
            pose = self.command_rpc.get_right_ee_pose()
        else:
            pose = state["right_ee_pose"]
        return np.asarray(pose.parameters(), dtype=float)

    def _start_recording(self, effective_correction: np.ndarray) -> None:
        baseline = self.baselines[self.cycle.task]
        context = {
            "task": self.cycle.task,
            "target_selection": {
                "source": "reviewed_success_baseline",
                "baseline_hdf5": baseline.source,
                "baseline_sha256": baseline.source_sha256,
            },
            "initial_bias_m": {
                "left": [0.0, 0.0, 0.0],
                "right": effective_correction.tolist(),
            },
            "policy_mode": "deterministic_operator_guided",
            "inference_enabled": False,
            "required_cameras": ["head", "right"],
            "control_frequency_hz": 30,
            "placement_right_mm": self.cycle.placement_right_mm,
            "operator_correction_m": self.cycle.active_attempt_correction_m.tolist(),
            "effective_pick_correction_m": effective_correction.tolist(),
            "descent_speed_m_s": float(self.profile["motion"]["descent_speed_m_s"]),
            "automatic_depth_search": False,
        }
        self.recorder.configure_episode(context)
        self.recorder.start_episode()
        self.recorder.set_sampling_active(True)

    def _execute(self, commands: Sequence[PoseCommand]) -> dict[str, Any]:
        self._rearm_local_pressure_guard("cartesian_pose_stream")
        if self.recorder.is_recording:
            self.recorder.set_sampling_active(True)
        try:
            return self.executor.execute(commands)
        finally:
            if self.recorder.is_recording:
                self.recorder.set_sampling_active(False)

    def _rearm_local_pressure_guard(self, stage: str) -> None:
        """Rebase pressure safety on the current static gravity load.

        The global absolute envelope was calibrated at other poses.  At a low
        wrist pose its joint-1 gravity torque can already exceed the global
        hard threshold before motion.  Keep a hard stop, but place it only
        0.08 Nm beyond ten fresh stationary samples; sustained local residual
        protection remains the tighter contact-sensitive guard.
        """
        path = _resolve(self.profile["motion"]["pressure_guard"]["config"])
        config = json.loads(path.read_text())
        original = np.asarray(config["thresholds"]["right"], dtype=float)
        recovery = config["recovery_teleop"]
        hard_multiplier = float(recovery["hard_limit_multiplier"])
        samples = []
        for _ in range(10):
            samples.append(np.abs(np.asarray(
                self.command_rpc.get_right_joint_torque(), dtype=float
            )))
            time.sleep(0.005)
        observed = np.max(np.stack(samples), axis=0)
        adjusted = np.maximum(
            original, (observed + 0.08) / hard_multiplier
        )
        thresholds = {"left": original.copy(), "right": adjusted}
        self.guard = RecoveryTorqueGuard(
            self.command_rpc,
            thresholds,
            consecutive_samples=int(config.get("consecutive_samples", 5)),
            preview_time_s=0.05,
            residual_fraction=float(recovery["residual_fraction"]),
            residual_floor_nm=float(recovery["residual_floor_nm"]),
            residual_ceiling_nm=float(recovery["residual_ceiling_nm"]),
            residual_duration_s=float(recovery["residual_duration_s"]),
            baseline_slew_nm_per_s=float(recovery["baseline_slew_nm_per_s"]),
            hard_limit_multiplier=hard_multiplier,
            enforce=True,
            audit_path=(
                "/var/tmp/piper-agent-collection/"
                "guided_lid_local_pressure.jsonl"
            ),
            provenance={
                "primitive": "guided_pose_locked_motion",
                "stage": stage,
                "observed_static_abs_nm": observed.tolist(),
                "adjusted_threshold_nm": adjusted.tolist(),
            },
        )
        self.executor.guard = self.guard

    def _execute_joint_path(
        self,
        expected_start: Sequence[float],
        path: Sequence[Sequence[float]],
        *,
        stage: str,
        convergence_tolerance_rad: float = 0.08,
    ) -> dict[str, Any]:
        if len(path) == 0:
            raise ValueError("joint path is empty")
        expected = np.asarray(expected_start, dtype=float)
        measured = np.asarray(
            self.command_rpc.get_right_joint_positions(), dtype=float
        )
        start_error = float(np.max(np.abs(measured - expected)))
        if start_error > 0.05:
            raise RuntimeError(
                f"pose-locked {stage} start changed by {start_error:.3f} rad"
            )
        # Time-scale the geometric path to a precise low-speed stream.  The
        # previous one-sample-per-millimetre version outran the low-gain arm,
        # so the final measured jaw attitude lagged behind the audited plan.
        maximum_speed_rad_s = 0.06
        maximum_step = maximum_speed_rad_s / 30.0
        dense_path = []
        previous = expected.copy()
        for waypoint in path:
            waypoint = np.asarray(waypoint, dtype=float)
            count = max(
                1,
                int(math.ceil(
                    float(np.max(np.abs(waypoint - previous))) / maximum_step
                )),
            )
            for index in range(1, count + 1):
                dense_path.append(
                    previous + (waypoint - previous) * (index / count)
                )
            previous = waypoint
        self._rearm_local_pressure_guard(stage)
        self.motion_stop.clear()
        self.rpc_hold_for_joint_path()
        refresh_right_mit_mode()
        self.command_rpc.set_right_gain(DEFAULT_MOTION_KP, DEFAULT_MOTION_KD)
        started = time.monotonic()
        sent = 0
        try:
            for index, target in enumerate(dense_path, start=1):
                if self.motion_stop.is_set():
                    raise RuntimeError("operator stop")
                torque, age_s = self._pressure_sample()
                if torque is None or age_s > 0.15:
                    raise RuntimeError("fresh pressure unavailable")
                if not self.guard.check("right", torque_sample=torque):
                    raise RuntimeError(
                        "pressure guard stopped pose-locked motion: "
                        + json.dumps(self.guard.latched["right"])
                    )
                self.command_rpc.set_right_joint_target(
                    np.asarray(target, dtype=float),
                    gripper_target=None,
                    preview_time=0.05,
                )
                sent += 1
                deadline = started + index / 30.0
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            # Give the low-gain physical controller time to reach the exact
            # final joint target before freezing measured state.  Without
            # this dwell, the following level checkpoint measures tracking
            # lag rather than the planned wrist attitude.
            final_target = np.asarray(dense_path[-1], dtype=float)
            converged = 0
            final_error = float("inf")
            for dwell in range(45):
                torque, age_s = self._pressure_sample()
                if torque is None or age_s > 0.15:
                    raise RuntimeError("fresh pressure unavailable during settle")
                if not self.guard.check("right", torque_sample=torque):
                    raise RuntimeError(
                        "pressure guard stopped pose-locked settle: "
                        + json.dumps(self.guard.latched["right"])
                    )
                self.command_rpc.set_right_joint_target(
                    final_target, gripper_target=None, preview_time=0.05
                )
                sent += 1
                measured_now = np.asarray(
                    self.command_rpc.get_right_joint_positions(), dtype=float
                )
                final_error = float(np.max(np.abs(measured_now - final_target)))
                # Joint-space calibration has ~0.03 rad steady-state error at
                # this low-load branch.  The authoritative gate below is the
                # measured Cartesian fingertip height difference, not raw q.
                converged = (
                    converged + 1
                    if final_error <= convergence_tolerance_rad
                    else 0
                )
                if converged >= 3:
                    break
                deadline = started + (len(dense_path) + dwell + 1) / 30.0
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            if converged < 3:
                raise RuntimeError(
                    f"pose-locked {stage} tracking did not converge: "
                    f"error={final_error:.3f} rad"
                )
        finally:
            self.executor._hold()
        return {
            "stage": stage,
            "commands_sent": sent,
            "geometric_waypoints": len(path),
            "streamed_waypoints": len(dense_path),
            "start_error_rad": start_error,
            "final_tracking_error_rad": final_error,
            "maximum_speed_rad_s": maximum_speed_rad_s,
        }

    def rpc_hold_for_joint_path(self) -> None:
        measured = np.asarray(
            self.command_rpc.get_right_joint_positions(), dtype=float
        )
        self.command_rpc.set_right_joint_target(
            measured, gripper_target=None, preview_time=0.2
        )
        self.command_rpc.set_right_gain(DEFAULT_HOLDING_KP, DEFAULT_HOLDING_KD)

    def _handle_level_descend(self, payload: Mapping[str, Any]) -> None:
        if self.cycle.phase not in {CollectionPhase.READY, CollectionPhase.STOPPED}:
            raise RuntimeError("level descent requires ready/stopped")
        start_q = np.asarray(
            self.command_rpc.get_right_joint_positions(), dtype=float
        )
        locked = self.profile["pose_locked_descent"]
        lift_m = float(payload.get("lift_m", locked["recovery_lift_m"]))
        descent_m = float(payload.get("descent_m", locked["descent_m"]))
        level_only = bool(payload.get("level_only", False))
        if (
            not 0.0005 <= lift_m <= 0.04
            or (not level_only and not 0.001 <= descent_m <= 0.05)
        ):
            raise ValueError("pose-locked lift/descent override is invalid")
        if level_only:
            self._handle_feedback_level_only(start_q)
            return
        if bool(payload.get("resume_hover", False)):
            self._handle_feedback_descent_from_hover(start_q, descent_m)
            return
        plan = self.pose_locked_planner.plan(
            start_q,
            lift_m=lift_m,
            descent_m=descent_m,
        )
        lift = np.asarray(plan.lift_q_path, dtype=float)
        rebranch = np.asarray(plan.rebranch_q_path, dtype=float)
        descent = np.asarray(plan.descent_q_path, dtype=float)
        reports = [self._execute_joint_path(
            start_q, lift, stage="pose_locked_vertical_recovery"
        )]
        reports.append(self._execute_joint_path(
            lift[-1], rebranch, stage="stopped_hover_level_rebranch"
        ))
        measured_hover = np.asarray(
            self.command_rpc.get_right_ee_pose().parameters(), dtype=float
        )
        assessment = assess_jaw_level(measured_hover, self.level_reference)
        if (
            assessment.tip_height_difference_m
            > self.level_reference.maximum_tip_height_difference_m
        ):
            raise RuntimeError(
                "measured left/right jaw level rejected before descent: "
                f"{assessment.tip_height_difference_m * 1000:.2f} mm"
            )
        reports.append(self._execute_joint_path(
            rebranch[-1], descent, stage="pose_locked_vertical_descent"
        ))
        with self.lock:
            self.metrics["last_pose_locked_descent"] = {
                "reports": reports,
                "predicted_hover": plan.hover_assessment,
                "predicted_final": plan.final_assessment,
                "predicted_final_delta_xyz_m": list(
                    plan.predicted_final_delta_xyz_m
                ),
                "chosen_joint5_rebranch_rad": (
                    plan.chosen_joint5_rebranch_rad
                ),
                "measured_hover": assessment.to_dict(),
            }

    def _handle_feedback_descent_from_hover(
        self, start_q: np.ndarray, descent_m: float
    ) -> None:
        """Close the loop on measured jaw level and wrist margin at hover."""

        reports = []
        level_records = []
        for attempt in range(1, 6):
            measured_q = np.asarray(
                self.command_rpc.get_right_joint_positions(), dtype=float
            )
            measured_pose = np.asarray(
                self.command_rpc.get_right_ee_pose().parameters(), dtype=float
            )
            assessment = assess_jaw_level(
                measured_pose, self.level_reference
            )
            pitch = signed_outward_tip_pitch_deg(
                measured_pose, self.level_reference
            )
            level_records.append({
                "attempt": attempt,
                "joint5_rad": float(measured_q[4]),
                "tip_pitch_deg": float(pitch),
                "assessment": assessment.to_dict(),
            })
            level_ready = (
                assessment.tip_height_difference_m
                <= self.level_reference.maximum_tip_height_difference_m
            )
            if level_ready and pitch < 0.0:
                break
            # Correct only the left/right jaw baseline.  Do not force joint 5
            # back into the narrower MJCF nominal range: this calibrated arm
            # has repeatedly and safely reached -1.311 rad during teleop and
            # the required vertical path ends above that measured limit.
            target = self.pose_locked_planner._level_rebranch_target(
                measured_q,
                0.0,
            )
            path = self.pose_locked_planner._minimum_jerk_path(
                measured_q, target, maximum_step_rad=math.radians(1.0)
            )
            reports.append(self._execute_joint_path(
                measured_q,
                path,
                stage=f"feedback_hover_level_rebranch_{attempt}",
            ))
        else:
            raise RuntimeError("measured hover level/rebranch did not converge")

        measured_q = np.asarray(
            self.command_rpc.get_right_joint_positions(), dtype=float
        )
        descent = self.pose_locked_planner._vertical_path(
            measured_q, -abs(float(descent_m))
        )
        descent_array = np.asarray(descent, dtype=float)
        non_wrist5 = np.asarray([0, 1, 2, 3, 5], dtype=int)
        ranges = self.pose_locked_planner.ranges
        if (
            np.any(descent_array[:, non_wrist5]
                   < ranges[non_wrist5, 0] + 0.005)
            or np.any(descent_array[:, non_wrist5]
                      > ranges[non_wrist5, 1] - 0.005)
            or float(np.min(descent_array[:, 4])) < float(
                self.profile["pose_locked_descent"][
                    "measured_joint5_lower_rad"
                ]
            )
            or float(np.max(descent_array[:, 4])) > 1.20
        ):
            raise RuntimeError("measured hover has insufficient joint margin")
        reports.append(self._execute_joint_path(
            measured_q, descent, stage="feedback_pose_locked_vertical_descent"
        ))

        # One measured low-pose roll correction is allowed, but it must not
        # alter joint 5 or the already reached XYZ position.
        for attempt in range(1, 4):
            measured_q = np.asarray(
                self.command_rpc.get_right_joint_positions(), dtype=float
            )
            measured_pose = np.asarray(
                self.command_rpc.get_right_ee_pose().parameters(), dtype=float
            )
            assessment = assess_jaw_level(
                measured_pose, self.level_reference
            )
            if (
                assessment.tip_height_difference_m
                <= self.level_reference.maximum_tip_height_difference_m
            ):
                break
            target = self.pose_locked_planner._level_rebranch_target(
                measured_q, 0.0
            )
            # This wrist has a repeatable ~50% small-command deadband under
            # the low-load holding gain.  Apply the measured correction gain;
            # the next loop still accepts/rejects from the physical pose.
            target = measured_q + 2.0 * (target - measured_q)
            path = self.pose_locked_planner._minimum_jerk_path(
                measured_q, target, maximum_step_rad=math.radians(0.75)
            )
            reports.append(self._execute_joint_path(
                measured_q,
                path,
                stage=f"feedback_low_level_correction_{attempt}",
                convergence_tolerance_rad=0.02,
            ))
        final_q = np.asarray(
            self.command_rpc.get_right_joint_positions(), dtype=float
        )
        final_pose = np.asarray(
            self.command_rpc.get_right_ee_pose().parameters(), dtype=float
        )
        final_assessment = assess_jaw_level(final_pose, self.level_reference)
        if (
            final_assessment.tip_height_difference_m
            > self.level_reference.maximum_tip_height_difference_m
        ):
            raise RuntimeError(
                "measured low jaw level did not converge: "
                f"{final_assessment.tip_height_difference_m * 1000:.2f} mm"
            )
        with self.lock:
            self.metrics["last_pose_locked_descent"] = {
                "mode": "measured_feedback_from_hover",
                "start_q_rad": start_q.tolist(),
                "reports": reports,
                "level_records": level_records,
                "final_q_rad": final_q.tolist(),
                "final_assessment": final_assessment.to_dict(),
                "final_tip_pitch_deg": signed_outward_tip_pitch_deg(
                    final_pose, self.level_reference
                ),
            }

    def _handle_feedback_level_only(self, start_q: np.ndarray) -> None:
        """Level the left/right fingertips at the current Cartesian pose."""

        reports = []
        records = []
        for attempt in range(1, 6):
            measured_q = np.asarray(
                self.command_rpc.get_right_joint_positions(), dtype=float
            )
            measured_pose = np.asarray(
                self.command_rpc.get_right_ee_pose().parameters(), dtype=float
            )
            assessment = assess_jaw_level(measured_pose, self.level_reference)
            pitch = signed_outward_tip_pitch_deg(
                measured_pose, self.level_reference
            )
            records.append({
                "attempt": attempt,
                "assessment": assessment.to_dict(),
                "tip_pitch_deg": float(pitch),
            })
            if (
                assessment.tip_height_difference_m
                <= self.level_reference.maximum_tip_height_difference_m
                and pitch < 0.0
            ):
                break
            target = self.pose_locked_planner._level_rebranch_target(
                measured_q, 0.0
            )
            target = measured_q + 2.0 * (target - measured_q)
            path = self.pose_locked_planner._minimum_jerk_path(
                measured_q, target, maximum_step_rad=math.radians(0.5)
            )
            reports.append(self._execute_joint_path(
                measured_q,
                path,
                stage=f"feedback_level_only_{attempt}",
                convergence_tolerance_rad=0.02,
            ))
        final_pose = np.asarray(
            self.command_rpc.get_right_ee_pose().parameters(), dtype=float
        )
        final_assessment = assess_jaw_level(final_pose, self.level_reference)
        if (
            final_assessment.tip_height_difference_m
            > self.level_reference.maximum_tip_height_difference_m
        ):
            raise RuntimeError(
                "measured jaw level did not converge: "
                f"{final_assessment.tip_height_difference_m * 1000:.2f} mm"
            )
        with self.lock:
            self.metrics["last_pose_locked_descent"] = {
                "mode": "measured_level_only",
                "start_q_rad": start_q.tolist(),
                "reports": reports,
                "level_records": records,
                "final_assessment": final_assessment.to_dict(),
                "final_tip_pitch_deg": signed_outward_tip_pitch_deg(
                    final_pose, self.level_reference
                ),
            }

    def _handle_close_review(self) -> None:
        """Close at measured q; reopen automatically on an empty closure."""

        if self.cycle.phase not in {CollectionPhase.READY, CollectionPhase.STOPPED}:
            raise RuntimeError("close review requires ready/stopped")
        q_hold = np.asarray(
            self.command_rpc.get_right_joint_positions(), dtype=float
        )
        opening_start = float(self.command_rpc.get_right_gripper_exact())
        self._rearm_local_pressure_guard("level_grasp_close")
        self.motion_stop.clear()
        self.rpc_hold_for_joint_path()
        refresh_right_mit_mode()
        self.command_rpc.set_right_gain(DEFAULT_MOTION_KP, DEFAULT_MOTION_KD)
        try:
            started = time.monotonic()
            for index in range(1, 31):
                torque, age_s = self._pressure_sample()
                if torque is None or age_s > 0.15:
                    raise RuntimeError("fresh pressure unavailable during close")
                if not self.guard.check("right", torque_sample=torque):
                    raise RuntimeError(
                        "pressure guard stopped close: "
                        + json.dumps(self.guard.latched["right"])
                    )
                ratio = opening_start * (1.0 - index / 30.0)
                self.command_rpc.set_right_joint_target(
                    q_hold, gripper_target=ratio, preview_time=0.05
                )
                remaining = started + index / 30.0 - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            # The Dynamixel exact read briefly reports a large transient while
            # the jaws are still travelling.  Decide only after the readings
            # have settled; otherwise an empty close can look like a grasp.
            opening_samples = []
            for _ in range(10):
                time.sleep(0.1)
                opening_samples.append(float(
                    self.command_rpc.get_right_gripper_exact()
                ))
            measured_opening = float(np.median(opening_samples[-5:]))
            threshold = float(self.profile["motion"]["minimum_blocked_opening"])
            grasped = measured_opening >= threshold
            if not grasped:
                self.command_rpc.set_right_joint_target(
                    q_hold, gripper_target=1.0, preview_time=0.35
                )
                time.sleep(0.4)
            with self.lock:
                self.metrics["last_close_review"] = {
                    "grasped": grasped,
                    "measured_opening": measured_opening,
                    "blocked_opening_threshold": threshold,
                    "empty_close_reopened": not grasped,
                    "settled_opening_samples": opening_samples,
                }
        finally:
            self.executor._hold()

    def _handle_open_review(self) -> None:
        if self.cycle.phase not in {CollectionPhase.READY, CollectionPhase.STOPPED}:
            raise RuntimeError("open review requires ready/stopped")
        q_hold = np.asarray(
            self.command_rpc.get_right_joint_positions(), dtype=float
        )
        self.command_rpc.set_right_joint_target(
            q_hold, gripper_target=1.0, preview_time=0.35
        )
        time.sleep(0.5)
        self.executor._hold()

    def _handle_start(self) -> None:
        self.cycle.start_attempt()
        baseline = self.baselines[self.cycle.task]
        correction = self.cycle.active_attempt_correction_m.copy()
        if self.cycle.task == "lid_open":
            correction += self.cycle.placement_robot_xyz_m
        self._start_recording(correction)
        motion = self.profile["motion"]
        commands = build_grasp_prefix(
            self._measured_pose(), baseline, correction,
            hover_clearance_m=float(motion["hover_clearance_m"]),
            transit_speed_m_s=float(motion["transit_speed_m_s"]),
            descent_speed_m_s=float(motion["descent_speed_m_s"]),
            close_duration_s=float(motion["close_duration_s"]),
            verification_lift_m=float(motion["verification_lift_m"]),
        )
        self.attempt_started_s = time.monotonic()
        report = self._execute(commands)
        self.cycle.enter_review()
        self.review_started_s = time.monotonic()
        self.recorder.log_event(
            "grasp_review_ready",
            correction_m=correction.tolist(), report=report,
        )
        with self.lock:
            self.metrics["runner"] = "awaiting_grasp_review"
        if self.cycle.auto_enabled:
            opening = float(self.command_rpc.get_right_gripper_exact())
            threshold = float(motion["minimum_blocked_opening"])
            self.command_queue.put_nowait(
                (0, "success" if opening >= threshold else "failure",
                 {} if opening >= threshold else {"reason": "grasp_miss"})
            )

    def _recovery(self) -> None:
        pose = self._measured_pose()
        opening = float(self.command_rpc.get_right_gripper_exact())
        lifted = pose.copy(); lifted[6] += 0.02
        commands = sample_pose_segment(
            pose, lifted, duration_s=0.4, stage="failure_vertical_recovery",
            gripper_start=opening,
        )
        commands += sample_pose_segment(
            lifted, lifted, duration_s=0.5, stage="failure_open_at_hover",
            gripper_start=opening, gripper_end=1.0,
            start_t_s=commands[-1].t_s,
        )
        self._execute(commands)

    def _handle_failure(self, reason: str) -> None:
        if self.cycle.phase != CollectionPhase.REVIEW:
            raise RuntimeError("FAIL is accepted only at grasp review")
        self.recorder.log_event("operator_failure", reason=reason)
        self._recovery()
        self.cycle.fail()
        self.review_started_s = None
        self.recorder.end_episode()
        destination = self.recorder.finalize("failure", reason=reason)
        with self.lock:
            self.metrics.update({"runner": "ready", "last_episode": str(destination)})
        if self.cycle.auto_enabled and not self.motion_stop.is_set():
            self.command_queue.put_nowait((0, "start", {}))

    def _handle_success(self) -> None:
        if self.cycle.phase != CollectionPhase.REVIEW:
            raise RuntimeError("SUCCESS is accepted only at grasp review")
        self.cycle.succeed()
        self.review_started_s = None
        self.recorder.log_event("operator_success")
        measured_review = self._measured_pose()
        baseline = self.baselines[self.cycle.task]
        commands = rebase_post_grasp(baseline, measured_review)
        report = self._execute(commands)
        completed_task = self.cycle.task
        self.recorder.log_event("task_transport_complete", report=report)
        self.recorder.end_episode()
        destination = self.recorder.finalize("success")
        transition = self.cycle.task_complete()
        with self.lock:
            self.metrics.update({
                "runner": transition,
                "last_episode": str(destination),
                "last_task": completed_task,
            })
        if transition == "reposition":
            current = self._measured_pose()
            source_review = np.asarray(
                baseline.review_pose_wxyz_xyz, dtype=float
            )
            source_release = np.asarray(
                baseline.release_pose_wxyz_xyz, dtype=float
            )
            placement = source_release.copy()
            placement[4:7] = measured_review[4:7] + (
                source_release[4:7] - source_review[4:7]
            )
            displacement = (
                self.cycle.next_placement_robot_xyz_m
                - self.cycle.placement_robot_xyz_m
            )
            commands = build_reposition_commands(
                current,
                placement,
                displacement,
                lift_m=float(self.profile["motion"]["reposition_lift_m"]),
                speed_m_s=float(self.profile["motion"]["transit_speed_m_s"]),
            )
            reposition_report = self.executor.execute(commands)
            self.cycle.reposition_complete()
            with self.lock:
                self.metrics["last_reposition"] = {
                    "delta_m": displacement.tolist(),
                    "report": reposition_report,
                }
        if not self.motion_stop.is_set():
            self.command_queue.put_nowait((0, "start", {}))

    def _safe_home(self) -> None:
        if self.cycle.phase == CollectionPhase.REVIEW:
            self._handle_failure("abort")
        if self.cycle.phase not in {CollectionPhase.READY, CollectionPhase.STOPPED}:
            raise RuntimeError("HOME is allowed only while ready/stopped")
        report = run_agent_auto_home(
            self.command_rpc,
            torque_config_path=_resolve(
                self.profile["motion"]["pressure_guard"]["config"]
            ),
            audit_path="/var/tmp/piper-agent-collection/guided_lid_home.jsonl",
            control_hz=30.0,
        )
        with self.lock:
            self.metrics["last_home"] = report

    def _safe_jog(self, payload: Mapping[str, Any]) -> None:
        """Execute one explicit Cartesian displacement while ready.

        Semantic direction is deliberately fixed here: on Pasteur, physical
        ``toward the operator`` is robot X-negative.  Keeping the mapping in
        the adapter prevents UI wording from silently changing robot motion.
        """
        if self.cycle.phase not in {CollectionPhase.READY, CollectionPhase.STOPPED}:
            raise RuntimeError("JOG is allowed only while ready/stopped")
        delta = np.asarray(payload.get("delta_xyz_m"), dtype=float)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("jog delta_xyz_m must be three finite values")
        if float(np.linalg.norm(delta)) > 0.10 + 1e-9:
            raise ValueError("jog displacement exceeds 100 mm")
        start = self._measured_pose()
        target = start.copy()
        target[4:7] += delta
        duration = max(0.5, float(np.linalg.norm(delta)) / 0.03)
        commands = sample_pose_segment(
            start, target, duration_s=duration, stage="operator_metric_jog",
            gripper_start=float(self.command_rpc.get_right_gripper_exact()),
        )
        report = self._execute(commands)
        measured = self._measured_pose()
        with self.lock:
            self.metrics["last_jog"] = {
                "requested_delta_xyz_m": delta.tolist(),
                "measured_delta_xyz_m": (measured[4:7] - start[4:7]).tolist(),
                "report": report,
            }

    def _command_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                request_id, command, payload = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self.lock:
                    self.metrics.update({
                        "runner": f"processing:{command}",
                        "last_request_id": request_id,
                        "last_error": None,
                    })
                if command == "adjust":
                    step = float(payload["step_mm"])
                    direction = int(payload["direction"])
                    if direction not in {-1, 1} or not 0 < step <= 80:
                        raise ValueError("invalid adjustment step/direction")
                    value = self.cycle.adjust(payload["axis"], direction * step)
                    with self.lock:
                        self.metrics["next_correction_m"] = value.tolist()
                        self.metrics["runner"] = "ready"
                elif command == "start":
                    self._handle_start()
                elif command == "success":
                    self._handle_success()
                elif command == "failure":
                    self._handle_failure(str(payload.get("reason", "grasp_miss")))
                elif command == "enable_auto":
                    self.cycle.enable_auto()
                    with self.lock:
                        self.metrics["runner"] = "auto_enabled"
                    if self.cycle.phase == CollectionPhase.REVIEW:
                        opening = float(
                            self.command_rpc.get_right_gripper_exact()
                        )
                        threshold = float(
                            self.profile["motion"]["minimum_blocked_opening"]
                        )
                        self.command_queue.put_nowait(
                            (0, "success" if opening >= threshold else "failure",
                             {} if opening >= threshold else {"reason": "grasp_miss"})
                        )
                elif command == "home":
                    self._safe_home()
                    with self.lock:
                        self.metrics["runner"] = "ready"
                elif command == "jog":
                    self._safe_jog(payload)
                    with self.lock:
                        self.metrics["runner"] = "ready"
                elif command == "level_descend":
                    self._handle_level_descend(payload)
                    with self.lock:
                        self.metrics["runner"] = "ready_for_close_review"
                elif command == "close_review":
                    self._handle_close_review()
                    with self.lock:
                        self.metrics["runner"] = "close_review_complete"
                elif command == "open_review":
                    self._handle_open_review()
                    with self.lock:
                        self.metrics["runner"] = "open_review_complete"
                elif command == "stop":
                    self.motion_stop.set()
                    self.cycle.stop()
                    self.executor._hold()
                    if self.recorder.is_recording:
                        self.recorder.set_sampling_active(False)
                        self.recorder.end_episode()
                        self.recorder.finalize("failure", reason="abort")
                    with self.lock:
                        self.metrics["runner"] = "stopped_holding"
            except Exception as error:
                self.motion_stop.set()
                try:
                    self.executor._hold()
                except Exception:
                    pass
                with self.lock:
                    self.metrics["last_error"] = f"{type(error).__name__}: {error}"
                    self.metrics["runner"] = "error_holding"
            finally:
                self.command_queue.task_done()

    def close(self) -> None:
        self.motion_stop.set()
        self.stop_event.set()
        try:
            self.executor._hold()
        except Exception:
            pass
        self.ui.stop()
        self.recorder.stop()
        # All camera workers share stop_event; stop each native session so its
        # detached Record3D thread is joined before interpreter teardown.
        for camera in (self.head, self.right, self.left):
            try:
                camera.stop()
            except Exception:
                pass
        self.claim.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="src/configs/guided_lid_collection.json"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    profile = json.loads(_resolve(args.config).read_text())
    baselines = {
        task: BaselineTrajectory.load(_resolve(path), task=task)
        for task, path in profile["baselines"].items()
    }
    audit = {
        "schema": profile["schema"],
        "inference_enabled": profile["policy"]["inference_enabled"],
        "descent_speed_m_s": profile["motion"]["descent_speed_m_s"],
        "baselines": {
            task: {
                "source": value.source,
                "sha256": value.source_sha256,
                "grasp_index": value.source_grasp_index,
                "review_index": value.source_review_index,
                "post_review_commands": len(value.post_review),
            }
            for task, value in baselines.items()
        },
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return
    runtime = GuidedLidRuntime(profile)
    print(runtime.metrics["ui_url"], flush=True)
    stopping = threading.Event()

    def stop_signal(*_):
        stopping.set()

    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        while not stopping.wait(0.5):
            pass
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

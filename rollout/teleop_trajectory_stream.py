"""Stream planned right-arm motion through ConeE's proven teleop command path.

The planner works in physical joint coordinates because they are convenient
for MuJoCo collision checking.  ConeE teleoperation, however, has proven most
reliable when it receives a fresh Cartesian target at 30 Hz with a 50 ms
preview.  This module bridges the two without replaying a demonstration:

* production-CAD FK converts every physical-right joint sample to an EE pose;
* every pose is sent through ``set_right_ee_target``;
* MIT mode and gains are prepared once for the whole trajectory;
* there is no workspace clamp or calibration probe;
* torque telemetry follows the configured enforce/observe-only policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import socket
import struct
import time
from typing import Callable, Iterable, Sequence

import mink
import numpy as np

from robot.arm.ik_solver import SingleArmIK


CONTROL_HZ = 30.0
COMMAND_PREVIEW_S = 0.05
PIPER_MIT_MODE_CAN_ID = 0x151
PIPER_MIT_MODE_PAYLOAD = bytes.fromhex("010400AD00000000")
DEFAULT_HOLDING_KP = np.full(6, 2.5)
DEFAULT_HOLDING_KD = np.full(6, 0.2)
DEFAULT_MOTION_KP = np.array([7.0, 7.0, 7.0, 5.0, 5.0, 5.0])
DEFAULT_MOTION_KD = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3])


class TrajectoryStreamError(RuntimeError):
    """Raised after the arm has been latched at its measured state."""


@dataclass(frozen=True)
class JointTrajectorySample:
    t_s: float
    stage: str
    right_q_physical_rad: np.ndarray
    right_gripper_open_ratio: float | None


def _vector(values, *, name: str, maximum: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain six finite values")
    if np.any(result < 0.0) or np.any(result > maximum):
        raise ValueError(f"{name} is outside the tested range")
    return result


def refresh_right_mit_mode(
    can_interface: str = "can_right",
    *,
    socket_factory=socket.socket,
) -> None:
    """Reassert right-arm MIT mode without enabling, homing, or moving it."""

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


class ProductionRightFK:
    """Exact physical-right FK used by the ConeE RPC server."""

    JOINT_NAMES = tuple(f"left_arm_joint{index}" for index in range(1, 7))
    EE_FRAME = "left_arm_ee"

    def __init__(self, production_model: str | Path):
        self.production_model = Path(production_model).resolve()
        self.solver = SingleArmIK(
            str(self.production_model),
            joint_names=list(self.JOINT_NAMES),
            ee_frame=self.EE_FRAME,
        )

    @staticmethod
    def _q(values) -> np.ndarray:
        q = np.asarray(values, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("physical right q must contain six finite values")
        return q

    def pose(self, right_q_physical_rad) -> mink.SE3:
        q = self._q(right_q_physical_rad)
        self.solver.init(q)
        return self.solver.forward_kinematics()

    def validate_measured(
        self,
        right_q_physical_rad,
        measured_ee,
        *,
        maximum_position_error_m: float = 1e-6,
        maximum_rotation_error_rad: float = 1e-6,
    ) -> dict:
        predicted = self.pose(right_q_physical_rad)
        measured_parameters = np.asarray(measured_ee.parameters(), dtype=float)
        predicted_parameters = np.asarray(predicted.parameters(), dtype=float)
        if (
            measured_parameters.shape != (7,)
            or predicted_parameters.shape != (7,)
            or not np.all(np.isfinite(measured_parameters))
        ):
            raise TrajectoryStreamError("invalid measured right EE pose")
        position_error = float(
            np.linalg.norm(
                measured_parameters[4:7] - predicted_parameters[4:7]
            )
        )
        quaternion_dot = float(
            abs(
                np.dot(
                    measured_parameters[:4],
                    predicted_parameters[:4],
                )
            )
        )
        rotation_error = float(
            2.0 * math.acos(np.clip(quaternion_dot, -1.0, 1.0))
        )
        accepted = bool(
            position_error <= float(maximum_position_error_m)
            and rotation_error <= float(maximum_rotation_error_rad)
        )
        if not accepted:
            raise TrajectoryStreamError(
                "production FK does not match the RPC right EE pose: "
                f"position_error={position_error:.6g}m, "
                f"rotation_error={rotation_error:.6g}rad"
            )
        return {
            "accepted": True,
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
            "physical_right_model_branch": "left_arm_*",
            "ee_frame": self.EE_FRAME,
        }


def _minimum_jerk(fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5


def sample_joint_knots(
    knots: Sequence[dict],
    *,
    control_hz: float = CONTROL_HZ,
) -> list[JointTrajectorySample]:
    """Interpolate a planner's physical joint knots at the teleop rate."""

    if not math.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if len(knots) < 2:
        raise ValueError("trajectory needs at least two knots")
    result: list[JointTrajectorySample] = []
    cursor = 0.0
    for first, second in zip(knots, knots[1:]):
        q0 = ProductionRightFK._q(first["right_q_physical_rad"])
        q1 = ProductionRightFK._q(second["right_q_physical_rad"])
        duration = float(second["minimum_duration_s"])
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("each knot duration must be positive")
        count = max(1, int(math.ceil(duration * control_hz)))
        gripper0 = first.get("right_gripper_open_ratio")
        gripper1 = second.get("right_gripper_open_ratio")
        if (gripper0 is None) != (gripper1 is None):
            raise ValueError("gripper targets must be present on both knots")
        for index in range(1, count + 1):
            fraction = index / count
            blend = _minimum_jerk(fraction)
            gripper = None
            if gripper0 is not None:
                gripper = float(
                    float(gripper0)
                    + blend * (float(gripper1) - float(gripper0))
                )
                if not 0.0 <= gripper <= 1.0:
                    raise ValueError("gripper target must be within [0, 1]")
            result.append(
                JointTrajectorySample(
                    t_s=cursor + fraction * duration,
                    stage=str(second["stage"]),
                    right_q_physical_rad=q0 + blend * (q1 - q0),
                    right_gripper_open_ratio=gripper,
                )
            )
        cursor += duration
    return result


class TeleopTrajectoryStreamer:
    """Execute one uninterrupted planned trajectory through teleop EE targets."""

    def __init__(
        self,
        rpc,
        fk: ProductionRightFK,
        *,
        torque_limit_nm: Sequence[float],
        consecutive_torque_samples: int,
        enforce_torque_stop: bool,
        holding_kp=DEFAULT_HOLDING_KP,
        holding_kd=DEFAULT_HOLDING_KD,
        motion_kp=DEFAULT_MOTION_KP,
        motion_kd=DEFAULT_MOTION_KD,
        gain_ramp_s: float = 1.0,
        mode_settle_s: float = 0.5,
        hold_settle_s: float = 0.25,
        maximum_start_joint_error_rad: float = 0.08,
        maximum_tracking_joint_error_rad: float = 0.45,
        maximum_tracking_position_error_m: float = 0.10,
        maximum_tracking_rotation_error_rad: float = 1.5,
        tracking_check_interval: int = 15,
        mode_refresher: Callable[[], None] = refresh_right_mit_mode,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpc = rpc
        self.fk = fk
        self.torque_limit = np.asarray(torque_limit_nm, dtype=float)
        if (
            self.torque_limit.shape != (6,)
            or not np.all(np.isfinite(self.torque_limit))
            or np.any(self.torque_limit <= 0.0)
        ):
            raise ValueError("torque limits must contain six positive values")
        self.consecutive_torque_samples = int(consecutive_torque_samples)
        if self.consecutive_torque_samples <= 0:
            raise ValueError("consecutive_torque_samples must be positive")
        self.enforce_torque_stop = bool(enforce_torque_stop)
        self.holding_kp = _vector(
            holding_kp, name="holding kp", maximum=DEFAULT_MOTION_KP
        )
        self.holding_kd = _vector(
            holding_kd, name="holding kd", maximum=DEFAULT_MOTION_KD
        )
        self.motion_kp = _vector(
            motion_kp, name="motion kp", maximum=DEFAULT_MOTION_KP
        )
        self.motion_kd = _vector(
            motion_kd, name="motion kd", maximum=DEFAULT_MOTION_KD
        )
        self.gain_ramp_s = float(gain_ramp_s)
        self.mode_settle_s = float(mode_settle_s)
        self.hold_settle_s = float(hold_settle_s)
        if (
            min(self.gain_ramp_s, self.mode_settle_s, self.hold_settle_s) < 0.0
            or self.gain_ramp_s > 5.0
            or max(self.mode_settle_s, self.hold_settle_s) > 2.0
        ):
            raise ValueError("motion preparation timing is outside range")
        self.maximum_start_joint_error_rad = float(
            maximum_start_joint_error_rad
        )
        self.maximum_tracking_joint_error_rad = float(
            maximum_tracking_joint_error_rad
        )
        self.maximum_tracking_position_error_m = float(
            maximum_tracking_position_error_m
        )
        self.maximum_tracking_rotation_error_rad = float(
            maximum_tracking_rotation_error_rad
        )
        self.tracking_check_interval = int(tracking_check_interval)
        if (
            self.maximum_start_joint_error_rad <= 0.0
            or self.maximum_tracking_joint_error_rad <= 0.0
            or self.maximum_tracking_position_error_m <= 0.0
            or self.maximum_tracking_rotation_error_rad <= 0.0
            or self.tracking_check_interval <= 0
        ):
            raise ValueError("trajectory tracking limits must be positive")
        self.mode_refresher = mode_refresher
        self.clock = clock
        self.sleep = sleep
        self.torque_warning_count = 0
        self.last_torque_warning = None
        self._torque_strikes = 0

    def _state(self) -> tuple[np.ndarray, mink.SE3]:
        q = np.asarray(self.rpc.get_right_joint_positions(), dtype=float)
        ee = self.rpc.get_right_ee_pose()
        parameters = np.asarray(ee.parameters(), dtype=float)
        if (
            q.shape != (6,)
            or parameters.shape != (7,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(parameters))
        ):
            raise TrajectoryStreamError("invalid measured right-arm state")
        return q, ee

    def hold_measured(self) -> None:
        q, _ = self._state()
        accepted = self.rpc.set_right_joint_target(
            q, gripper_target=None, preview_time=0.2
        )
        if accepted is False:
            raise TrajectoryStreamError("right measured-pose hold was rejected")

    def _check_torque(self, stage: str) -> None:
        torque = np.abs(
            np.asarray(self.rpc.get_right_joint_torque(), dtype=float)
        )
        invalid = (
            torque.shape != self.torque_limit.shape
            or not np.all(np.isfinite(torque))
        )
        exceeded = (
            False if invalid else bool(np.any(torque > self.torque_limit))
        )
        self._torque_strikes = (
            self._torque_strikes + 1 if invalid or exceeded else 0
        )
        if self._torque_strikes < self.consecutive_torque_samples:
            return
        details = {
            "stage": stage,
            "sample": torque.tolist(),
            "limit": self.torque_limit.tolist(),
            "invalid": invalid,
        }
        if self.enforce_torque_stop:
            raise TrajectoryStreamError(
                f"right torque stop {stage}: {details}"
            )
        self.torque_warning_count += 1
        self.last_torque_warning = details
        self._torque_strikes = 0

    def _wait(self, duration_s: float, stage: str) -> None:
        deadline = self.clock() + float(duration_s)
        while True:
            self._check_torque(stage)
            remaining = deadline - self.clock()
            if remaining <= 0.0:
                return
            self.sleep(min(0.05, remaining))

    def _prepare(self) -> dict:
        start_q, start_ee = self._state()
        fk_check = self.fk.validate_measured(start_q, start_ee)
        self.hold_measured()
        self.rpc.set_right_gain(self.holding_kp, self.holding_kd)
        self._wait(self.hold_settle_s, "while latching measured right pose")
        self.mode_refresher()
        self._wait(self.mode_settle_s, "after right MIT mode refresh")
        steps = max(1, int(math.ceil(self.gain_ramp_s / 0.1)))
        for index in range(steps):
            self.hold_measured()
            alpha = 1.0 if self.gain_ramp_s <= 0.0 else (index + 1) / steps
            self.rpc.set_right_gain(
                self.holding_kp * (1.0 - alpha) + self.motion_kp * alpha,
                self.holding_kd * (1.0 - alpha) + self.motion_kd * alpha,
            )
            if self.gain_ramp_s > 0.0:
                self._wait(
                    self.gain_ramp_s / steps,
                    f"during right gain ramp {index + 1}/{steps}",
                )
        self.mode_refresher()
        self._check_torque("after final right MIT mode refresh")
        return {"start_q": start_q, "fk_check": fk_check}

    def _finish(self) -> None:
        hold_error = None
        try:
            self.hold_measured()
        except BaseException as error:
            hold_error = error
        self.rpc.set_right_gain(self.holding_kp, self.holding_kd)
        if hold_error is not None:
            raise hold_error

    def execute(
        self,
        samples: Iterable[JointTrajectorySample],
        *,
        xyz_correction_provider: Callable[[str, float], Sequence[float]]
        | None = None,
        stage_gate: Callable[[str], None] | None = None,
    ) -> dict:
        """Stream samples continuously; optional corrections never pause it."""

        samples = list(samples)
        if not samples:
            raise ValueError("cannot execute an empty trajectory")
        first_q = samples[0].right_q_physical_rad
        prepared = None
        motion_error = None
        stages = []
        gated_stages = []
        maximum_tracking_error = 0.0
        maximum_tracking_position_error = 0.0
        maximum_tracking_rotation_error = 0.0
        try:
            prepared = self._prepare()
            start_error = float(
                np.max(np.abs(prepared["start_q"] - first_q))
            )
            if start_error > self.maximum_start_joint_error_rad:
                raise TrajectoryStreamError(
                    "right arm is not at the planned initial state: "
                    f"max_joint_error={start_error:.3f}rad"
                )
            started = self.clock()
            previous_time = 0.0
            for index, sample in enumerate(samples, start=1):
                if sample.t_s <= previous_time:
                    raise ValueError("trajectory sample times must increase")
                previous_time = sample.t_s
                self._check_torque(f"during {sample.stage}")
                if not stages or stages[-1] != sample.stage:
                    if stage_gate is not None:
                        stage_gate(sample.stage)
                        gated_stages.append(sample.stage)
                pose = self.fk.pose(sample.right_q_physical_rad)
                if xyz_correction_provider is not None:
                    correction = np.asarray(
                        xyz_correction_provider(sample.stage, sample.t_s),
                        dtype=float,
                    )
                    if (
                        correction.shape != (3,)
                        or not np.all(np.isfinite(correction))
                    ):
                        raise TrajectoryStreamError(
                            "live XYZ correction is invalid"
                        )
                    parameters = np.asarray(pose.parameters(), dtype=float)
                    parameters[4:7] += correction
                    pose = mink.SE3(parameters)
                accepted = self.rpc.set_right_ee_target(
                    pose,
                    gripper_target=sample.right_gripper_open_ratio,
                    preview_time=COMMAND_PREVIEW_S,
                )
                if accepted is not True:
                    raise TrajectoryStreamError(
                        f"right teleop setpoint {index}/{len(samples)} rejected"
                    )
                if not stages or stages[-1] != sample.stage:
                    stages.append(sample.stage)
                if index % self.tracking_check_interval == 0:
                    measured_q, measured_ee = self._state()
                    joint_error = float(
                        np.max(
                            np.abs(
                                measured_q - sample.right_q_physical_rad
                            )
                        )
                    )
                    maximum_tracking_error = max(
                        maximum_tracking_error, joint_error
                    )
                    measured_parameters = np.asarray(
                        measured_ee.parameters(), dtype=float
                    )
                    commanded_parameters = np.asarray(
                        pose.parameters(), dtype=float
                    )
                    position_error = float(
                        np.linalg.norm(
                            measured_parameters[4:7]
                            - commanded_parameters[4:7]
                        )
                    )
                    quaternion_dot = float(
                        abs(
                            np.dot(
                                measured_parameters[:4],
                                commanded_parameters[:4],
                            )
                        )
                    )
                    rotation_error = float(
                        2.0
                        * math.acos(
                            np.clip(quaternion_dot, -1.0, 1.0)
                        )
                    )
                    maximum_tracking_position_error = max(
                        maximum_tracking_position_error,
                        position_error,
                    )
                    maximum_tracking_rotation_error = max(
                        maximum_tracking_rotation_error,
                        rotation_error,
                    )
                    # Cartesian teleop is free to follow a different IK
                    # branch than the planner.  Joint error is useful
                    # telemetry, but only EE error indicates that the
                    # commanded physical motion is not being followed.
                    if (
                        position_error
                        > self.maximum_tracking_position_error_m
                        or rotation_error
                        > self.maximum_tracking_rotation_error_rad
                    ):
                        raise TrajectoryStreamError(
                            "right arm stopped following the Cartesian "
                            "teleop trajectory: "
                            f"position_error={position_error:.3f}m, "
                            f"rotation_error={rotation_error:.3f}rad"
                        )
                deadline = started + sample.t_s
                remaining = deadline - self.clock()
                if remaining < -2.0 / CONTROL_HZ:
                    raise TrajectoryStreamError(
                        f"teleop stream missed deadline at {sample.stage}"
                    )
                if remaining > 0.0:
                    self.sleep(remaining)
            self._wait(
                COMMAND_PREVIEW_S + 0.15,
                "while settling after trajectory",
            )
        except BaseException as error:
            motion_error = error
            raise
        finally:
            try:
                self._finish()
            except BaseException as cleanup_error:
                if motion_error is None:
                    raise
                note = getattr(motion_error, "add_note", None)
                if note is not None:
                    note(
                        "right-arm hold/gain cleanup also failed: "
                        f"{cleanup_error!r}"
                    )
        final_q, final_ee = self._state()
        final_gripper = None
        if hasattr(self.rpc, "get_right_gripper_exact"):
            final_gripper = float(
                np.asarray(
                    self.rpc.get_right_gripper_exact(), dtype=float
                ).reshape(-1)[0]
            )
        return {
            "commands_sent": True,
            "command_path": "set_right_ee_target",
            "control_hz": CONTROL_HZ,
            "preview_time_s": COMMAND_PREVIEW_S,
            "sample_count": len(samples),
            "stages": stages,
            "gated_stages": gated_stages,
            "fk_validation": prepared["fk_check"],
            "maximum_tracking_joint_error_rad": maximum_tracking_error,
            "maximum_tracking_position_error_m": (
                maximum_tracking_position_error
            ),
            "maximum_tracking_rotation_error_rad": (
                maximum_tracking_rotation_error
            ),
            "torque_stop_enforced": self.enforce_torque_stop,
            "torque_warning_count": self.torque_warning_count,
            "last_torque_warning": self.last_torque_warning,
            "final_right_q_physical_rad": final_q.tolist(),
            "final_right_ee_wxyz_xyz": np.asarray(
                final_ee.parameters(), dtype=float
            ).tolist(),
            "final_right_gripper_open_ratio": final_gripper,
        }

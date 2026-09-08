"""Shared Piper preparation for short-horizon real-time motion.

The working lid-servo path does more than send targets at 30 Hz: it latches
measured joints, refreshes MIT mode, ramps tested gains, then restores a
measured joint hold.  Calibration and teleoperation must use the same sequence
instead of relying on whichever mode a previous process happened to leave.
"""

from __future__ import annotations

import socket
import struct
import time
from typing import Callable

import numpy as np


PIPER_MIT_MODE_CAN_ID = 0x151
PIPER_MIT_MODE_PAYLOAD = bytes.fromhex("010400AD00000000")
DEFAULT_HOLDING_KP = np.full(6, 2.5)
DEFAULT_HOLDING_KD = np.full(6, 0.2)
DEFAULT_MOTION_KP = np.array([7.0, 7.0, 7.0, 5.0, 5.0, 5.0])
DEFAULT_MOTION_KD = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3])


def refresh_piper_mit_mode(
    can_interface: str,
    *,
    socket_factory=socket.socket,
) -> None:
    """Reassert MIT mode without reset, enable, or homing."""

    if can_interface not in {"can_left", "can_right"}:
        raise ValueError("MIT refresh is restricted to can_left/can_right")
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
            f"short {can_interface} MIT mode CAN write: {sent}/{len(frame)}"
        )


class PiperRealtimeMotionPreparation:
    """Prepare one arm for real-time targets and restore a measured hold."""

    def __init__(
        self,
        rpc,
        arm: str,
        *,
        holding_kp=DEFAULT_HOLDING_KP,
        holding_kd=DEFAULT_HOLDING_KD,
        motion_kp=DEFAULT_MOTION_KP,
        motion_kd=DEFAULT_MOTION_KD,
        hold_settle_s: float = 0.25,
        mode_settle_s: float = 0.50,
        gain_ramp_s: float = 1.00,
        maximum_joint_drift_rad: float = 0.02,
        maximum_ee_drift_m: float = 0.002,
        mode_refresher: Callable[[str], None] = refresh_piper_mit_mode,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if arm not in {"left", "right"}:
            raise ValueError("arm must be left or right")
        self.rpc = rpc
        self.arm = arm
        self.can_interface = f"can_{arm}"
        self.holding_kp = self._gain(
            holding_kp, "holding kp", DEFAULT_MOTION_KP
        )
        self.holding_kd = self._gain(
            holding_kd, "holding kd", DEFAULT_MOTION_KD
        )
        self.motion_kp = self._gain(
            motion_kp, "motion kp", DEFAULT_MOTION_KP
        )
        self.motion_kd = self._gain(
            motion_kd, "motion kd", DEFAULT_MOTION_KD
        )
        self.hold_settle_s = float(hold_settle_s)
        self.mode_settle_s = float(mode_settle_s)
        self.gain_ramp_s = float(gain_ramp_s)
        self.maximum_joint_drift_rad = float(maximum_joint_drift_rad)
        self.maximum_ee_drift_m = float(maximum_ee_drift_m)
        self.mode_refresher = mode_refresher
        self.clock = clock
        self.sleep = sleep
        timing = np.asarray(
            [self.hold_settle_s, self.mode_settle_s, self.gain_ramp_s],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(timing))
            or np.any(timing < 0.0)
            or self.gain_ramp_s > 5.0
            or max(self.hold_settle_s, self.mode_settle_s) > 2.0
            or self.maximum_joint_drift_rad <= 0.0
            or self.maximum_ee_drift_m <= 0.0
        ):
            raise ValueError("motion-preparation bounds are invalid")

    @staticmethod
    def _gain(value, name, maximum):
        result = np.asarray(value, dtype=float)
        if result.shape != (6,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain six finite values")
        if np.any(result < 0.0) or np.any(result > maximum):
            raise ValueError(f"{name} exceeds the tested gain envelope")
        return result

    def _state(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(
            getattr(self.rpc, f"get_{self.arm}_joint_positions")(),
            dtype=float,
        )
        xyz = np.asarray(
            getattr(self.rpc, f"get_{self.arm}_ee_pose")().translation(),
            dtype=float,
        )
        if (
            qpos.shape != (6,)
            or xyz.shape != (3,)
            or not np.all(np.isfinite(qpos))
            or not np.all(np.isfinite(xyz))
        ):
            raise RuntimeError(f"invalid measured {self.arm} state")
        return qpos, xyz

    def hold_measured(self) -> None:
        qpos, _ = self._state()
        getattr(self.rpc, f"set_{self.arm}_joint_target")(
            qpos,
            gripper_target=None,
            preview_time=0.2,
        )

    def _set_gain(self, kp, kd) -> None:
        getattr(self.rpc, f"set_{self.arm}_gain")(kp, kd)

    def _wait(
        self,
        duration_s: float,
        check_torque: Callable[[], None],
        check_drift: Callable[[], None],
    ) -> None:
        deadline = self.clock() + duration_s
        while True:
            check_torque()
            check_drift()
            remaining = deadline - self.clock()
            if remaining <= 0.0:
                return
            self.sleep(min(0.05, remaining))

    def prepare(self, check_torque: Callable[[], None]) -> None:
        start_q, start_xyz = self._state()

        def check_drift():
            qpos, xyz = self._state()
            if (
                np.max(np.abs(qpos - start_q)) > self.maximum_joint_drift_rad
                or np.linalg.norm(xyz - start_xyz) > self.maximum_ee_drift_m
            ):
                raise RuntimeError(
                    f"unexpected {self.arm} motion during mode preparation"
                )

        self.hold_measured()
        self._set_gain(self.holding_kp, self.holding_kd)
        self._wait(
            self.hold_settle_s,
            check_torque,
            check_drift,
        )
        self.mode_refresher(self.can_interface)
        self._wait(
            self.mode_settle_s,
            check_torque,
            check_drift,
        )
        steps = max(1, int(np.ceil(self.gain_ramp_s / 0.1)))
        duration = self.gain_ramp_s / steps if steps else 0.0
        for index in range(steps):
            self.hold_measured()
            alpha = (index + 1) / steps
            kp = self.holding_kp * (1.0 - alpha) + self.motion_kp * alpha
            kd = self.holding_kd * (1.0 - alpha) + self.motion_kd * alpha
            self._set_gain(kp, kd)
            self._wait(duration, check_torque, check_drift)
        self.mode_refresher(self.can_interface)
        check_torque()
        check_drift()

    def finish(self) -> None:
        hold_error = None
        try:
            self.hold_measured()
        except BaseException as exc:
            hold_error = exc
        try:
            self._set_gain(self.holding_kp, self.holding_kd)
        except BaseException:
            if hold_error is None:
                raise
        if hold_error is not None:
            raise hold_error

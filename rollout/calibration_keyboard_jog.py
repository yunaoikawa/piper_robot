"""Fail-closed Cartesian jog controller for camera/robot calibration.

This module deliberately contains no terminal handling and no camera access.
The RGB-D capture process must remain observation-only so its manifest can
truthfully record ``commands_sent=false``.  A small CLI in
``src/calibration_keyboard_jog.py`` uses this controller from a second
terminal while ``capture_record3d_multiview.py`` owns Record3D.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable

import mink
import numpy as np

from rollout.torque_safety import TorqueWatchdog


MOVE_KEYS = {
    "a": np.array([-1.0, 0.0, 0.0]),
    "d": np.array([1.0, 0.0, 0.0]),
    "s": np.array([0.0, -1.0, 0.0]),
    "w": np.array([0.0, 1.0, 0.0]),
    "f": np.array([0.0, 0.0, -1.0]),
    "r": np.array([0.0, 0.0, 1.0]),
}


class CalibrationJogStop(RuntimeError):
    """Latched stop that prevents any later motion in the same process."""


@dataclass(frozen=True)
class PendingJog:
    arm: str
    kind: str
    key: str
    delta_xyz_m: tuple[float, float, float] | None = None
    joint_index: int | None = None
    joint_delta_rad: float | None = None
    gripper_target: float | None = None


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def load_torque_thresholds(
    path: str | Path,
    *,
    allow_symmetric_left_fallback: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load positive six-joint limits, optionally mirroring right to left.

    The fallback is intentionally explicit.  It is useful for the two
    mechanically identical Pasteur Piper arms before a dedicated left-arm
    envelope exists, but is never silently enabled.
    """

    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    raw = payload.get("thresholds")
    if not isinstance(raw, dict):
        raise ValueError("torque config lacks thresholds")
    thresholds = {}
    for arm in ("left", "right"):
        value = raw.get(arm)
        if value is not None:
            parsed = _finite_vector(value, 6, f"{arm} torque threshold")
            if np.any(parsed <= 0.0):
                raise ValueError(f"{arm} torque thresholds must be positive")
            thresholds[arm] = parsed
    fallback = None
    if "left" not in thresholds:
        if not allow_symmetric_left_fallback:
            raise ValueError(
                "left torque thresholds are missing; pass the explicit "
                "symmetric fallback only for identical Piper arms"
            )
        if "right" not in thresholds:
            raise ValueError("right torque thresholds are unavailable for fallback")
        thresholds["left"] = thresholds["right"].copy()
        fallback = {
            "arm": "left",
            "source_arm": "right",
            "reason": "explicit identical-Piper conservative fallback",
        }
    if "right" not in thresholds:
        raise ValueError("right torque thresholds are missing")
    provenance = {
        "path": str(path),
        "method": payload.get("method"),
        "consecutive_samples": int(payload.get("consecutive_samples", 5)),
        "fallback": fallback,
    }
    if provenance["consecutive_samples"] < 1:
        raise ValueError("consecutive_samples must be positive")
    return thresholds, provenance


def _se3_record(value: Any) -> dict[str, list[float]]:
    return {
        "translation_xyz_m": _finite_vector(
            value.translation(), 3, "EE translation"
        ).tolist(),
        "quaternion_wxyz": _finite_vector(
            value.rotation().wxyz, 4, "EE quaternion"
        ).tolist(),
    }


class CalibrationJogController:
    """Execute confirmed, bounded jogs without initialization or homing."""

    def __init__(
        self,
        rpc: Any,
        *,
        torque_thresholds: dict[str, np.ndarray],
        torque_consecutive_samples: int = 5,
        step_m: float = 0.005,
        maximum_step_m: float = 0.010,
        preview_time_s: float = 0.6,
        monitor_time_s: float = 0.9,
        monitor_hz: float = 30.0,
        cartesian_move_time_s: float = 1.0,
        cartesian_command_preview_s: float = 0.05,
        cartesian_tracking_tolerance_m: float = 0.003,
        joint_step_rad: float = 0.005,
        joint_move_time_s: float = 1.0,
        joint_command_preview_s: float = 0.05,
        joint_minimum_progress: float = 0.25,
        joint_maximum_progress: float = 1.50,
        motion_preparers: dict[str, Any] | None = None,
        audit_path: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpc = rpc
        self.thresholds = {
            arm: _finite_vector(value, 6, f"{arm} torque thresholds")
            for arm, value in torque_thresholds.items()
        }
        if set(self.thresholds) != {"left", "right"}:
            raise ValueError("torque thresholds are required for both arms")
        if any(np.any(value <= 0.0) for value in self.thresholds.values()):
            raise ValueError("torque thresholds must be positive")
        self.torque_consecutive_samples = int(torque_consecutive_samples)
        self.step_m = float(step_m)
        self.maximum_step_m = float(maximum_step_m)
        self.preview_time_s = float(preview_time_s)
        self.monitor_time_s = float(monitor_time_s)
        self.monitor_hz = float(monitor_hz)
        self.cartesian_move_time_s = float(cartesian_move_time_s)
        self.cartesian_command_preview_s = float(cartesian_command_preview_s)
        self.cartesian_tracking_tolerance_m = float(
            cartesian_tracking_tolerance_m
        )
        self.joint_step_rad = float(joint_step_rad)
        self.joint_move_time_s = float(joint_move_time_s)
        self.joint_command_preview_s = float(joint_command_preview_s)
        self.joint_minimum_progress = float(joint_minimum_progress)
        self.joint_maximum_progress = float(joint_maximum_progress)
        self.motion_preparers = dict(motion_preparers or {})
        if not set(self.motion_preparers).issubset({"left", "right"}):
            raise ValueError("motion preparers contain an unknown arm")
        if (
            self.step_m <= 0.0
            or self.step_m > self.maximum_step_m
            or self.maximum_step_m > 0.010
        ):
            raise ValueError("step must be positive and no larger than 10 mm")
        if (
            self.preview_time_s <= 0.0
            or self.monitor_time_s < self.preview_time_s
            or self.monitor_hz <= 0.0
            or self.cartesian_move_time_s < 0.5
            or self.cartesian_command_preview_s <= 0.0
            or self.cartesian_command_preview_s > 0.25
            or self.cartesian_tracking_tolerance_m <= 0.0
            or self.cartesian_tracking_tolerance_m >= self.step_m
            or self.joint_step_rad <= 0.0
            or self.joint_step_rad > 0.005
            or self.joint_move_time_s < 0.5
            or self.joint_command_preview_s <= 0.0
            or self.joint_command_preview_s > 0.25
            or self.joint_minimum_progress <= 0.0
            or self.joint_minimum_progress > 1.0
            or self.joint_maximum_progress < 1.0
            or self.torque_consecutive_samples < 1
        ):
            raise ValueError("invalid timing or torque watchdog configuration")
        self.audit_path = Path(audit_path).resolve() if audit_path else None
        self.clock = clock
        self.sleep = sleep
        self.selected_arm = "left"
        self.selected_joint = 0
        self.pending: PendingJog | None = None
        self.enabled = False
        self.latched_stop: dict[str, Any] | None = None

    def health_snapshot(self) -> dict[str, Any]:
        """Read all required state and reject an uninitialized RPC server."""

        result = {}
        for arm in ("left", "right"):
            pose = getattr(self.rpc, f"get_{arm}_ee_pose")()
            qpos = _finite_vector(
                getattr(self.rpc, f"get_{arm}_joint_positions")(),
                6,
                f"{arm} qpos",
            )
            torque = _finite_vector(
                getattr(self.rpc, f"get_{arm}_joint_torque")(),
                6,
                f"{arm} torque",
            )
            if np.any(np.abs(torque) > self.thresholds[arm]):
                raise CalibrationJogStop(
                    f"{arm} stationary torque already exceeds its limit"
                )
            result[arm] = {
                "ee_pose": _se3_record(pose),
                "joint_positions_rad": qpos.tolist(),
                "joint_torque": torque.tolist(),
                "torque_limit": self.thresholds[arm].tolist(),
                "cartesian_jog_available": True,
                "cartesian_limit_source": "ik_configuration_limits",
            }
        return result

    def enable(self) -> dict[str, Any]:
        snapshot = self.health_snapshot()
        self.enabled = True
        self._audit({"event": "enabled", "state": snapshot})
        return snapshot

    def select_arm(self, arm: str) -> None:
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        self.selected_arm = arm
        self.pending = None

    def select_joint(self, joint_index: int) -> None:
        joint_index = int(joint_index)
        if joint_index < 0 or joint_index >= 6:
            raise ValueError("joint index must be in [0, 5]")
        self.selected_joint = joint_index
        self.pending = None

    def propose(self, key: str) -> PendingJog:
        """Stage a command.  Staging never sends a robot command."""

        if self.latched_stop is not None:
            raise CalibrationJogStop("torque stop is latched; restart after inspection")
        key = str(key).lower()
        if key in MOVE_KEYS:
            delta = MOVE_KEYS[key] * self.step_m
            pending = PendingJog(
                arm=self.selected_arm,
                kind="cartesian",
                key=key,
                delta_xyz_m=tuple(float(value) for value in delta),
            )
        elif key in ("-", "=", "+"):
            delta = -self.joint_step_rad if key == "-" else self.joint_step_rad
            pending = PendingJog(
                arm=self.selected_arm,
                kind="joint",
                key=key,
                joint_index=self.selected_joint,
                joint_delta_rad=delta,
            )
        elif key in ("o", "c"):
            pending = PendingJog(
                arm=self.selected_arm,
                kind="gripper",
                key=key,
                gripper_target=1.0 if key == "o" else 0.0,
            )
        else:
            raise ValueError(f"unsupported jog key {key!r}")
        self.pending = pending
        return pending

    def cancel_pending(self) -> None:
        self.pending = None

    def prepare_pending_motion(self) -> str | None:
        """Apply the proven gain/MIT preparation before a pending motion."""

        if self.pending is None or self.pending.kind not in {
            "cartesian",
            "joint",
        }:
            return None
        arm = self.pending.arm
        preparer = self.motion_preparers.get(arm)
        if preparer is None:
            return None
        watchdog = TorqueWatchdog(
            {arm: self.thresholds[arm]},
            consecutive_limit=self.torque_consecutive_samples,
        )

        def check_torque():
            torque = _finite_vector(
                getattr(self.rpc, f"get_{arm}_joint_torque")(),
                6,
                f"{arm} torque",
            )
            if watchdog.check(arm, torque):
                return
            self.latched_stop = dict(watchdog.tripped or {})
            raise CalibrationJogStop(
                f"{arm} sustained torque limit during motion preparation"
            )

        try:
            preparer.prepare(check_torque)
        except BaseException:
            try:
                preparer.finish()
            except BaseException:
                pass
            self._audit(
                {
                    "event": "motion_preparation_failed",
                    "arm": arm,
                    "trip": self.latched_stop,
                }
            )
            raise
        self._audit({"event": "motion_prepared", "arm": arm})
        return arm

    def finish_motion(self, arm: str | None) -> None:
        if arm is None:
            return
        preparer = self.motion_preparers.get(arm)
        if preparer is None:
            return
        preparer.finish()
        self._audit({"event": "motion_finished_in_joint_hold", "arm": arm})

    def confirm(self) -> dict[str, Any]:
        """Execute exactly one staged command and monitor joint torque."""

        if not self.enabled:
            raise CalibrationJogStop("controller is not enabled")
        if self.latched_stop is not None:
            raise CalibrationJogStop("torque stop is latched; restart after inspection")
        if self.pending is None:
            raise ValueError("no pending command")
        command = self.pending
        self.pending = None
        arm = command.arm
        before_pose = getattr(self.rpc, f"get_{arm}_ee_pose")()
        before_qpos = _finite_vector(
            getattr(self.rpc, f"get_{arm}_joint_positions")(),
            6,
            f"{arm} qpos",
        )
        before_torque = _finite_vector(
            getattr(self.rpc, f"get_{arm}_joint_torque")(),
            6,
            f"{arm} torque",
        )
        if np.any(np.abs(before_torque) > self.thresholds[arm]):
            self._latch(
                arm,
                "pre_command_torque_limit",
                before_torque,
                self.thresholds[arm],
            )

        watchdog = TorqueWatchdog(
            {arm: self.thresholds[arm]},
            consecutive_limit=self.torque_consecutive_samples,
        )
        samples = [before_torque]

        def check_torque() -> None:
            torque = _finite_vector(
                getattr(self.rpc, f"get_{arm}_joint_torque")(),
                6,
                f"{arm} torque",
            )
            samples.append(torque)
            if watchdog.check(arm, torque):
                return
            self._hold_arm(arm)
            self.latched_stop = dict(watchdog.tripped or {})
            record = {
                "event": "torque_stop",
                "command": command.__dict__,
                "trip": self.latched_stop,
                "torque_samples": np.asarray(samples).tolist(),
            }
            self._audit(record)
            raise CalibrationJogStop(
                f"{arm} sustained torque limit exceeded; joint hold sent"
            )

        target_qpos = None
        target_xyz = None
        if command.kind == "cartesian":
            delta = _finite_vector(command.delta_xyz_m, 3, "jog delta")
            if np.linalg.norm(delta) > self.maximum_step_m + 1e-12:
                raise CalibrationJogStop("pending jog exceeds maximum step")
            start_parameters = _finite_vector(
                before_pose.parameters(), 7, f"{arm} EE pose"
            ).copy()
            target_xyz = start_parameters[4:7] + delta
            steps = max(
                2, int(np.ceil(self.cartesian_move_time_s * self.monitor_hz))
            )
            start = self.clock()
            for index in range(1, steps + 1):
                fraction = index / steps
                waypoint = start_parameters.copy()
                waypoint[4:7] += fraction * delta
                accepted = getattr(self.rpc, f"set_{arm}_ee_target")(
                    mink.SE3(waypoint),
                    preview_time=self.cartesian_command_preview_s,
                )
                if accepted is False:
                    self._hold_arm(arm)
                    raise CalibrationJogStop(
                        f"{arm} IK rejected Cartesian waypoint {index}/{steps}"
                    )
                check_torque()
                deadline = start + index * self.cartesian_move_time_s / steps
                delay = deadline - self.clock()
                if delay > 0.0:
                    self.sleep(delay)
            monitor_time = 0.3
        elif command.kind == "joint":
            joint_index = int(command.joint_index)
            delta_rad = float(command.joint_delta_rad)
            if abs(delta_rad) > 0.005 + 1e-12:
                raise CalibrationJogStop("joint jog exceeds 0.005 rad")
            target_qpos = before_qpos.copy()
            target_qpos[joint_index] += delta_rad
            # Piper's timestamp is a short command horizon, not a request to
            # synthesize a long trajectory.  A single target five seconds in
            # the future may be ignored.  Follow the proven teleop pattern:
            # stream closely-spaced targets with a short preview horizon.
            steps = max(2, int(np.ceil(self.joint_move_time_s * self.monitor_hz)))
            start = self.clock()
            for index in range(1, steps + 1):
                fraction = index / steps
                waypoint = before_qpos + fraction * (target_qpos - before_qpos)
                getattr(self.rpc, f"set_{arm}_joint_target")(
                    waypoint,
                    preview_time=self.joint_command_preview_s,
                )
                check_torque()
                deadline = start + index * self.joint_move_time_s / steps
                delay = deadline - self.clock()
                if delay > 0.0:
                    self.sleep(delay)
            monitor_time = 0.3
        elif command.kind == "gripper":
            method = "open" if command.gripper_target == 1.0 else "close"
            getattr(self.rpc, f"{method}_{arm}_gripper")()
            monitor_time = self.monitor_time_s
        else:
            raise AssertionError(command.kind)

        deadline = self.clock() + monitor_time
        next_sample = self.clock()
        while self.clock() < deadline:
            check_torque()
            next_sample += 1.0 / self.monitor_hz
            delay = next_sample - self.clock()
            if delay > 0.0:
                self.sleep(delay)

        after_pose = getattr(self.rpc, f"get_{arm}_ee_pose")()
        after_qpos = _finite_vector(
            getattr(self.rpc, f"get_{arm}_joint_positions")(),
            6,
            f"{arm} qpos",
        )
        if target_xyz is not None:
            measured_xyz = _finite_vector(
                after_pose.translation(), 3, f"{arm} EE translation"
            )
            error = float(np.linalg.norm(measured_xyz - target_xyz))
            if error > self.cartesian_tracking_tolerance_m:
                self._hold_arm(arm)
                record = {
                    "event": "tracking_stop",
                    "command": command.__dict__,
                    "target_translation_xyz_m": target_xyz.tolist(),
                    "measured_translation_xyz_m": measured_xyz.tolist(),
                    "tracking_error_m": error,
                    "tracking_tolerance_m": self.cartesian_tracking_tolerance_m,
                }
                self._audit(record)
                raise CalibrationJogStop(
                    f"{arm} Cartesian target was not reached "
                    f"(error {error*1000.0:.1f} mm); joint hold sent"
                )
        if target_qpos is not None:
            joint_index = int(command.joint_index)
            requested = float(command.joint_delta_rad)
            achieved = float(after_qpos[joint_index] - before_qpos[joint_index])
            progress = achieved / requested
            if (
                not np.isfinite(progress)
                or progress < self.joint_minimum_progress
                or progress > self.joint_maximum_progress
            ):
                self._hold_arm(arm)
                record = {
                    "event": "tracking_stop",
                    "command": command.__dict__,
                    "target_joint_positions_rad": target_qpos.tolist(),
                    "measured_joint_positions_rad": after_qpos.tolist(),
                    "requested_delta_rad": requested,
                    "achieved_delta_rad": achieved,
                    "progress_fraction": progress,
                    "minimum_progress": self.joint_minimum_progress,
                    "maximum_progress": self.joint_maximum_progress,
                }
                self._audit(record)
                raise CalibrationJogStop(
                    f"{arm} joint target progress was {progress:.2f}; "
                    "joint hold sent"
                )
        record = {
            "event": "command_completed",
            "command": command.__dict__,
            "before": {
                "ee_pose": _se3_record(before_pose),
                "joint_positions_rad": before_qpos.tolist(),
                "joint_torque": before_torque.tolist(),
            },
            "after": {
                "ee_pose": _se3_record(after_pose),
                "joint_positions_rad": after_qpos.tolist(),
                "joint_torque_max_abs": np.max(
                    np.abs(np.asarray(samples)), axis=0
                ).tolist(),
            },
            "torque_limit": self.thresholds[arm].tolist(),
        }
        self._audit(record)
        return record

    def hold(self) -> None:
        self.pending = None
        errors = []
        for arm in ("left", "right"):
            try:
                self._hold_arm(arm)
            except Exception as exc:  # best-effort stop of the other arm
                errors.append(f"{arm}: {type(exc).__name__}: {exc}")
        self._audit({"event": "hold", "errors": errors})
        if errors:
            raise CalibrationJogStop("; ".join(errors))

    def _hold_arm(self, arm: str) -> None:
        # Joint-space hold is mandatory here.  A measured EE position can sit
        # outside ConeE's historical shared Cartesian clamp (notably the left
        # arm's mirrored Y frame); sending that EE pose would make the server
        # clamp and move instead of hold.
        current = _finite_vector(
            getattr(self.rpc, f"get_{arm}_joint_positions")(),
            6,
            f"{arm} qpos",
        )
        getattr(self.rpc, f"set_{arm}_joint_target")(
            current,
            preview_time=0.1,
        )

    def _latch(
        self,
        arm: str,
        reason: str,
        torque: np.ndarray,
        threshold: np.ndarray,
    ) -> None:
        self.latched_stop = {
            "arm": arm,
            "reason": reason,
            "torque": np.asarray(torque, dtype=float).tolist(),
            "threshold": np.asarray(threshold, dtype=float).tolist(),
        }
        self._audit({"event": "torque_stop", "trip": self.latched_stop})
        raise CalibrationJogStop(f"{arm} torque safety rejected the command")

    def _audit(self, payload: dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "piper_robot.calibration_keyboard_jog_event/v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

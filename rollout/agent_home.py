"""Pressure-guarded, gripper-neutral homing for agent data collection."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from robot.arm.home import physical_home_q
from .recovery_teleop_safety import (
    RecoveryTorqueGuard,
    extend_fallback_threshold_for_stationary_pose,
)


def _agent_pressure_guard(rpc, torque_config_path, audit_path, provenance):
    config = json.loads(Path(torque_config_path).read_text())
    right_threshold = np.asarray(config["thresholds"]["right"], dtype=float)
    thresholds = {"left": right_threshold.copy(), "right": right_threshold.copy()}
    left_fallback = extend_fallback_threshold_for_stationary_pose(
        rpc,
        thresholds,
        arm="left",
        sample_count=25,
        sample_interval_s=0.01,
        margin=1.20,
    )
    recovery = config["recovery_teleop"]
    home = config.get("agent_auto_home", {})
    return RecoveryTorqueGuard(
        rpc,
        thresholds,
        consecutive_samples=int(config.get("consecutive_samples", 5)),
        preview_time_s=0.05,
        residual_fraction=float(recovery["residual_fraction"]),
        residual_floor_nm=float(home.get(
            "residual_floor_nm", recovery["residual_floor_nm"]
        )),
        # Planned, slow gravity-compensated homing changes joint load even
        # without contact.  Use its separately calibrated residual envelope;
        # the absolute hard limit and sustained-residual stop remain enforced.
        residual_ceiling_nm=float(home.get(
            "residual_ceiling_nm", recovery["residual_ceiling_nm"]
        )),
        residual_duration_s=float(home.get(
            "residual_duration_s", recovery["residual_duration_s"]
        )),
        baseline_slew_nm_per_s=float(recovery["baseline_slew_nm_per_s"]),
        hard_limit_multiplier=float(recovery["hard_limit_multiplier"]),
        enforce=True,
        audit_path=audit_path,
        provenance={
            **dict(provenance),
            "left_fallback_calibration": left_fallback,
        },
    )


def agent_home_trajectory(
    start_by_arm: dict[str, np.ndarray],
    *,
    control_hz: float = 30.0,
    maximum_speed_rad_s: float = 0.18,
    minimum_duration_s: float = 2.5,
    maximum_duration_s: float = 30.0,
) -> tuple[dict[str, np.ndarray], float]:
    """Build synchronized minimum-jerk joint paths for both physical arms."""

    if set(start_by_arm) != {"left", "right"}:
        raise ValueError("left and right start joints are required")
    if control_hz <= 0 or maximum_speed_rad_s <= 0:
        raise ValueError("control rate and maximum speed must be positive")
    starts = {}
    largest_delta = 0.0
    for arm in ("left", "right"):
        value = np.asarray(start_by_arm[arm], dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid {arm} joint state")
        starts[arm] = value.copy()
        largest_delta = max(
            largest_delta,
            float(np.max(np.abs(physical_home_q(arm) - value))),
        )
    duration = float(np.clip(
        largest_delta / maximum_speed_rad_s,
        minimum_duration_s,
        maximum_duration_s,
    ))
    sample_count = max(2, int(np.ceil(duration * control_hz)))
    phase = np.arange(1, sample_count + 1, dtype=float) / sample_count
    blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
    paths = {
        arm: starts[arm][None, :] + blend[:, None] * (
            physical_home_q(arm) - starts[arm]
        )[None, :]
        for arm in ("left", "right")
    }
    return paths, duration


def run_agent_auto_home(
    rpc: Any,
    *,
    torque_config_path: str | Path,
    audit_path: str | Path,
    control_hz: float = 30.0,
) -> dict[str, Any]:
    """Slowly home both arms without changing either gripper."""

    starts = {
        arm: np.asarray(
            getattr(rpc, f"get_{arm}_joint_positions")(), dtype=float
        )
        for arm in ("left", "right")
    }
    config = json.loads(Path(torque_config_path).read_text())
    maximum_speed_rad_s = float(
        config.get("agent_auto_home", {}).get("maximum_speed_rad_s", 0.18)
    )
    paths, duration = agent_home_trajectory(
        starts,
        control_hz=control_hz,
        maximum_speed_rad_s=maximum_speed_rad_s,
    )
    guard = _agent_pressure_guard(
        rpc,
        torque_config_path,
        audit_path,
        {"primitive": "agent_auto_home"},
    )

    started = time.monotonic()
    completed = False
    try:
        for index in range(len(paths["left"])):
            for arm in ("left", "right"):
                if not guard.check(arm):
                    raise RuntimeError(
                        f"pressure guard stopped agent auto-home on {arm}: "
                        + json.dumps(guard.latched[arm])
                    )
            # No gripper_target: homing must not open, close, or depend on a
            # Dynamixel gripper serial device.
            rpc.set_left_joint_target(
                paths["left"][index], gripper_target=None, preview_time=0.05
            )
            rpc.set_right_joint_target(
                paths["right"][index], gripper_target=None, preview_time=0.05
            )
            remaining = started + (index + 1) / control_hz - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        completed = True
    finally:
        for arm in ("left", "right"):
            measured = np.asarray(
                getattr(rpc, f"get_{arm}_joint_positions")(), dtype=float
            )
            getattr(rpc, f"set_{arm}_joint_target")(
                measured, gripper_target=None, preview_time=0.2
            )

    final = {
        arm: np.asarray(
            getattr(rpc, f"get_{arm}_joint_positions")(), dtype=float
        )
        for arm in ("left", "right")
    }
    errors = {
        arm: float(np.max(np.abs(final[arm] - physical_home_q(arm))))
        for arm in ("left", "right")
    }
    maximum_final_error_rad = float(
        config.get("agent_auto_home", {}).get("maximum_final_error_rad", 0.12)
    )
    if any(error > maximum_final_error_rad for error in errors.values()):
        raise RuntimeError(f"agent auto-home did not converge: {errors}")
    return {
        "completed": completed,
        "duration_s": duration,
        "control_hz": control_hz,
        "maximum_speed_rad_s": maximum_speed_rad_s,
        "maximum_final_error_rad": errors,
        "grippers_commanded": False,
        "pressure_stop_enforced": True,
    }

#!/usr/bin/env python3
"""Return only a near-home physical right Piper to canonical home slowly.

This intentionally does not call ``home_right_arm()``.  It starts at the
measured joint state, streams a minimum-jerk joint path at 30 Hz, and enforces
the residual pressure guard.  It is only valid after a human or audited path
has already brought the arm into the near-home envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.arm.home import physical_home_q
from robot.rpc import RPCClient
from rollout.grasp_orchestration import ControllerLease
from rollout.recovery_teleop_safety import (
    RecoveryTorqueGuard,
    extend_fallback_threshold_for_stationary_pose,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--duration-s", type=float, default=9.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--near-home-limit-rad", type=float, default=0.5)
    parser.add_argument(
        "--torque-config", default="src/configs/pasteur_lid_torque.json"
    )
    parser.add_argument(
        "--controller-lock",
        default="/tmp/piper_robot_right_arm_controller.lock",
    )
    parser.add_argument(
        "--audit-log", default="/var/tmp/piper-slow-right-home/torque.jsonl"
    )
    args = parser.parse_args()
    if not 3.0 <= args.duration_s <= 30.0:
        raise ValueError("duration must be within [3, 30] seconds")
    if args.control_hz != 30.0:
        raise ValueError("this recovery primitive is pinned to 30 Hz")

    rpc = RPCClient(args.host, args.port, timeout_ms=3000)
    config = json.loads((ROOT / args.torque_config).read_text())
    right_threshold = np.asarray(config["thresholds"]["right"], dtype=float)
    # RecoveryTorqueGuard has a bimanual schema, but this primitive checks and
    # commands only physical right.  The unused left entry is a schema filler.
    thresholds = {"left": right_threshold.copy(), "right": right_threshold.copy()}
    recovery = config["recovery_teleop"]
    output = Path(args.audit_log).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with ControllerLease(
        args.controller_lock,
        owner={"task": "slow_right_home", "physical_arm": "right"},
    ):
        start = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        target = physical_home_q("right")
        maximum_delta = float(np.max(np.abs(target - start)))
        if maximum_delta > args.near_home_limit_rad:
            raise RuntimeError(
                "right arm is outside the near-home recovery envelope: "
                f"{maximum_delta:.4f}rad"
            )
        original_gain = rpc.get_right_gain()
        original_kp = np.asarray(original_gain["kp"], dtype=float)
        original_kd = np.asarray(original_gain["kd"], dtype=float)
        rpc.set_right_joint_target(start, preview_time=0.05)
        calibration = extend_fallback_threshold_for_stationary_pose(
            rpc,
            thresholds,
            arm="right",
            sample_count=25,
            sample_interval_s=0.01,
            margin=1.20,
        )
        guard = RecoveryTorqueGuard(
            rpc,
            thresholds,
            consecutive_samples=int(config.get("consecutive_samples", 5)),
            preview_time_s=0.05,
            residual_fraction=float(recovery["residual_fraction"]),
            residual_floor_nm=float(recovery["residual_floor_nm"]),
            residual_ceiling_nm=float(recovery["residual_ceiling_nm"]),
            residual_duration_s=float(recovery["residual_duration_s"]),
            baseline_slew_nm_per_s=float(
                recovery["baseline_slew_nm_per_s"]
            ),
            hard_limit_multiplier=float(recovery["hard_limit_multiplier"]),
            enforce=True,
            audit_path=output,
            provenance={"primitive": "slow_right_home", "calibration": calibration},
        )
        count = int(round(args.duration_s * args.control_hz))
        started = time.monotonic()
        maximum_tracking_error = 0.0
        completed = False
        try:
            for index in range(1, count + 1):
                if not guard.check("right"):
                    raise RuntimeError(
                        "pressure guard stopped slow right-home motion: "
                        + json.dumps(guard.latched["right"])
                    )
                t = index / count
                blend = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
                command = start + blend * (target - start)
                rpc.set_right_joint_target(command, preview_time=0.05)
                if index % 5 == 0:
                    measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
                    error = float(np.max(np.abs(measured - command)))
                    maximum_tracking_error = max(maximum_tracking_error, error)
                    if error > 0.12:
                        rpc.set_right_joint_target(measured, preview_time=0.05)
                        raise RuntimeError(
                            f"tracking error stopped slow right-home motion: {error:.4f}rad"
                        )
                remaining = started + index / args.control_hz - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
            completed = True
        finally:
            measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
            rpc.set_right_joint_target(measured, preview_time=0.2)
            rpc.set_right_gain(original_kp, original_kd)

        final_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        report = {
            "completed": completed,
            "control_hz": args.control_hz,
            "duration_s": args.duration_s,
            "start_q_rad": start.tolist(),
            "target_q_rad": target.tolist(),
            "final_q_rad": final_q.tolist(),
            "maximum_start_delta_rad": maximum_delta,
            "maximum_tracking_error_rad": maximum_tracking_error,
            "maximum_final_error_rad": float(np.max(np.abs(final_q - target))),
            "pressure_stop_enforced": True,
            "left_arm_commanded": False,
        }
        print(json.dumps(report, indent=2), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

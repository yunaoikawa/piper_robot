#!/usr/bin/env python3
"""Temporarily make only the physical right Piper hand-guidable.

The command never homes either arm.  It latches the measured right-arm pose,
ramps only the right position gain down, then leaves the controller in damped
manual mode.  On Enter/SIGINT/SIGTERM it latches the *new* measured pose before
restoring the original gains, avoiding a snap back to the old command.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.rpc import RPCClient
from rollout.grasp_orchestration import ControllerLease


def _vector(value: float) -> np.ndarray:
    return np.full(6, float(value), dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--manual-kp", type=float, default=0.5)
    parser.add_argument("--manual-kd", type=float, default=0.3)
    parser.add_argument("--ramp-s", type=float, default=2.5)
    parser.add_argument(
        "--controller-lock",
        default="/tmp/piper_robot_right_arm_controller.lock",
    )
    args = parser.parse_args()
    if not 0.0 <= args.manual_kp <= 1.0:
        raise ValueError("manual-kp must be within [0, 1]")
    if not 0.05 <= args.manual_kd <= 1.0:
        raise ValueError("manual-kd must be within [0.05, 1]")
    if not 0.5 <= args.ramp_s <= 10.0:
        raise ValueError("ramp-s must be within [0.5, 10]")

    rpc = RPCClient(args.host, args.port, timeout_ms=3000)
    stop = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with ControllerLease(
        args.controller_lock,
        owner={"task": "manual_right_arm_damping", "physical_arm": "right"},
    ):
        original = rpc.get_right_gain()
        original_kp = np.asarray(original["kp"], dtype=float)
        original_kd = np.asarray(original["kd"], dtype=float)
        start_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        rpc.set_right_joint_target(start_q, preview_time=0.2)
        manual_kp = _vector(args.manual_kp)
        manual_kd = _vector(args.manual_kd)
        steps = max(1, int(round(args.ramp_s * 10.0)))
        for index in range(steps):
            alpha = (index + 1) / steps
            rpc.set_right_gain(
                (1.0 - alpha) * original_kp + alpha * manual_kp,
                (1.0 - alpha) * original_kd + alpha * manual_kd,
            )
            time.sleep(args.ramp_s / steps)

        print(
            "MANUAL_READY right only; move slowly while supporting the arm. "
            "Press Enter when done.",
            flush=True,
        )
        while not stop:
            readable = __import__("select").select([sys.stdin], [], [], 0.1)[0]
            if readable:
                sys.stdin.readline()
                break

        # Capture the new pose before stiffness returns, so restoration cannot
        # pull the arm toward the pre-manual target.
        final_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        rpc.set_right_joint_target(final_q, preview_time=0.2)
        restore_steps = max(1, int(round(args.ramp_s * 10.0)))
        for index in range(restore_steps):
            alpha = (index + 1) / restore_steps
            rpc.set_right_joint_target(final_q, preview_time=0.2)
            rpc.set_right_gain(
                (1.0 - alpha) * manual_kp + alpha * original_kp,
                (1.0 - alpha) * manual_kd + alpha * original_kd,
            )
            time.sleep(args.ramp_s / restore_steps)
        rpc.set_right_joint_target(final_q, preview_time=0.2)
        print("HOLD_RESTORED right_q=" + repr(final_q.tolist()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

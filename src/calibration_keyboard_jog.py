#!/usr/bin/env python3
"""Confirmed 5 mm keyboard jogging for fixed-head calibration.

Run this in a terminal separate from ``capture_record3d_multiview.py``.  This
process may command the robot; the capture process remains read-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import termios
import tty

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.rpc import RPCClient
from rollout.calibration_keyboard_jog import (
    CalibrationJogController,
    CalibrationJogStop,
    load_torque_thresholds,
)
from rollout.piper_realtime_motion import PiperRealtimeMotionPreparation


HELP = """
Keys (a direction key only stages a command; Enter executes it):
  1 / 2       select left / right arm
  a / d       robot-frame -X / +X by one step
  s / w       robot-frame -Y / +Y by one step
  f / r       robot-frame -Z / +Z by one step
  [ / ]       select previous / next joint (1..6)
  - / +       stage -0.005 / +0.005 rad joint jog
  o / c       stage gripper open / close
  Enter       execute the staged command
  Esc         cancel the staged command
  Space       hold both arms immediately
  p           print "pose ready" (no command; capture in the other terminal)
  ?           show this help
  q           hold both arms and quit; never home
""".strip()


def _default_audit_path() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"/var/tmp/piper-head-calibration/jog-{stamp}-{os.getpid()}.jsonl"


def _describe_pending(value) -> str:
    if value.kind == "cartesian":
        millimetres = [round(item * 1000.0, 1) for item in value.delta_xyz_m]
        return f"{value.arm} Cartesian delta XYZ={millimetres} mm"
    if value.kind == "joint":
        return (
            f"{value.arm} joint {value.joint_index + 1} "
            f"delta={value.joint_delta_rad:+.4f} rad"
        )
    target = "OPEN" if value.gripper_target == 1.0 else "CLOSE"
    return f"{value.arm} gripper {target}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-host", default="localhost")
    parser.add_argument("--robot-port", type=int, default=8081)
    parser.add_argument("--step-mm", type=float, default=5.0)
    parser.add_argument("--maximum-step-mm", type=float, default=10.0)
    parser.add_argument("--preview-time-s", type=float, default=0.6)
    parser.add_argument("--monitor-time-s", type=float, default=0.9)
    parser.add_argument("--monitor-hz", type=float, default=30.0)
    parser.add_argument("--cartesian-move-time-s", type=float, default=1.0)
    parser.add_argument("--cartesian-command-preview-s", type=float, default=0.05)
    parser.add_argument("--joint-step-rad", type=float, default=0.005)
    parser.add_argument("--joint-move-time-s", type=float, default=1.0)
    parser.add_argument("--joint-command-preview-s", type=float, default=0.05)
    parser.add_argument(
        "--torque-config",
        default="src/configs/pasteur_lid_torque.json",
    )
    parser.add_argument(
        "--allow-symmetric-left-torque-fallback",
        action="store_true",
        help=(
            "explicitly reuse the right thresholds for the mechanically "
            "identical left Piper when no left envelope is present"
        ),
    )
    parser.add_argument("--audit-log", default=_default_audit_path())
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="read state and torque without enabling or sending commands",
    )
    parser.add_argument(
        "--lock-path",
        default="/var/tmp/piper-head-calibration/keyboard-jog.lock",
    )
    args = parser.parse_args(argv)
    if not sys.stdin.isatty() and not args.check_only:
        parser.error("interactive jog requires a TTY")

    lock_path = Path(args.lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another calibration keyboard jog process is active") from exc
    lock_stream.write(f"{os.getpid()}\n")
    lock_stream.flush()

    thresholds, provenance = load_torque_thresholds(
        args.torque_config,
        allow_symmetric_left_fallback=args.allow_symmetric_left_torque_fallback,
    )
    rpc = RPCClient(args.robot_host, args.robot_port, timeout_ms=4000)

    motion_preparers = {
        arm: PiperRealtimeMotionPreparation(rpc, arm)
        for arm in ("left", "right")
    }
    controller = CalibrationJogController(
        rpc,
        torque_thresholds=thresholds,
        torque_consecutive_samples=provenance["consecutive_samples"],
        step_m=args.step_mm / 1000.0,
        maximum_step_m=args.maximum_step_mm / 1000.0,
        preview_time_s=args.preview_time_s,
        monitor_time_s=args.monitor_time_s,
        monitor_hz=args.monitor_hz,
        cartesian_move_time_s=args.cartesian_move_time_s,
        cartesian_command_preview_s=args.cartesian_command_preview_s,
        joint_step_rad=args.joint_step_rad,
        joint_move_time_s=args.joint_move_time_s,
        joint_command_preview_s=args.joint_command_preview_s,
        motion_preparers=motion_preparers,
        audit_path=args.audit_log,
    )
    health = controller.health_snapshot()
    print(
        json.dumps(
            {
                "status": "healthy_read_only",
                "audit_log": str(Path(args.audit_log).resolve()),
                "torque_provenance": provenance,
                "state": health,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.check_only:
        return 0

    print(
        "\nThis tool never initializes or homes the robot. "
        "Confirm the robot is visible and the emergency stop is reachable."
    )
    if input("Type ENABLE and press Enter to permit confirmed 5 mm jogs: ") != "ENABLE":
        print("Not enabled; no command sent.")
        return 2
    controller.enable()
    print(HELP)
    print("\nSelected arm: left")
    old_settings = termios.tcgetattr(sys.stdin)
    exit_code = 0
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = sys.stdin.read(1)
            if key == "q":
                controller.hold()
                print("\nHeld both arms; exiting without home.")
                break
            if key == " ":
                controller.hold()
                print("\nHeld both arms; pending command cleared.")
                continue
            if key == "\x1b":
                controller.cancel_pending()
                print("\nPending command cancelled.")
                continue
            if key in ("1", "2"):
                arm = "left" if key == "1" else "right"
                controller.select_arm(arm)
                print(f"\nSelected arm: {arm}")
                continue
            if key in ("[", "]"):
                offset = -1 if key == "[" else 1
                controller.select_joint(
                    (controller.selected_joint + offset) % 6
                )
                print(
                    f"\nSelected {controller.selected_arm} "
                    f"joint {controller.selected_joint + 1}"
                )
                continue
            if key == "?":
                print("\n" + HELP)
                continue
            if key == "p":
                print("\nPOSE READY — press Enter in the read-only capture terminal.")
                continue
            if key in ("\r", "\n"):
                if controller.pending is None:
                    print("\nNo pending command.")
                    continue
                prepared_arm = None
                try:
                    prepared_arm = controller.prepare_pending_motion()
                    record = controller.confirm()
                    after = record["after"]["ee_pose"]["translation_xyz_m"]
                    print(f"\nCommand complete; {record['command']['arm']} EE XYZ={after}")
                except CalibrationJogStop as exc:
                    print(f"\nSTOPPED: {exc}", file=sys.stderr)
                    exit_code = 3
                    break
                except Exception as exc:
                    print(
                        f"\nSTOPPED: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    exit_code = 3
                    break
                finally:
                    try:
                        controller.finish_motion(prepared_arm)
                    except Exception as exc:
                        print(
                            f"\nSTOPPED restoring joint hold: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        exit_code = 4
                if exit_code == 4:
                    break
                continue
            try:
                pending = controller.propose(key)
            except ValueError:
                continue
            print(
                f"\nSTAGED (no motion yet): {_describe_pending(pending)}. "
                "Press Enter to execute or Esc to cancel."
            )
    except KeyboardInterrupt:
        print("\nKeyboard interrupt; holding both arms.")
        exit_code = 130
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        try:
            controller.hold()
        except Exception as exc:
            print(f"HOLD ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = exit_code or 4
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

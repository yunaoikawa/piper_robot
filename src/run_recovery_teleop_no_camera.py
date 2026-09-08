#!/usr/bin/env python3
"""Run the established Quest recovery loop without opening Record3D.

This is for interleaving fixed-head RGB-D calibration captures with manual
robot repositioning.  Control semantics remain those of
``teleop_collect_example.py --mode recovery``; only camera ownership is
removed so the read-only capture process can keep the head stream.
"""

from __future__ import annotations

import argparse
import atexit
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teleop_collect_example
from teleop_collect_example import MinimalTeleopCollector


class NoCameraRecoveryTeleop(MinimalTeleopCollector):
    def _init_cameras(self):
        self.cameras = {}
        self.camera_threads = []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-relay",
        action="store_true",
        help="use a remote Tailscale relay instead of the direct lab Quest",
    )
    parser.add_argument("--quest-host", default="192.168.1.106")
    parser.add_argument("--quest-port", type=int, default=5555)
    parser.add_argument("--relay-host", default="100.125.255.41")
    parser.add_argument("--relay-port", type=int, default=6006)
    parser.add_argument("--relay-topic", default="oculus_controller")
    parser.add_argument("--vr-timeout", type=float, default=0.75)
    parser.add_argument("--safety-config")
    parser.add_argument(
        "--torque-config",
        default="src/configs/pasteur_lid_torque.json",
    )
    parser.add_argument(
        "--recovery-audit-log",
        default="/var/tmp/piper-recovery-teleop/torque-latest.jsonl",
    )
    args = parser.parse_args(argv)
    if (
        (args.use_relay and not args.relay_host)
        or (not args.use_relay and not args.quest_host)
        or args.vr_timeout <= 0.0
    ):
        parser.error("selected source host and positive VR timeout are required")
    # Namespace fields consumed by the unchanged recovery collector.
    if not args.use_relay:
        # The base recovery loop already has a direct-Quest branch.  Make the
        # lab endpoint explicit instead of requiring source edits.
        teleop_collect_example.VR_TCP_HOST = args.quest_host
        teleop_collect_example.VR_TCP_PORT = args.quest_port
    args.mode = "recovery"
    args.start_step = None
    args.start_index = 0
    args.no_display = True
    args.no_head_stream = True
    args.head_stream_host = "127.0.0.1"
    args.head_stream_port = 0
    args.head_stream_token = None
    args.head_stream_fps = 1.0
    # The two Pasteur arms are mechanically identical.  The dedicated left
    # envelope is collected after this calibration session; until then the
    # user explicitly chose the audited right-to-left fallback.
    args.allow_symmetric_left_torque_fallback = True
    # Human recovery control keeps Quest freshness, IK continuity, and the
    # measured hold on explicit disengagement. Torque is recorded as telemetry
    # only; pose-dependent gravity changes are not a reliable contact stop.
    args.enforce_recovery_torque_stop = False

    collector = NoCameraRecoveryTeleop(args)
    atexit.register(collector.stop)
    try:
        collector.control_loop()
    except KeyboardInterrupt:
        print("\n[RECOVERY] Stopped; current poses held, no home.", flush=True)
    finally:
        collector.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

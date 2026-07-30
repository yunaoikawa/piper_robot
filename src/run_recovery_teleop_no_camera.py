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

from teleop_collect_example import MinimalTeleopCollector


class NoCameraRecoveryTeleop(MinimalTeleopCollector):
    def _init_cameras(self):
        self.cameras = {}
        self.camera_threads = []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-host", default="100.125.255.41")
    parser.add_argument("--relay-port", type=int, default=6006)
    parser.add_argument("--relay-topic", default="oculus_controller")
    parser.add_argument("--vr-timeout", type=float, default=0.75)
    parser.add_argument("--safety-config")
    args = parser.parse_args(argv)
    if not args.relay_host or args.vr_timeout <= 0.0:
        parser.error("relay host and positive VR timeout are required")
    # Namespace fields consumed by the unchanged recovery collector.
    args.use_relay = True
    args.mode = "recovery"
    args.start_step = None
    args.start_index = 0
    args.no_display = True
    args.no_head_stream = True
    args.head_stream_host = "127.0.0.1"
    args.head_stream_port = 0
    args.head_stream_token = None
    args.head_stream_fps = 1.0

    collector = NoCameraRecoveryTeleop(args)
    atexit.register(collector.stop)
    try:
        collector.control_loop()
    finally:
        collector.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

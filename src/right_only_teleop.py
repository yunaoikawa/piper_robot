#!/usr/bin/env python3
"""Move only the right arm with a Quest controller.

The ConeE RPC server must already be running.  This process deliberately does
not call ``init`` or any left-arm RPC method.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import zmq
from loop_rate_limiters import RateLimiter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.rpc import RPCClient
from robot.teleop.oculus_msgs import parse_controller_state
from rollout.right_only_teleop import RightOnlyTeleop, RightTeleopEvent
from rollout.safety import SafetyLayer


def main():
    parser = argparse.ArgumentParser(description="Quest teleoperation for the right arm only")
    parser.add_argument("--relay-host", default="100.125.255.41")
    parser.add_argument("--relay-port", type=int, default=6006)
    parser.add_argument("--relay-topic", default="oculus_controller")
    parser.add_argument("--vr-timeout", type=float, default=0.75)
    parser.add_argument("--safety-config", default=None)
    args = parser.parse_args()

    rpc = RPCClient("localhost", 8081, timeout_ms=3000)
    safety = SafetyLayer.from_config(args.safety_config)
    teleop = RightOnlyTeleop(rpc, timeout_s=args.vr_timeout, safety=safety)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 10)
    socket.setsockopt(zmq.SUBSCRIBE, args.relay_topic.encode("utf-8"))
    endpoint = f"tcp://{args.relay_host}:{args.relay_port}"
    socket.connect(endpoint)

    print("[RIGHT-ONLY] No robot init, home, or left-arm RPC calls.", flush=True)
    print("[RIGHT-ONLY] A: engage right arm; B: disengage and hold; trigger: gripper.", flush=True)
    print(f"[RIGHT-ONLY] Waiting for Quest messages from {endpoint}", flush=True)

    latest = None
    last_message_at = time.monotonic()
    warned = False
    rate = RateLimiter(30.0)
    try:
        while True:
            while True:
                try:
                    parts = socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                payload = parts[1] if len(parts) >= 2 else parts[0]
                latest = parse_controller_state(payload.decode(errors="replace"))
                last_message_at = time.monotonic()
                warned = False

            if latest is not None:
                event = teleop.step(latest)
                if event == RightTeleopEvent.ENGAGED:
                    print("[RIGHT-ONLY] Right arm engaged.", flush=True)
                elif event == RightTeleopEvent.DISENGAGED:
                    print("[RIGHT-ONLY] Right arm disengaged; holding.", flush=True)
                elif event == RightTeleopEvent.STALE:
                    print("[RIGHT-ONLY] Quest stream stale; right arm disengaged.", flush=True)

            if time.monotonic() - last_message_at > 2.0 and not warned:
                print("[RIGHT-ONLY] WARNING: no Quest messages for >2s.", flush=True)
                warned = True
            rate.sleep()
    except KeyboardInterrupt:
        teleop.disengage()
        print("\n[RIGHT-ONLY] Stopped; no home command sent.", flush=True)
    finally:
        socket.close(0)
        context.destroy(linger=0)


if __name__ == "__main__":
    main()

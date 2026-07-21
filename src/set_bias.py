#!/usr/bin/env python3
"""Read or change the running controller's EE bias without restarting it.

Tuning a bias used to mean killing the inference server and relaunching with a
new --z-bias, which cost a minute and lost the action buffer. This talks to the
controller's bias port (rollout/controller.py `_bias_control_loop`) instead.

Runs against the robot host, so use the uv .venv (it has pyzmq; no torch needed).

  python src/set_bias.py                          # show current bias
  python src/set_bias.py --z -0.025               # right arm, 2.5cm deeper
  python src/set_bias.py --arm left --x 0.01
  python src/set_bias.py --reset

Values are metres in the robot frame and are clamped server-side to MAX_BIAS_M.
"""

import argparse
import sys

import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--arm", default="right", choices=["left", "right"])
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--z", type=float)
    ap.add_argument("--reset", action="store_true", help="set this arm's bias to 0")
    ap.add_argument("--timeout", type=int, default=3000, help="ms")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout)
    sock.setsockopt(zmq.SNDTIMEO, args.timeout)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{args.host}:{args.port}")

    def rpc(msg):
        sock.send_pyobj(msg)
        return sock.recv_pyobj()

    try:
        current = rpc({"command": "get_bias"})
    except zmq.Again:
        sys.exit(f"no response from {args.host}:{args.port} -- is the controller running?")

    if current.get("status") != "ok":
        sys.exit(f"error: {current}")

    if not args.reset and args.x is None and args.y is None and args.z is None:
        for arm, b in current["bias"].items():
            print(f"{arm:5s} bias = {b} m")
        print(f"safety rejections so far: {current['safety_rejected']}")
        return

    # Start from the arm's existing bias so a single axis can be nudged alone.
    bias = [0.0, 0.0, 0.0] if args.reset else list(current["bias"][args.arm])
    for i, v in enumerate((args.x, args.y, args.z)):
        if v is not None:
            bias[i] = v

    reply = rpc({"command": "set_bias", "arm": args.arm, "bias": bias})
    if reply.get("status") != "ok":
        sys.exit(f"error: {reply}")
    print(f"{reply['arm']} bias = {reply['bias']} m")


if __name__ == "__main__":
    main()

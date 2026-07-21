#!/usr/bin/env python3
"""Inspect or control a paused replay_demo checkpoint."""

import argparse
import sys

import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5561)
    ap.add_argument("--arm", choices=("left", "right"), default="right")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--snapshot", action="store_true")
    group.add_argument("--bias", type=float, nargs=3, metavar=("X", "Y", "Z"))
    group.add_argument("--resume", action="store_true")
    group.add_argument("--abort", action="store_true")
    ap.add_argument("--timeout", type=int, default=5000, help="milliseconds")
    args = ap.parse_args()

    if args.status:
        command = {"command": "status"}
    elif args.snapshot:
        command = {"command": "snapshot"}
    elif args.bias is not None:
        command = {"command": "adjust", "arm": args.arm, "bias": args.bias}
    elif args.resume:
        command = {"command": "resume"}
    else:
        command = {"command": "abort"}

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout)
    socket.connect(f"tcp://{args.host}:{args.port}")
    try:
        socket.send_pyobj(command)
        reply = socket.recv_pyobj()
    except zmq.Again:
        sys.exit(f"no checkpoint response from {args.host}:{args.port}")
    finally:
        socket.close()
        context.term()

    if reply.get("status") != "ok":
        sys.exit(f"error: {reply.get('message', reply)}")
    print(reply)


if __name__ == "__main__":
    main()

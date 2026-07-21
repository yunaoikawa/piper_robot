#!/usr/bin/env python3
"""Closed-loop XY exploration at a paused replay checkpoint."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.lid_vision import VisionProfile, detect_blue_marker
from rollout.apriltag_retarget import detect_tags


def bounded_step(jacobian, error, max_step_m=0.025, damping=1e-3):
    jacobian = np.asarray(jacobian, dtype=float).reshape(2, 2)
    error = np.asarray(error, dtype=float).reshape(2)
    lhs = jacobian.T @ jacobian + damping * np.eye(2)
    step = np.linalg.solve(lhs, jacobian.T @ error)
    norm = float(np.linalg.norm(step))
    if norm > max_step_m:
        step *= max_step_m / norm
    return step


def feature_from_reply(reply, profile, args):
    path = reply.get("images", {}).get("right")
    image = cv2.imread(path) if path else None
    if image is None:
        raise RuntimeError(f"checkpoint did not provide a right image: {reply}")
    if args.tracker == "tag":
        tags = detect_tags(image, args.tag_family)
        tag = next((item for item in tags if item.tag_id == args.tag_id), None)
        if tag is None:
            raise RuntimeError(f"tag {args.tag_family}/{args.tag_id} not found in {path}")
        return np.asarray(tag.center, dtype=float), path
    marker, _ = detect_blue_marker(image, profile)
    if marker is None:
        raise RuntimeError(f"blue cross not found in {path}")
    return np.asarray(marker, dtype=float), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5561)
    ap.add_argument("--vision-profile", default="src/configs/pasteur_lid_vision.json")
    ap.add_argument("--probe-mm", type=float, default=10.0)
    ap.add_argument("--step-mm", type=float, default=25.0)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--tolerance-px", type=float, default=5.0)
    ap.add_argument("--tracker", choices=("blue", "tag"), default="blue")
    ap.add_argument("--target-px", nargs=2, type=float)
    ap.add_argument("--tag-family", default="DICT_5X5_50")
    ap.add_argument("--tag-id", type=int, default=0)
    args = ap.parse_args()

    profile = VisionProfile.load(args.vision_profile)
    target = np.asarray(args.target_px if args.target_px else profile.marker_center,
                        dtype=float)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 15000)
    socket.setsockopt(zmq.SNDTIMEO, 5000)
    socket.connect(f"tcp://{args.host}:{args.port}")

    def request(command):
        try:
            socket.send_pyobj(command)
            reply = socket.recv_pyobj()
        except zmq.Again as exc:
            raise RuntimeError("checkpoint stopped responding (possibly torque stop)") from exc
        if reply.get("status") != "ok":
            raise RuntimeError(reply.get("message", str(reply)))
        return reply

    def adjust(bias):
        reply = request({"command": "adjust", "arm": "right",
                         "bias": [float(bias[0]), float(bias[1]), 0.0]})
        marker, path = feature_from_reply(reply, profile, args)
        print(f"bias={np.round(bias*1000, 1)}mm marker={np.round(marker, 1)}px "
              f"error={np.round(target-marker, 1)}px image={path}", flush=True)
        return marker

    try:
        status = request({"command": "status"})
        base = np.asarray(status["bias"]["right"][:2], dtype=float)
        base_marker, _ = feature_from_reply(status, profile, args)
        probe = args.probe_mm / 1000.0
        jacobian = np.empty((2, 2), dtype=float)
        for axis in range(2):
            trial = base.copy()
            trial[axis] += probe
            observed = adjust(trial)
            jacobian[:, axis] = (observed - base_marker) / probe
        current_bias = base.copy()
        current_marker = adjust(current_bias)
        print(f"initial Jacobian px/m:\n{np.round(jacobian, 1)}", flush=True)

        for iteration in range(args.max_iters):
            error = target - current_marker
            if np.linalg.norm(error) <= args.tolerance_px:
                print(f"CONVERGED iteration={iteration} bias={current_bias.tolist()}")
                return
            step = bounded_step(jacobian, error, args.step_mm / 1000.0)
            candidate = current_bias + step
            observed = adjust(candidate)
            feature_delta = observed - current_marker
            denominator = float(step @ step)
            if denominator > 1e-10:
                jacobian += np.outer(feature_delta - jacobian @ step, step) / denominator
            current_bias, current_marker = candidate, observed
        raise RuntimeError(f"did not converge in {args.max_iters} iterations")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()

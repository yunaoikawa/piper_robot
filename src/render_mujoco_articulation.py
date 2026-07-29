#!/usr/bin/env python3
"""Render a small, simulation-only articulation check for a MuJoCo scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def render_articulation(
    model_path: str | Path,
    output_path: str | Path,
    *,
    frames: int = 60,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nkey:
        try:
            key_id = int(model.key("synchronized").id)
        except KeyError:
            key_id = 0
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    initial = data.qpos.copy()
    movable = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id])
        in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }
    ]
    if not movable:
        return {"ok": False, "reason": "model_has_no_articulated_joint"}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the articulation video")
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for frame_index in range(frames):
            phase = 2.0 * np.pi * frame_index / max(1, frames - 1)
            data.qpos[:] = initial
            for joint_rank, joint_id in enumerate(movable):
                address = int(model.jnt_qposadr[joint_id])
                low, high = model.jnt_range[joint_id]
                span = float(high - low)
                amplitude = min(0.10, max(0.01, 0.08 * abs(span)))
                proposed = initial[address] + amplitude * np.sin(
                    phase + 0.35 * joint_rank
                )
                if bool(model.jnt_limited[joint_id]):
                    proposed = float(np.clip(proposed, low, high))
                data.qpos[address] = proposed
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        renderer.close()
        writer.release()
    return {
        "ok": True,
        "path": str(output.resolve()),
        "frames": int(frames),
        "fps": int(fps),
        "joint_count": len(movable),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args(argv)
    print(
        render_articulation(
            args.model,
            args.output,
            frames=args.frames,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()

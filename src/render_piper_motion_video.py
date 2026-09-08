#!/usr/bin/env python3
"""Render a short MuJoCo-native Piper articulation verification video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def render(args):
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    manifest = json.loads((Path(args.capture) / "manifest.json").read_text())
    state = manifest["robot_state"]["after"]
    scene = json.loads(Path(args.scene_report).read_text())
    object_report = scene["objects"]
    yaw = np.deg2rad(
        object_report["right_robot"]["shared_upright_yaw_deg"]
    )
    base_quaternion = np.array(
        [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    )
    try:
        model.joint("right/joint1")
        complete_model = True
    except KeyError:
        complete_model = False
    for side in ("right", "left"):
        if complete_model:
            body = model.body(f"{side}/base_link")
        else:
            prefix = "left_" if side == "left" else ""
            body = model.body(f"{prefix}base_link")
        pose = object_report[f"{side}_robot"]["base_xyz_level_m"]
        model.body_pos[body.id] = pose
        model.body_quat[body.id] = base_quaternion

    q_targets = {
        "right": np.asarray(state["right_joint_positions_rad"], float),
        "left": np.asarray(state["left_joint_positions_rad"], float),
    }
    renderer = mujoco.Renderer(
        model, height=args.height, width=args.width
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.10, 0.24, 0.02]
    camera.distance = 1.05
    camera.azimuth = 145
    camera.elevation = -32
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[2] = 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}",
        "-r", str(args.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_count = int(args.seconds * args.fps)
    try:
        for frame_index in range(frame_count):
            phase = 2 * np.pi * frame_index / max(frame_count - 1, 1)
            envelope = np.sin(phase)
            for side, target in q_targets.items():
                direction = -1.0 if side == "left" else 1.0
                offsets = direction * envelope * np.array(
                    [0.18, 0.10, -0.08, 0.12, 0.16, 0.14]
                )
                for joint_index, value in enumerate(
                    target[:len(offsets)] + offsets, 1
                ):
                    if complete_model:
                        joint = model.joint(f"{side}/joint{joint_index}")
                    else:
                        prefix = "left_" if side == "left" else ""
                        if joint_index == 6:
                            continue
                        joint = model.joint(
                            f"{prefix}joint{joint_index}"
                        )
                    data.qpos[int(joint.qposadr[0])] = np.clip(
                        value,
                        model.jnt_range[joint.id, 0] + 0.01,
                        model.jnt_range[joint.id, 1] - 0.01,
                    )
            mujoco.mj_forward(model, data)
            renderer.update_scene(
                data, camera=camera, scene_option=scene_option
            )
            encoder.stdin.write(renderer.render().tobytes())
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        return_code = encoder.wait()
        renderer.close()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with status {return_code}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="robot/arm/mujoco/bimanual_piper_table.xml",
    )
    parser.add_argument("--capture", required=True)
    parser.add_argument("--scene-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    print(render(args))


if __name__ == "__main__":
    main()

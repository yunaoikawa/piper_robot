#!/usr/bin/env python3
"""Render a phone-readable MuJoCo preview of one dish air-transport plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.arm.home import physical_to_semantic_model_q_offset, semantic_model_home_q
from rollout.dish_transport_rehearsal import _insert_virtual_dish


def render(plan_path: Path, output: Path, *, width: int, height: int, fps: int) -> Path:
    manifest = json.loads(plan_path.read_text())
    if not manifest.get("plans"):
        raise ValueError("plan manifest contains no transport")
    plan = manifest["plans"][0]
    scene = Path(json.loads(Path(manifest["config"]).read_text())["planning_model"])
    if not scene.is_absolute():
        scene = ROOT / scene
    dish = plan["collision_audit"]["virtual_dish"]
    generated = _insert_virtual_dish(
        scene, float(dish["radius_m"]), float(dish["thickness_m"])
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(generated))
    finally:
        generated.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    q_path = np.asarray(plan["q_physical_rad"], dtype=float)
    checkpoint_indices = [item["pose_index"] for item in plan["checkpoints"]]
    carry_start, carry_stop = checkpoint_indices[0], checkpoint_indices[-1]
    right_ids = np.asarray(
        [model.joint(f"right/joint{i}").qposadr[0] for i in range(1, 7)]
    )
    left_ids = np.asarray(
        [model.joint(f"left/joint{i}").qposadr[0] for i in range(1, 7)]
    )
    data.qpos[left_ids] = semantic_model_home_q("left")
    dish_body = model.body("virtual-carried-dish")
    dish_mocap = int(dish_body.mocapid[0])
    right_ee = model.site("right/ee")
    tool_offset = np.asarray(dish["center_offset_ee_m"], dtype=float)

    renderer = mujoco.Renderer(model, height=height, width=width)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.geomgroup[2] = 1
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.14, 0.82, -0.28]
    camera.distance = 1.45
    camera.azimuth = 45
    camera.elevation = -24

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_indices = []
    checkpoint_set = set(checkpoint_indices)
    for index in range(len(q_path)):
        frame_indices.append(index)
        if index in checkpoint_set:
            frame_indices.extend([index] * max(1, fps // 2))
    frame_indices.extend(range(len(q_path) - 1, -1, -1))
    try:
        for index in frame_indices:
            data.qpos[right_ids] = (
                q_path[index] + physical_to_semantic_model_q_offset("right")
            )
            data.mocap_pos[dish_mocap] = np.asarray([0.0, 0.0, 2.0])
            mujoco.mj_forward(model, data)
            if carry_start <= index <= carry_stop:
                rotation = np.asarray(data.site_xmat[right_ee.id]).reshape(3, 3)
                data.mocap_pos[dish_mocap] = data.site_xpos[right_ee.id] + rotation @ tool_offset
                data.mocap_quat[dish_mocap] = np.asarray([1.0, 0.0, 0.0, 0.0])
                mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            encoder.stdin.write(renderer.render().tobytes())
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        status = encoder.wait()
        renderer.close()
    if status:
        raise RuntimeError(f"ffmpeg failed with status {status}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    print(render(args.plan, args.output, width=args.width, height=args.height, fps=args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

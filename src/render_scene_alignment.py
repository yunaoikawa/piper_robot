#!/usr/bin/env python3
"""Render a stopped, display-only audit of the aligned semantic scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.render_home_lid_trajectory import (
    _add_objects,
    _apply_visibility_policy,
    _arm_mapping,
)


def _visible_extent(model: mujoco.MjModel, data: mujoco.MjData) -> tuple:
    points = []
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) in (3, 5):
            continue
        points.append(np.asarray(data.geom_xpos[geom_id], dtype=float))
    values = np.asarray(points)
    return values.mean(axis=0), float(np.ptp(values, axis=0).max())


def render(args) -> dict:
    report = json.loads(Path(args.alignment_report).read_text())
    object_scene = json.loads(Path(args.object_scene).read_text())
    spec = mujoco.MjSpec.from_file(str(Path(args.model).resolve()))
    _add_objects(spec, object_scene)
    model = spec.compile()
    data = mujoco.MjData(model)
    mapping = _arm_mapping(model)
    _apply_visibility_policy(model, mapping)
    model.vis.headlight.ambient[:] = (0.55, 0.55, 0.55)
    model.vis.headlight.diffuse[:] = (0.85, 0.85, 0.85)
    for geom_id in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom_id]).name
        if body_name.startswith("microscope-"):
            model.geom_rgba[geom_id] = (0.85, 0.15, 0.85, 1.0)
        elif body_name.startswith("support-"):
            model.geom_rgba[geom_id] = (0.42, 0.46, 0.52, 1.0)
    key_id = int(model.key(mapping["key"]).id)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    left_ee = np.asarray(data.site_xpos[model.site("left/ee").id])
    right_ee = np.asarray(data.site_xpos[model.site("right/ee").id])
    center, span = _visible_extent(model, data)
    contact_positions = []
    for contact in data.contact:
        first = model.body(model.geom_bodyid[contact.geom1]).name
        second = model.body(model.geom_bodyid[contact.geom2]).name
        first_robot = first.startswith(("left/", "right/"))
        second_robot = second.startswith(("left/", "right/"))
        if first_robot != second_robot:
            contact_positions.append(np.asarray(contact.pos, dtype=float))
    if args.focus == "contacts" and contact_positions:
        center = np.median(np.asarray(contact_positions), axis=0)
        span = min(span, 0.65)
        for geom_id in range(model.ngeom):
            body_name = model.body(model.geom_bodyid[geom_id]).name
            if body_name.startswith(("right/", "incubator-")):
                model.geom_group[geom_id] = 5
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = (
        max(0.7, span * 1.65)
        if args.focus == "contacts"
        else max(1.4, span * 1.65)
    )
    camera.elevation = args.elevation_deg
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[0] = 1
    options.geomgroup[2] = 1

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_total = max(1, int(round(args.duration_s * args.fps)))
    with mujoco.Renderer(
        model,
        height=args.height,
        width=args.width,
    ) as renderer:
        for frame_index in range(frame_total):
            phase = frame_index / max(1, frame_total - 1)
            camera.azimuth = (
                args.azimuth_deg
                + args.orbit_deg * np.sin(phase * 2.0 * np.pi)
            )
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame = renderer.render()
            frame = np.asarray(frame, dtype=np.uint8)
            cv2.rectangle(
                frame,
                (12, 12),
                (args.width - 12, 82),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                "HOME FIXED | LATEST LID | DISPLAY ONLY",
                (24, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            reasons = report.get("trajectory_gate", {}).get("reasons", [])
            cv2.putText(
                frame,
                "BLOCKED: " + ", ".join(reasons),
                (24, 69),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (120, 190, 255),
                1,
                cv2.LINE_AA,
            )
            encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")

    result = {
        "schema": "piper_robot.scene_alignment_render/v1",
        "display_only": True,
        "commands_sent": False,
        "trajectory_authorized": bool(
            report.get("trajectory_gate", {}).get("authorized", False)
        ),
        "trajectory_gate_reasons": report.get(
            "trajectory_gate", {}
        ).get("reasons", []),
        "model": str(Path(args.model).resolve()),
        "object_scene": str(Path(args.object_scene).resolve()),
        "video": str(output),
        "home": {
            "keyframe": mapping["key"],
            "left_ee_z_m": float(left_ee[2]),
            "right_ee_z_m": float(right_ee[2]),
            "ee_height_difference_m": float(abs(left_ee[2] - right_ee[2])),
        },
        "focus": args.focus,
        "robot_environment_contact_count": len(contact_positions),
    }
    Path(args.report).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--object-scene", required=True)
    parser.add_argument("--alignment-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--azimuth-deg", type=float, default=135.0)
    parser.add_argument("--elevation-deg", type=float, default=-22.0)
    parser.add_argument("--orbit-deg", type=float, default=12.0)
    parser.add_argument(
        "--focus",
        choices=("scene", "contacts"),
        default="scene",
    )
    args = parser.parse_args(argv)
    print(json.dumps(render(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

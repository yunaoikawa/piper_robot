#!/usr/bin/env python3
"""Render a planned home-to-lid trajectory without sending robot commands."""

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

from robot.arm.home import mujoco_home_qpos, physical_home_q
from rollout.home_lid_trajectory import SCHEMA


def _add_objects(spec, object_scene):
    records = {}
    for item in object_scene.get("objects", ()):
        geometry = item.get("geometry", {})
        if geometry.get("type") != "cylinder":
            continue
        role = str(item.get("role"))
        pose = item.get("pose_scene", item.get("pose_robot"))
        if pose is not None:
            position = np.asarray(pose, dtype=float)[:3, 3]
        elif role == "target_lid":
            position = np.asarray(
                item["grasp_ee_pose_robot_wxyz_xyz"][4:],
                dtype=float,
            )
        else:
            continue
        radius = float(geometry["radius_m"])
        height = float(geometry["height_m"])
        color = (
            [0.15, 0.55, 1.0, 0.75]
            if role == "target_lid"
            else [0.95, 0.65, 0.15, 0.70]
        )
        body = spec.worldbody.add_body(
            name=f"trajectory_{role}",
            pos=position,
        )
        body.add_geom(
            name=f"trajectory_{role}_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, height / 2.0, 0.0],
            rgba=color,
            contype=0,
            conaffinity=0,
        )
        records[role] = body.name
    return records


def _joint_ids(model, prefix):
    return [
        int(model.joint(f"{prefix}joint{index}").qposadr[0])
        for index in range(1, 7)
    ]


def _arm_mapping(model):
    if mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left/joint1"
    ) >= 0:
        right_ids = [
            int(model.joint(f"left/joint{index}").qposadr[0])
            for index in range(1, 7)
        ]
        left_ids = [
            int(model.joint(f"right/joint{index}").qposadr[0])
            for index in range(1, 7)
        ]
        return {
            "variant": "sam_reconstruction_upright_nyu",
            "key": "home",
            "right_ids": right_ids,
            "left_ids": left_ids,
            "ee": "left/ee",
            "right_body_prefix": "left/",
            "left_body_prefix": "right/",
        }
    return {
        "variant": "legacy_cone_e",
        "key": "lab_home",
        "right_ids": _joint_ids(model, "left_arm_"),
        "left_ids": _joint_ids(model, "right_arm_"),
        "ee": "left_arm_ee",
        "right_body_prefix": "left_arm_",
        "left_body_prefix": "right_arm_",
    }


def _stage_text(frame, stage, display_only):
    label = f"HOME FIXED | {stage}"
    if display_only:
        label += " | DISPLAY ONLY"
    cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 570), 62), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (25, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _apply_visibility_policy(model, mapping):
    """Keep semantic observed meshes visible; hide only the dense raw scan."""

    for geom_id in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom_id]).name
        if body_name == "measured-static-scene-observed":
            model.geom_group[geom_id] = 5
            continue
        if body_name.startswith(mapping["right_body_prefix"]):
            model.geom_rgba[geom_id] = [0.95, 0.45, 0.10, 1.0]
        elif body_name.startswith(mapping["left_body_prefix"]):
            model.geom_rgba[geom_id] = [0.05, 0.75, 0.85, 1.0]


def render(args):
    payload = json.loads(Path(args.plan).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported trajectory plan schema")
    q_waypoints = np.asarray(payload["mujoco_q_waypoints"], dtype=float)
    waypoints = payload["plan"]["waypoints"]
    if q_waypoints.ndim != 2 or q_waypoints.shape[1] != 6:
        raise ValueError("mujoco_q_waypoints must be Nx6")
    if len(q_waypoints) != len(waypoints):
        raise ValueError("waypoint pose/qpos counts differ")

    spec = mujoco.MjSpec.from_file(str(Path(args.model).resolve()))
    added = _add_objects(spec, payload["object_scene"])
    model = spec.compile()
    data = mujoco.MjData(model)
    mapping = _arm_mapping(model)
    # Simulation-only colours make the two otherwise dark, overlapping CAD
    # branches legible on a phone: physical right/model-left moves; physical
    # left/model-right remains at home.
    _apply_visibility_policy(model, mapping)
    key_id = int(model.key(mapping["key"]).id)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    right_ids = mapping["right_ids"]
    left_ids = mapping["left_ids"]
    right_physical_home = physical_home_q("right")
    left_physical_home = physical_home_q("left")
    right_model_home = right_physical_home.copy()
    left_model_home = left_physical_home.copy()
    data.qpos[right_ids] = right_model_home
    data.qpos[left_ids] = left_model_home
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    # Display only physical object/CAD visuals. Dense group-3 collision voxels
    # remain active in planning but otherwise occlude the phone-sized render.
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.geomgroup[2] = 1
    render_camera = args.camera
    if args.camera == "trajectory":
        render_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(render_camera)
        render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        robot_points = np.asarray([
            data.xpos[body_id]
            for body_id in range(1, model.nbody)
            if model.body(body_id).name.startswith(
                (
                    mapping["right_body_prefix"],
                    mapping["left_body_prefix"],
                )
            )
        ])
        render_camera.lookat[:] = np.mean(robot_points, axis=0)
        render_camera.distance = max(
            1.0, float(np.max(np.ptp(robot_points, axis=0))) * 2.8
        )
        render_camera.azimuth = 135.0
        render_camera.elevation = -20.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
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
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    close_t = next(
        (
            float(item["t_s"])
            for item in payload["gripper_events"]
            if item["action"] == "close"
        ),
        float("inf"),
    )
    lid_body_id = (
        int(model.body(added["target_lid"]).id)
        if "target_lid" in added
        else None
    )
    ee_site_id = int(model.site(mapping["ee"]).id)
    if lid_body_id is not None:
        lid = next(
            item
            for item in payload["object_scene"]["objects"]
            if item.get("role") == "target_lid"
        )
        if "grasp_right_q_rad" in lid:
            grasp_q = np.asarray(lid["grasp_right_q_rad"], dtype=float)
            data.qpos[right_ids] = grasp_q
            data.qpos[left_ids] = left_model_home
            mujoco.mj_forward(model, data)
            model.body_pos[lid_body_id] = np.asarray(
                data.site_xpos[ee_site_id], dtype=float
            )
            data.qpos[right_ids] = right_model_home
            data.qpos[left_ids] = left_model_home
            mujoco.mj_forward(model, data)
    home_frames = max(1, int(args.home_hold_s * args.fps))
    final_frames = max(1, int(args.final_hold_s * args.fps))
    frame_count = 0
    left_values = []

    def emit(stage):
        nonlocal frame_count
        renderer.update_scene(data, camera=render_camera, scene_option=option)
        frame = renderer.render()
        _stage_text(frame, stage, bool(payload["display_only"]))
        encoder.stdin.write(frame.tobytes())
        frame_count += 1
        left_values.append(data.qpos[left_ids].copy())

    try:
        for _ in range(home_frames):
            emit("HOME")
        for q, waypoint in zip(q_waypoints, waypoints):
            data.qpos[right_ids] = q
            data.qpos[left_ids] = left_model_home
            mujoco.mj_forward(model, data)
            if lid_body_id is not None and float(waypoint["t_s"]) >= close_t:
                model.body_pos[lid_body_id] = np.asarray(
                    data.site_xpos[ee_site_id], dtype=float
                )
                mujoco.mj_forward(model, data)
            emit(str(waypoint["stage"]).upper())
        for _ in range(final_frames):
            emit("GRASP VERIFIED / LIFT")
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        return_code = encoder.wait()
        renderer.close()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with status {return_code}")

    left_values = np.asarray(left_values)
    report = {
        "schema": "piper_robot.home_lid_trajectory_render/v1",
        "plan": str(Path(args.plan).resolve()),
        "video": str(output.resolve()),
        "camera": args.camera,
        "frame_count": frame_count,
        "fps": args.fps,
        "model_variant": mapping["variant"],
        "model_path": str(Path(args.model).resolve()),
        "first_mujoco_qpos": np.concatenate(
            [right_model_home, left_model_home]
        ).tolist(),
        "physical_home_q": mujoco_home_qpos().tolist(),
        "left_home_q": left_model_home.tolist(),
        "left_maximum_drift_rad": float(
            np.max(np.abs(left_values - left_model_home))
        ),
        "right_first_waypoint_delta_from_home_rad": float(
            np.max(np.abs(q_waypoints[0] - right_physical_home))
        ),
        "display_only": bool(payload["display_only"]),
        "commands_sent": False,
    }
    report_path = Path(args.report) if args.report else output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return output, report_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--model",
        default="robot/pasteur-calibrated-scene/scene.mjcf",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--camera", default="trajectory")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--home-hold-s", type=float, default=1.0)
    parser.add_argument("--final-hold-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    if (
        args.width <= 0
        or args.height <= 0
        or args.fps <= 0
        or args.home_hold_s < 0
        or args.final_hold_s < 0
    ):
        parser.error("invalid render dimensions, rate, or hold duration")
    output, report = render(args)
    print(output)
    print(report)


if __name__ == "__main__":
    main()

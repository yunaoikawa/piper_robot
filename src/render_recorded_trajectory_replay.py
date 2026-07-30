#!/usr/bin/env python3
"""Render the audited stopped-state replay without commanding hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.arm.home import physical_home_q


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _add_geom(scene, mujoco, *, geom_type, size, pos, mat, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float),
        np.asarray(mat, dtype=float).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_connector(scene, mujoco, start, finish, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.004,
        np.asarray(start, dtype=float),
        np.asarray(finish, dtype=float),
    )
    scene.ngeom += 1


def _object_centers(object_scene: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    target = object_scene["objects"][0]
    latest = np.asarray(target["pose_scene"], dtype=float)[:3, 3]
    episodes = object_scene.get("source", {}).get(
        "episode_targets_scene_xyz_m",
        {},
    )
    recorded = np.asarray(
        episodes.get("successful_grasp_before_lift", latest),
        dtype=float,
    )
    # All transparent target centers share the measured support surface.
    recorded[2] = latest[2]
    return recorded, latest, target["geometry"]


def render(
    model_path: Path,
    trajectory_path: Path,
    output_dir: Path,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 20,
    camera_azimuth_deg: float = 155.0,
    camera_elevation_deg: float = -24.0,
    camera_distance_m: float = 1.75,
) -> dict:
    import mujoco

    model_path = model_path.resolve()
    trajectory_path = trajectory_path.resolve()
    replay = json.loads(trajectory_path.read_text())
    if replay.get("commands_sent") is not False:
        raise ValueError("replay lacks commands_sent=false provenance")
    if not replay["validation"]["all_keyframes_exact"]:
        raise ValueError("replay keyframe exactness gate failed")
    if not replay["validation"]["moving_arm_path_clear"]:
        raise ValueError("moving-arm collision gate failed")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    # The fused static observation is useful as provenance but forms a large,
    # dark one-sided shell that occludes CAD from many synthetic cameras.
    # Hide only that display geom; collision geometry and semantic objects are
    # unchanged.
    try:
        model.geom("measured-static-scene-visual").group = 5
    except KeyError:
        pass
    data = mujoco.MjData(model)
    moving_branch = replay["physical_to_model_branch"]["right"]
    static_branch = replay["physical_to_model_branch"]["left"]
    # The current ConeE-derived model historically calls the physical-right
    # branch left/. Make physical identity visually unambiguous instead of
    # asking the viewer to infer it from the internal MJCF name.
    for geom_id in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_id])).name
        if body_name.startswith(f"{moving_branch}/"):
            model.geom_rgba[geom_id, :3] = [0.10, 0.82, 0.95]
            model.geom_rgba[geom_id, 3] = max(
                0.82,
                float(model.geom_rgba[geom_id, 3]),
            )
        elif body_name.startswith(f"{static_branch}/"):
            model.geom_rgba[geom_id, :3] = [0.32, 0.35, 0.40]
            model.geom_rgba[geom_id, 3] = min(
                0.42,
                float(model.geom_rgba[geom_id, 3]),
            )
    moving_ids = [
        int(model.joint(f"{moving_branch}/joint{index}").qposadr[0])
        for index in range(1, 7)
    ]
    static_ids = [
        int(model.joint(f"{static_branch}/joint{index}").qposadr[0])
        for index in range(1, 7)
    ]
    site_id = int(model.site(f"{moving_branch}/ee").id)
    samples = replay["samples"]
    duration = float(samples[-1]["t_s"])
    frame_times = np.arange(0.0, duration, 1.0 / fps)
    if not len(frame_times) or frame_times[-1] != duration:
        frame_times = np.r_[frame_times, duration]
    sample_times = np.asarray([item["t_s"] for item in samples], dtype=float)
    selected_indices = np.searchsorted(sample_times, frame_times, side="left")
    selected_indices = np.clip(selected_indices, 0, len(samples) - 1)
    previous = np.maximum(selected_indices - 1, 0)
    choose_previous = (
        np.abs(sample_times[previous] - frame_times)
        < np.abs(sample_times[selected_indices] - frame_times)
    )
    selected_indices[choose_previous] = previous[choose_previous]

    object_scene = replay["object_scene"]
    recorded_target, latest_target, geometry = _object_centers(object_scene)
    radius = float(geometry["radius_m"])
    half_height = float(geometry["height_m"]) / 2.0
    static_home = physical_home_q("left")

    # Compute the complete EE path once, and the measured closed attachment.
    ee_poses = []
    closed_sample_index = None
    for index, sample in enumerate(samples):
        data.qpos[moving_ids] = sample["right_q_model_rad"]
        data.qpos[static_ids] = static_home
        mujoco.mj_forward(model, data)
        ee_poses.append(
            _transform(
                data.site_xmat[site_id].reshape(3, 3),
                data.site_xpos[site_id],
            )
        )
        if (
            closed_sample_index is None
            and sample["stage"] == "closed_nonempty"
            and sample["measured_endpoint"]
        ):
            closed_sample_index = index
    if closed_sample_index is None:
        raise ValueError("closed measured keyframe not found")
    world_from_target_at_close = np.eye(4)
    world_from_target_at_close[:3, 3] = recorded_target
    ee_from_target = (
        np.linalg.inv(ee_poses[closed_sample_index])
        @ world_from_target_at_close
    )
    path_positions = np.asarray([pose[:3, 3] for pose in ee_poses])
    path_indices = np.unique(
        np.linspace(
            0,
            len(path_positions) - 1,
            min(70, len(path_positions)),
            dtype=int,
        )
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([-0.05, 1.02, -0.42])
    camera.distance = float(camera_distance_m)
    camera.azimuth = float(camera_azimuth_deg)
    camera.elevation = float(camera_elevation_deg)
    option = mujoco.MjvOption()
    option.geomgroup[:] = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    renderer = mujoco.Renderer(model, height=height, width=width)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "recorded_replay.mp4"
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
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    first_frame = None
    final_frame = None
    try:
        for frame_number, (timestamp, sample_index) in enumerate(
            zip(frame_times, selected_indices)
        ):
            sample = samples[int(sample_index)]
            data.qpos[moving_ids] = sample["right_q_model_rad"]
            data.qpos[static_ids] = static_home
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            # Full planned EE trajectory: display only, never collision input.
            for first, second in zip(
                path_positions[path_indices][:-1],
                path_positions[path_indices][1:],
            ):
                _add_connector(
                    renderer.scene,
                    mujoco,
                    first,
                    second,
                    [0.10, 0.85, 0.95, 0.42],
                )
            if int(sample_index) >= closed_sample_index:
                target_pose = ee_poses[int(sample_index)] @ ee_from_target
            else:
                target_pose = np.eye(4)
                target_pose[:3, 3] = recorded_target
            _add_geom(
                renderer.scene,
                mujoco,
                geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[radius, half_height, 0.0],
                pos=target_pose[:3, 3],
                mat=target_pose[:3, :3],
                rgba=[0.98, 0.72, 0.08, 0.86],
            )
            # Latest post-drop estimate, shown as a non-physical blue ghost.
            _add_geom(
                renderer.scene,
                mujoco,
                geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[radius, half_height * 0.45, 0.0],
                pos=latest_target + np.array([0.0, 0.0, 0.004]),
                mat=np.eye(3),
                rgba=[0.10, 0.65, 1.0, 0.28],
            )
            frame = renderer.render()
            cv2.putText(
                frame,
                (
                    f"OFFLINE REPLAY  t={timestamp:05.2f}s  "
                    f"{sample['stage']}"
                ),
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                (
                    "CYAN ROBOT = PHYSICAL RIGHT ARM "
                    f"(internal MJCF {moving_branch}/)"
                ),
                (14, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (90, 235, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                (
                    "cyan robot/path=physical RIGHT  gray=physical LEFT home  "
                    "yellow=recorded target  blue=latest"
                ),
                (14, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            if first_frame is None:
                first_frame = frame.copy()
            final_frame = frame.copy()
            assert encoder.stdin is not None
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
        renderer.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    assert first_frame is not None and final_frame is not None
    cv2.imwrite(
        str(output_dir / "recorded_replay_start.png"),
        cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(output_dir / "recorded_replay_final.png"),
        cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR),
    )
    report = {
        "schema": "piper_robot.recorded_trajectory_render/v1",
        "commands_sent": False,
        "observation_only": True,
        "model": str(model_path),
        "trajectory": str(trajectory_path),
        "video": str(video_path.resolve()),
        "start_png": str(
            (output_dir / "recorded_replay_start.png").resolve()
        ),
        "final_png": str(
            (output_dir / "recorded_replay_final.png").resolve()
        ),
        "duration_s": duration,
        "fps": fps,
        "frames": len(frame_times),
        "physical_right_model_branch": moving_branch,
        "moving_physical_arm": "right",
        "static_physical_arm": "left",
        "arm_display_policy": (
            "physical-right robot is cyan; static physical-left robot is gray"
        ),
        "static_physical_left_home_q_rad": static_home.tolist(),
        "recorded_target_center_scene_m": recorded_target.tolist(),
        "latest_target_center_scene_m": latest_target.tolist(),
        "target_shift_recorded_to_latest_m": (
            latest_target - recorded_target
        ).tolist(),
        "render_is_display_only": True,
        "camera": {
            "azimuth_deg": camera_azimuth_deg,
            "elevation_deg": camera_elevation_deg,
            "distance_m": camera_distance_m,
        },
    }
    (output_dir / "render_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-azimuth-deg", type=float, default=155.0)
    parser.add_argument("--camera-elevation-deg", type=float, default=-24.0)
    parser.add_argument("--camera-distance-m", type=float, default=1.75)
    args = parser.parse_args(argv)
    report = render(
        args.model,
        args.trajectory,
        args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_azimuth_deg=args.camera_azimuth_deg,
        camera_elevation_deg=args.camera_elevation_deg,
        camera_distance_m=args.camera_distance_m,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

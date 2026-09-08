#!/usr/bin/env python3
"""Build the Pasteur MuJoCo scene from persisted fixture and RGB-D evidence.

The world origin is the right Piper base.  The measured right-Piper-to-
incubator vector is the primary XY constraint; SAM is used for semantic
exclusion and object bounds, never as an unconstrained global pose solver.
Both Piper bases are positioned independently from RGB-D anchors and remain
upright.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


def _fixture_registration(calibration, fixture):
    """Align levelled RGB-D using the base-to-incubator correspondence."""
    anchors = calibration["anchor_xyz_level_m"]
    source_base = np.asarray(anchors["right_piper_base"], dtype=float)
    source_incubator = np.asarray(anchors["incubator"], dtype=float)
    target_base = np.asarray(fixture["right_piper_base_xyz_m"], dtype=float)
    target_incubator = np.asarray(
        fixture["incubator"]["center_xyz_from_mount_m"], dtype=float
    )
    source_vector = source_incubator[:2] - source_base[:2]
    target_vector = target_incubator[:2] - target_base[:2]
    yaw = np.arctan2(target_vector[1], target_vector[0]) - np.arctan2(
        source_vector[1], source_vector[0]
    )
    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    right_platform = float(
        calibration["level_heights_m"]["right_platform"]
    )
    target_platform = float(
        fixture["vertical_offsets_m"][
            "right_platform_above_right_piper_mount"
        ]
    )
    translation = target_base - rotation @ source_base
    translation[2] = target_platform - right_platform
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mapped_incubator = rotation @ source_incubator + translation
    return transform, {
        "method": "right_piper_base_to_incubator_vector",
        "primary_constraint": "incubator_robot_distance_and_bearing",
        "yaw_deg": float(np.rad2deg(yaw)),
        "translation_xyz_m": translation.tolist(),
        "source_distance_m": float(np.linalg.norm(source_vector)),
        "target_distance_m": float(np.linalg.norm(target_vector)),
        "distance_disagreement_m": float(
            abs(np.linalg.norm(source_vector) - np.linalg.norm(target_vector))
        ),
        "mapped_incubator_rgbd_anchor_m": mapped_incubator.tolist(),
        "configured_incubator_center_m": target_incubator.tolist(),
        "xy_residual_m": float(
            np.linalg.norm(mapped_incubator[:2] - target_incubator[:2])
        ),
    }


def _write_positioned_robot_model(
    source_path, destination, left_base, right_base
):
    """Write a derived independent-arm model with measured base positions."""
    source_path = Path(source_path).resolve()
    xml = source_path.read_text()
    xml = xml.replace(
        '<compiler angle="radian" meshdir="assets" />',
        f'<compiler angle="radian" meshdir="{source_path.parent / "assets"}" />',
    )
    replacements = {
        '<body name="left/base_link" pos="0 0.25 0"':
            f'<body name="left/base_link" pos="{left_base[0]:.6f} '
            f'{left_base[1]:.6f} {left_base[2]:.6f}"',
        '<body name="right/base_link" pos="0 -0.25 0"':
            f'<body name="right/base_link" pos="{right_base[0]:.6f} '
            f'{right_base[1]:.6f} {right_base[2]:.6f}"',
    }
    for old, new in replacements.items():
        if old not in xml:
            raise RuntimeError(f"robot base declaration not found: {old}")
        xml = xml.replace(old, new, 1)
    Path(destination).write_text(xml)


def _capture_geometry(capture_dir, report):
    capture_dir = Path(capture_dir)
    sequence = int(report["sequence"])
    frame_dir = capture_dir / "raw" / "head" / f"{sequence:06d}"
    metadata = json.loads((frame_dir / "meta.json").read_text())
    rgb = cv2.rotate(
        cv2.imread(str(frame_dir / "rgb.png")),
        cv2.ROTATE_90_CLOCKWISE,
    )
    depth = cv2.rotate(
        np.load(frame_dir / "depth.npy"), cv2.ROTATE_90_CLOCKWISE
    ).astype(float)
    confidence = cv2.rotate(
        np.load(frame_dir / "confidence.npy"), cv2.ROTATE_90_CLOCKWISE
    )
    matrix = np.asarray(
        metadata["intrinsics"]["K_rgb_rotated_clockwise"], dtype=float
    )
    matrix[0] *= depth.shape[1] / rgb.shape[1]
    matrix[1] *= depth.shape[0] / rgb.shape[0]
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    camera_points = np.stack(
        [
            (xx - matrix[0, 2]) * depth / matrix[0, 0],
            (yy - matrix[1, 2]) * depth / matrix[1, 1],
            depth,
        ],
        axis=-1,
    )
    transform = np.asarray(report["T_level_camera"], dtype=float)
    level_points = (
        transform[:3, :3] @ camera_points.reshape(-1, 3).T
        + transform[:3, 3:4]
    ).T.reshape(camera_points.shape)
    valid = (
        np.isfinite(depth)
        & (depth > 0.3)
        & (depth < 1.8)
        & (confidence >= 1)
    )
    return rgb, depth, valid, level_points


def _lowres_mask(path, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (
        cv2.resize(
            mask, (shape[1], shape[0]), interpolation=cv2.INTER_AREA
        )
        > 100
    )


def _write_obj(path, vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    compact_vertices = vertices[used]
    compact_faces = remap[faces]
    lines = [
        f"v {x:.8f} {y:.8f} {z:.8f}"
        for x, y, z in compact_vertices
    ]
    lines.extend(
        f"f {a + 1} {b + 1} {c + 1}"
        for a, b, c in compact_faces
    )
    Path(path).write_text("\n".join(lines) + "\n")
    return int(len(compact_vertices)), int(len(compact_faces))


def _transform_obj(source, destination, transform):
    vertices = []
    other = []
    for line in Path(source).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        else:
            other.append(line)
    vertices = np.asarray(vertices)
    vertices = (
        transform[:3, :3] @ vertices.T
    ).T + transform[:3, 3]
    vertex_lines = [
        f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices
    ]
    Path(destination).write_text(
        "\n".join(vertex_lines + other) + "\n"
    )


def _mask_bounds(mask_path, valid, level_points, transform):
    mask = _lowres_mask(mask_path, valid.shape) & valid
    points = level_points[mask]
    world = (
        transform[:3, :3] @ points.T
    ).T + transform[:3, 3]
    lower = np.quantile(world, 0.03, axis=0)
    upper = np.quantile(world, 0.97, axis=0)
    return {
        "center_m": ((lower + upper) / 2).tolist(),
        "size_m": (upper - lower).tolist(),
        "lower_m": lower.tolist(),
        "upper_m": upper.tolist(),
        "points": int(len(world)),
    }


def _fridge_xml(center, platform_height):
    x, y = center[:2]
    z = platform_height + 0.175
    return f"""
    <body name="measured_fridge" pos="{x:.6f} {y:.6f} {z:.6f}"
          euler="0 0 2.136">
      <geom type="box" size="0.14 0.004 0.175" pos="0 -0.121 0"
            material="fridge_white" mass="1.5"/>
      <geom type="box" size="0.004 0.125 0.175" pos="-0.136 0 0"
            material="fridge_white" mass="0.8"/>
      <geom type="box" size="0.004 0.125 0.175" pos="0.136 0 0"
            material="fridge_white" mass="0.8"/>
      <geom type="box" size="0.14 0.125 0.004" pos="0 0 0.171"
            material="fridge_white" mass="0.5"/>
      <geom type="box" size="0.14 0.125 0.004" pos="0 0 -0.171"
            material="fridge_white" mass="0.5"/>
      <body name="measured_fridge_door" pos="0.132 0.121 0"
            euler="0 0 1.30">
        <geom type="box" size="0.132 0.004 0.17"
              pos="-0.132 0.004 0" material="fridge_white" mass="0.6"/>
      </body>
    </body>"""


def _microscope_xml(bounds):
    center = np.asarray(bounds["center_m"])
    size = np.asarray(bounds["size_m"])
    # Decompose the visible bounds rather than using one blocking AABB.
    base_z = center[2] - size[2] / 2
    stage_z = base_z + 0.42 * size[2]
    return f"""
    <body name="measured_microscope" pos="{center[0]:.6f} {center[1]:.6f} 0">
      <geom name="microscope_base_proxy" type="box"
            size="{0.34 * size[0]:.6f} {0.36 * size[1]:.6f} 0.025"
            pos="0 0 {base_z + 0.025:.6f}" material="microscope_white"/>
      <geom name="microscope_stage_proxy" type="box"
            size="{0.34 * size[0]:.6f} {0.38 * size[1]:.6f} 0.018"
            pos="0 0 {stage_z:.6f}" material="microscope_black"/>
      <geom name="microscope_column_proxy" type="box"
            size="0.025 {0.32 * size[1]:.6f} {0.34 * size[2]:.6f}"
            pos="{0.35 * size[0]:.6f} 0 {base_z + 0.38 * size[2]:.6f}"
            material="microscope_white"/>
    </body>"""


def build(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(Path(args.calibration_report).read_text())
    fixture = json.loads(Path(args.fixture_alignment).read_text())
    capture_manifest = json.loads(
        (Path(args.capture) / "manifest.json").read_text()
    )
    state = capture_manifest["robot_state"]["after"]
    right_q = np.asarray(state["right_joint_positions_rad"], dtype=float)
    left_q = np.asarray(state["left_joint_positions_rad"], dtype=float)
    _, _, valid, level_points = _capture_geometry(
        args.capture, calibration
    )
    transform, registration = _fixture_registration(calibration, fixture)
    right_base = np.asarray(fixture["right_piper_base_xyz_m"], dtype=float)
    left_base_level = np.asarray(
        calibration["anchor_xyz_level_m"]["left_piper_base"], dtype=float
    )
    left_base = transform[:3, :3] @ left_base_level + transform[:3, 3]
    left_base[2] = right_base[2]

    mesh_archive = np.load(calibration["measured_scene_mesh"])
    vertices_level = np.asarray(mesh_archive["vertices_xyz_m"], dtype=float)
    faces = np.asarray(mesh_archive["faces"], dtype=np.int32)
    robot_full = cv2.imread(
        str(args.robot_mask), cv2.IMREAD_GRAYSCALE
    )
    robot_low = cv2.resize(
        robot_full,
        (valid.shape[1], valid.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    robot_low = cv2.dilate(
        (robot_low > 80).astype(np.uint8), np.ones((5, 5), np.uint8)
    ).reshape(-1).astype(bool)
    static_faces = faces[~np.any(robot_low[faces], axis=1)]
    vertices_world = (
        transform[:3, :3] @ vertices_level.T
    ).T + transform[:3, 3]
    static_obj = output_dir / "measured_static_visual.obj"
    static_vertices, static_face_count = _write_obj(
        static_obj, vertices_world, static_faces
    )

    support_assets = []
    support_geoms = []
    for index, exported in enumerate(
        calibration["exported_collision_surfaces"], 1
    ):
        source = Path(calibration["measured_support_mjcf"]).parent / exported[
            "obj"
        ]
        destination = output_dir / f"support_{index}.obj"
        _transform_obj(source, destination, transform)
        support_assets.append(
            f'    <mesh name="support_{index}" file="{destination.resolve()}"/>'
        )
        support_geoms.append(
            f'    <geom name="measured_support_{index}" type="mesh" '
            f'mesh="support_{index}" material="support_{index}_mat" '
            'contype="1" conaffinity="1" friction="1 0.01 0.01"/>'
        )

    microscope = _mask_bounds(
        args.microscope_mask, valid, level_points, transform
    )
    platform_height = float(
        transform[2, 3] + calibration["level_heights_m"]["right_platform"]
    )
    left_table_height = float(
        transform[2, 3] + calibration["level_heights_m"]["left_table"]
    )
    fridge = np.asarray(
        fixture["incubator"]["center_xyz_from_mount_m"], dtype=float
    )
    bottle_level = np.asarray(
        calibration["semantic_observations"][
            "culture_media_bottle_left"
        ]["proxy_center_level_m"]
    )
    bottle = (
        transform[:3, :3] @ bottle_level + transform[:3, 3]
    )
    dish_level = np.asarray(
        calibration["anchor_xyz_level_m"]["petri_dish"]
    )
    dish = transform[:3, :3] @ dish_level + transform[:3, 3]
    model_path = Path(args.robot_model).resolve()
    positioned_model = output_dir / "positioned_bimanual_piper.xml"
    _write_positioned_robot_model(
        model_path, positioned_model, left_base, right_base
    )
    qpos = " ".join(
        f"{value:.9f}" for value in np.r_[left_q, right_q]
    )
    ctrl = qpos
    xml_path = output_dir / "measured_full_lab_scene.mjcf"
    xml_path.write_text(
        f"""<mujoco model="pasteur_measured_full_lab">
  <include file="{positioned_model.resolve()}"/>
  <statistic center="0.2 0 0.65" extent="1.35"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7"/>
    <global azimuth="145" elevation="-24"/>
  </visual>
  <asset>
    <mesh name="measured_static_visual" file="{static_obj.resolve()}"/>
{chr(10).join(support_assets)}
    <material name="measured_rgbd" rgba="0.55 0.60 0.66 0.14"/>
    <material name="support_1_mat" rgba="0.06 0.08 0.10 1"/>
    <material name="support_2_mat" rgba="0.28 0.31 0.35 1"/>
    <material name="support_3_mat" rgba="0.28 0.31 0.35 1"/>
    <material name="fridge_white" rgba="0.95 0.95 0.92 1"/>
    <material name="microscope_white" rgba="0.88 0.88 0.84 1"/>
    <material name="microscope_black" rgba="0.04 0.04 0.04 1"/>
    <material name="bottle_mat" rgba="0.65 0.80 0.92 0.55"/>
    <material name="dish_mat" rgba="0.30 0.65 0.95 0.45"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1"/>
    <geom name="measured_static_rgbd" type="mesh"
          mesh="measured_static_visual" material="measured_rgbd"
          group="4" contype="0" conaffinity="0"/>
{chr(10).join(support_geoms)}
{_fridge_xml(fridge, platform_height)}
{_microscope_xml(microscope)}
    <body name="culture_media_bottle" pos="{bottle[0]:.6f} {bottle[1]:.6f} {bottle[2]:.6f}">
      <geom type="cylinder" size="0.0375 0.080" material="bottle_mat"/>
    </body>
    <body name="petri_dish" pos="{dish[0]:.6f} {dish[1]:.6f} {platform_height + 0.008:.6f}">
      <geom type="cylinder" size="0.045 0.008" material="dish_mat"/>
    </body>
    <geom name="black_back_wall" type="box" size="1.8 0.005 0.9"
          pos="0 -0.55 0.75" rgba="0.04 0.04 0.04 0.16"
          contype="0" conaffinity="0"/>
    <geom name="white_right_wall" type="box" size="0.005 1.2 0.9"
          pos="1.1 0 0.75" rgba="0.93 0.93 0.91 0.16"
          contype="0" conaffinity="0"/>
    <camera name="measured_overview" pos="-0.65 1.65 1.55"
            xyaxes="0.93 0.36 0 -0.22 0.57 0.79" fovy="58"/>
    <camera name="measured_side" pos="-0.9 -0.9 1.25"
            xyaxes="0.75 -0.66 0 0.35 0.40 0.85" fovy="58"/>
  </worldbody>
  <keyframe>
    <key name="measured_capture" qpos="{qpos}" ctrl="{ctrl}"/>
  </keyframe>
</mujoco>
"""
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    report = {
        "schema": "piper_robot.measured_full_mujoco/v1",
        "capture": str(Path(args.capture).resolve()),
        "base_robot_model": str(model_path),
        "static_scene_prior": str(
            Path("robot/pasteur-calibrated-scene/scene.mjcf").resolve()
        ),
        "robot_layout": "two_fully_separate_upright_6dof_pipers",
        "registration": registration,
        "T_fixture_level": transform.tolist(),
        "fixture_alignment": str(Path(args.fixture_alignment).resolve()),
        "right_piper_base_m": right_base.tolist(),
        "left_piper_base_m": left_base.tolist(),
        "piper_base_separation_m": float(
            np.linalg.norm(left_base[:2] - right_base[:2])
        ),
        "right_platform_height_robot_m": platform_height,
        "left_table_height_robot_m": left_table_height,
        "height_difference_m": left_table_height - platform_height,
        "synchronized_q": {
            "assignment": "model_left_is_physical_left",
            "model_left_arm": left_q.tolist(),
            "model_right_arm": right_q.tolist(),
        },
        "static_visual_mesh": {
            "vertices": static_vertices,
            "faces": static_face_count,
            "collision_enabled": False,
        },
        "collision_support_count": len(support_geoms),
        "microscope_visible_bounds": microscope,
        "incubator_pose_source": "fixture primary anchor",
        "incubator_center_m": fridge.tolist(),
        "right_piper_to_incubator_center_distance_m": float(
            np.linalg.norm(fridge[:2] - right_base[:2])
        ),
        "right_piper_to_incubator_front_distance_m": float(
            fixture["incubator"][
                "front_face_distance_from_right_piper_m"
            ]
        ),
        "bottle_center_m": bottle.tolist(),
        "dish_center_m": [
            float(dish[0]),
            float(dish[1]),
            platform_height + 0.008,
        ],
        "compiled": {
            "ngeom": int(model.ngeom),
            "nmesh": int(model.nmesh),
            "nq": int(model.nq),
        },
        "mjcf": str(xml_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--robot-mask", required=True)
    parser.add_argument("--microscope-mask", required=True)
    parser.add_argument("--fridge-mask")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--robot-model",
        default="robot/arm/mujoco/bimanual_piper_table.xml",
    )
    parser.add_argument(
        "--fixture-alignment",
        default="src/configs/pasteur_fixture_alignment.json",
    )
    args = parser.parse_args(argv)
    report = build(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

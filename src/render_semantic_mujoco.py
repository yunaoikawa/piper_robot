#!/usr/bin/env python3
"""Render SAM-labelled RGB-D geometry beside the complete MuJoCo robot CAD.

The measured scene is deliberately placed beside, not on top of, the robot
until a camera-to-robot extrinsic and synchronized joint state are available.
All measured triangle geoms are visual-only; static collision clearance stays
in the ESDF because a whole-scene MuJoCo mesh would collide as a convex hull.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_semantics import (
    LABEL_BACKGROUND,
    LABEL_LID,
    LABEL_NAMES,
    LABEL_ROBOT,
)


def classify_faces(vertex_labels, faces) -> np.ndarray:
    labels = np.asarray(vertex_labels, dtype=np.uint8)
    face_labels = labels[np.asarray(faces, dtype=np.int32)]
    result = np.full(len(face_labels), LABEL_BACKGROUND, dtype=np.uint8)
    result[np.count_nonzero(face_labels == LABEL_ROBOT, axis=1) >= 2] = (
        LABEL_ROBOT
    )
    result[np.count_nonzero(face_labels == LABEL_LID, axis=1) >= 2] = LABEL_LID
    return result


def compact_mesh(vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    if not len(faces):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3).astype(np.int32)


def add_semantic_capture(spec, vertices, faces, vertex_labels, offset):
    face_labels = classify_faces(vertex_labels, faces)
    materials = {
        LABEL_BACKGROUND: ("observed_background", (0.62, 0.65, 0.69, 0.78)),
        LABEL_ROBOT: ("sam_robot_observation", (0.13, 0.83, 0.93, 0.82)),
        LABEL_LID: ("sam_lid_observation", (0.23, 0.51, 0.96, 0.92)),
    }
    for _, (name, rgba) in materials.items():
        spec.add_material(name=name, rgba=rgba)
    body = spec.worldbody.add_body(
        name="UNREGISTERED_camera_local_capture", pos=np.asarray(offset)
    )
    counts = {}
    for label, (material, _) in materials.items():
        subset_vertices, subset_faces = compact_mesh(
            vertices, faces[face_labels == label]
        )
        counts[LABEL_NAMES[label]] = int(len(subset_faces))
        if len(subset_faces) < 2 or len(subset_vertices) < 4:
            continue
        mesh_name = f"capture_{LABEL_NAMES[label]}"
        spec.add_mesh(
            name=mesh_name,
            uservert=subset_vertices.ravel(),
            userface=subset_faces.ravel(),
        )
        body.add_geom(
            name=mesh_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            material=material,
            contype=0,
            conaffinity=0,
            group=4,
        )
    return counts


def free_camera(lookat, *, distance, azimuth, elevation):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def render_view(renderer, data, camera, groups):
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    for group in groups:
        option.geomgroup[int(group)] = 1
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render().copy()


def labelled_panel(image_rgb, title, subtitle):
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 64), (12, 18, 28), -1)
    cv2.putText(
        image,
        title,
        (16, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        subtitle,
        (16, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (110, 200, 255),
        1,
    )
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-mesh", required=True)
    parser.add_argument(
        "--robot-model",
        default="robot/cone-e-description/lab-scene.mjcf",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robot-keyframe", default="lab_home")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = np.load(args.scene_mesh)
    vertices = np.asarray(scene["vertices_xyz_m"], dtype=float)
    faces = np.asarray(scene["faces"], dtype=np.int32)
    labels = np.asarray(scene["semantic_labels"], dtype=np.uint8)

    robot_model = Path(args.robot_model).resolve()
    spec = mujoco.MjSpec.from_file(str(robot_model))
    spec.meshdir = str(robot_model.parent)
    lower = np.percentile(vertices, 1.0, axis=0)
    upper = np.percentile(vertices, 99.0, axis=0)
    center = 0.5 * (lower + upper)
    # Side-by-side placement is intentional: no camera-to-robot extrinsic or
    # synchronized qpos exists for this saved SAM frame.
    capture_center_world = np.array([1.55, 0.0, 0.5 * (lower[2] + upper[2])])
    offset = capture_center_world - center
    face_counts = add_semantic_capture(
        spec, vertices, faces, labels, offset
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, args.robot_keyframe
    )
    if keyframe_id < 0:
        raise ValueError(f"MuJoCo keyframe not found: {args.robot_keyframe}")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    xml_path = output_dir / "semantic_comparison_UNREGISTERED.mjcf"
    xml_path.write_text(
        "<!-- OFFLINE DIAGNOSTIC ONLY: camera-to-robot registration and "
        "synchronized qpos are unavailable. Measured meshes are visual-only. -->\n"
        + spec.to_xml()
    )

    renderer = mujoco.Renderer(
        model, height=args.height, width=args.width
    )
    cad_camera = "topdown"
    capture_camera = free_camera(
        capture_center_world,
        distance=max(1.9, 1.05 * float(np.linalg.norm(upper - lower))),
        azimuth=180,
        elevation=-28,
    )
    combined_camera = free_camera(
        [0.78, 0.0, 0.30],
        distance=3.35,
        azimuth=180,
        elevation=-54,
    )
    cad = render_view(renderer, data, cad_camera, (0, 1, 2))
    collision = render_view(renderer, data, cad_camera, (0, 1, 3))
    capture = render_view(renderer, data, capture_camera, (4,))
    combined = render_view(renderer, data, combined_camera, (0, 1, 2, 4))
    renderer.close()

    panels = [
        labelled_panel(
            cad,
            "MuJoCo CAD robot",
            "complete STL; lab_home, not the saved SAM-time pose",
        ),
        labelled_panel(
            capture,
            "SAM + synchronized RGB-D",
            "cyan robot / blue lid / gray static; camera-local frame",
        ),
        labelled_panel(
            collision,
            "MuJoCo collision proxies",
            "robot proxies only; measured whole-scene mesh is NOT collision",
        ),
        labelled_panel(
            combined,
            "Diagnostic comparison",
            "side-by-side UNREGISTERED placement; do not use for clearance",
        ),
    ]
    diagnostic = np.vstack(
        (np.hstack(panels[:2]), np.hstack(panels[2:]))
    )
    warning = np.full(
        (58, diagnostic.shape[1], 3), (18, 18, 45), dtype=np.uint8
    )
    cv2.putText(
        warning,
        "OFFLINE MuJoCo DIAGNOSTIC - CAD and capture are NOT spatially registered",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (80, 180, 255),
        2,
    )
    diagnostic = np.vstack((warning, diagnostic))
    image_path = output_dir / "mujoco_semantic_diagnostic.png"
    cv2.imwrite(str(image_path), diagnostic)
    report = {
        "offline_only": True,
        "robot_model": str(robot_model),
        "robot_keyframe": args.robot_keyframe,
        "synchronized_joint_state": False,
        "camera_to_robot_extrinsic": False,
        "spatially_registered": False,
        "measured_mesh_collision_enabled": False,
        "collision_strategy": (
            "robot CAD/proxies in MuJoCo; static clearance in external ESDF"
        ),
        "capture_offset_for_side_by_side_display_m": offset.tolist(),
        "semantic_face_counts": face_counts,
        "mujoco_ngeom": int(model.ngeom),
        "mujoco_nmesh": int(model.nmesh),
        "mjcf": str(xml_path),
        "diagnostic_image": str(image_path),
    }
    (output_dir / "mujoco_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()

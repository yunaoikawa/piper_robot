#!/usr/bin/env python3
"""Render a SAM-calibrated RGB-D scene with a nominal MuJoCo lab prior.

The MacBook WetRobo model supplies nominal articulation, fixture shapes,
materials, and gripper sites.  SAM plus synchronized RGB-D is the runtime pose
authority.  Without an accepted camera-to-robot transform, the measured scene
is deliberately placed beside the nominal model rather than overlaid.

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
from wetrobo.perception.sam import (
    CalibrationRejected,
    SamCalibrationArtifact,
)


def load_nominal_spec(requested_path, fallback_path=None):
    """Load a nominal lab model, retaining a transparent fallback record."""

    requested = Path(requested_path).resolve()
    candidates = [requested]
    if fallback_path is not None:
        fallback = Path(fallback_path).resolve()
        if fallback != requested:
            candidates.append(fallback)
    failures = []
    for candidate in candidates:
        try:
            spec = mujoco.MjSpec.from_file(str(candidate))
            # Compile once before adding measured geometry so missing nominal
            # assets are diagnosed separately and can trigger the fallback.
            spec.compile()
            return spec, candidate, failures
        except (OSError, ValueError, RuntimeError) as error:
            failures.append(
                {"model": str(candidate), "error": str(error).splitlines()[0]}
            )
    raise RuntimeError(
        "could not compile any nominal MuJoCo model: "
        + "; ".join(f"{f['model']}: {f['error']}" for f in failures)
    )


def validate_transform(matrix, *, name):
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation is not proper")
    return transform


def load_transform_file(path):
    """Load an explicit 4x4 transform from NPY, NPZ, or JSON."""

    transform_path = Path(path)
    if transform_path.suffix == ".npy":
        value = np.load(transform_path)
    elif transform_path.suffix == ".npz":
        archive = np.load(transform_path)
        keys = list(archive.keys())
        if len(keys) != 1:
            raise ValueError(
                f"{transform_path} must contain exactly one transform array"
            )
        value = archive[keys[0]]
    else:
        payload = json.loads(transform_path.read_text())
        if isinstance(payload, dict):
            for key in (
                "T_robot_camera",
                "robot_from_camera",
                "transform",
                "matrix",
            ):
                if key in payload:
                    payload = payload[key]
                    break
        value = payload
    return validate_transform(value, name="T_robot_camera")


def classify_faces(vertex_labels, faces) -> np.ndarray:
    labels = np.asarray(vertex_labels, dtype=np.uint8)
    face_labels = labels[np.asarray(faces, dtype=np.int32)]
    result = np.full(len(face_labels), LABEL_BACKGROUND, dtype=np.uint8)
    result[np.count_nonzero(face_labels == LABEL_LID, axis=1) >= 2] = LABEL_LID
    result[np.count_nonzero(face_labels == LABEL_ROBOT, axis=1) >= 2] = (
        LABEL_ROBOT
    )
    return result


def compact_mesh(vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    if not len(faces):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3).astype(np.int32)


def add_semantic_capture(
    spec,
    vertices,
    faces,
    vertex_labels,
    transform_world_from_level,
    *,
    registered,
):
    face_labels = classify_faces(vertex_labels, faces)
    materials = {
        LABEL_BACKGROUND: ("observed_background", (0.62, 0.65, 0.69, 0.78)),
        LABEL_ROBOT: ("sam_robot_observation", (0.13, 0.83, 0.93, 0.82)),
        LABEL_LID: ("sam_lid_observation", (0.23, 0.51, 0.96, 0.92)),
    }
    for _, (name, rgba) in materials.items():
        spec.add_material(name=name, rgba=rgba)
    transform = validate_transform(
        transform_world_from_level,
        name="transform_world_from_level",
    )
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, transform[:3, :3].reshape(9))
    body = spec.worldbody.add_body(
        name=(
            "SAM_REGISTERED_level_capture"
            if registered
            else "UNREGISTERED_level_capture"
        ),
        pos=transform[:3, 3],
        quat=quat,
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
        "--sam-artifact-dir",
        help="directory containing esdf_report.json and scene_esdf.npz",
    )
    parser.add_argument(
        "--robot-model",
        default="robot/piper-mujoco/xml/lab-scene.xml",
        help="nominal shape/material/articulation prior; never pose authority",
    )
    parser.add_argument(
        "--fallback-robot-model",
        default="robot/cone-e-description/lab-scene.mjcf",
    )
    parser.add_argument(
        "--robot-from-camera",
        help="explicit T_robot_camera 4x4 JSON/NPY; omit for fail-closed side-by-side view",
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

    artifact_dir = Path(
        args.sam_artifact_dir or Path(args.scene_mesh).resolve().parent
    )
    artifact = SamCalibrationArtifact.load(artifact_dir)
    artifact_mesh = (artifact_dir / "scene_mesh_levelled.npz").resolve()
    if Path(args.scene_mesh).resolve() != artifact_mesh:
        if (
            artifact.vertices_level_m.shape != vertices.shape
            or artifact.semantic_labels.shape != labels.shape
            or not np.array_equal(artifact.semantic_labels, labels)
        ):
            raise ValueError(
                "--scene-mesh does not match the SAM calibration artifact"
            )

    spec, compiled_nominal_model, nominal_failures = load_nominal_spec(
        args.robot_model, args.fallback_robot_model
    )
    lower = np.percentile(vertices, 1.0, axis=0)
    upper = np.percentile(vertices, 99.0, axis=0)
    center = 0.5 * (lower + upper)

    T_robot_level = None
    registration_error = None
    if args.robot_from_camera:
        try:
            T_robot_level = artifact.compute_T_robot_level(
                load_transform_file(args.robot_from_camera)
            )
        except CalibrationRejected as error:
            registration_error = str(error)
    registered = T_robot_level is not None
    if registered:
        capture_transform = T_robot_level
        capture_center_world = (
            T_robot_level[:3, :3] @ center + T_robot_level[:3, 3]
        )
        display_offset = None
    else:
        # Side-by-side placement is intentional. A camera-local capture must
        # never look as if it were calibrated to the nominal lab.
        capture_center_world = np.array(
            [1.65, 0.0, 0.5 * (lower[2] + upper[2])]
        )
        display_offset = capture_center_world - center
        capture_transform = np.eye(4)
        capture_transform[:3, 3] = display_offset
    face_counts = add_semantic_capture(
        spec,
        vertices,
        faces,
        labels,
        capture_transform,
        registered=registered,
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
    registration_name = "REGISTERED" if registered else "UNREGISTERED"
    xml_path = output_dir / f"semantic_comparison_{registration_name}.mjcf"
    diagnostic_comment = (
        "SAM capture is registered to robot coordinates, but synchronized "
        "joint state is unavailable; measured meshes remain visual-only."
        if registered
        else "Camera-to-robot registration is unavailable. The SAM capture is "
        "shown beside the nominal lab and measured meshes are visual-only."
    )
    xml_path.write_text(
        f"<!-- OFFLINE DIAGNOSTIC ONLY: {diagnostic_comment} -->\n"
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
            "MacBook MuJoCo nominal lab",
            "shape/material/gripper sites only; nominal pose is not measurement",
        ),
        labelled_panel(
            capture,
            "SAM + synchronized RGB-D",
            (
                "cyan robot / blue lid / gray static; robot-base registered"
                if registered
                else "cyan robot / blue lid / gray static; levelled camera frame"
            ),
        ),
        labelled_panel(
            collision,
            "Nominal MuJoCo collision proxies",
            "diagnostic only; static clearance remains in external ESDF",
        ),
        labelled_panel(
            combined,
            f"Diagnostic comparison ({registration_name})",
            (
                "registered capture; robot keyframe is still not synchronized"
                if registered
                else "side-by-side by design; do not infer relative pose"
            ),
        ),
    ]
    diagnostic = np.vstack(
        (np.hstack(panels[:2]), np.hstack(panels[2:]))
    )
    warning = np.full(
        (58, diagnostic.shape[1], 3), (18, 18, 45), dtype=np.uint8
    )
    warning_text = (
        "SAM REGISTERED - robot qpos still unavailable; no motion/clearance use"
        if registered
        else (
            "SAM QUALITY CHECKED, UNREGISTERED - no robot-frame pose or clearance"
            if artifact.quality.accepted
            else "SAM QUALITY REJECTED - diagnostic display only"
        )
    )
    cv2.putText(
        warning,
        warning_text,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (80, 180, 255),
        2,
    )
    diagnostic = np.vstack((warning, diagnostic))
    image_path = output_dir / "mujoco_semantic_diagnostic.png"
    cv2.imwrite(str(image_path), diagnostic)

    footprint_estimates = {}
    for label in (LABEL_LID, LABEL_ROBOT):
        try:
            footprint = artifact.estimate_horizontal_footprint(
                label, T_robot_level=T_robot_level
            )
            footprint_estimates[LABEL_NAMES[label]] = footprint.to_dict()
        except CalibrationRejected as error:
            footprint_estimates[LABEL_NAMES[label]] = {
                "accepted": False,
                "reason": str(error),
            }
    report = {
        "offline_only": True,
        "calibration_authority": (
            "quality-gated SAM + synchronized RGB-D + explicit extrinsic"
        ),
        "nominal_model_role": "geometry/material/articulation prior only",
        "requested_nominal_model": str(Path(args.robot_model).resolve()),
        "compiled_nominal_model": str(compiled_nominal_model),
        "nominal_model_failures_before_fallback": nominal_failures,
        "robot_keyframe": args.robot_keyframe,
        "synchronized_joint_state": False,
        "camera_to_robot_extrinsic_provided": bool(args.robot_from_camera),
        "camera_to_robot_extrinsic_accepted": registered,
        "spatially_registered": registered,
        "registration_error": registration_error,
        "robot_collision_ready": False,
        "object_bench_state_ready": bool(
            registered
            and artifact.quality.accepted
            and artifact.provenance["sam_scores_available"]
            and artifact.provenance["capture_sync_verified"]
        ),
        "motion_scene_ready": False,
        "actionable_blockers": [
            blocker
            for blocker, blocked in (
                (
                    "camera-to-robot transform missing or rejected",
                    not registered,
                ),
                ("synchronized robot joint state missing", True),
                (
                    "raw SAM score/model provenance missing",
                    not artifact.provenance["sam_scores_available"],
                ),
                (
                    "capture timestamps not verified (file mtime only)",
                    not artifact.provenance["capture_sync_verified"],
                ),
            )
            if blocked
        ],
        "measured_mesh_collision_enabled": False,
        "collision_strategy": (
            "static background in external ESDF; dynamic target separate; "
            "robot CAD only after synchronized qpos"
        ),
        "sam_artifact": artifact.summary(),
        "semantic_pose_estimates": footprint_estimates,
        "capture_offset_for_side_by_side_display_m": (
            None if display_offset is None else display_offset.tolist()
        ),
        "T_robot_level": (
            None if T_robot_level is None else T_robot_level.tolist()
        ),
        "semantic_face_counts": face_counts,
        "mujoco_ngeom": int(model.ngeom),
        "mujoco_nmesh": int(model.nmesh),
        "mjcf": str(xml_path),
        "diagnostic_image": str(image_path),
    }
    (output_dir / "mujoco_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

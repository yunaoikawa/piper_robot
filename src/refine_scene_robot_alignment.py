#!/usr/bin/env python3
"""Refine a semantic scene from a fixed tag and stopped RGB-D views.

This is an offline, observation-only stage.  A fixed metric tag connects the
original multiview reconstruction to a later fixed-head RGB-D capture.  The
two robot bases are then moved as one rigid planar pair; robot/environment
contacts are evaluated only after the image-derived fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from robot.arm.home import physical_home_q, semantic_model_home_q
from rollout.scene_registration import (
    apply_independent_base_translations_to_mjcf,
    apply_shared_planar_transform_to_mjcf,
    assign_components_by_joint_excitation,
    assign_independent_base_translations,
    assign_named_base_translations,
    assign_visible_base_translations,
    backproject_depth,
    bridge_camera_from_fixed_tag,
    component_base_centers,
    depth_layer_foreground_mask,
    fit_shared_planar_robot_transform,
    intersect_pixel_with_horizontal_plane,
    persistent_depth_component_centers,
    reject_base_candidates_inside_semantic_objects,
    rigid_transform_consensus,
    scaled_camera_matrix,
    tag_pose_camera,
    transform_points,
)


SCHEMA = "piper_robot.scene_robot_alignment/v1"


def _matrix(meta: dict) -> np.ndarray:
    return np.asarray(meta["intrinsics"]["K_raw_rgb"], dtype=float)


def _reference_tag_consensus(
    report_path: Path,
    capture: Path,
    *,
    tag_id: int,
    tag_size_m: float,
) -> tuple[object, list[dict]]:
    report = json.loads(report_path.read_text())
    registrations = {
        item["name"]: np.asarray(item["reference_from_camera"], dtype=float)
        for item in report["registration"]["views"]
        if item.get("accepted", False)
    }
    transforms = []
    records = []
    for view_name, reference_from_camera in registrations.items():
        image_path = (
            capture / "derived" / view_name / "rgb_landscape.jpg"
        )
        frame_dir = capture / "raw" / "head" / view_name / "000001"
        meta = json.loads((frame_dir / "meta.json").read_text())
        image = cv2.imread(str(image_path))
        try:
            camera_from_tag, rms = tag_pose_camera(
                image,
                _matrix(meta),
                tag_id=tag_id,
                tag_size_m=tag_size_m,
            )
        except ValueError as error:
            records.append(
                {
                    "view": view_name,
                    "accepted": False,
                    "reason": str(error),
                }
            )
            continue
        reference_from_tag = reference_from_camera @ camera_from_tag
        transforms.append(reference_from_tag)
        records.append(
            {
                "view": view_name,
                "accepted": True,
                "tag_reprojection_rms_px": rms,
            }
        )
    if len(transforms) < 3:
        raise ValueError("fixed tag must be observed in at least three reference views")
    return rigid_transform_consensus(transforms), records


def _last_frame(capture: Path, view: dict) -> tuple[Path, dict]:
    frame = view["frames"][-1]
    relative = Path(frame["files"]["rgb_png"]["path"])
    frame_dir = capture / relative.parent
    return frame_dir, frame


def _contact_records(scene_model: Path, keyframe: str = "home") -> list[dict]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_model))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    mujoco.mj_forward(model, data)
    records = []
    for contact in data.contact:
        first = model.body(model.geom_bodyid[contact.geom1]).name
        second = model.body(model.geom_bodyid[contact.geom2]).name
        robot_first = first.startswith(("left/", "right/"))
        robot_second = second.startswith(("left/", "right/"))
        if robot_first == robot_second:
            continue
        environment = second if robot_first else first
        robot = first if robot_first else second
        records.append(
            {
                "robot_body": robot,
                "environment_body": environment,
                "penetration_depth_m": max(0.0, -float(contact.dist)),
            }
        )
    return sorted(
        records,
        key=lambda item: item["penetration_depth_m"],
        reverse=True,
    )


def _derived_scene(
    source_scene: Path,
    positioned_model: Path,
    output_scene: Path,
) -> None:
    source_scene = source_scene.resolve()
    tree = ET.parse(source_scene)
    root = tree.getroot()
    include = root.find("include")
    if include is None:
        raise ValueError("semantic scene lacks positioned robot include")
    include.set("file", str(positioned_model.resolve()))
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir"))
        if not meshdir.is_absolute():
            compiler.set(
                "meshdir",
                str((source_scene.parent / meshdir).resolve()),
            )
    tree.write(output_scene, encoding="unicode")


def _pin_canonical_physical_home(positioned_model: Path) -> None:
    """Pin physical home after mapping into the semantic model joint frame."""

    tree = ET.parse(positioned_model)
    root = tree.getroot()
    keyframe = root.find("./keyframe/key[@name='home']")
    if keyframe is None:
        raise ValueError("positioned Piper model lacks keyframe home")
    values = np.r_[
        semantic_model_home_q("left"),
        semantic_model_home_q("right"),
    ]
    serialized = " ".join(f"{value:.10g}" for value in values)
    keyframe.set("qpos", serialized)
    if keyframe.get("ctrl") is not None:
        keyframe.set("ctrl", serialized)
    tree.write(positioned_model, encoding="unicode")


def _view_qpos(manifest: dict) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for view in manifest["views"]:
        state = view.get("robot_state", {}).get("after", {})
        result[view["name"]] = {
            arm: np.asarray(
                state[f"{arm}_joint_positions_rad"],
                dtype=float,
            )
            for arm in ("left", "right")
        }
    return result


def _project_scene_points(
    points_scene: list[np.ndarray],
    scene_from_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[np.ndarray]:
    camera_from_scene = np.linalg.inv(scene_from_camera)
    points_camera = transform_points(
        np.asarray(points_scene, dtype=float),
        camera_from_scene,
    )
    if np.any(points_camera[:, 2] <= 0):
        raise ValueError("persistent robot base projects behind the camera")
    homogeneous = points_camera @ np.asarray(camera_matrix, dtype=float).T
    return [
        value[:2] / value[2]
        for value in homogeneous
    ]


def _write_depth_mask_montage(
    frame_dirs: dict[str, Path],
    mask_dir: Path,
    output: Path,
) -> None:
    tiles = []
    for view_name, frame_dir in frame_dirs.items():
        image = cv2.imread(str(frame_dir / "rgb.png"))
        mask = cv2.imread(
            str(mask_dir / f"{view_name}_robot.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        color = np.zeros_like(image)
        color[:, :, 1] = mask
        tile = cv2.addWeighted(image, 0.65, color, 0.5, 0.0)
        tile = cv2.resize(tile, (360, 480))
        cv2.putText(
            tile,
            view_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    columns = min(3, len(tiles))
    rows = int(np.ceil(len(tiles) / columns))
    canvas = np.zeros((rows * 480, columns * 360, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * 480:(row + 1) * 480,
               column * 360:(column + 1) * 360] = tile
    cv2.imwrite(str(output), canvas)


def _write_arm_identity_overlay(
    image_path: Path,
    projected_centers: list[np.ndarray],
    arm_identity: dict,
    output_path: Path,
    *,
    retained_centers_px: dict[str, np.ndarray] | None = None,
    rejected_centers_px: list[tuple[np.ndarray, str]] | None = None,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    colors = {"left": (255, 0, 255), "right": (255, 255, 0)}
    radius = max(12, int(round(min(image.shape[:2]) * 0.035)))
    for arm, index in arm_identity[
        "physical_arm_to_component_index"
    ].items():
        center = tuple(
            np.rint(projected_centers[index]).astype(int).tolist()
        )
        cv2.circle(image, center, radius, colors[arm], 6, cv2.LINE_AA)
        cv2.putText(
            image,
            f"physical {arm.upper()} = semantic {arm}/",
            (max(10, center[0] - radius), max(35, center[1] - radius - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors[arm],
            3,
            cv2.LINE_AA,
        )
    for arm, value in (retained_centers_px or {}).items():
        center = tuple(np.rint(value).astype(int).tolist())
        cv2.drawMarker(
            image,
            center,
            colors[arm],
            cv2.MARKER_TILTED_CROSS,
            radius * 2,
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"physical {arm.upper()}: retained reviewed base",
            (max(10, center[0] - radius), max(35, center[1] - radius - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors[arm],
            3,
            cv2.LINE_AA,
        )
    for value, label in rejected_centers_px or []:
        center = tuple(np.rint(value).astype(int).tolist())
        cv2.drawMarker(
            image,
            center,
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            radius * 2,
            7,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"REJECTED: {label}",
            (max(10, center[0] - radius), max(35, center[1] - radius - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), image)


def build(args) -> dict:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_capture = Path(args.reference_capture).resolve()
    current_capture = Path(args.current_capture).resolve()
    scene_json_path = Path(args.scene_json).resolve()
    scene = json.loads(scene_json_path.read_text())
    axis_sign = np.asarray(args.scene_axis_sign, dtype=float)
    if axis_sign.shape != (3,) or not np.all(np.isin(axis_sign, (-1, 1))):
        raise ValueError("scene axis sign must contain three +/-1 values")

    tag_consensus, reference_tag_records = _reference_tag_consensus(
        Path(args.reference_report).resolve(),
        reference_capture,
        tag_id=args.tag_id,
        tag_size_m=args.tag_size_m,
    )
    manifest = json.loads((current_capture / "manifest.json").read_text())
    if manifest.get("commands_sent") is not False:
        raise ValueError("current capture command provenance is unsafe")
    mask_dir = Path(args.robot_mask_dir).resolve()
    initial_bases = {
        name: np.asarray(value, dtype=float)
        for name, value in scene["robot_placement"][
            "base_xyz_level_m"
        ].items()
    }
    per_view = []
    scene_from_camera_by_view = {}
    fits = []
    depth_robot_points_by_view = {}
    frame_dirs = {
        view["name"]: _last_frame(current_capture, view)[0]
        for view in manifest["views"]
    }
    depth_stack = []
    for frame_dir in frame_dirs.values():
        depth = np.load(frame_dir / "depth.npy").astype(float)
        confidence = np.load(frame_dir / "confidence.npy")
        depth[confidence < args.minimum_confidence] = np.nan
        depth[depth <= 0] = np.nan
        depth_stack.append(depth)
    depth_values = np.asarray(depth_stack)
    depth_valid = np.any(np.isfinite(depth_values), axis=0)
    temporal_background_depth = np.max(
        np.where(np.isfinite(depth_values), depth_values, -np.inf),
        axis=0,
    )
    temporal_background_depth[~depth_valid] = np.nan
    depth_mask_output = output / "depth_layer_robot_masks"
    depth_mask_output.mkdir(exist_ok=True)
    for view in manifest["views"]:
        view_name = view["name"]
        frame_dir = frame_dirs[view_name]
        image = cv2.imread(str(frame_dir / "rgb.png"))
        meta = json.loads((frame_dir / "meta.json").read_text())
        mask_path = mask_dir / f"{view_name}_robot.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        depth = np.load(frame_dir / "depth.npy")
        confidence = np.load(frame_dir / "confidence.npy")
        depth_mask, depth_components = depth_layer_foreground_mask(
            mask,
            depth,
            confidence,
            temporal_background_depth,
            minimum_confidence=args.minimum_confidence,
            maximum_neighbor_depth_jump_m=(
                args.maximum_neighbor_depth_jump_m
            ),
            minimum_foreground_delta_m=args.minimum_foreground_delta_m,
            minimum_component_pixels=args.minimum_depth_component_pixels,
            minimum_dynamic_pixels=args.minimum_dynamic_pixels,
        )
        cv2.imwrite(
            str(depth_mask_output / f"{view_name}_robot.png"),
            depth_mask,
        )
        camera_from_tag, tag_rms = tag_pose_camera(
            image,
            _matrix(meta),
            tag_id=args.tag_id,
            tag_size_m=args.tag_size_m,
        )
        level_from_camera = bridge_camera_from_fixed_tag(
            tag_consensus.transform,
            camera_from_tag,
        )
        scaled_matrix = scaled_camera_matrix(
            _matrix(meta),
            source_shape_hw=image.shape[:2],
            target_shape_hw=depth.shape,
        )
        valid_depth_robot = (
            (depth_mask > 0)
            & np.isfinite(depth)
            & (depth > 0)
            & (confidence >= args.minimum_confidence)
        )
        depth_robot_points_by_view[view_name] = (
            transform_points(
                backproject_depth(depth, scaled_matrix)[valid_depth_robot],
                level_from_camera,
            )
            * axis_sign
        )
        try:
            centers = component_base_centers(
                depth_mask,
                depth,
                confidence,
                _matrix(meta),
                level_from_camera,
                expected_base_xyz=[
                    np.asarray(value) / axis_sign
                    for value in initial_bases.values()
                ],
                minimum_confidence=args.minimum_confidence,
                base_height_band_m=args.base_height_band_m,
                maximum_anchor_distance_m=args.maximum_anchor_distance_m,
            )
        except ValueError as error:
            per_view.append(
                {
                    "view": view_name,
                    "accepted": False,
                    "tag_reprojection_rms_px": tag_rms,
                    "depth_components": depth_components,
                    "reason": str(error),
                }
            )
            scene_from_camera_by_view[view_name] = level_from_camera
            continue
        centers = [np.asarray(value) * axis_sign for value in centers]
        fit = fit_shared_planar_robot_transform(initial_bases, centers)
        fits.append(fit)
        scene_from_camera_by_view[view_name] = level_from_camera
        per_view.append(
            {
                "view": view_name,
                "accepted": True,
                "tag_reprojection_rms_px": tag_rms,
                "depth_components": depth_components,
                "observed_base_xyz_m": [
                    value.tolist() for value in centers
                ],
                "fit": fit,
            }
        )

    depth_montage_path = output / "depth_layer_robot_montage.png"
    _write_depth_mask_montage(
        frame_dirs,
        depth_mask_output,
        depth_montage_path,
    )
    holdout_name = manifest["views"][-1]["name"]
    train = [
        item["fit"]
        for item in per_view
        if item.get("accepted") and item["view"] != holdout_name
    ]
    holdout = next(
        (
            item["fit"]
            for item in per_view
            if item.get("accepted") and item["view"] == holdout_name
        ),
        None,
    )
    fit_available = len(train) >= 3 and holdout is not None
    if fit_available:
        translation_values = np.asarray(
            [item["translation_xy_m"] for item in train],
            dtype=float,
        )
        yaw_values = np.asarray(
            [item["yaw_delta_rad"] for item in train],
            dtype=float,
        )
        translation = np.median(translation_values, axis=0)
        yaw = float(
            np.arctan2(
                np.mean(np.sin(yaw_values)),
                np.mean(np.cos(yaw_values)),
            )
        )
        translation_spread = float(
            np.max(
                np.linalg.norm(
                    translation_values - translation,
                    axis=1,
                )
            )
        )
        yaw_spread = float(
            np.degrees(
                np.max(
                    np.abs(
                        np.arctan2(
                            np.sin(yaw_values - yaw),
                            np.cos(yaw_values - yaw),
                        )
                    )
                )
            )
        )
        holdout_translation_error = float(
            np.linalg.norm(
                np.asarray(holdout["translation_xy_m"]) - translation
            )
        )
        holdout_yaw_error = float(
            np.degrees(
                abs(
                    np.arctan2(
                        np.sin(holdout["yaw_delta_rad"] - yaw),
                        np.cos(holdout["yaw_delta_rad"] - yaw),
                    )
                )
            )
        )
    else:
        # Fail closed and retain the reviewed scene.  A missing base is not
        # permission to infer a large transform from moving links.
        translation = np.zeros(2, dtype=float)
        yaw = 0.0
        translation_spread = None
        yaw_spread = None
        holdout_translation_error = None
        holdout_yaw_error = None
    current_tag_rms = max(
        item["tag_reprojection_rms_px"] for item in per_view
    )
    camera_registration_accepted = bool(
        tag_consensus.sample_count >= 3
        and tag_consensus.translation_spread_m
        <= args.maximum_tag_translation_spread_m
        and tag_consensus.rotation_spread_deg
        <= args.maximum_tag_rotation_spread_deg
        and current_tag_rms <= args.maximum_tag_reprojection_rms_px
    )
    shared_fit_accepted = bool(
        fit_available
        and translation_spread <= args.maximum_translation_spread_m
        and yaw_spread <= args.maximum_yaw_spread_deg
        and holdout_translation_error <= args.maximum_holdout_translation_m
        and holdout_yaw_error <= args.maximum_holdout_yaw_deg
        and camera_registration_accepted
    )
    train_names = [view["name"] for view in manifest["views"][:-1]]
    raw_persistent_centers, persistent_components = (
        persistent_depth_component_centers(
            [
                depth_robot_points_by_view[name]
                for name in train_names
            ],
            voxel_size_m=args.persistence_voxel_size_m,
            minimum_views=args.persistence_minimum_views,
            minimum_voxels=args.persistence_minimum_voxels,
            expected_base_z_m=float(
                np.median(
                    [value[2] for value in initial_bases.values()]
                )
            ),
            maximum_base_plane_gap_m=args.maximum_base_plane_gap_m,
        )
    )
    persistent_centers, semantic_exclusion = (
        reject_base_candidates_inside_semantic_objects(
            raw_persistent_centers,
            scene["objects"],
            margin_m=args.semantic_exclusion_margin_m,
        )
    )
    persistent_fit = None
    arm_identity = None
    arm_identity_overlay = None
    persistent_holdout_errors = []
    if persistent_centers:
        baseline_name = manifest["views"][0]["name"]
        baseline_frame = frame_dirs[baseline_name]
        baseline_meta = json.loads(
            (baseline_frame / "meta.json").read_text()
        )
        projected_centers = _project_scene_points(
            persistent_centers,
            scene_from_camera_by_view[baseline_name],
            _matrix(baseline_meta),
        )
        if len(persistent_centers) == 2:
            robot_masks = {}
            for view in manifest["views"]:
                view_name = view["name"]
                path = mask_dir / f"{view_name}_robot.png"
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(path)
                robot_masks[view_name] = mask > 0
            arm_identity = assign_components_by_joint_excitation(
                qpos_by_view=_view_qpos(manifest),
                robot_masks_by_view=robot_masks,
                component_centers_px=projected_centers,
                baseline_view=baseline_name,
                minimum_joint_excitation_rad=(
                    args.minimum_joint_excitation_rad
                ),
                minimum_joint_dominance_ratio=(
                    args.minimum_joint_dominance_ratio
                ),
                motion_radius_fraction=args.motion_radius_fraction,
                minimum_motion_density=args.minimum_motion_density,
                minimum_assignment_ratio=args.minimum_assignment_ratio,
            )
            physical_centers = {
                f"{arm}/base_link": persistent_centers[index]
                for arm, index in arm_identity[
                    "physical_arm_to_component_index"
                ].items()
            }
            persistent_fit = assign_named_base_translations(
                initial_bases,
                physical_centers,
            )
            excessive = {
                name: float(np.linalg.norm(value))
                for name, value in persistent_fit[
                    "translations_xy_m"
                ].items()
                if np.linalg.norm(value)
                > args.maximum_independent_base_translation_m
            }
            if excessive:
                raise ValueError(
                    "semantic base assignment requires excessive "
                    f"translation: {excessive}"
                )
        else:
            persistent_fit = assign_visible_base_translations(
                initial_bases,
                persistent_centers,
                maximum_translation_m=(
                    args.maximum_independent_base_translation_m
                ),
                minimum_nearest_ratio=args.minimum_base_nearest_ratio,
            )
            observed_name = persistent_fit["observed_bases"][0]
            observed_arm = observed_name.split("/", 1)[0]
            arm_identity = {
                "accepted": True,
                "baseline_view": baseline_name,
                "physical_arm_to_component_index": {
                    observed_arm: 0,
                },
                "policy": (
                    "single semantically clean component assigned only when "
                    "close and unambiguous relative to reviewed base; "
                    "unobserved base retained"
                ),
                "evidence": persistent_fit["evidence"],
            }
        arm_identity_overlay = output / "arm_identity_overlay.png"
        retained_centers_px = {}
        for name in persistent_fit.get("retained_unobserved_bases", []):
            arm = name.split("/", 1)[0]
            retained_centers_px[arm] = _project_scene_points(
                [initial_bases[name]],
                scene_from_camera_by_view[baseline_name],
                _matrix(baseline_meta),
            )[0]
        rejected_centers_px = []
        for record in semantic_exclusion:
            if record["accepted"]:
                continue
            projected = _project_scene_points(
                [np.asarray(record["center_xyz_m"], dtype=float)],
                scene_from_camera_by_view[baseline_name],
                _matrix(baseline_meta),
            )[0]
            labels = sorted(
                {
                    item["semantic_name"]
                    for item in record["overlapping_semantic_objects"]
                }
            )
            rejected_centers_px.append((projected, "+".join(labels)))
        _write_arm_identity_overlay(
            baseline_frame / "rgb.png",
            projected_centers,
            arm_identity,
            arm_identity_overlay,
            retained_centers_px=retained_centers_px,
            rejected_centers_px=rejected_centers_px,
        )
        holdout_points = depth_robot_points_by_view[holdout_name]
        for center in persistent_centers:
            persistent_holdout_errors.append(
                float(
                    np.min(
                        np.linalg.norm(
                            holdout_points - center,
                            axis=1,
                        )
                    )
                )
            )
    persistence_accepted = bool(
        camera_registration_accepted
        and persistent_fit is not None
        and max(persistent_holdout_errors, default=float("inf"))
        <= args.maximum_persistent_holdout_error_m
    )
    robot_alignment_accepted = persistence_accepted

    source_positioned = Path(args.positioned_robot).resolve()
    candidate_positioned = output / "positioned_robot.mjcf"
    if persistence_accepted:
        apply_independent_base_translations_to_mjcf(
            source_positioned,
            candidate_positioned,
            translations_xy_m=persistent_fit["translations_xy_m"],
        )
    else:
        apply_shared_planar_transform_to_mjcf(
            source_positioned,
            candidate_positioned,
            root_bodies=("left/base_link", "right/base_link"),
            translation_xy_m=(0.0, 0.0),
            yaw_delta_rad=0.0,
        )
    _pin_canonical_physical_home(candidate_positioned)
    candidate_scene = output / "scene.mjcf"
    _derived_scene(
        Path(args.scene_model).resolve(),
        candidate_positioned,
        candidate_scene,
    )

    lid_scene = None
    object_scene_path = None
    overlay_path = None
    if args.lid_center_px is not None:
        # Legacy/manual compatibility only.  The unattended pipeline omits
        # this argument and runs the tool-relative wrist RGB-D target stage.
        final_view = manifest["views"][-1]
        final_frame_dir, _ = _last_frame(current_capture, final_view)
        final_meta = json.loads((final_frame_dir / "meta.json").read_text())
        level_from_camera = scene_from_camera_by_view[final_view["name"]]
        lid_level = intersect_pixel_with_horizontal_plane(
            tuple(args.lid_center_px),
            _matrix(final_meta),
            level_from_camera,
            plane_z_m=args.support_plane_z_m,
        )
        lid_scene = lid_level * axis_sign
        lid_scene[2] = (
            args.support_plane_z_m + args.lid_height_m / 2.0
        )
        object_scene = {
            "schema": "piper_robot.dynamic_dish_lid_scene/v2",
            "source": {
                "kind": "fixed_tag_bridge_plus_operator_confirmed_rgb_roi",
                "capture_manifest": str(
                    (current_capture / "manifest.json").resolve()
                ),
                "reference_report": str(
                    Path(args.reference_report).resolve()
                ),
                "tag_id": args.tag_id,
            },
            "camera_to_scene_accepted": camera_registration_accepted,
            "operator_confirmed": True,
            "objects": [
                {
                    "instance_id": "petri-lid-target",
                    "semantic_name": "petri dish lid",
                    "role": "target_lid",
                    "status": "confirmed",
                    "pose_scene": [
                        [1.0, 0.0, 0.0, float(lid_scene[0])],
                        [0.0, 1.0, 0.0, float(lid_scene[1])],
                        [0.0, 0.0, 1.0, float(lid_scene[2])],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "geometry": {
                        "type": "cylinder",
                        "radius_m": args.lid_radius_m,
                        "height_m": args.lid_height_m,
                        "pose_anchor": "center",
                    },
                    "perception": {
                        "center_px": list(args.lid_center_px),
                        "support_plane_z_m": args.support_plane_z_m,
                        "selection": (
                            "operator-confirmed RGB ROI containing blue cross"
                        ),
                        "current_frame_sam_accepted": False,
                    },
                }
            ],
        }
        object_scene_path = output / "latest_lid_scene.json"
        object_scene_path.write_text(
            json.dumps(object_scene, indent=2, ensure_ascii=False) + "\n"
        )
        final_image = cv2.imread(str(final_frame_dir / "rgb.png"))
        angles = np.linspace(0.0, 2.0 * np.pi, 181)
        circle_scene = np.c_[
            lid_scene[0] + args.lid_radius_m * np.cos(angles),
            lid_scene[1] + args.lid_radius_m * np.sin(angles),
            np.full_like(angles, args.support_plane_z_m),
        ]
        circle_level = circle_scene / axis_sign
        circle_camera = transform_points(
            circle_level,
            np.linalg.inv(level_from_camera),
        )
        projected = circle_camera @ _matrix(final_meta).T
        projected = np.rint(
            projected[:, :2] / projected[:, 2:3]
        ).astype(np.int32)
        cv2.polylines(
            final_image,
            [projected],
            True,
            (0, 255, 255),
            5,
            cv2.LINE_AA,
        )
        center = tuple(np.rint(args.lid_center_px).astype(int))
        cv2.drawMarker(
            final_image,
            center,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            40,
            5,
        )
        overlay_path = output / "latest_lid_alignment.png"
        cv2.imwrite(str(overlay_path), final_image)

    current_contacts = _contact_records(
        Path(args.scene_model).resolve()
    )
    candidate_contacts = _contact_records(candidate_scene)
    gate_reasons = []
    home_pose_provenance_accepted = bool(args.baseline_is_home)
    if not camera_registration_accepted:
        gate_reasons.append("camera_to_scene_registration_not_accepted")
    if not robot_alignment_accepted:
        gate_reasons.append("persistent_depth_robot_alignment_not_accepted")
    if not home_pose_provenance_accepted:
        gate_reasons.append("home_pose_provenance_not_accepted")
    if candidate_contacts:
        gate_reasons.append("home_robot_environment_contacts_present")
    trajectory_authorized = not gate_reasons
    report = {
        "schema": SCHEMA,
        "accepted": robot_alignment_accepted,
        "display_only": not trajectory_authorized,
        "commands_sent": False,
        "home_pose_provenance": {
            "accepted": home_pose_provenance_accepted,
            "source": (
                "repository_physical_home_q_with_semantic_identity_mapping"
                if args.baseline_is_home
                else None
            ),
            "baseline_view": manifest["views"][0]["name"],
            "physical_right_q_rad": physical_home_q("right").tolist(),
            "physical_left_q_rad": physical_home_q("left").tolist(),
            "model_qpos_order": [
                "physical_left_on_semantic_left",
                "physical_right_on_semantic_right",
            ],
        },
        "fixed_tag_reference": {
            "tag_id": args.tag_id,
            "sample_count": tag_consensus.sample_count,
            "translation_spread_m": tag_consensus.translation_spread_m,
            "rotation_spread_deg": tag_consensus.rotation_spread_deg,
            "views": reference_tag_records,
        },
        "shared_robot_fit": {
            "available": fit_available,
            "accepted": shared_fit_accepted,
            "reason": (
                None
                if fit_available
                else "insufficient_views_with_two_observable_bases"
            ),
            "translation_xy_m": translation.tolist(),
            "yaw_delta_rad": yaw,
            "translation_spread_m": translation_spread,
            "yaw_spread_deg": yaw_spread,
            "holdout_translation_error_m": holdout_translation_error,
            "holdout_yaw_error_deg": holdout_yaw_error,
            "preserves_base_baseline": True,
        },
        "persistent_depth_robot_fit": {
            "available": persistent_fit is not None,
            "accepted": persistence_accepted,
            "fit": persistent_fit,
            "train_views": train_names,
            "holdout_view": holdout_name,
            "holdout_nearest_point_errors_m": persistent_holdout_errors,
            "components": persistent_components,
            "semantic_object_exclusion": semantic_exclusion,
            "arm_identity": arm_identity,
            "microscope_rejection": (
                "Depth-layer filtering followed by explicit rejection of "
                "persistent robot candidates inside known non-robot "
                "semantic volumes."
            ),
            "yaw_modified": False,
        },
        "per_view": per_view,
        "lid_pose_scene_xyz_m": (
            None if lid_scene is None else lid_scene.tolist()
        ),
        "trajectory_gate": {
            "authorized": trajectory_authorized,
            "reasons": gate_reasons,
            "policy": (
                "Require accepted camera registration, accepted persistent "
                "depth robot alignment, explicit home-pose provenance, and "
                "zero home environment contacts."
            ),
        },
        "ablation": {
            "current_geometry": {
                "robot_environment_contacts": current_contacts,
            },
            "depth_persistence_fitted_robot_bases": {
                "robot_environment_contacts": candidate_contacts,
            },
            "microscope_geometry_modified": False,
        },
        "thresholds": {
            "maximum_translation_spread_m": args.maximum_translation_spread_m,
            "maximum_yaw_spread_deg": args.maximum_yaw_spread_deg,
            "maximum_holdout_translation_m": args.maximum_holdout_translation_m,
            "maximum_holdout_yaw_deg": args.maximum_holdout_yaw_deg,
            "maximum_tag_reprojection_rms_px": (
                args.maximum_tag_reprojection_rms_px
            ),
            "maximum_tag_translation_spread_m": (
                args.maximum_tag_translation_spread_m
            ),
            "maximum_tag_rotation_spread_deg": (
                args.maximum_tag_rotation_spread_deg
            ),
            "persistence_voxel_size_m": args.persistence_voxel_size_m,
            "persistence_minimum_views": args.persistence_minimum_views,
            "persistence_minimum_voxels": args.persistence_minimum_voxels,
            "maximum_persistent_holdout_error_m": (
                args.maximum_persistent_holdout_error_m
            ),
            "maximum_base_plane_gap_m": args.maximum_base_plane_gap_m,
            "minimum_joint_excitation_rad": (
                args.minimum_joint_excitation_rad
            ),
            "minimum_joint_dominance_ratio": (
                args.minimum_joint_dominance_ratio
            ),
            "motion_radius_fraction": args.motion_radius_fraction,
            "minimum_motion_density": args.minimum_motion_density,
            "minimum_assignment_ratio": args.minimum_assignment_ratio,
            "semantic_exclusion_margin_m": (
                args.semantic_exclusion_margin_m
            ),
            "maximum_independent_base_translation_m": (
                args.maximum_independent_base_translation_m
            ),
            "minimum_base_nearest_ratio": args.minimum_base_nearest_ratio,
        },
        "artifacts": {
            "scene_model": str(candidate_scene),
            "positioned_robot": str(candidate_positioned),
            "object_scene": (
                None if object_scene_path is None else str(object_scene_path)
            ),
            "lid_alignment_overlay": (
                None if overlay_path is None else str(overlay_path)
            ),
            "depth_layer_robot_masks": str(depth_mask_output),
            "depth_layer_robot_montage": str(depth_montage_path),
            "arm_identity_overlay": (
                None
                if arm_identity_overlay is None
                else str(arm_identity_overlay)
            ),
        },
    }
    (output / "alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-report", required=True)
    parser.add_argument("--reference-capture", required=True)
    parser.add_argument("--current-capture", required=True)
    parser.add_argument("--robot-mask-dir", required=True)
    parser.add_argument("--scene-json", required=True)
    parser.add_argument("--scene-model", required=True)
    parser.add_argument("--positioned-robot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag-id", type=int, default=3)
    parser.add_argument("--tag-size-m", type=float, default=0.06)
    parser.add_argument(
        "--scene-axis-sign",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
    )
    parser.add_argument(
        "--lid-center-px",
        type=float,
        nargs=2,
        help=(
            "legacy manual target center; omit for unattended wrist RGB-D "
            "target localization"
        ),
    )
    parser.add_argument("--support-plane-z-m", type=float, required=True)
    parser.add_argument("--lid-radius-m", type=float, default=0.047)
    parser.add_argument("--lid-height-m", type=float, default=0.006)
    parser.add_argument("--minimum-confidence", type=int, default=1)
    parser.add_argument(
        "--baseline-is-home",
        action="store_true",
        help="Record the operator assertion that the first stopped view is home.",
    )
    parser.add_argument(
        "--maximum-neighbor-depth-jump-m",
        type=float,
        default=0.045,
    )
    parser.add_argument(
        "--minimum-foreground-delta-m",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--minimum-depth-component-pixels",
        type=int,
        default=20,
    )
    parser.add_argument("--minimum-dynamic-pixels", type=int, default=8)
    parser.add_argument(
        "--persistence-voxel-size-m",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--persistence-minimum-views",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--persistence-minimum-voxels",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--maximum-persistent-holdout-error-m",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--maximum-base-plane-gap-m",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--minimum-joint-excitation-rad",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--minimum-joint-dominance-ratio",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--motion-radius-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--minimum-motion-density",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--minimum-assignment-ratio",
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "--semantic-exclusion-margin-m",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--maximum-independent-base-translation-m",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--minimum-base-nearest-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument("--base-height-band-m", type=float, default=0.18)
    parser.add_argument("--maximum-anchor-distance-m", type=float, default=0.30)
    parser.add_argument("--maximum-translation-spread-m", type=float, default=0.03)
    parser.add_argument("--maximum-yaw-spread-deg", type=float, default=5.0)
    parser.add_argument("--maximum-holdout-translation-m", type=float, default=0.03)
    parser.add_argument("--maximum-holdout-yaw-deg", type=float, default=5.0)
    parser.add_argument(
        "--maximum-tag-reprojection-rms-px",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--maximum-tag-translation-spread-m",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--maximum-tag-rotation-spread-deg",
        type=float,
        default=2.0,
    )
    args = parser.parse_args(argv)
    report = build(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

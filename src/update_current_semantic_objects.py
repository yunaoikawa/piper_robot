#!/usr/bin/env python3
"""Refresh movable semantic objects in an accepted Pasteur MuJoCo scene.

This stage is observation-only.  SAM supplies instance masks, a fixed metric
tag bridges the current head camera into the accepted scene, and transparent
object Z is constrained by its measured support plane.  Existing static scene
geometry, robot placement, physical arm identity, and NYU grippers are never
refit here.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.arm.home import semantic_model_home_q
from rollout.sam_segmentation import (
    SamSegmentationClient,
    mask_geometry,
)
from rollout.scene_registration import (
    bridge_camera_from_fixed_tag,
    intersect_pixel_with_horizontal_plane,
    tag_pose_camera,
)
from src.refine_scene_robot_alignment import _reference_tag_consensus


SCHEMA = "piper_robot.current_semantic_object_refresh/v1"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def _selected_frame(capture: Path) -> tuple[Path, Path, dict, dict]:
    manifest = json.loads((capture / "manifest.json").read_text())
    preview = manifest.get("derived_preview", {})
    sequence = int(preview.get("selected_sequence", -1))
    if sequence < 0:
        raise ValueError("capture has no selected derived preview sequence")
    camera = str(manifest.get("camera_label", "head"))
    frame = capture / "raw" / camera / f"{sequence:06d}"
    rgb = capture / str(preview["rgb"]["path"])
    if not rgb.is_file() or not (frame / "meta.json").is_file():
        raise FileNotFoundError("selected RGB-D frame is incomplete")
    meta = json.loads((frame / "meta.json").read_text())
    return rgb, frame, meta, manifest


def _model_objects(model_path: Path, definitions: list[dict]) -> list[dict]:
    root = ET.parse(model_path).getroot()
    result = []
    for definition in definitions:
        body_name = str(definition["body_name"])
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"base model lacks body {body_name!r}")
        geom = body.find("geom")
        if geom is None or geom.get("type") != "cylinder":
            raise ValueError(f"{body_name!r} must contain a cylinder geom")
        size = np.fromstring(geom.get("size", ""), sep=" ")
        position = np.fromstring(body.get("pos", ""), sep=" ")
        if size.size < 2 or position.shape != (3,):
            raise ValueError(f"{body_name!r} has invalid cylinder geometry")
        result.append(
            {
                **definition,
                "radius_m": float(size[0]),
                "height_m": float(size[1] * 2.0),
                "previous_center_scene_xyz_m": position.tolist(),
            }
        )
    return result


def assign_instances_by_previous_pose(
    objects: list[dict],
    candidates: list[dict],
    *,
    minimum_margin_in_radii: float,
    maximum_displacement_in_radii: float,
) -> tuple[list[dict], dict]:
    """Assign unlabeled SAM instances without using image-left/right rules."""

    if len(objects) != len(candidates):
        raise ValueError(
            f"expected {len(objects)} SAM instances, got {len(candidates)}"
        )
    costs = np.empty((len(objects), len(candidates)), dtype=float)
    for row, item in enumerate(objects):
        previous = np.asarray(item["previous_center_scene_xyz_m"], dtype=float)
        radius = float(item["radius_m"])
        for column, candidate in enumerate(candidates):
            current = np.asarray(candidate["center_scene_xyz_m"], dtype=float)
            costs[row, column] = (
                np.linalg.norm(current[:2] - previous[:2]) / radius
            )
    ranked = []
    for order in permutations(range(len(candidates))):
        total = float(sum(costs[row, column] for row, column in enumerate(order)))
        ranked.append((total, order))
    ranked.sort(key=lambda item: item[0])
    best_cost, best_order = ranked[0]
    margin = (
        float("inf")
        if len(ranked) == 1
        else float(ranked[1][0] - ranked[0][0])
    )
    assigned = []
    displacements = []
    for row, column in enumerate(best_order):
        displacement = float(costs[row, column])
        displacements.append(displacement)
        assigned.append({**candidates[column], **objects[row]})
    accepted = bool(
        margin >= minimum_margin_in_radii
        and max(displacements, default=0.0) <= maximum_displacement_in_radii
    )
    audit = {
        "method": "minimum_normalized_motion_from_accepted_model",
        "cost_matrix_in_object_radii": costs.tolist(),
        "best_total_cost": best_cost,
        "second_best_margin_in_radii": margin,
        "maximum_assigned_displacement_in_radii": max(
            displacements, default=0.0
        ),
        "thresholds": {
            "minimum_margin_in_radii": minimum_margin_in_radii,
            "maximum_displacement_in_radii": maximum_displacement_in_radii,
        },
        "accepted": accepted,
    }
    if not accepted:
        raise ValueError(f"ambiguous SAM instance assignment: {audit}")
    return assigned, audit


def _accepted_mask_candidates(
    image: np.ndarray,
    mask_records: dict[str, dict],
    objects: list[dict],
) -> tuple[list[dict], dict]:
    candidates = []
    for item in objects:
        semantic_name = str(item["semantic_name"])
        record = mask_records.get(semantic_name)
        if record is None:
            raise ValueError(f"accepted mask missing for {semantic_name!r}")
        path = _resolve(record["path"])
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError(f"accepted mask shape mismatch: {path}")
        geometry = mask_geometry(mask > 0, min_area_px=200)
        if geometry is None:
            raise ValueError(f"accepted mask is not a stable circle: {path}")
        candidates.append(
            {
                "semantic_name": semantic_name,
                "mask": mask > 0,
                "score": float(record.get("score", 1.0)),
                "model": str(record.get("model", "accepted_sam_mask")),
                "prompt": str(record.get("prompt", semantic_name)),
                "center_px": geometry.center_px.tolist(),
                "radius_px": geometry.radius_px,
                "area_px": geometry.area_px,
                "circularity": geometry.circularity,
                "mask_path": str(path),
            }
        )
    return candidates, {
        "mode": "accepted_masks",
        "instance_identity": "explicit semantic mask mapping",
    }


def _live_sam_candidates(
    image: np.ndarray,
    *,
    endpoint: str,
    prompt: str,
    expected_count: int,
    confidence_threshold: float,
    timeout_ms: int,
) -> tuple[list[dict], dict]:
    client = SamSegmentationClient(endpoint, timeout_ms=timeout_ms)
    try:
        result = client.segment(
            image,
            frame_id=1,
            timestamp=time.time(),
            prompt=prompt,
            confidence_threshold=confidence_threshold,
            jpeg_quality=95,
        )
    finally:
        client.close()
    ranked = sorted(result.candidates, key=lambda item: item.score, reverse=True)
    candidates = []
    for candidate in ranked:
        geometry = mask_geometry(candidate.mask, min_area_px=200)
        if geometry is None:
            continue
        candidates.append(
            {
                "mask": candidate.mask,
                "score": float(candidate.score),
                "model": result.model,
                "prompt": prompt,
                "center_px": geometry.center_px.tolist(),
                "radius_px": geometry.radius_px,
                "area_px": geometry.area_px,
                "circularity": geometry.circularity,
            }
        )
    if len(candidates) != expected_count:
        raise ValueError(
            f"SAM produced {len(candidates)} stable circles; expected "
            f"{expected_count}"
        )
    return candidates, {
        "mode": "live_sam",
        "endpoint": endpoint,
        "model": result.model,
        "prompt": prompt,
        "inference_ms": result.inference_ms,
    }


def _metric_centers(
    candidates: list[dict],
    camera_matrix: np.ndarray,
    scene_from_camera: np.ndarray,
    *,
    support_plane_z_m: float,
) -> list[dict]:
    result = []
    for item in candidates:
        center = intersect_pixel_with_horizontal_plane(
            tuple(item["center_px"]),
            camera_matrix,
            scene_from_camera,
            plane_z_m=support_plane_z_m,
        )
        result.append({**item, "center_scene_xyz_m": center.tolist()})
    return result


def _write_model(
    source: Path,
    output: Path,
    assignments: list[dict],
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    for item in assignments:
        body = root.find(f".//body[@name='{item['body_name']}']")
        center = np.asarray(item["center_scene_xyz_m"], dtype=float)
        center[2] = float(item["support_plane_z_m"]) + float(item["height_m"]) / 2
        body.set("pos", " ".join(f"{value:.8f}" for value in center))
        color = item.get("display_rgba")
        if color is not None:
            body.find("geom").set("rgba", " ".join(str(value) for value in color))
        item["center_scene_xyz_m"] = center.tolist()
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode")


def _validate_model(model_path: Path, assignments: list[dict]) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    required = [
        "left/nyu_gripper_visual",
        "right/nyu_gripper_visual",
    ]
    missing = [
        name
        for name in required
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0
    ]
    key = model.key("home")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key.id)
    expected = {
        side: float(semantic_model_home_q(side)[5])
        for side in ("left", "right")
    }
    actual = {
        side: float(
            data.qpos[model.joint(f"{side}/joint6").qposadr[0]]
        )
        for side in ("left", "right")
    }
    position_errors = {}
    for item in assignments:
        body = model.body(str(item["body_name"]))
        position_errors[str(item["body_name"])] = float(
            np.linalg.norm(
                np.asarray(body.pos, dtype=float)
                - np.asarray(item["center_scene_xyz_m"], dtype=float)
            )
        )
    accepted = bool(
        not missing
        and max(
            abs(actual[side] - expected[side]) for side in expected
        )
        <= 1e-7
        and max(position_errors.values(), default=0.0) <= 1e-7
    )
    return {
        "accepted": accepted,
        "compiled": True,
        "required_gripper_geoms_missing": missing,
        "home_joint6_model_rad": actual,
        "expected_home_joint6_model_rad": expected,
        "object_position_error_m": position_errors,
    }


def _object_scene(
    assignments: list[dict],
    *,
    source: dict,
    previous_scene: dict | None,
) -> dict:
    objects = []
    ordered = sorted(
        assignments,
        key=lambda item: (item.get("role") != "target_lid", item["semantic_name"]),
    )
    for item in ordered:
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(item["center_scene_xyz_m"], dtype=float)
        objects.append(
            {
                "instance_id": f"{item['semantic_name']}-current",
                "semantic_name": item["semantic_name"],
                "role": item["role"],
                "status": "auto_observed",
                "pose_scene": pose.tolist(),
                "geometry": {
                    "type": "cylinder",
                    "radius_m": item["radius_m"],
                    "height_m": item["height_m"],
                    "pose_anchor": "center",
                },
                "perception": {
                    "sam_model": item["model"],
                    "sam_prompt": item["prompt"],
                    "sam_score": item["score"],
                    "sam_center_px_landscape": item["center_px"],
                    "support_plane_z_m": item["support_plane_z_m"],
                    "fixed_pixel_roi": False,
                },
            }
        )
    previous_source = {} if previous_scene is None else previous_scene.get("source", {})
    return {
        "schema": "piper_robot.dynamic_dish_lid_scene/v2",
        "source": {
            **source,
            "episode_targets_scene_xyz_m": previous_source.get(
                "episode_targets_scene_xyz_m", {}
            ),
        },
        "camera_to_scene_accepted": True,
        "operator_confirmed": False,
        "objects": objects,
    }


def _write_overlay(
    image: np.ndarray,
    assignments: list[dict],
    path: Path,
) -> None:
    output = image.copy()
    for item in assignments:
        color = tuple(int(value) for value in item.get("overlay_bgr", [0, 255, 0]))
        center = tuple(int(round(value)) for value in item["center_px"])
        radius = int(round(item["radius_px"]))
        cv2.circle(output, center, radius, color, 5)
        cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 34, 4)
        cv2.putText(
            output,
            str(item["semantic_name"]),
            (max(10, center[0] - radius), max(35, center[1] - radius - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), output)


def run(args) -> dict:
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text())
    settings = config["current_object_refresh"]
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = _resolve(args.model)
    capture = _resolve(settings["current_capture"])
    rgb_path, _, meta, manifest = _selected_frame(capture)
    image = cv2.imread(str(rgb_path))
    if image is None:
        raise FileNotFoundError(rgb_path)
    quality = meta.get("quality", {})
    mean_gray = float(quality.get("rgb_mean_gray", np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))))
    near_black = float(quality.get("rgb_near_black_fraction", 1.0))
    if (
        mean_gray < float(settings.get("minimum_mean_gray", 35.0))
        or near_black > float(settings.get("maximum_near_black_fraction", 0.35))
    ):
        raise ValueError("current capture failed the lighting gate")

    objects = _model_objects(model_path, list(settings["objects"]))
    accepted_masks = settings.get("accepted_masks")
    if accepted_masks:
        candidates, segmentation = _accepted_mask_candidates(
            image, accepted_masks, objects
        )
        explicit_identity = True
    else:
        endpoint = args.sam_endpoint or settings.get("sam_endpoint")
        if not endpoint:
            raise ValueError("live current-object refresh requires a SAM endpoint")
        candidates, segmentation = _live_sam_candidates(
            image,
            endpoint=str(endpoint),
            prompt=str(settings["shared_prompt"]),
            expected_count=len(objects),
            confidence_threshold=float(
                settings.get("sam_confidence_threshold", 0.80)
            ),
            timeout_ms=int(settings.get("sam_timeout_ms", 120000)),
        )
        explicit_identity = False

    reference_consensus, reference_records = _reference_tag_consensus(
        _resolve(settings["reference_report"]),
        _resolve(settings["reference_capture"]),
        tag_id=int(settings.get("tag_id", 3)),
        tag_size_m=float(settings.get("tag_size_m", 0.06)),
    )
    camera_matrix = np.asarray(meta["intrinsics"]["K_raw_rgb"], dtype=float)
    camera_from_tag, tag_rms = tag_pose_camera(
        image,
        camera_matrix,
        tag_id=int(settings.get("tag_id", 3)),
        tag_size_m=float(settings.get("tag_size_m", 0.06)),
    )
    scene_from_camera = bridge_camera_from_fixed_tag(
        reference_consensus.transform,
        camera_from_tag,
    )
    candidates = _metric_centers(
        candidates,
        camera_matrix,
        scene_from_camera,
        support_plane_z_m=float(settings["support_plane_z_m"]),
    )
    if explicit_identity:
        by_name = {item["semantic_name"]: item for item in candidates}
        assignments = [{**by_name[item["semantic_name"]], **item} for item in objects]
        assignment_audit = {
            "method": "explicit_accepted_semantic_masks",
            "accepted": True,
        }
    else:
        assignments, assignment_audit = assign_instances_by_previous_pose(
            objects,
            candidates,
            minimum_margin_in_radii=float(
                settings.get("minimum_assignment_margin_in_radii", 0.25)
            ),
            maximum_displacement_in_radii=float(
                settings.get("maximum_displacement_in_radii", 5.0)
            ),
        )
    for item in assignments:
        item["support_plane_z_m"] = float(settings["support_plane_z_m"])

    output_model = output / "scene.mjcf"
    _write_model(model_path, output_model, assignments)
    validation = _validate_model(output_model, assignments)
    if not validation["accepted"]:
        raise ValueError(f"updated MuJoCo model failed validation: {validation}")

    previous_scene_path = args.previous_object_scene
    previous_scene = (
        None
        if not previous_scene_path
        else json.loads(_resolve(previous_scene_path).read_text())
    )
    source = {
        "kind": "fixed_head_sam_plus_fixed_tag_support_plane",
        "capture_manifest": str((capture / "manifest.json").resolve()),
        "frame_sequence": int(manifest["derived_preview"]["selected_sequence"]),
        "tag_id": int(settings.get("tag_id", 3)),
        "tag_reprojection_rms_px": tag_rms,
    }
    object_scene = _object_scene(
        assignments, source=source, previous_scene=previous_scene
    )
    object_scene_path = output / "latest_target_scene.json"
    object_scene_path.write_text(
        json.dumps(object_scene, indent=2, ensure_ascii=False) + "\n"
    )
    overlay = output / "current_objects_overlay.png"
    _write_overlay(image, assignments, overlay)
    report = {
        "schema": SCHEMA,
        "accepted": True,
        "commands_sent": False,
        "hardware_motion_authorized": False,
        "config": str(config_path),
        "base_model": str(model_path),
        "lighting_gate": {
            "accepted": True,
            "rgb_mean_gray": mean_gray,
            "rgb_near_black_fraction": near_black,
        },
        "segmentation": segmentation,
        "registration": {
            "accepted": True,
            "tag_reprojection_rms_px": tag_rms,
            "reference_translation_spread_m": (
                reference_consensus.translation_spread_m
            ),
            "reference_rotation_spread_deg": (
                reference_consensus.rotation_spread_deg
            ),
            "reference_views": reference_records,
        },
        "assignment": assignment_audit,
        "objects": [
            {
                key: item[key]
                for key in (
                    "semantic_name",
                    "body_name",
                    "role",
                    "center_scene_xyz_m",
                    "radius_m",
                    "height_m",
                    "score",
                    "prompt",
                    "model",
                )
            }
            for item in assignments
        ],
        "model_validation": validation,
        "artifacts": {
            "scene_model": str(output_model),
            "object_scene": str(object_scene_path),
            "overlay": str(overlay),
        },
    }
    report_path = output / "current_object_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--previous-object-scene")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam-endpoint")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

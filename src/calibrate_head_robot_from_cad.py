#!/usr/bin/env python3
"""Estimate one fixed head-camera-to-bimanual-Piper transform.

The camera remains fixed while an operator teleoperates four distinct robot
poses.  RGB-D bursts and read-only qpos snapshots are synchronized by
``capture_record3d_multiview.py``.  SAM supplies robot masks; geometry comes
from the exact Piper CAD with the pinned NYU grippers.  This program never
sends robot commands and fails closed when train/holdout gates are not met.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from rollout.multiview_scene import gravity_level_transform
from rollout.scene_3d import backproject
from rollout.semantic_scene_pipeline import load_mask, load_profile, sha256_file
from src.build_multiview_semantic_scene import _pin_nyu_grippers, transform_points
from src.reconstruct_multiview_scene import _temporal_view


SCHEMA = "piper_robot.camera_robot_calibration/v1"


def _parse_masks(specs: list[str]) -> dict[str, Path]:
    result = {}
    for spec in specs:
        left, separator, path = spec.partition("=")
        view, colon, label = left.partition(":")
        if not separator or not colon or label != "robot":
            raise ValueError("accepted masks must be VIEW:robot=/path/mask.png")
        result[view] = Path(path).resolve()
    return result


def _qpos(view: dict) -> list[float]:
    state = view.get("robot_state", {})
    stability = state.get("stability", {})
    qpos = stability.get("representative_qpos_rad")
    if (
        not stability.get("accepted", False)
        or qpos is None
        or len(qpos) != 12
    ):
        raise ValueError(
            f"{view.get('name')}: stable synchronized 12-joint qpos is missing"
        )
    return [float(item) for item in qpos]


def _cad_geometry(model_path: Path, qpos: list[float]) -> tuple[np.ndarray, dict]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nq != 12:
        raise ValueError(f"expected 12 robot qpos, model has nq={model.nq}")
    data.qpos[:] = np.asarray(qpos, dtype=float)
    mujoco.mj_forward(model, data)

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(model.nbody)
    ]
    geoms = {"left": [], "right": []}
    all_points = []
    mesh_kind = int(mujoco.mjtGeom.mjGEOM_MESH)
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 2:
            continue
        body_name = body_names[int(model.geom_bodyid[geom_id])] or ""
        arm = "left" if body_name.startswith("left/") else (
            "right" if body_name.startswith("right/") else None
        )
        if arm is None or int(model.geom_type[geom_id]) != mesh_kind:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        stride = max(1, len(local) // 1200)
        local = local[::stride]
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world = (
            rotation @ local.T
        ).T + np.asarray(data.geom_xpos[geom_id], dtype=float)
        geoms[arm].append(world)
        all_points.append(world)
    if not all_points:
        raise RuntimeError("Piper CAD produced no visual mesh vertices")
    return np.vstack(all_points), geoms


def _matrix(parameters: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    result[:3, 3] = parameters[3:]
    return result


def _parameters(transform: np.ndarray) -> np.ndarray:
    return np.r_[
        Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
        transform[:3, 3],
    ]


def _initial_transform(
    camera_pose: dict,
    observations: list[np.ndarray],
    cad_points: list[np.ndarray],
) -> np.ndarray:
    level_from_camera = gravity_level_transform(camera_pose)
    observed_level = transform_points(np.vstack(observations), level_from_camera)
    observed_center = np.median(observed_level, axis=0)
    cad_center = np.median(np.vstack(cad_points), axis=0)
    best = None
    for yaw in np.linspace(-np.pi, np.pi, 36, endpoint=False):
        yaw_rotation = Rotation.from_euler("z", yaw).as_matrix()
        rotation = yaw_rotation @ level_from_camera[:3, :3]
        translation = cad_center - rotation @ observed_center
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        score = 0.0
        for observed, cad in zip(observations, cad_points):
            query = transform_points(observed[:: max(1, len(observed) // 800)], transform)
            distance, _ = cKDTree(cad).query(query, workers=-1)
            score += float(np.median(distance))
        if best is None or score < best[0]:
            best = (score, transform)
    assert best is not None
    return best[1]


def fit_transform(
    observations: list[np.ndarray],
    cad_points: list[np.ndarray],
    initial: np.ndarray,
    *,
    maximum_points_per_view: int = 2500,
    residual_clip_m: float = 0.08,
    maximum_evaluations: int = 120,
) -> np.ndarray:
    """Fit camera points to pose-specific CAD surfaces with robust residuals."""

    sampled = [
        points[:: max(1, len(points) // maximum_points_per_view)][
            :maximum_points_per_view
        ]
        for points in observations
    ]
    trees = [cKDTree(points) for points in cad_points]

    def residual(parameters):
        transform = _matrix(parameters)
        values = []
        for observed, cad, tree in zip(sampled, cad_points, trees):
            transformed = transform_points(observed, transform)
            _, indices = tree.query(transformed, workers=-1)
            vector = transformed - cad[indices]
            length = np.linalg.norm(vector, axis=1)
            scale = np.minimum(1.0, residual_clip_m / np.maximum(length, 1e-9))
            values.append((vector * scale[:, None]).ravel())
        return np.concatenate(values)

    result = least_squares(
        residual,
        _parameters(initial),
        loss="soft_l1",
        f_scale=0.012,
        max_nfev=maximum_evaluations,
    )
    return _matrix(result.x)


def _projected_mask(
    geoms: dict[str, list[np.ndarray]],
    robot_from_camera: np.ndarray,
    camera_matrix: np.ndarray,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    camera_from_robot = np.linalg.inv(robot_from_camera)
    masks = {
        "left": np.zeros(shape_hw, dtype=np.uint8),
        "right": np.zeros(shape_hw, dtype=np.uint8),
    }
    for arm, arm_geoms in geoms.items():
        for vertices in arm_geoms:
            camera = transform_points(vertices, camera_from_robot)
            valid = camera[:, 2] > 0.05
            camera = camera[valid]
            if len(camera) < 3:
                continue
            projected = (camera_matrix @ camera.T).T
            pixels = projected[:, :2] / projected[:, 2:3]
            finite = np.all(np.isfinite(pixels), axis=1)
            pixels = np.rint(pixels[finite]).astype(np.int32)
            inside = (
                (pixels[:, 0] >= -20)
                & (pixels[:, 0] < shape_hw[1] + 20)
                & (pixels[:, 1] >= -20)
                & (pixels[:, 1] < shape_hw[0] + 20)
            )
            pixels = pixels[inside]
            if len(pixels) >= 3:
                cv2.fillConvexPoly(masks[arm], cv2.convexHull(pixels), 1)
    union = np.logical_or(masks["left"], masks["right"])
    return union, {key: value.astype(bool) for key, value in masks.items()}


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second)
    return (
        float(np.count_nonzero(np.logical_and(first, second)))
        / float(np.count_nonzero(union))
        if np.any(union)
        else 0.0
    )


def _best_component_iou(
    observed_union: np.ndarray, projected_arm: np.ndarray
) -> float:
    count, labels = cv2.connectedComponents(
        observed_union.astype(np.uint8), connectivity=8
    )
    candidates = [labels == index for index in range(1, count)]
    return max((_iou(mask, projected_arm) for mask in candidates), default=0.0)


def _metrics(
    observed_points: np.ndarray,
    observed_mask: np.ndarray,
    cad_points: np.ndarray,
    cad_geoms: dict,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
) -> dict:
    transformed = transform_points(observed_points, transform)
    distances, _ = cKDTree(cad_points).query(transformed, workers=-1)
    projected, per_arm = _projected_mask(
        cad_geoms, transform, camera_matrix, observed_mask.shape
    )
    return {
        "depth_median_m": float(np.median(distances)),
        "depth_p90_m": float(np.quantile(distances, 0.90)),
        "mask_union_iou": _iou(observed_mask, projected),
        # SAM supplies an unlabeled robot union.  Match each CAD arm to its
        # best connected SAM component; requiring both catches a missing or
        # mirrored arm without inventing left/right image coordinates.
        "mask_per_arm_iou": {
            arm: _best_component_iou(observed_mask, mask)
            for arm, mask in per_arm.items()
        },
    }


def _mask_overlay(
    rgb_bgr: np.ndarray,
    sam_mask: np.ndarray,
    projected: np.ndarray,
    path: Path,
) -> None:
    image = rgb_bgr.copy()
    sam_rgb = cv2.resize(
        sam_mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    projected_rgb = cv2.resize(
        projected.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    image[sam_rgb] = (
        0.55 * image[sam_rgb] + 0.45 * np.array([40, 220, 40])
    ).astype(np.uint8)
    image[projected_rgb] = (
        0.55 * image[projected_rgb] + 0.45 * np.array([220, 80, 40])
    ).astype(np.uint8)
    overlap = sam_rgb & projected_rgb
    image[overlap] = np.array([40, 220, 220], dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def build(args) -> dict:
    from src.build_semantic_scene import _run_sam

    capture = Path(args.capture).resolve()
    manifest_path = capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "piper_robot.rgbd_multiview_capture/v1":
        raise ValueError("unsupported capture schema")
    if manifest.get("operator_action") != "move-robot":
        raise ValueError("calibration requires operator_action=move-robot")
    if manifest.get("commands_sent") is not False:
        raise ValueError("capture command provenance is unsafe")
    views = list(manifest.get("views", ()))
    if len(views) < 4:
        raise ValueError("four or more stopped robot poses are required")
    profile, catalog = load_profile(args.profile)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = output.with_suffix("")
    audit_dir.mkdir(parents=True, exist_ok=True)
    accepted_masks = _parse_masks(args.mask)
    unknown = sorted(set(accepted_masks) - {view["name"] for view in views})
    if unknown:
        raise ValueError(f"accepted masks reference unknown views: {unknown}")

    robot_profile = dict(profile)
    robot_profile["objects"] = ["robot"]
    robot_catalog = {"robot": catalog["robot"]}
    temporal = [
        _temporal_view(capture, view, minimum_confidence=args.minimum_confidence)
        for view in views
    ]
    qposes = [_qpos(view) for view in views]
    masks = []
    mask_sources = []
    for item in temporal:
        mask_path = accepted_masks.get(item["name"])
        if mask_path is None:
            observations = _run_sam(
                item["rgb_bgr"],
                robot_profile,
                robot_catalog,
                args.sam_endpoint,
                audit_dir / item["name"],
            )
            robot_masks = [
                load_mask(record.mask_path, item["rgb_bgr"].shape[:2])
                for record in observations
                if record.semantic_name == "robot"
            ]
            if not robot_masks:
                raise RuntimeError(f"{item['name']}: SAM robot mask missing")
            rgb_mask = np.logical_or.reduce(robot_masks)
            mask_sources.append([record.mask_path for record in observations])
        else:
            rgb_mask = load_mask(mask_path, item["rgb_bgr"].shape[:2])
            mask_sources.append([str(mask_path)])
        depth_mask = cv2.resize(
            rgb_mask.astype(np.uint8),
            (item["depth_m"].shape[1], item["depth_m"].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        masks.append(depth_mask)

    # Fixed-camera clutter mistakenly labelled as robot (e.g. microscope
    # struts) persists at identical pixels while articulated links move.
    # Remove only the eroded all-view intersection, retaining boundaries and
    # allowing the stationary robot bases to remain partially observed.
    persistent = np.logical_and.reduce(masks).astype(np.uint8)
    persistent = cv2.erode(persistent, np.ones((5, 5), np.uint8), iterations=1)
    cleaned_masks = [mask & ~persistent.astype(bool) for mask in masks]

    with tempfile.TemporaryDirectory(prefix="piper-cad-calibration-") as temporary:
        model_path = Path(temporary) / "pinned_piper.xml"
        _pin_nyu_grippers(
            Path(profile["robot_model"]).resolve(),
            model_path,
            profile,
            None,
            0.0,
        )
        cad = [_cad_geometry(model_path, qpos) for qpos in qposes]

    observed_points = []
    for item, mask in zip(temporal, cleaned_masks):
        valid = (
            mask
            & np.isfinite(item["depth_m"])
            & (item["depth_m"] >= args.min_depth)
            & (item["depth_m"] <= args.max_depth)
            & (item["confidence"] >= args.minimum_confidence)
        )
        points = backproject(item["depth_m"], item["camera_matrix"])[valid]
        if len(points) < 300:
            raise RuntimeError(
                f"{item['name']}: only {len(points)} robot RGB-D points after cleanup"
            )
        observed_points.append(points)
    cad_points = [value[0] for value in cad]
    initial = _initial_transform(
        temporal[0]["camera_pose"], observed_points[:3], cad_points[:3]
    )
    fitted = fit_transform(observed_points[:-1], cad_points[:-1], initial)

    metrics = [
        _metrics(
            observed,
            mask,
            cad_points[index],
            cad[index][1],
            fitted,
            temporal[index]["camera_matrix"],
        )
        for index, (observed, mask) in enumerate(
            zip(observed_points, cleaned_masks)
        )
    ]
    projected_masks = [
        _projected_mask(
            cad[index][1],
            fitted,
            temporal[index]["camera_matrix"],
            cleaned_masks[index].shape,
        )[0]
        for index in range(len(views))
    ]
    for item, sam_mask, projected in zip(
        temporal, cleaned_masks, projected_masks
    ):
        _mask_overlay(
            item["rgb_bgr"],
            sam_mask,
            projected,
            audit_dir / f"{item['name']}_overlay.png",
        )

    independent = [
        fit_transform([observed_points[index]], [cad_points[index]], fitted)
        for index in range(len(views))
    ]
    translations = np.asarray([item[:3, 3] for item in independent])
    rotation_deltas = [
        Rotation.from_matrix(item[:3, :3] @ fitted[:3, :3].T).magnitude()
        for item in independent
    ]
    repeatability_translation = float(
        np.max(np.linalg.norm(translations - np.median(translations, axis=0), axis=1))
    )
    repeatability_rotation = float(np.degrees(max(rotation_deltas)))
    train = metrics[:-1]
    holdout = metrics[-1]
    thresholds = {
        "train_depth_median_max_m": 0.010,
        "train_depth_p90_max_m": 0.025,
        "holdout_depth_median_max_m": 0.015,
        "holdout_depth_p90_max_m": 0.030,
        "mask_union_iou_min": 0.70,
        "mask_per_arm_iou_min": 0.60,
        "repeatability_translation_max_m": 0.005,
        "repeatability_rotation_max_deg": 1.0,
    }
    decisions = {
        "train_depth": all(
            item["depth_median_m"] <= thresholds["train_depth_median_max_m"]
            and item["depth_p90_m"] <= thresholds["train_depth_p90_max_m"]
            for item in train
        ),
        "holdout_depth": (
            holdout["depth_median_m"]
            <= thresholds["holdout_depth_median_max_m"]
            and holdout["depth_p90_m"]
            <= thresholds["holdout_depth_p90_max_m"]
        ),
        "mask_union": all(
            item["mask_union_iou"] >= thresholds["mask_union_iou_min"]
            for item in metrics
        ),
        "mask_per_arm": all(
            min(item["mask_per_arm_iou"].values())
            >= thresholds["mask_per_arm_iou_min"]
            for item in metrics
        ),
        "repeatability": (
            repeatability_translation
            <= thresholds["repeatability_translation_max_m"]
            and repeatability_rotation
            <= thresholds["repeatability_rotation_max_deg"]
        ),
    }
    accepted = all(decisions.values())
    report = {
        "schema": SCHEMA,
        "calibration_id": f"head-piper-cad-{manifest['session_id']}",
        "accepted": accepted,
        "T_robot_camera": fitted.tolist(),
        "transform_convention": "p_robot = T_robot_camera @ p_camera",
        "method": (
            "fixed_head_rgbd_sam_masks_plus_synchronized_qpos_exact_piper_cad"
        ),
        "commands_sent": False,
        "fit_view_names": [view["name"] for view in views[:-1]],
        "holdout_view_name": views[-1]["name"],
        "metrics": {
            "per_view": {
                view["name"]: value for view, value in zip(views, metrics)
            },
            "repeatability_translation_m": repeatability_translation,
            "repeatability_rotation_deg": repeatability_rotation,
        },
        "thresholds": thresholds,
        "decisions": decisions,
        "static_false_positive_cleanup": {
            "method": "eroded_all_view_robot_mask_intersection",
            "persistent_pixels_removed": int(np.count_nonzero(persistent)),
            "purpose": "remove fixed clutter such as microscope-arm confusion",
        },
        "source": {
            "capture_manifest": str(manifest_path),
            "capture_manifest_sha256": sha256_file(manifest_path),
            "profile": str(Path(args.profile).resolve()),
            "record3d_udid": manifest.get("device", {}).get("udid"),
            "mask_sources": {
                view["name"]: sources
                for view, sources in zip(views, mask_sources)
            },
            "qpos_rad": {
                view["name"]: qpos for view, qpos in zip(views, qposes)
            },
        },
        "artifacts": {
            "audit_directory": str(audit_dir),
            "overlays": {
                view["name"]: str(
                    (audit_dir / f"{view['name']}_overlay.png").resolve()
                )
                for view in views
            },
        },
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        help="accepted VIEW:robot=/absolute/mask.png",
    )
    parser.add_argument("--minimum-confidence", type=int, default=1)
    parser.add_argument("--min-depth", type=float, default=0.12)
    parser.add_argument("--max-depth", type=float, default=3.0)
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

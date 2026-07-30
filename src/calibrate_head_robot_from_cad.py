#!/usr/bin/env python3
"""Estimate fixed head-camera transforms to two independently mounted Pipers.

The camera remains fixed while an operator teleoperates four distinct robot
poses.  RGB-D bursts and read-only qpos snapshots are synchronized by
``capture_record3d_multiview.py``.  SAM supplies robot masks; geometry comes
from the exact Piper CAD with the pinned NYU grippers.  This program never
sends robot commands and fails closed when train/holdout gates are not met.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from rollout.multiview_scene import gravity_level_transform
from rollout.scene_3d import backproject
from rollout.semantic_scene_pipeline import load_mask, load_profile, sha256_file
from src.build_multiview_semantic_scene import transform_points
from src.reconstruct_multiview_scene import _temporal_view


# Keep the established envelope so existing fail-closed consumers can read the
# authoritative right-arm transform. ``schema_version`` records the additive
# per-arm extension below.
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


def _qpos_diversity(
    qposes: list[list[float]],
    *,
    minimum_joint_range_rad: float,
    minimum_moving_joints_per_arm: int,
    minimum_holdout_distance_rad: float,
) -> dict:
    values = np.asarray(qposes, dtype=float)
    if values.ndim != 2 or values.shape[1] != 12 or len(values) < 3:
        raise ValueError("qpos diversity requires three or more 12-joint poses")
    train = values[:-1]
    holdout = values[-1]
    per_arm = {}
    accepted = True
    for arm, joint_slice in (
        ("left", slice(0, 6)),
        ("right", slice(6, 12)),
    ):
        ranges = np.ptp(train[:, joint_slice], axis=0)
        moving = int(np.count_nonzero(ranges >= minimum_joint_range_rad))
        holdout_distance = float(
            np.min(
                np.linalg.norm(
                    train[:, joint_slice] - holdout[joint_slice],
                    axis=1,
                )
            )
        )
        arm_accepted = bool(
            moving >= minimum_moving_joints_per_arm
            and holdout_distance >= minimum_holdout_distance_rad
        )
        accepted &= arm_accepted
        per_arm[arm] = {
            "accepted": arm_accepted,
            "train_joint_range_rad": ranges.tolist(),
            "moving_joint_count": moving,
            "holdout_nearest_train_distance_rad": holdout_distance,
        }
    return {
        "accepted": bool(accepted),
        "minimum_joint_range_rad": float(minimum_joint_range_rad),
        "minimum_moving_joints_per_arm": int(minimum_moving_joints_per_arm),
        "minimum_holdout_distance_rad": float(minimum_holdout_distance_rad),
        "per_arm": per_arm,
    }


def _cad_geometry(
    model_path: Path,
    qpos: list[float],
    physical_to_model: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Return CAD vertices using the same physical/model mapping as ConeE.

    ConeE deliberately drives the model's ``right_arm_*`` branch from the
    physical left controller and its ``left_arm_*`` branch from the physical
    right controller.  The two physical Piper bases are independently mounted,
    so their geometry must later be fitted with independent camera transforms.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nq != 12:
        raise ValueError(f"expected 12 robot qpos, model has nq={model.nq}")
    qpos = np.asarray(qpos, dtype=float)
    physical_to_model = physical_to_model or {
        "left": "right_arm_",
        "right": "left_arm_",
    }
    for physical_arm, model_prefix in physical_to_model.items():
        values = qpos[:6] if physical_arm == "left" else qpos[6:]
        for index, value in enumerate(values, start=1):
            joint = model.joint(f"{model_prefix}joint{index}")
            data.qpos[int(joint.qposadr[0])] = value
    mujoco.mj_forward(model, data)

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(model.nbody)
    ]
    geoms = {"left": [], "right": []}
    base_from_model = {}
    for physical_arm, model_prefix in physical_to_model.items():
        body_id = model.body(f"{model_prefix}link0").id
        model_from_base = np.eye(4)
        model_from_base[:3, :3] = np.asarray(
            data.xmat[body_id], dtype=float
        ).reshape(3, 3)
        model_from_base[:3, 3] = np.asarray(
            data.xpos[body_id], dtype=float
        )
        base_from_model[physical_arm] = np.linalg.inv(model_from_base)
    anchors = {}
    for physical_arm, model_prefix in physical_to_model.items():
        site_id = model.site(f"{model_prefix}ee").id
        model_from_site = np.eye(4)
        model_from_site[:3, :3] = np.asarray(
            data.site_xmat[site_id], dtype=float
        ).reshape(3, 3)
        model_from_site[:3, 3] = np.asarray(
            data.site_xpos[site_id], dtype=float
        )
        anchors[physical_arm] = (
            base_from_model[physical_arm] @ model_from_site
        )
    all_points = []
    mesh_kind = int(mujoco.mjtGeom.mjGEOM_MESH)
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 2:
            continue
        body_name = body_names[int(model.geom_bodyid[geom_id])] or ""
        physical_arm = next(
            (
                arm
                for arm, prefix in physical_to_model.items()
                if body_name.startswith(prefix)
            ),
            None,
        )
        if physical_arm is None or int(model.geom_type[geom_id]) != mesh_kind:
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
        base_local = transform_points(world, base_from_model[physical_arm])
        geoms[physical_arm].append(base_local)
        all_points.append(base_local)
    if not all_points:
        raise RuntimeError("Piper CAD produced no visual mesh vertices")
    return np.vstack(all_points), geoms, anchors


def _large_components(mask: np.ndarray, minimum_pixels: int = 120) -> list[np.ndarray]:
    """Keep plausible SAM instances without using image-side assumptions."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    result = [
        labels == index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_pixels
    ]
    result = sorted(result, key=np.count_nonzero, reverse=True)
    if not result:
        return []
    largest = np.count_nonzero(result[0])
    # Tiny fragments make one-way ICP deceptively cheap. This relative test
    # adapts to resolution and distance while preserving an occluded arm.
    return [
        component
        for component in result[:4]
        if np.count_nonzero(component) >= 0.30 * largest
    ]


def _component_groups_from_parts(
    parts: list[np.ndarray],
) -> list[tuple[np.ndarray, frozenset[int]]]:
    """Form instance hypotheses from SAM parts split by self-occlusion."""

    groups = []
    for size in range(1, min(3, len(parts)) + 1):
        for indices in combinations(range(len(parts)), size):
            # Two physical arms are expected. The two largest SAM components
            # are therefore instance cores; smaller disconnected components
            # may augment a core but may not become an arm by themselves.
            if not set(indices).intersection(range(min(2, len(parts)))):
                continue
            groups.append(
                (
                    np.logical_or.reduce([parts[index] for index in indices]),
                    frozenset(indices),
                )
            )
    return groups


def _tracked_core_by_arm(
    parts_by_view: list[list[np.ndarray]],
    qposes: list[list[float]],
) -> dict[str, list[int]]:
    """Track two fixed-base instances and label them by qpos-correlated motion."""

    if any(len(parts) < 2 for parts in parts_by_view):
        raise RuntimeError("two robot instance cores are required in every view")
    tracks = [[0], [1]]
    previous = [parts_by_view[0][0], parts_by_view[0][1]]
    for parts in parts_by_view[1:]:
        direct = _iou(previous[0], parts[0]) + _iou(previous[1], parts[1])
        swapped = _iou(previous[0], parts[1]) + _iou(previous[1], parts[0])
        assignment = (0, 1) if direct >= swapped else (1, 0)
        for track, part_index in zip(tracks, assignment):
            track.append(part_index)
        previous = [parts[assignment[0]], parts[assignment[1]]]

    visual_motion = []
    for track_index, track in enumerate(tracks):
        visual_motion.append(
            np.asarray(
                [
                    1.0
                    - _iou(
                        parts_by_view[index - 1][track[index - 1]],
                        parts_by_view[index][track[index]],
                    )
                    for index in range(1, len(parts_by_view))
                ],
                dtype=float,
            )
        )
    joint_motion = {
        "left": np.asarray(
            [
                np.linalg.norm(
                    np.asarray(qposes[index][:6])
                    - np.asarray(qposes[index - 1][:6])
                )
                for index in range(1, len(qposes))
            ]
        ),
        "right": np.asarray(
            [
                np.linalg.norm(
                    np.asarray(qposes[index][6:])
                    - np.asarray(qposes[index - 1][6:])
                )
                for index in range(1, len(qposes))
            ]
        ),
    }

    # Use the transition with the strongest one-arm excitation. Later poses can
    # move both arms and SAM visibility can change abruptly, which makes a
    # global correlation spuriously swap identities.
    left_motion = joint_motion["left"]
    right_motion = joint_motion["right"]
    excitation = np.abs(
        np.log((left_motion + 1e-3) / (right_motion + 1e-3))
    )
    transition = int(np.argmax(excitation))
    track_with_more_motion = int(
        visual_motion[1][transition] > visual_motion[0][transition]
    )
    arm_with_more_motion = (
        "left"
        if left_motion[transition] > right_motion[transition]
        else "right"
    )
    if arm_with_more_motion == "left":
        left_track = track_with_more_motion
    else:
        left_track = 1 - track_with_more_motion
    return {
        "left": tracks[left_track],
        "right": tracks[1 - left_track],
    }


def _component_points(
    item: dict,
    mask: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    minimum_confidence: int,
) -> np.ndarray:
    valid = (
        mask
        & np.isfinite(item["depth_m"])
        & (item["depth_m"] >= min_depth)
        & (item["depth_m"] <= max_depth)
        & (item["confidence"] >= minimum_confidence)
    )
    return backproject(item["depth_m"], item["camera_matrix"])[valid]


def _blue_tool_anchor(item: dict, arm_mask: np.ndarray) -> np.ndarray | None:
    """Return a robust RGB-D centroid of the cyan NYU gripper fingers."""

    hsv = cv2.cvtColor(item["rgb_bgr"], cv2.COLOR_BGR2HSV)
    blue_rgb = cv2.inRange(hsv, (80, 80, 60), (110, 255, 255))
    height, width = item["depth_m"].shape
    blue = cv2.resize(
        blue_rgb,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    region = cv2.dilate(
        arm_mask.astype(np.uint8),
        np.ones((5, 5), np.uint8),
        iterations=1,
    ).astype(bool)
    candidate = blue & region
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    indices = sorted(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:2]
    if not indices:
        return None
    selected = np.isin(labels, indices)
    valid = (
        selected
        & np.isfinite(item["depth_m"])
        & (item["depth_m"] > 0.12)
        & (item["confidence"] >= 1)
    )
    points = backproject(item["depth_m"], item["camera_matrix"])[valid]
    return np.median(points, axis=0) if len(points) >= 20 else None


def _anchor_initial_transform(
    camera_pose: dict,
    observed: list[np.ndarray],
    cad: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit upright camera pose and one rigid tool-point offset."""

    level = gravity_level_transform(camera_pose)
    level_observed = transform_points(np.asarray(observed), level)
    cad_positions = np.asarray([transform[:3, 3] for transform in cad])
    first = level_observed[:, :2] - np.mean(
        level_observed[:, :2], axis=0
    )
    second = cad_positions[:, :2] - np.mean(cad_positions[:, :2], axis=0)
    cosine = float(np.sum(first[:, 0] * second[:, 0] + first[:, 1] * second[:, 1]))
    sine = float(np.sum(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]))
    yaw = float(np.arctan2(sine, cosine))
    seed_rotation = (
        Rotation.from_euler("z", yaw).as_matrix() @ level[:3, :3]
    )
    seed_translation = np.median(
        cad_positions - (seed_rotation @ np.asarray(observed).T).T,
        axis=0,
    )

    def values(parameters):
        rotation = (
            Rotation.from_euler("z", parameters[0]).as_matrix()
            @ level[:3, :3]
        )
        translation = parameters[1:4]
        offset = parameters[4:7]
        residual = []
        for camera_point, base_from_ee in zip(observed, cad):
            predicted = (
                base_from_ee[:3, :3] @ offset + base_from_ee[:3, 3]
            )
            residual.extend(rotation @ camera_point + translation - predicted)
        return np.asarray(residual)

    solution = least_squares(
        values,
        np.r_[yaw, seed_translation, np.zeros(3)],
        bounds=(
            np.r_[yaw - 0.6, seed_translation - 0.35, [-0.20] * 3],
            np.r_[yaw + 0.6, seed_translation + 0.35, [0.20] * 3],
        ),
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=160,
    )
    result = np.eye(4)
    result[:3, :3] = (
        Rotation.from_euler("z", solution.x[0]).as_matrix()
        @ level[:3, :3]
    )
    result[:3, 3] = solution.x[1:4]
    offset = solution.x[4:7]
    residuals = []
    for camera_point, base_from_ee in zip(observed, cad):
        predicted = base_from_ee[:3, :3] @ offset + base_from_ee[:3, 3]
        residuals.append(
            np.linalg.norm(
                transform_points(camera_point[None, :], result)[0] - predicted
            )
        )
    return result, offset, float(np.median(residuals))


def _component_fit_cost(
    observed: np.ndarray,
    cad: np.ndarray,
    transform: np.ndarray,
) -> float:
    """Robust one-way surface distance with a small coverage penalty."""

    sampled = observed[:: max(1, len(observed) // 900)][:900]
    transformed = transform_points(sampled, transform)
    distance, _ = cKDTree(cad).query(transformed, workers=-1)
    cad_sampled = cad[:: max(1, len(cad) // 900)][:900]
    reverse, _ = cKDTree(transformed).query(cad_sampled, workers=-1)
    # Bidirectional coverage prevents one disconnected base/link fragment from
    # winning merely because all of its points touch some CAD surface.
    return float(
        0.50 * np.median(distance)
        + 0.25 * np.quantile(distance, 0.80)
        + 0.25 * np.median(reverse)
    )


def _motion_inconsistency(
    arm: str,
    view_index: int,
    candidate_index: int,
    component_masks: list[list[np.ndarray]],
    qposes: list[list[float]],
) -> float:
    """Penalize a mask that stays fixed while that arm's joints move."""

    joint_slice = slice(0, 6) if arm == "left" else slice(6, 12)
    reference_q = np.asarray(qposes[view_index][joint_slice], dtype=float)
    reference_mask = component_masks[view_index][candidate_index]
    worst = 0.0
    for other_index, other_masks in enumerate(component_masks):
        if other_index == view_index:
            continue
        delta = np.linalg.norm(
            reference_q - np.asarray(qposes[other_index][joint_slice], dtype=float)
        )
        if delta < 0.12:
            continue
        best_overlap = max(
            (_iou(reference_mask, other) for other in other_masks),
            default=0.0,
        )
        worst = max(worst, best_overlap)
    return 5.0 * max(0.0, worst - 0.72)


def _fit_independent_arms(
    component_points: list[list[np.ndarray]],
    component_masks: list[list[np.ndarray]],
    component_members: list[list[frozenset[int]]],
    required_core: dict[str, list[int]],
    cad: list[
        tuple[np.ndarray, dict[str, list[np.ndarray]], dict[str, np.ndarray]]
    ],
    camera_pose: dict,
    qposes: list[list[float]],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Jointly assign unlabeled SAM components and fit each physical base.

    Initialization tests every sizeable first-view component.  EM refinement
    then enforces that the two arms cannot consume the same component in a
    view.  Identity is recovered from the arm-specific qpos motion, not from a
    hard-coded left/right pixel coordinate.
    """

    arms = ("left", "right")
    level_from_camera = gravity_level_transform(camera_pose)
    transforms: dict[str, np.ndarray] = {}
    for arm in arms:
        cad_points = [entry[1][arm] for entry in cad]
        cad_points = [np.vstack(parts) for parts in cad_points]
        hypotheses = []
        for view_index in range(min(3, len(component_points) - 1)):
            for candidate_index, observed in enumerate(component_points[view_index]):
                if (
                    required_core[arm][view_index]
                    not in component_members[view_index][candidate_index]
                ):
                    continue
                initial = _initial_transform(
                    camera_pose,
                    [observed],
                    [cad_points[view_index]],
                )
                score = 0.0
                for score_view, (candidate_view, view_cad) in enumerate(zip(
                    component_points[:-1], cad_points[:-1]
                )):
                    score += min(
                        _component_fit_cost(candidate, view_cad, initial)
                        + _motion_inconsistency(
                            arm,
                            score_view,
                            candidate_index,
                            component_masks,
                            qposes,
                        )
                        for candidate_index, candidate in enumerate(candidate_view)
                        if required_core[arm][score_view]
                        in component_members[score_view][candidate_index]
                    )
                hypotheses.append((score, initial))
        if not hypotheses:
            raise RuntimeError(f"{arm}: no SAM component hypothesis")
        transforms[arm] = min(hypotheses, key=lambda item: item[0])[1]

    assignments: dict[str, list[int]] = {arm: [] for arm in arms}
    for _ in range(5):
        assignments = {arm: [] for arm in arms}
        for view_index, candidates in enumerate(component_points):
            costs = {
                arm: [
                    (
                        _component_fit_cost(
                            candidate,
                            np.vstack(cad[view_index][1][arm]),
                            transforms[arm],
                        )
                        + _motion_inconsistency(
                            arm,
                            view_index,
                            candidate_index,
                            component_masks,
                            qposes,
                        )
                        if required_core[arm][view_index]
                        in component_members[view_index][candidate_index]
                        else float("inf")
                    )
                    for candidate_index, candidate in enumerate(candidates)
                ]
                for arm in arms
            }
            choices = [
                (costs["left"][left] + costs["right"][right], left, right)
                for left in range(len(candidates))
                for right in range(len(candidates))
                if component_members[view_index][left].isdisjoint(
                    component_members[view_index][right]
                )
            ]
            if not choices:
                raise RuntimeError(
                    f"view {view_index}: two distinct robot components required"
                )
            _, left, right = min(choices)
            assignments["left"].append(left)
            assignments["right"].append(right)
        for arm in arms:
            selected = [
                component_points[index][assignments[arm][index]]
                for index in range(len(component_points) - 1)
            ]
            arm_cad = [
                np.vstack(cad[index][1][arm])
                for index in range(len(component_points) - 1)
            ]
            transforms[arm] = fit_bounded_transform(
                selected,
                arm_cad,
                transforms[arm],
            )
    return transforms, assignments


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
    # ``rotation`` below maps the original camera-frame observations into the
    # robot frame.  Keep their center in that same source frame.  Applying the
    # gravity-level rotation to the center here and again in ``rotation`` would
    # level it twice, producing a metre-scale bad seed that can saturate the
    # robust nearest-neighbour residual.
    observed_center = np.median(np.vstack(observations), axis=0)
    cad_center = np.median(np.vstack(cad_points), axis=0)
    best = None
    for yaw in np.linspace(-np.pi, np.pi, 72, endpoint=False):
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


def fit_upright_transform(
    observations: list[np.ndarray],
    cad_points: list[np.ndarray],
    initial: np.ndarray,
    level_from_camera: np.ndarray,
    *,
    maximum_points_per_view: int = 2500,
    residual_clip_m: float = 0.08,
    maximum_evaluations: int = 120,
) -> np.ndarray:
    """Fit yaw and translation while preserving the measured gravity axis."""

    sampled = [
        points[:: max(1, len(points) // maximum_points_per_view)][
            :maximum_points_per_view
        ]
        for points in observations
    ]
    trees = [cKDTree(points) for points in cad_points]
    relative = initial[:3, :3] @ level_from_camera[:3, :3].T
    initial_yaw = Rotation.from_matrix(relative).as_euler("zyx")[0]

    def matrix(parameters):
        result = np.eye(4)
        result[:3, :3] = (
            Rotation.from_euler("z", parameters[0]).as_matrix()
            @ level_from_camera[:3, :3]
        )
        result[:3, 3] = parameters[1:]
        return result

    def residual(parameters):
        transform = matrix(parameters)
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
        np.r_[initial_yaw, initial[:3, 3]],
        bounds=(
            np.r_[initial_yaw - 0.35, initial[:3, 3] - 0.25],
            np.r_[initial_yaw + 0.35, initial[:3, 3] + 0.25],
        ),
        loss="soft_l1",
        f_scale=0.012,
        max_nfev=maximum_evaluations,
    )
    return matrix(result.x)


def fit_bounded_transform(
    observations: list[np.ndarray],
    cad_points: list[np.ndarray],
    initial: np.ndarray,
    *,
    maximum_points_per_view: int = 2500,
    residual_clip_m: float = 0.08,
    maximum_evaluations: int = 160,
) -> np.ndarray:
    """Refine a gravity/yaw seed without permitting unconstrained divergence."""

    sampled = [
        points[:: max(1, len(points) // maximum_points_per_view)][
            :maximum_points_per_view
        ]
        for points in observations
    ]
    trees = [cKDTree(points) for points in cad_points]

    def matrix(parameters):
        result = np.eye(4)
        result[:3, :3] = (
            Rotation.from_rotvec(parameters[:3]).as_matrix()
            @ initial[:3, :3]
        )
        result[:3, 3] = initial[:3, 3] + parameters[3:]
        return result

    def residual(parameters):
        transform = matrix(parameters)
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
        np.zeros(6),
        bounds=(
            np.array([-0.85, -0.85, -0.45, -0.30, -0.30, -0.30]),
            np.array([0.85, 0.85, 0.45, 0.30, 0.30, 0.30]),
        ),
        loss="soft_l1",
        f_scale=0.012,
        max_nfev=maximum_evaluations,
    )
    return matrix(result.x)


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


def _projected_mask_independent(
    geoms: dict[str, list[np.ndarray]],
    robot_from_camera: dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per_arm = {}
    for arm in ("left", "right"):
        _, projected = _projected_mask(
            {arm: geoms[arm], **{
                other: [] for other in ("left", "right") if other != arm
            }},
            robot_from_camera[arm],
            camera_matrix,
            shape_hw,
        )
        per_arm[arm] = projected[arm]
    return np.logical_or(per_arm["left"], per_arm["right"]), per_arm


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


def _metrics_independent(
    observed_points: dict[str, np.ndarray],
    observed_masks: dict[str, np.ndarray],
    cad_geoms: dict[str, list[np.ndarray]],
    transforms: dict[str, np.ndarray],
    camera_matrix: np.ndarray,
) -> dict:
    depth = []
    for arm in ("left", "right"):
        transformed = transform_points(observed_points[arm], transforms[arm])
        distance, _ = cKDTree(np.vstack(cad_geoms[arm])).query(
            transformed, workers=-1
        )
        depth.append(distance)
    distances = np.concatenate(depth)
    observed_union = np.logical_or(
        observed_masks["left"], observed_masks["right"]
    )
    projected, per_arm = _projected_mask_independent(
        cad_geoms,
        transforms,
        camera_matrix,
        observed_union.shape,
    )
    return {
        "depth_median_m": float(np.median(distances)),
        "depth_p90_m": float(np.quantile(distances, 0.90)),
        "mask_union_iou": _iou(observed_union, projected),
        "mask_per_arm_iou": {
            arm: _iou(observed_masks[arm], per_arm[arm])
            for arm in ("left", "right")
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
    if len(views) < args.minimum_views:
        raise ValueError(
            f"{args.minimum_views} or more stopped robot poses are required"
        )
    profile, catalog = load_profile(args.profile)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = output.with_suffix("")
    audit_dir.mkdir(parents=True, exist_ok=True)
    accepted_masks = _parse_masks(args.mask)
    unknown = sorted(set(accepted_masks) - {view["name"] for view in views})
    if unknown:
        raise ValueError(f"accepted masks reference unknown views: {unknown}")

    calibration_profile = dict(profile.get("robot_calibration", {}))
    robot_profile = dict(profile)
    robot_profile["objects"] = ["robot"]
    robot_record = catalog["robot"]
    if calibration_profile.get("sam_prompts"):
        robot_record = replace(
            robot_record,
            prompts=tuple(calibration_profile["sam_prompts"]),
        )
    robot_catalog = {"robot": robot_record}
    temporal = [
        _temporal_view(capture, view, minimum_confidence=args.minimum_confidence)
        for view in views
    ]
    qposes = [_qpos(view) for view in views]
    qpos_diversity = _qpos_diversity(
        qposes,
        minimum_joint_range_rad=args.minimum_joint_range_rad,
        minimum_moving_joints_per_arm=args.minimum_moving_joints_per_arm,
        minimum_holdout_distance_rad=args.minimum_holdout_distance_rad,
    )
    if not qpos_diversity["accepted"]:
        raise ValueError(
            "robot poses lack per-arm fit/holdout diversity: "
            + json.dumps(qpos_diversity, sort_keys=True)
        )
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

    if not calibration_profile.get("model"):
        raise ValueError(
            "robot_calibration.model is required; camera calibration must use "
            "the production robot model rather than a scene approximation"
        )
    calibration_model = Path(calibration_profile["model"]).resolve()
    physical_to_model = dict(
        calibration_profile.get(
            "physical_to_model_branch",
            {"left": "right_arm_", "right": "left_arm_"},
        )
    )
    if set(physical_to_model) != {"left", "right"}:
        raise ValueError(
            "robot_calibration.physical_to_model_branch must map left and right"
        )
    cad = [
        _cad_geometry(calibration_model, qpos, physical_to_model)
        for qpos in qposes
    ]

    component_masks: list[list[np.ndarray]] = []
    component_members: list[list[frozenset[int]]] = []
    component_points: list[list[np.ndarray]] = []
    parts_by_view: list[list[np.ndarray]] = []
    for item, mask in zip(temporal, cleaned_masks):
        parts = _large_components(mask)
        parts_by_view.append(parts)
        groups_for_view = _component_groups_from_parts(parts)
        masks_for_view = [value[0] for value in groups_for_view]
        points_for_view = [
            _component_points(
                item,
                component,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                minimum_confidence=args.minimum_confidence,
            )
            for component in masks_for_view
        ]
        retained = [
            (component, points, members)
            for component, points, (_, members) in zip(
                masks_for_view, points_for_view, groups_for_view
            )
            if len(points) >= 120
        ]
        if len(retained) < 2:
            raise RuntimeError(
                f"{item['name']}: fewer than two robot RGB-D components after cleanup"
            )
        component_masks.append([value[0] for value in retained])
        component_points.append([value[1] for value in retained])
        component_members.append([value[2] for value in retained])
    required_core = _tracked_core_by_arm(parts_by_view, qposes)
    fitted, assignments = _fit_independent_arms(
        component_points,
        component_masks,
        component_members,
        required_core,
        cad,
        temporal[0]["camera_pose"],
        qposes,
    )

    selected_masks = [
        {
            arm: component_masks[index][assignments[arm][index]]
            for arm in ("left", "right")
        }
        for index in range(len(views))
    ]
    selected_points = [
        {
            arm: component_points[index][assignments[arm][index]]
            for arm in ("left", "right")
        }
        for index in range(len(views))
    ]
    dynamic_points: list[dict[str, np.ndarray]] = [
        {} for _ in range(len(views))
    ]
    persistent_by_arm = {}
    tool_anchor_views = {}
    tool_anchor_diagnostics = {}
    for arm in ("left", "right"):
        frequency = np.mean(
            np.stack([mask_pair[arm] for mask_pair in selected_masks]),
            axis=0,
        )
        persistent_arm = (frequency >= 0.80).astype(np.uint8)
        persistent_arm = cv2.erode(
            persistent_arm, np.ones((3, 3), np.uint8), iterations=1
        ).astype(bool)
        persistent_by_arm[arm] = persistent_arm
        for index, item in enumerate(temporal):
            dynamic_mask = selected_masks[index][arm] & ~persistent_arm
            points = _component_points(
                item,
                dynamic_mask,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                minimum_confidence=args.minimum_confidence,
            )
            dynamic_points[index][arm] = (
                points if len(points) >= 120 else selected_points[index][arm]
            )
        arm_cad = [
            np.vstack(cad[index][1][arm]) for index in range(len(views) - 1)
        ]
        anchor_indices = []
        observed_anchors = []
        cad_anchors = []
        for index, item in enumerate(temporal):
            anchor = _blue_tool_anchor(item, selected_masks[index][arm])
            if anchor is not None:
                anchor_indices.append(index)
                observed_anchors.append(anchor)
                cad_anchors.append(cad[index][2][arm])
        tool_anchor_views[arm] = anchor_indices
        anchor_span = (
            max(
                np.linalg.norm(
                    cad_anchors[first][:3, 3] - cad_anchors[second][:3, 3]
                )
                for first in range(len(cad_anchors))
                for second in range(first)
            )
            if len(cad_anchors) >= 2
            else 0.0
        )
        anchor_accepted = False
        anchor_residual = None
        anchor_offset = None
        # Two samples can fit an apparently excellent but geometrically
        # ambiguous camera pose. Require three stopped poses before an anchor
        # may influence the transform; the full CAD fit still performs the
        # final refinement below.
        if len(observed_anchors) >= 3 and anchor_span >= 0.05:
            anchor_initial, anchor_offset, anchor_residual = _anchor_initial_transform(
                temporal[0]["camera_pose"],
                observed_anchors,
                cad_anchors,
            )
            anchor_accepted = bool(
                anchor_residual <= 0.035
                and np.linalg.norm(anchor_offset) <= 0.18
            )
        tool_anchor_diagnostics[arm] = {
            "view_indices": anchor_indices,
            "cad_span_m": float(anchor_span),
            "median_correspondence_residual_m": anchor_residual,
            "tool_point_offset_ee_m": (
                None if anchor_offset is None else anchor_offset.tolist()
            ),
            "accepted": anchor_accepted,
        }
        initial = (
            anchor_initial
            if anchor_accepted
            else _initial_transform(
                temporal[0]["camera_pose"],
                [dynamic_points[index][arm] for index in range(3)],
                arm_cad[:3],
            )
        )
        fitted[arm] = fit_bounded_transform(
            [dynamic_points[index][arm] for index in range(len(views) - 1)],
            arm_cad,
            initial,
        )
    metrics = [
        _metrics_independent(
            dynamic_points[index],
            selected_masks[index],
            cad[index][1],
            fitted,
            temporal[index]["camera_matrix"],
        )
        for index in range(len(views))
    ]
    projected_masks = [
        _projected_mask_independent(
            cad[index][1],
            fitted,
            temporal[index]["camera_matrix"],
            cleaned_masks[index].shape,
        )[0]
        for index in range(len(views))
    ]
    for item, mask_pair, projected in zip(
        temporal, selected_masks, projected_masks
    ):
        _mask_overlay(
            item["rgb_bgr"],
            np.logical_or(mask_pair["left"], mask_pair["right"]),
            projected,
            audit_dir / f"{item['name']}_overlay.png",
        )

    independent = {
        arm: [
            fit_bounded_transform(
                [dynamic_points[index][arm]],
                # Per-pose repeatability uses the moving-link observations;
                # fixed mounts and stands are not part of the Piper CAD.
                [np.vstack(cad[index][1][arm])],
                fitted[arm],
            )
            for index in range(len(views))
        ]
        for arm in ("left", "right")
    }
    repeatability_by_arm = {}
    for arm, values in independent.items():
        translations = np.asarray([item[:3, 3] for item in values])
        rotation_deltas = [
            Rotation.from_matrix(
                item[:3, :3] @ fitted[arm][:3, :3].T
            ).magnitude()
            for item in values
        ]
        repeatability_by_arm[arm] = {
            "translation_m": float(
                np.max(
                    np.linalg.norm(
                        translations - np.median(translations, axis=0), axis=1
                    )
                )
            ),
            "rotation_deg": float(np.degrees(max(rotation_deltas))),
        }
    repeatability_translation = max(
        value["translation_m"] for value in repeatability_by_arm.values()
    )
    repeatability_rotation = max(
        value["rotation_deg"] for value in repeatability_by_arm.values()
    )
    level_rotation = gravity_level_transform(
        temporal[0]["camera_pose"]
    )[:3, :3]
    base_tilt_by_arm = {}
    for arm, transform in fitted.items():
        relative = transform[:3, :3] @ level_rotation.T
        base_tilt_by_arm[arm] = float(
            np.degrees(
                np.arccos(np.clip(abs(relative[2, 2]), -1.0, 1.0))
            )
        )
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
        "base_tilt_max_deg": 12.0,
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
        "qpos_diversity": qpos_diversity["accepted"],
        "base_upright": all(
            value <= thresholds["base_tilt_max_deg"]
            for value in base_tilt_by_arm.values()
        ),
    }
    accepted = all(decisions.values())
    report = {
        "schema": SCHEMA,
        "schema_version": 2,
        "calibration_id": f"head-piper-cad-{manifest['session_id']}",
        "accepted": accepted,
        "accepted_at_s": time.time() if accepted else None,
        "record3d_udid": manifest.get("device", {}).get("udid"),
        # Backward-compatible authority frame: physical right Piper's nominal
        # ConeE model frame. New consumers should use the explicit per-arm map.
        "T_robot_camera": fitted["right"].tolist(),
        "T_physical_arm_camera": {
            arm: transform.tolist() for arm, transform in fitted.items()
        },
        "T_right_frame_left_model": (
            fitted["right"] @ np.linalg.inv(fitted["left"])
        ).tolist(),
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
            "repeatability_by_arm": repeatability_by_arm,
            "base_tilt_by_arm_deg": base_tilt_by_arm,
            "qpos_diversity": qpos_diversity,
        },
        "thresholds": thresholds,
        "decisions": decisions,
        "static_false_positive_cleanup": {
            "method": "eroded_all_view_robot_mask_intersection",
            "persistent_pixels_removed": int(np.count_nonzero(persistent)),
            "per_arm_persistent_pixels_removed": {
                arm: int(np.count_nonzero(mask))
                for arm, mask in persistent_by_arm.items()
            },
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
            "component_assignment": {
                view["name"]: {
                    arm: {
                        "hypothesis_index": int(assignments[arm][index]),
                        "source_components": sorted(
                            component_members[index][assignments[arm][index]]
                        ),
                    }
                    for arm in ("left", "right")
                }
                for index, view in enumerate(views)
            },
            "tracked_core_component": {
                view["name"]: {
                    arm: int(required_core[arm][index])
                    for arm in ("left", "right")
                }
                for index, view in enumerate(views)
            },
            "blue_tool_anchor_view_indices": tool_anchor_views,
            "blue_tool_anchor_diagnostics": tool_anchor_diagnostics,
            "audit_display": {
                "rotation": "none",
                "note": (
                    "RGB, depth, masks, and intrinsics remain in the captured "
                    "sensor coordinates; viewer rotation is presentation-only"
                ),
            },
            "calibration_robot_model": str(calibration_model),
            "physical_to_model_branch": physical_to_model,
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
    parser.add_argument("--minimum-views", type=int, default=5)
    parser.add_argument("--minimum-joint-range-rad", type=float, default=0.02)
    parser.add_argument("--minimum-moving-joints-per-arm", type=int, default=2)
    parser.add_argument("--minimum-holdout-distance-rad", type=float, default=0.03)
    args = parser.parse_args(argv)
    if (
        args.minimum_views < 4
        or args.minimum_joint_range_rad <= 0.0
        or args.minimum_moving_joints_per_arm < 1
        or args.minimum_moving_joints_per_arm > 6
        or args.minimum_holdout_distance_rad <= 0.0
    ):
        parser.error("invalid pose-diversity thresholds")
    report = build(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

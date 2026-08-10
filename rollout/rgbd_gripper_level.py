"""Measure the physical blue gripper against gravity from one RGB-D frame.

This module deliberately does not require a camera-to-robot transform.  The
Record3D pose supplies gravity in the OpenCV camera frame, while the physical
blue jaw supplies the tool direction in that same frame.  A separate local
joint-space planner uses production FK only to find a translation-null motion;
its sign and gain must be verified by a stopped RGB-D probe before correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from rollout.multiview_scene import depth_points_and_normals, record3d_pose_matrix


def _unit(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(result))
    if not np.all(np.isfinite(result)) or norm < 1e-9:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return result / norm


def record3d_gravity_up_camera(camera_pose: dict) -> np.ndarray:
    """Return gravity-up expressed in the raw OpenCV RGB/depth frame."""

    session_from_camera = record3d_pose_matrix(camera_pose)
    return _unit(
        session_from_camera[:3, :3].T @ np.array([0.0, 1.0, 0.0]),
        "Record3D gravity up",
    )


@dataclass(frozen=True)
class RGBDGripperLevelMeasurement:
    accepted: bool
    signed_long_axis_angle_deg: float
    absolute_long_axis_angle_deg: float
    long_axis_camera: tuple[float, float, float]
    gravity_up_camera: tuple[float, float, float]
    support_normal_camera: tuple[float, float, float]
    support_gravity_disagreement_deg: float
    blue_component_bbox_normalized_xywh: tuple[float, float, float, float]
    blue_component_area_fraction: float
    blue_axis_anisotropy: float
    blue_depth_points: int
    horizontal_support_points: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RGBDLevelConsensus:
    """Independent stopped bursts agreeing that the physical jaw is level."""

    accepted: bool
    sample_count: int
    median_angle_deg: float
    interburst_range_deg: float
    maximum_individual_mad_deg: float
    maximum_accepted_angle_deg: float
    maximum_allowed_interburst_range_deg: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def confirm_stopped_level_bursts(
    signed_angles_deg,
    burst_mads_deg,
    *,
    minimum_bursts: int = 3,
    maximum_accepted_angle_deg: float = 0.25,
    maximum_interburst_range_deg: float = 0.75,
    maximum_individual_mad_deg: float = 0.60,
) -> RGBDLevelConsensus:
    """Require a consensus across independent stopped RGB-D connections.

    A low MAD inside one burst does not catch a coherent depth/PCA branch
    change between Record3D connections.  The independent-burst range closes
    that gap while the median prevents one noisy connection from deciding the
    physical level state.
    """

    angles = np.asarray(signed_angles_deg, dtype=float).reshape(-1)
    mads = np.asarray(burst_mads_deg, dtype=float).reshape(-1)
    if angles.shape != mads.shape:
        raise ValueError("angle and MAD lists must have the same length")
    if len(angles) < int(minimum_bursts):
        raise ValueError(
            f"physical level confirmation needs at least {int(minimum_bursts)} bursts"
        )
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(mads)):
        raise ValueError("physical level confirmation values must be finite")
    if np.any(mads < 0.0):
        raise ValueError("burst MAD values must be non-negative")
    median = float(np.median(angles))
    interburst_range = float(np.max(angles) - np.min(angles))
    maximum_mad = float(np.max(mads))
    reasons = []
    if abs(median) > float(maximum_accepted_angle_deg):
        reasons.append("physical_blue_jaw_consensus_not_horizontal")
    if interburst_range > float(maximum_interburst_range_deg):
        reasons.append("independent_rgbd_bursts_disagree")
    if maximum_mad > float(maximum_individual_mad_deg):
        reasons.append("individual_rgbd_burst_unstable")
    return RGBDLevelConsensus(
        accepted=not reasons,
        sample_count=int(len(angles)),
        median_angle_deg=median,
        interburst_range_deg=interburst_range,
        maximum_individual_mad_deg=maximum_mad,
        maximum_accepted_angle_deg=float(maximum_accepted_angle_deg),
        maximum_allowed_interburst_range_deg=float(
            maximum_interburst_range_deg
        ),
        reasons=tuple(reasons),
    )


def _camera_matrix_for_depth(camera_matrix_rgb, rgb_shape, depth_shape):
    matrix = np.asarray(camera_matrix_rgb, dtype=float).reshape(3, 3).copy()
    scale_x = float(depth_shape[1]) / float(rgb_shape[1])
    scale_y = float(depth_shape[0]) / float(rgb_shape[0])
    matrix[0, :] *= scale_x
    matrix[1, :] *= scale_y
    matrix[2, :] = (0.0, 0.0, 1.0)
    return matrix


def _blue_component(rgb: np.ndarray, hue_range, minimum_area_fraction: float):
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    low_hue, high_hue = (int(value) for value in hue_range)
    mask = cv2.inRange(
        hsv,
        np.array([low_hue, 70, 45], dtype=np.uint8),
        np.array([high_hue, 255, 255], dtype=np.uint8),
    )
    scale = max(3, int(round(min(rgb.shape[:2]) * 0.004)))
    if scale % 2 == 0:
        scale += 1
    kernel = np.ones((scale, scale), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    pixels = float(rgb.shape[0] * rgb.shape[1])
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        area_fraction = float(area) / pixels
        if area_fraction < minimum_area_fraction:
            continue
        component_y, component_x = np.nonzero(labels == index)
        coordinates = np.c_[component_x, component_y].astype(float)
        singular = np.linalg.svd(
            coordinates - np.mean(coordinates, axis=0),
            compute_uv=False,
        )
        anisotropy = float(singular[0] / max(singular[1], 1e-9))
        score = float(area) * min(anisotropy, 8.0)
        candidates.append(
            (score, index, (x, y, width, height), area_fraction, anisotropy)
        )
    if not candidates:
        raise ValueError("no elongated blue gripper component was found")
    candidates.sort(reverse=True)
    score, index, bbox, area_fraction, anisotropy = candidates[0]
    if anisotropy < 1.7:
        raise ValueError("largest blue component is not sufficiently elongated")
    if len(candidates) > 1 and score < 1.20 * candidates[1][0]:
        raise ValueError("blue gripper component selection is ambiguous")
    return labels == index, bbox, area_fraction, anisotropy


def measure_blue_gripper_level(
    rgb,
    depth_m,
    camera_matrix_rgb,
    *,
    camera_pose: dict | None = None,
    gravity_up_camera: Sequence[float] | None = None,
    blue_hue_range=(82, 112),
    minimum_blue_area_fraction: float = 1.5e-4,
    minimum_blue_depth_points: int = 35,
    minimum_support_points: int = 100,
    maximum_support_gravity_disagreement_deg: float = 8.0,
    maximum_accepted_angle_deg: float = 0.25,
) -> RGBDGripperLevelMeasurement:
    """Measure the signed physical jaw angle and independently check support."""

    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must have shape HxWx3")
    if depth.ndim != 2:
        raise ValueError("depth must have shape HxW")
    if gravity_up_camera is None:
        if camera_pose is None:
            raise ValueError("camera pose or gravity-up vector is required")
        up = record3d_gravity_up_camera(camera_pose)
    else:
        up = _unit(gravity_up_camera, "gravity up")
    component, bbox, area_fraction, component_anisotropy = _blue_component(
        rgb, blue_hue_range, minimum_blue_area_fraction
    )
    matrix = _camera_matrix_for_depth(camera_matrix_rgb, rgb.shape, depth.shape)
    coverage = cv2.resize(
        component.astype(np.float32),
        (depth.shape[1], depth.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    valid = np.isfinite(depth) & (depth >= 0.20) & (depth <= 2.00)
    selected = valid & (coverage >= 0.20)
    rows, columns = np.indices(depth.shape)
    x = (columns - matrix[0, 2]) * depth / matrix[0, 0]
    y = (rows - matrix[1, 2]) * depth / matrix[1, 1]
    organized = np.dstack((x, y, depth))
    points = organized[selected]
    if len(points) < minimum_blue_depth_points:
        raise ValueError(
            f"blue gripper has only {len(points)} valid depth points"
        )
    median_depth = float(np.median(points[:, 2]))
    depth_deviation = np.abs(points[:, 2] - median_depth)
    mad = float(np.median(depth_deviation))
    depth_band = max(0.018, min(0.045, 4.5 * mad))
    points = points[depth_deviation <= depth_band]
    if len(points) < minimum_blue_depth_points:
        raise ValueError("blue gripper depth cluster is too sparse after filtering")
    center = np.mean(points, axis=0)
    _, singular, vectors = np.linalg.svd(points - center, full_matrices=False)
    axis = vectors[0]
    axis_anisotropy = float(singular[0] / max(singular[1], 1e-9))
    if axis_anisotropy < 2.0:
        raise ValueError("blue gripper 3D axis is not observable")

    # Fix PCA sign through its image derivative.  This is arbitrary but stable
    # across the sub-degree stopped probes used by the empirical controller.
    z = float(center[2])
    du = matrix[0, 0] * (axis[0] * z - center[0] * axis[2]) / (z * z)
    dv = matrix[1, 1] * (axis[1] * z - center[1] * axis[2]) / (z * z)
    dominant = du if abs(du) >= abs(dv) else dv
    if dominant < 0.0:
        axis = -axis
    signed_angle = math.degrees(
        math.asin(float(np.clip(axis @ up, -1.0, 1.0)))
    )

    support_points, support_normals = depth_points_and_normals(
        depth,
        matrix,
        stride=2,
        maximum_depth_step_m=0.025,
    )
    alignment = np.abs(support_normals @ up)
    horizontal = alignment >= math.cos(math.radians(12.0))
    horizontal_normals = support_normals[horizontal]
    if len(horizontal_normals):
        signs = np.sign(horizontal_normals @ up)
        support_normal = _unit(
            np.mean(horizontal_normals * signs[:, None], axis=0),
            "support normal",
        )
        support_disagreement = math.degrees(
            math.acos(float(np.clip(support_normal @ up, -1.0, 1.0)))
        )
    else:
        support_normal = np.full(3, np.nan)
        support_disagreement = float("inf")

    reasons = []
    if len(horizontal_normals) < minimum_support_points:
        reasons.append("horizontal_support_not_observed")
    if support_disagreement > maximum_support_gravity_disagreement_deg:
        reasons.append("support_normal_disagrees_with_gravity")
    if abs(signed_angle) > maximum_accepted_angle_deg:
        reasons.append("physical_blue_jaw_not_horizontal")
    x0, y0, width, height = bbox
    normalized_bbox = (
        float(x0) / rgb.shape[1],
        float(y0) / rgb.shape[0],
        float(width) / rgb.shape[1],
        float(height) / rgb.shape[0],
    )
    return RGBDGripperLevelMeasurement(
        accepted=not reasons,
        signed_long_axis_angle_deg=float(signed_angle),
        absolute_long_axis_angle_deg=abs(float(signed_angle)),
        long_axis_camera=tuple(float(value) for value in axis),
        gravity_up_camera=tuple(float(value) for value in up),
        support_normal_camera=tuple(float(value) for value in support_normal),
        support_gravity_disagreement_deg=float(support_disagreement),
        blue_component_bbox_normalized_xywh=normalized_bbox,
        blue_component_area_fraction=float(area_fraction),
        blue_axis_anisotropy=min(float(axis_anisotropy), float(component_anisotropy)),
        blue_depth_points=int(len(points)),
        horizontal_support_points=int(len(horizontal_normals)),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class JointLevelProbePlan:
    accepted: bool
    direction_q_rad: tuple[float, ...]
    probe_delta_q_rad: tuple[float, ...]
    predicted_approach_change_deg: float
    predicted_xyz_change_m: float
    predicted_baseline_change_deg: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def plan_translation_null_level_probe(
    pose_function,
    q_rad,
    *,
    approach_axis_ee=(1.0, 0.0, 0.0),
    baseline_axis_ee=(0.0, 0.0, 1.0),
    up_robot=(0.0, 0.0, 1.0),
    target_probe_angle_deg: float = 0.40,
    maximum_joint_delta_rad: float = math.radians(1.5),
    finite_difference_rad: float = 2e-4,
) -> JointLevelProbePlan:
    """Plan a joint probe that changes pitch while nulling XYZ and jaw roll."""

    from scipy.spatial.transform import Rotation

    q = np.asarray(q_rad, dtype=float).reshape(6)
    pose0 = np.asarray(pose_function(q), dtype=float).reshape(7)
    rotation0 = Rotation.from_quat(pose0[[1, 2, 3, 0]])
    matrix0 = rotation0.as_matrix()
    approach = matrix0 @ _unit(approach_axis_ee, "approach axis")
    baseline = matrix0 @ _unit(baseline_axis_ee, "baseline axis")
    up = _unit(up_robot, "robot up")
    position_jacobian = np.zeros((3, 6))
    angular_jacobian = np.zeros((3, 6))
    for joint in range(6):
        perturbed = q.copy()
        perturbed[joint] += finite_difference_rad
        pose = np.asarray(pose_function(perturbed), dtype=float).reshape(7)
        position_jacobian[:, joint] = (
            pose[4:] - pose0[4:]
        ) / finite_difference_rad
        rotation = Rotation.from_quat(pose[[1, 2, 3, 0]])
        angular_jacobian[:, joint] = (
            rotation * rotation0.inv()
        ).as_rotvec() / finite_difference_rad
    approach_row = np.cross(approach, up) @ angular_jacobian
    baseline_row = np.cross(baseline, up) @ angular_jacobian
    constraints = np.vstack((position_jacobian, baseline_row))
    _, singular, right = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.sum(singular > max(singular[0] * 1e-6, 1e-8)))
    nullspace = right[rank:].T
    reasons = []
    if nullspace.shape[1] == 0:
        reasons.append("no_translation_and_roll_nullspace")
        direction = np.zeros(6)
    else:
        projected = nullspace @ (nullspace.T @ approach_row)
        norm = float(np.linalg.norm(projected))
        if norm < 1e-5:
            reasons.append("pitch_unobservable_in_nullspace")
            direction = np.zeros(6)
        else:
            direction = projected / norm
    predicted_per_rad = float(approach_row @ direction)
    if abs(predicted_per_rad) < 1e-5:
        reasons.append("predicted_pitch_gain_too_small")
        scale = 0.0
    else:
        scale = math.radians(target_probe_angle_deg) / abs(predicted_per_rad)
        scale = min(scale, maximum_joint_delta_rad / max(np.max(np.abs(direction)), 1e-9))
    probe = direction * scale
    predicted_xyz = float(np.linalg.norm(position_jacobian @ probe))
    predicted_approach = math.degrees(float(approach_row @ probe))
    predicted_baseline = math.degrees(float(baseline_row @ probe))
    if predicted_xyz > 0.0008:
        reasons.append("predicted_probe_translation_too_large")
    if abs(predicted_baseline) > 0.10:
        reasons.append("predicted_probe_roll_too_large")
    if abs(predicted_approach) < 0.20:
        reasons.append("predicted_probe_pitch_too_small")
    return JointLevelProbePlan(
        accepted=not reasons,
        direction_q_rad=tuple(float(value) for value in direction),
        probe_delta_q_rad=tuple(float(value) for value in probe),
        predicted_approach_change_deg=float(predicted_approach),
        predicted_xyz_change_m=predicted_xyz,
        predicted_baseline_change_deg=float(predicted_baseline),
        reasons=tuple(reasons),
    )


def empirical_correction_from_probe(
    base_angle_deg: float,
    probe_angle_deg: float,
    probe_delta_q_rad,
    *,
    maximum_correction_angle_deg: float = 1.5,
    minimum_observed_probe_change_deg: float = 0.12,
    maximum_joint_delta_rad: float = math.radians(3.0),
) -> np.ndarray:
    """Scale a verified probe to cancel the observed physical jaw angle."""

    probe = np.asarray(probe_delta_q_rad, dtype=float).reshape(6)
    observed = float(probe_angle_deg) - float(base_angle_deg)
    if not math.isfinite(observed) or abs(observed) < minimum_observed_probe_change_deg:
        raise ValueError("RGB-D probe did not produce a measurable signed angle change")
    desired = float(np.clip(-base_angle_deg, -maximum_correction_angle_deg, maximum_correction_angle_deg))
    correction = probe * (desired / observed)
    maximum = float(np.max(np.abs(correction)))
    if maximum > maximum_joint_delta_rad:
        correction *= maximum_joint_delta_rad / maximum
    return correction


def robust_signed_angle(values, *, maximum_mad_deg: float = 0.60) -> dict:
    """Summarize a stopped depth burst and reject unstable measurements."""

    samples = np.asarray(values, dtype=float).reshape(-1)
    if len(samples) < 5 or not np.all(np.isfinite(samples)):
        raise ValueError("RGB-D angle burst needs at least five finite samples")
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    if mad > float(maximum_mad_deg):
        raise ValueError(
            f"RGB-D angle burst MAD {mad:.3f}deg exceeds "
            f"{float(maximum_mad_deg):.3f}deg"
        )
    return {
        "sample_count": int(len(samples)),
        "median_deg": median,
        "mad_deg": mad,
        "p10_deg": float(np.percentile(samples, 10.0)),
        "p90_deg": float(np.percentile(samples, 90.0)),
        "samples_deg": samples.tolist(),
    }

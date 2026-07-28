"""Geometry and online calibration for data-free real-time SAM servoing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from rollout.sam_segmentation import LidMaskGeometry, MaskCandidate, mask_geometry


DEPTH_SCALE = 1000.0  # metres -> millimetre-like feature units


@dataclass(frozen=True)
class SamSceneFeature:
    lid_candidate: MaskCandidate
    lid_geometry: LidMaskGeometry
    gripper_candidate: MaskCandidate
    lid_grasp_feature: np.ndarray
    gripper_feature: np.ndarray


@dataclass(frozen=True)
class ReachableFeatureModel:
    """Local image/depth response restricted to motions the arm actually made."""

    basis_xyz: np.ndarray
    feature_matrix: np.ndarray
    rank: int
    condition: float


def valid_depth_values(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(depth_m, dtype=float)[np.asarray(mask, dtype=bool)]
    return values[np.isfinite(values) & (values > 0.05) & (values < 5.0)]


def mask_depth_median(depth_m: np.ndarray, mask: np.ndarray) -> float:
    values = valid_depth_values(depth_m, mask)
    if values.size < 30:
        raise ValueError("SAM mask has insufficient valid depth")
    return float(np.median(values))


def choose_lid(
    candidates: Iterable[MaskCandidate],
    *,
    previous_center_px: np.ndarray | None = None,
) -> tuple[MaskCandidate, LidMaskGeometry]:
    ranked = []
    for candidate in candidates:
        geometry = mask_geometry(candidate.mask)
        if geometry is None or geometry.circularity < 0.40:
            continue
        height, width = candidate.mask.shape
        fraction = geometry.area_px / float(height * width)
        if not 0.001 <= fraction <= 0.08:
            continue
        jump = (
            0.0
            if previous_center_px is None
            else float(np.linalg.norm(geometry.center_px - previous_center_px))
        )
        if previous_center_px is not None and jump > 80.0:
            continue
        ranked.append((jump, -float(candidate.score), -geometry.circularity, candidate, geometry))
    if not ranked:
        raise ValueError("SAM did not produce a stable circular lid")
    *_, candidate, geometry = min(ranked, key=lambda item: item[:3])
    return candidate, geometry


def choose_right_gripper(
    candidates: Iterable[MaskCandidate],
    *,
    image_width: int,
    previous_center_px: np.ndarray | None = None,
) -> MaskCandidate:
    ranked = []
    for candidate in candidates:
        ys, xs = np.where(np.asarray(candidate.mask, dtype=bool))
        if xs.size < 50:
            continue
        center = np.array([np.median(xs), np.median(ys)], dtype=float)
        if center[0] < 0.32 * image_width:
            continue
        if previous_center_px is not None:
            jump = float(np.linalg.norm(center - previous_center_px))
            if jump > 100.0:
                continue
            rank = (jump, -float(candidate.score), -float(xs.size))
        else:
            # The right-arm claw is the right-most blue clamp in the initial
            # head image. Temporal proximity takes over after this first frame.
            rank = (-center[0], -float(candidate.score), -float(xs.size))
        ranked.append((rank, candidate))
    if not ranked:
        raise ValueError("SAM did not produce a stable right gripper")
    return min(ranked, key=lambda item: item[0])[1]


def gripper_mask_center(candidate: MaskCandidate) -> np.ndarray:
    ys, xs = np.where(np.asarray(candidate.mask, dtype=bool))
    if xs.size < 50:
        raise ValueError("gripper mask is too small")
    return np.array([np.median(xs), np.median(ys)], dtype=float)


def gripper_tip_px(candidate: MaskCandidate) -> np.ndarray:
    ys, xs = np.where(np.asarray(candidate.mask, dtype=bool))
    if xs.size < 50:
        raise ValueError("gripper mask is too small")
    points = np.column_stack((xs, ys)).astype(float)
    centered = points - np.mean(points, axis=0)
    _, singular_values, axes = np.linalg.svd(
        centered, full_matrices=False
    )
    if (
        singular_values.shape != (2,)
        or singular_values[1] <= 0.0
        or singular_values[0] / singular_values[1] < 1.25
    ):
        raise ValueError("gripper mask has no stable longitudinal axis")

    # A single extreme contour pixel is unstable between otherwise equivalent
    # SAM masks, especially at the angled jaw tip. Estimate the tool axis from
    # the whole instance and use the median of its terminal two-percent band.
    # The head-camera task convention is that the right-arm contact end is the
    # longitudinal endpoint with the greater image x coordinate.
    axis = axes[0]
    if axis[0] < 0.0:
        axis = -axis
    projection = centered @ axis
    terminal = projection >= np.percentile(projection, 98.0)
    if np.count_nonzero(terminal) < 10:
        raise ValueError("gripper tip band is too small")
    return np.median(points[terminal], axis=0)


def lid_left_grasp_px(candidate: MaskCandidate, geometry: LidMaskGeometry) -> np.ndarray:
    ys, xs = np.where(np.asarray(candidate.mask, dtype=bool))
    if xs.size < 50:
        raise ValueError("lid mask is too small")
    contour = np.asarray(geometry.contour, dtype=np.float32).reshape(-1, 1, 2)
    if len(contour) >= 5:
        # A tilted circular lid appears as an ellipse.  Fitting the whole rim is
        # substantially less jittery than taking the left-most mask percentile.
        (cx, cy), (diameter_a, diameter_b), angle_deg = cv2.fitEllipse(contour)
        angle = np.deg2rad(angle_deg)
        phase = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        a = 0.5 * diameter_a
        b = 0.5 * diameter_b
        ellipse_x = (
            cx
            + a * np.cos(phase) * np.cos(angle)
            - b * np.sin(phase) * np.sin(angle)
        )
        ellipse_y = (
            cy
            + a * np.cos(phase) * np.sin(angle)
            + b * np.sin(phase) * np.cos(angle)
        )
        left = int(np.argmin(ellipse_x))
        return np.array([ellipse_x[left], ellipse_y[left]], dtype=float)
    edge = float(np.percentile(xs, 1))
    band_y = ys[xs <= edge + 3.0]
    y = float(np.median(band_y)) if band_y.size else float(geometry.center_px[1])
    return np.array([edge, y], dtype=float)


def scene_feature(
    *,
    lid_candidates: Iterable[MaskCandidate],
    gripper_candidates: Iterable[MaskCandidate],
    depth_m: np.ndarray,
    previous_lid_center_px: np.ndarray | None = None,
    previous_gripper_center_px: np.ndarray | None = None,
    clearance_m: float = 0.040,
    lid_support_depth_m: float | None = None,
    selected_lid: tuple[MaskCandidate, LidMaskGeometry] | None = None,
) -> SamSceneFeature:
    if selected_lid is None:
        lid_candidate, lid_geometry = choose_lid(
            lid_candidates, previous_center_px=previous_lid_center_px
        )
    else:
        lid_candidate, lid_geometry = selected_lid
    gripper_candidate = choose_right_gripper(
        gripper_candidates,
        image_width=depth_m.shape[1],
        previous_center_px=previous_gripper_center_px,
    )
    lid_px = lid_left_grasp_px(lid_candidate, lid_geometry)
    gripper_px = gripper_tip_px(gripper_candidate)
    lid_depth = (
        mask_depth_median(depth_m, lid_candidate.mask)
        if lid_support_depth_m is None
        else float(lid_support_depth_m)
    )
    gripper_depth = mask_depth_median(depth_m, gripper_candidate.mask)
    lid_feature = np.array(
        [lid_px[0], lid_px[1], DEPTH_SCALE * (lid_depth - clearance_m)], dtype=float
    )
    gripper_feature = np.array(
        [gripper_px[0], gripper_px[1], DEPTH_SCALE * gripper_depth], dtype=float
    )
    return SamSceneFeature(
        lid_candidate=lid_candidate,
        lid_geometry=lid_geometry,
        gripper_candidate=gripper_candidate,
        lid_grasp_feature=lid_feature,
        gripper_feature=gripper_feature,
    )


def estimate_feature_jacobian(robot_deltas_xyz, feature_deltas) -> np.ndarray:
    robot = np.asarray(robot_deltas_xyz, dtype=float)
    feature = np.asarray(feature_deltas, dtype=float)
    if robot.shape != (3, 3) or feature.shape != (3, 3):
        raise ValueError("three independent 3D probes are required")
    # Rows are probe observations; feature_delta = robot_delta @ J.T.
    if np.linalg.matrix_rank(robot) < 3:
        raise ValueError("robot probes are not independent")
    jacobian = np.linalg.lstsq(robot, feature, rcond=None)[0].T
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("non-finite feature Jacobian")
    condition = float(np.linalg.cond(jacobian))
    if condition > 500.0:
        raise ValueError(f"ill-conditioned feature Jacobian: {condition:.1f}")
    return jacobian


def estimate_reachable_feature_model(
    robot_deltas_xyz,
    feature_deltas,
    *,
    rcond: float = 0.15,
) -> ReachableFeatureModel:
    """Fit a local feature model without requiring three reachable XYZ axes.

    Each row is one measured motion.  Near a Cartesian singularity, requested
    Y and Z probes can collapse onto the same physical direction.  The SVD
    basis explicitly discards that unobserved direction instead of amplifying
    it through an ill-conditioned inverse.
    """

    robot = np.asarray(robot_deltas_xyz, dtype=float)
    feature = np.asarray(feature_deltas, dtype=float)
    if (
        robot.ndim != 2
        or feature.ndim != 2
        or robot.shape != feature.shape
        or robot.shape[1] != 3
        or robot.shape[0] < 2
    ):
        raise ValueError("at least two paired 3D motion observations are required")
    if not np.all(np.isfinite(robot)) or not np.all(np.isfinite(feature)):
        raise ValueError("motion observations must be finite")

    # Columns are observations: F = J B.
    motion_matrix = robot.T
    feature_delta_matrix = feature.T
    left, singular_values, _ = np.linalg.svd(motion_matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-6:
        raise ValueError("robot probes produced no measurable motion")
    rank = int(np.count_nonzero(singular_values > rcond * singular_values[0]))
    if rank < 2:
        raise ValueError("robot probes span fewer than two reachable directions")

    basis = left[:, :rank]
    jacobian = feature_delta_matrix @ np.linalg.pinv(
        motion_matrix, rcond=rcond
    )
    feature_matrix = jacobian @ basis
    weighted_singular = np.linalg.svd(feature_matrix, compute_uv=False)
    positive = weighted_singular[weighted_singular > 1e-8]
    condition = (
        float(positive[0] / positive[-1])
        if positive.size >= 2
        else float("inf")
    )
    return ReachableFeatureModel(
        basis_xyz=basis,
        feature_matrix=feature_matrix,
        rank=rank,
        condition=condition,
    )


def bounded_reachable_servo_step(
    model: ReachableFeatureModel,
    feature_error: np.ndarray,
    *,
    tolerances: np.ndarray,
    max_norm_m: float = 0.012,
    max_axis_m: float = 0.008,
) -> np.ndarray:
    """Solve a tolerance-weighted step within the measured motion subspace."""

    error = np.asarray(feature_error, dtype=float).reshape(3)
    tolerance = np.asarray(tolerances, dtype=float).reshape(3)
    if np.any(tolerance <= 0):
        raise ValueError("feature tolerances must be positive")
    weighted_matrix = model.feature_matrix / tolerance[:, None]
    weighted_error = error / tolerance
    coefficients = np.linalg.lstsq(
        weighted_matrix, weighted_error, rcond=0.10
    )[0]
    step = model.basis_xyz @ coefficients
    peak = float(np.max(np.abs(step)))
    if peak > max_axis_m:
        step *= float(max_axis_m) / peak
    norm = float(np.linalg.norm(step))
    if norm > max_norm_m:
        step *= float(max_norm_m) / norm
    if not np.all(np.isfinite(step)):
        raise ValueError("reachable servo produced a non-finite step")
    return step


def bounded_servo_step(
    jacobian: np.ndarray,
    feature_error: np.ndarray,
    *,
    max_norm_m: float = 0.012,
    max_axis_m: float = 0.008,
    damping: float = 1e-3,
) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=float).reshape(3, 3)
    error = np.asarray(feature_error, dtype=float).reshape(3)
    scale = max(float(np.linalg.norm(jacobian, ord=2)), 1.0)
    lhs = jacobian.T @ jacobian + damping * scale * scale * np.eye(3)
    step = np.linalg.solve(lhs, jacobian.T @ error)
    step = np.clip(step, -float(max_axis_m), float(max_axis_m))
    norm = float(np.linalg.norm(step))
    if norm > max_norm_m:
        step *= float(max_norm_m) / norm
    return step


def render_scene(image_bgr: np.ndarray, feature: SamSceneFeature, label: str) -> np.ndarray:
    out = image_bgr.copy()
    for candidate, color in (
        (feature.lid_candidate, (0, 255, 0)),
        (feature.gripper_candidate, (0, 255, 255)),
    ):
        mask = np.asarray(candidate.mask, dtype=bool)
        tint = np.zeros_like(out)
        tint[:] = color
        out[mask] = cv2.addWeighted(out[mask], 0.55, tint[mask], 0.45, 0)
    lid_px = tuple(np.rint(feature.lid_grasp_feature[:2]).astype(int))
    grip_px = tuple(np.rint(feature.gripper_feature[:2]).astype(int))
    cv2.drawMarker(out, lid_px, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
    cv2.drawMarker(out, grip_px, (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.line(out, grip_px, lid_px, (255, 255, 255), 2)
    cv2.putText(
        out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3
    )
    cv2.putText(
        out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1
    )
    return out

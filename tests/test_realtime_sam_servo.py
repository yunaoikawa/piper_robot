#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.realtime_sam_servo import (
    bounded_reachable_servo_step,
    bounded_servo_step,
    estimate_feature_jacobian,
    estimate_reachable_feature_model,
    gripper_tip_px,
    scene_feature,
)
from rollout.sam_segmentation import MaskCandidate


def circle_mask(center, radius, shape=(240, 320)):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2


lid_mask = circle_mask((220, 150), 28)
gripper_mask = np.zeros((240, 320), bool)
gripper_mask[130:170, 120:190] = True
left_gripper = np.zeros((240, 320), bool)
left_gripper[40:80, 20:90] = True

lid = MaskCandidate(lid_mask, np.array([192, 122, 248, 178]), 0.95)
right = MaskCandidate(gripper_mask, np.array([120, 130, 190, 170]), 0.85)
left = MaskCandidate(left_gripper, np.array([20, 40, 90, 80]), 0.99)
depth = np.ones((240, 320), float)
depth[lid_mask] = 1.10
depth[gripper_mask] = 0.95

feature = scene_feature(
    lid_candidates=[lid],
    gripper_candidates=[left, right],
    depth_m=depth,
    clearance_m=0.04,
)
assert np.allclose(feature.lid_grasp_feature[2], 1060)
assert np.allclose(feature.gripper_feature[2], 950)
assert gripper_tip_px(right)[0] >= 188
assert feature.lid_grasp_feature[0] < 200

true_jacobian = np.array(
    [[500, 20, -80], [-30, 420, 100], [100, -50, 900]], dtype=float
)
robot_deltas = np.eye(3) * 0.005
feature_deltas = robot_deltas @ true_jacobian.T
estimated = estimate_feature_jacobian(robot_deltas, feature_deltas)
assert np.allclose(estimated, true_jacobian)

error = np.array([40, -30, 50], dtype=float)
step = bounded_servo_step(estimated, error)
assert np.linalg.norm(step) <= 0.012001
assert np.max(np.abs(step)) <= 0.008001

# A near-singular arm can produce only two independent Cartesian directions.
# The controller must stay inside those measured directions, not invent the
# missing third one.
singular_robot = np.array(
    [
        [0.00445, 0.00022, 0.00025],
        [-0.00004, 0.00285, -0.00285],
        [0.00005, -0.00233, 0.00233],
    ]
)
singular_feature = np.array(
    [[2.0, -2.0, 0.0], [-4.0, -4.0, -10.25], [1.0, 1.0, 12.7]]
)
reachable = estimate_reachable_feature_model(
    singular_robot, singular_feature
)
assert reachable.rank == 2
reachable_step = bounded_reachable_servo_step(
    reachable,
    np.array([-8.0, 105.0, 78.0]),
    tolerances=np.array([6.0, 6.0, 8.0]),
)
projected = reachable.basis_xyz @ (
    reachable.basis_xyz.T @ reachable_step
)
assert np.allclose(reachable_step, projected)
assert 0.001 < np.linalg.norm(reachable_step) <= 0.012001
assert np.max(np.abs(reachable_step)) <= 0.008001

print("real-time SAM servo checks passed")

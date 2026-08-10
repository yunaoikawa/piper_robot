import math

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rollout.rgbd_gripper_level import (
    confirm_stopped_level_bursts,
    empirical_correction_from_probe,
    measure_blue_gripper_level,
    plan_translation_null_level_probe,
    robust_signed_angle,
)


def _synthetic_blue_bar(angle_deg=0.0):
    height, width = 240, 320
    rgb = np.full((height, width, 3), 35, dtype=np.uint8)
    center = (210, 90)
    length = 105
    delta_x = math.cos(math.radians(angle_deg)) * length / 2
    delta_y = -math.sin(math.radians(angle_deg)) * length / 2
    first = (round(center[0] - delta_x), round(center[1] - delta_y))
    second = (round(center[0] + delta_x), round(center[1] + delta_y))
    # RGB cyan; OpenCV drawing is done in RGB array deliberately.
    cv2.line(rgb, first, second, (20, 160, 230), 22, cv2.LINE_AA)
    depth = np.ones((120, 160), dtype=np.float32)
    # Give only the blue bar a depth slope, so its 3D long axis tilts out of
    # the otherwise fronto-parallel support plane.
    blue = cv2.inRange(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
        np.array([82, 70, 45], dtype=np.uint8),
        np.array([112, 255, 255], dtype=np.uint8),
    )
    blue_depth = cv2.resize(
        blue, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA
    ) > 50
    columns = np.indices(depth.shape)[1]
    depth[blue_depth] += (
        np.tan(np.deg2rad(angle_deg)) * (columns[blue_depth] - center[0] / 2) / 130.0
    )
    matrix = np.array([[260.0, 0.0, 160.0], [0.0, 260.0, 120.0], [0.0, 0.0, 1.0]])
    return rgb, depth, matrix


def test_rgbd_blue_axis_detects_level_and_tilt():
    rgb, depth, matrix = _synthetic_blue_bar(0.0)
    level = measure_blue_gripper_level(
        rgb,
        depth,
        matrix,
        gravity_up_camera=(0.0, 0.0, 1.0),
        minimum_support_points=10,
        maximum_accepted_angle_deg=1.0,
    )
    assert level.accepted
    assert level.absolute_long_axis_angle_deg < 0.5
    rgb, depth, matrix = _synthetic_blue_bar(5.0)
    tilted = measure_blue_gripper_level(
        rgb,
        depth,
        matrix,
        gravity_up_camera=(0.0, 0.0, 1.0),
        minimum_support_points=10,
        maximum_accepted_angle_deg=1.0,
    )
    assert not tilted.accepted
    assert 3.0 < tilted.absolute_long_axis_angle_deg < 7.0


def test_translation_null_probe_and_empirical_scaling():
    def pose(q):
        rotation = Rotation.from_euler("xyz", [q[3], q[4], q[5]]).as_quat()
        return np.r_[rotation[[3, 0, 1, 2]], q[:3]]

    plan = plan_translation_null_level_probe(pose, np.zeros(6))
    assert plan.accepted
    assert plan.predicted_xyz_change_m < 1e-8
    correction = empirical_correction_from_probe(
        0.8, 0.4, plan.probe_delta_q_rad
    )
    assert np.allclose(correction, 2.0 * np.asarray(plan.probe_delta_q_rad))


def test_stopped_angle_burst_uses_median_and_gates_noise():
    report = robust_signed_angle([4.9, 5.0, 5.1, 5.0, 8.0])
    assert report["median_deg"] == 5.0
    assert report["mad_deg"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="MAD"):
        robust_signed_angle([-2.0, -1.0, 0.0, 1.0, 2.0], maximum_mad_deg=0.5)


def test_independent_bursts_confirm_level_by_consensus():
    consensus = confirm_stopped_level_bursts(
        [-0.1728, 0.2275, 0.4478],
        [0.3942, 0.4341, 0.2340],
    )
    assert consensus.accepted
    assert consensus.median_angle_deg == pytest.approx(0.2275)
    assert consensus.interburst_range_deg == pytest.approx(0.6206)


def test_independent_bursts_reject_a_coherent_depth_branch_jump():
    consensus = confirm_stopped_level_bursts(
        [-0.1728, 0.2275, -4.4068],
        [0.3942, 0.4341, 0.1979],
    )
    assert not consensus.accepted
    assert "independent_rgbd_bursts_disagree" in consensus.reasons


def test_independent_bursts_reject_stable_but_tilted_result():
    consensus = confirm_stopped_level_bursts(
        [1.0, 1.1, 1.2],
        [0.2, 0.2, 0.2],
    )
    assert not consensus.accepted
    assert "physical_blue_jaw_consensus_not_horizontal" in consensus.reasons

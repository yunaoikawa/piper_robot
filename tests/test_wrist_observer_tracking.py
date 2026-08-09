import cv2
import numpy as np
import pytest

from rollout.wrist_observer_tracking import (
    assess_command_direction,
    blue_components,
    require_joint_limits,
    target_blue_components,
)


def test_dark_shadow_is_not_a_blue_target():
    image = np.zeros((120, 180, 3), np.uint8)
    cv2.rectangle(image, (20, 20), (100, 60), (12, 12, 12), -1)
    assert blue_components(image) == ()


def test_self_mask_removes_observer_jaw_but_keeps_remote_blue():
    image = np.zeros((120, 180, 3), np.uint8)
    cv2.rectangle(image, (10, 70), (80, 115), (255, 150, 0), -1)
    cv2.rectangle(image, (120, 15), (170, 45), (255, 150, 0), -1)
    self_mask = np.zeros(image.shape[:2], bool)
    self_mask[65:, :90] = True
    targets = target_blue_components(image, self_mask=self_mask)
    assert len(targets) == 1
    assert targets[0].centroid_xy[0] > 120


def test_reverse_motion_is_rejected():
    start = np.zeros(6)
    target = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    measured = np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0])
    result = assess_command_direction(start, target, measured)
    assert not result.accepted
    assert result.reverse_joint_indices == (4,)


def test_forward_motion_is_accepted_before_endpoint():
    start = np.zeros(6)
    target = np.array([0.0, 0.0, 0.0, 0.0, 0.20, 0.0])
    measured = np.array([0.0, 0.0, 0.0, 0.0, 0.08, 0.0])
    result = assess_command_direction(
        start, target, measured, maximum_tracking_error_rad=0.15
    )
    assert result.accepted
    assert result.progress_fraction == pytest.approx(0.4)


def test_joint_limit_gate_is_fail_closed():
    require_joint_limits(np.zeros(6), -np.ones(6), np.ones(6))
    with pytest.raises(ValueError, match=r"indices \[4\]"):
        require_joint_limits(
            np.array([0.0, 0.0, 0.0, 0.0, 1.1, 0.0]),
            -np.ones(6),
            np.ones(6),
        )

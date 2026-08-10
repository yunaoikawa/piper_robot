import cv2
import numpy as np
import pytest

from rollout.wrist_observer_tracking import (
    JointDirectionMonitor,
    assess_command_direction,
    blue_components,
    compare_side_view_shape,
    describe_blue_component,
    fit_local_image_jacobian,
    image_servo_step,
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


def test_stream_direction_uses_current_command_and_debounces_compliance():
    monitor = JointDirectionMonitor(
        np.zeros(6),
        minimum_command_excursion_rad=0.07,
        reverse_excursion_rad=0.02,
        consecutive_reverse_samples=3,
    )
    for _ in range(5):
        state = monitor.observe(
            np.array([0, 0, 0, 0, 0.03, 0]),
            np.array([0, 0, 0, 0, -0.03, 0]),
        )
        assert state.accepted
    state = monitor.observe(
        np.array([0, 0, 0, 0, 0.10, 0]),
        np.array([0, 0, 0, 0, -0.03, 0]),
    )
    assert state.accepted
    state = monitor.observe(
        np.array([0, 0, 0, 0, 0.11, 0]),
        np.array([0, 0, 0, 0, 0.02, 0]),
    )
    assert state.accepted
    assert state.reverse_counts[4] == 0


def test_stream_direction_rejects_sustained_material_reverse_motion():
    monitor = JointDirectionMonitor(
        np.zeros(6), consecutive_reverse_samples=3
    )
    command = np.array([0, 0, 0, 0, 0.12, 0])
    measured = np.array([0, 0, 0, 0, -0.04, 0])
    assert monitor.observe(command, measured).accepted
    assert monitor.observe(command, measured).accepted
    result = monitor.observe(command, measured)
    assert not result.accepted
    assert result.reverse_joint_indices == (4,)


def test_measured_image_jacobian_learns_camera_signs_and_bounded_step():
    # This camera happens to map robot -Y to image-right and +Z to image-up.
    true_jacobian = np.array([[-1500.0, 80.0], [120.0, -1100.0]])
    probes = np.array(
        [[0.04, 0.0], [-0.04, 0.0], [0.0, 0.03], [0.0, -0.03]]
    )
    calibration = fit_local_image_jacobian(
        probes,
        probes @ true_jacobian.T,
        motion_axes=("world_y_m", "world_z_m"),
    )
    assert calibration.matrix_px_per_unit == pytest.approx(true_jacobian)
    step = image_servo_step(
        np.array([300.0, -220.0]),
        calibration,
        maximum_abs_motion=np.array([0.05, 0.05]),
    )
    assert step.motion_delta[0] < 0.0
    assert step.motion_delta[1] > 0.0
    assert np.max(np.abs(step.motion_delta)) <= 0.05
    assert np.linalg.norm(step.residual_pixel_error) < np.linalg.norm(
        [300.0, -220.0]
    )


def test_image_jacobian_rejects_unexcited_probe_axis():
    with pytest.raises(ValueError, match="independently excite"):
        fit_local_image_jacobian(
            [[0.01, 0.0], [0.02, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
            motion_axes=("world_y_m", "world_z_m"),
        )


def test_side_view_shape_is_scale_independent_and_rejects_rotation():
    reference_image = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(reference_image, (90, 90), (230, 120), (255, 150, 0), -1)
    reference_component = blue_components(reference_image)[0]
    reference = describe_blue_component(reference_image, reference_component)

    smaller = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(smaller, (45, 45), (115, 60), (255, 150, 0), -1)
    current = describe_blue_component(smaller, blue_components(smaller)[0])
    assert compare_side_view_shape(current, reference).accepted

    rotated = np.zeros_like(reference_image)
    box = cv2.boxPoints(((160, 105), (140, 30), 15)).astype(np.int32)
    cv2.fillConvexPoly(rotated, box, (255, 150, 0))
    rotated_shape = describe_blue_component(rotated, blue_components(rotated)[0])
    result = compare_side_view_shape(rotated_shape, reference)
    assert not result.accepted
    assert "jaw_side_view_axis_changed" in result.reasons

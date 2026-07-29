import numpy as np

from rollout.descent_probe import assess_descent_probe


def test_normal_probe_progress_is_not_contact():
    result = assess_descent_probe(
        requested_distance_m=0.002,
        measured_delta_xyz_m=[0, 0, -0.0019],
        descent_direction_xyz=[0, 0, -1],
        torque_before_nm=np.zeros(6),
        torque_after_nm=np.full(6, 0.2),
        support_clearance_m=0.006,
        maximum_support_clearance_m=0.002,
        minimum_progress_ratio_at_contact=0.3,
        minimum_torque_change_nm=0.1,
    )
    assert not result.early_contact


def test_stall_with_torque_before_support_is_early_contact():
    result = assess_descent_probe(
        requested_distance_m=0.002,
        measured_delta_xyz_m=[0, 0, -0.0002],
        descent_direction_xyz=[0, 0, -1],
        torque_before_nm=np.zeros(6),
        torque_after_nm=np.full(6, 0.2),
        support_clearance_m=0.006,
        maximum_support_clearance_m=0.002,
        minimum_progress_ratio_at_contact=0.3,
        minimum_torque_change_nm=0.1,
    )
    assert result.early_contact


def test_stall_at_support_is_not_mislabelled_as_target_contact():
    result = assess_descent_probe(
        requested_distance_m=0.002,
        measured_delta_xyz_m=[0, 0, 0],
        descent_direction_xyz=[0, 0, -1],
        torque_before_nm=np.zeros(6),
        torque_after_nm=np.full(6, 0.2),
        support_clearance_m=0.001,
        maximum_support_clearance_m=0.002,
        minimum_progress_ratio_at_contact=0.3,
        minimum_torque_change_nm=0.1,
    )
    assert not result.early_contact


def test_zero_length_probe_direction_is_rejected():
    try:
        assess_descent_probe(
            requested_distance_m=0.002,
            measured_delta_xyz_m=[0, 0, 0],
            descent_direction_xyz=[0, 0, 0],
            torque_before_nm=np.zeros(6),
            torque_after_nm=np.zeros(6),
            support_clearance_m=0.01,
            maximum_support_clearance_m=0.002,
            minimum_progress_ratio_at_contact=0.3,
            minimum_torque_change_nm=0.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero-length descent direction was accepted")

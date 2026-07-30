import numpy as np

from rollout.recorded_trajectory_replay import (
    _minimum_jerk,
    _segment_duration,
)


def test_minimum_jerk_has_exact_endpoints_and_zero_endpoint_velocity():
    assert _minimum_jerk(0.0) == 0.0
    assert _minimum_jerk(1.0) == 1.0
    epsilon = 1e-5
    assert _minimum_jerk(epsilon) / epsilon < 1e-6
    assert (1.0 - _minimum_jerk(1.0 - epsilon)) / epsilon < 1e-6


def test_segment_duration_enforces_quintic_peak_joint_speed():
    start = np.zeros(6)
    finish = np.array([0.7, -0.2, 0.1, 0.0, 0.0, 0.0])
    duration = _segment_duration(start, finish, 0.35, 0.5)
    assert duration == 1.875 * 0.7 / 0.35


def test_offline_replay_source_does_not_import_hardware_control():
    source = __import__("pathlib").Path(
        "src/run_pasteur_offline_replay.py"
    ).read_text()
    forbidden = ("PiperClient", "robot_rpc", "teleop", "send_joint")
    assert not any(token in source for token in forbidden)

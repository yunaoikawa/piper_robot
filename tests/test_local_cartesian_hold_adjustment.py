import numpy as np

from src.execute_audited_scene_path_prefix import _teleop_samples
from src.plan_local_cartesian_hold_adjustment import _unit_horizontal


def test_retreat_direction_discards_vertical_component():
    direction = _unit_horizontal(np.asarray([3.0, 4.0, 9.0]))

    np.testing.assert_allclose(direction, [0.6, 0.8, 0.0])
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)


def test_audited_joint_branch_is_resampled_at_teleop_rate():
    q = np.zeros((3, 6), dtype=float)
    q[:, 0] = [0.0, 0.01, 0.02]
    poses = np.zeros((3, 7), dtype=float)
    poses[:, 0] = 1.0
    poses[:, 4] = [0.0, 0.001, 0.002]

    samples = _teleop_samples(
        q, poses, speed_m_s=0.005, gripper_open_ratio=1.0
    )

    assert len(samples) == 13
    assert samples[0].t_s > 0.0
    assert all(first.t_s < second.t_s for first, second in zip(samples, samples[1:]))
    np.testing.assert_allclose(samples[0].right_q_physical_rad, q[0])
    np.testing.assert_allclose(samples[-1].right_q_physical_rad, q[-1])

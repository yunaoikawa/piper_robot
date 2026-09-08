import numpy as np

from rollout.incubator_door_close import (
    nearest_pose_index,
    register_close_trajectory,
    reverse_opening_from_live_pose,
)


def _sample(frame, x, yaw_w=1.0):
    return {
        "frame": frame,
        "t_s": frame / 30.0,
        "pose_wxyz_xyz": [yaw_w, 0.0, 0.0, 0.0, x, 0.0, 0.0],
        "gripper": 0.3,
    }


def test_reverse_opening_is_live_anchored_and_open_jaw():
    opening = [_sample(10, 0.0), _sample(11, 0.1), _sample(12, 0.2)]
    live = [1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0]
    nearest = nearest_pose_index(live, opening)
    assert nearest["index"] == 2
    closed = reverse_opening_from_live_pose(
        live, opening, nearest_index=nearest["index"]
    )
    assert [item["frame"] for item in closed] == [12, 11, 10]
    np.testing.assert_allclose(closed[0]["pose_wxyz_xyz"], live)
    np.testing.assert_allclose(closed[-1]["pose_wxyz_xyz"][4:], [0.05, 0, 0])
    assert all(item["gripper"] == 1.0 for item in closed)
    assert all(b["t_s"] > a["t_s"] for a, b in zip(closed, closed[1:]))


def test_register_close_trajectory_applies_scene_translation():
    trajectory = [_sample(0, 0.2)]
    registered = register_close_trajectory(
        trajectory, [1.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3]
    )
    np.testing.assert_allclose(
        registered[0]["pose_wxyz_xyz"][4:], [0.3, -0.2, 0.3]
    )
    assert registered[0]["gripper"] == 1.0

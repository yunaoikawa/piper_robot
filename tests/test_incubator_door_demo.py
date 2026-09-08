import h5py
import mink
import numpy as np

from rollout.incubator_door_demo import (
    SCHEMA,
    compile_demonstrations,
    retarget_relative_pose,
    retarget_relative_trajectory,
)
from src.run_incubator_door_demo import world_yaw_pose


def _recording(path, x_offset):
    count = 12
    position = np.zeros((count, 3), dtype=float)
    position[:, 0] = x_offset
    position[4:, 0] += np.linspace(0.0, -0.08, count - 4)
    position[:, 2] = 0.9
    quaternion = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
    gripper = np.ones(count)
    gripper[4:11] = 0.0
    with h5py.File(path, "w") as recording:
        recording["right_ee_pos"] = position
        recording["right_ee_quat"] = quaternion
        recording["right_gripper"] = gripper
        recording["timestamps"] = np.arange(count) / 30.0


def test_compile_uses_contact_relative_pull(tmp_path):
    paths = []
    for index, offset in enumerate((0.30, 0.31, 0.29)):
        path = tmp_path / f"demo_{index}.hdf5"
        _recording(path, offset)
        paths.append(path)
    result = compile_demonstrations(paths)
    assert result["schema"] == SCHEMA
    assert result["verified_success_count"] == 3
    first = result["relative_pull_trajectory"][0]["pose_wxyz_xyz"]
    assert np.allclose(first, [1, 0, 0, 0, 0, 0, 0])
    final = result["relative_pull_trajectory"][-1]["pose_wxyz_xyz"]
    assert final[4] < -0.06


def test_retarget_moves_the_whole_pull_with_live_contact():
    contact = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_z_radians(np.pi / 2), np.asarray([0.2, -0.1, 0.9])
    )
    relative = mink.SE3.from_translation(np.asarray([-0.05, 0.0, 0.0]))
    result = retarget_relative_pose(contact.parameters(), relative.parameters())
    assert np.allclose(result[4:], [0.2, -0.15, 0.9], atol=1e-8)


def test_retarget_segment_uses_inclusive_frames_and_positive_time():
    contact = [1, 0, 0, 0, 0.2, -0.1, 0.9]
    trajectory = [
        {
            "frame": frame,
            "t_s": (frame - 10) / 30.0,
            "pose_wxyz_xyz": [1, 0, 0, 0, -0.001 * (frame - 10), 0, 0],
            "gripper": 0.0,
        }
        for frame in range(10, 16)
    ]
    result = retarget_relative_trajectory(
        contact, trajectory, first_frame=12, last_frame=14
    )
    assert [sample["frame"] for sample in result] == [12, 13, 14]
    assert result[0]["t_s"] > 0.0
    assert all(
        second["t_s"] > first["t_s"]
        for first, second in zip(result, result[1:])
    )
    assert np.allclose(result[-1]["pose_wxyz_xyz"][4:], [0.196, -0.1, 0.9])


def test_world_yaw_pose_keeps_tool_origin_fixed():
    start = np.asarray([1, 0, 0, 0, 0.2, -0.1, 0.9], dtype=float)
    result = world_yaw_pose(start, 90.0)
    assert np.allclose(result[4:], start[4:])
    expected = mink.SO3.from_z_radians(np.pi / 2).as_matrix()
    assert np.allclose(mink.SE3(result).rotation().as_matrix(), expected)

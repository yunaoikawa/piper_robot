import numpy as np

from src.build_measured_mujoco_scene import _fixture_registration


def test_fixture_registration_uses_robot_to_incubator_vector():
    calibration = {
        "anchor_xyz_level_m": {
            "right_piper_base": [0.0, 0.0, 0.0],
            "incubator": [-0.6521327612, -0.3466585845, 0.0],
        },
        "level_heights_m": {"right_platform": 0.0},
    }
    fixture = {
        "right_piper_base_xyz_m": [0.0, 0.0, 0.0],
        "incubator": {
            "center_xyz_from_mount_m": [0.711, -0.174, 0.372]
        },
        "vertical_offsets_m": {
            "right_platform_above_right_piper_mount": 0.197
        },
    }

    transform, report = _fixture_registration(calibration, fixture)
    mapped = transform[:3, :3] @ np.array(
        calibration["anchor_xyz_level_m"]["incubator"]
    ) + transform[:3, 3]

    assert report["method"] == "right_piper_base_to_incubator_vector"
    assert report["distance_disagreement_m"] < 0.007
    assert np.linalg.norm(mapped[:2] - np.array([0.711, -0.174])) < 0.007
    assert transform[2, 3] == 0.197

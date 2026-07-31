from pathlib import Path

import numpy as np
import pytest

from src.update_current_semantic_objects import (
    assign_instances_by_previous_pose,
)


def _objects():
    return [
        {
            "semantic_name": "dish",
            "body_name": "dish-body",
            "role": "dish_body",
            "radius_m": 0.045,
            "height_m": 0.014,
            "previous_center_scene_xyz_m": [0.092, 1.001, -0.559],
        },
        {
            "semantic_name": "lid",
            "body_name": "lid-body",
            "role": "target_lid",
            "radius_m": 0.047,
            "height_m": 0.006,
            "previous_center_scene_xyz_m": [0.140, 1.014, -0.563],
        },
    ]


def test_unlabelled_instances_use_model_motion_not_image_side():
    candidates = [
        {"center_scene_xyz_m": [0.059, 0.838, -0.566], "candidate": "right"},
        {"center_scene_xyz_m": [0.044, 0.939, -0.566], "candidate": "left"},
    ]
    assigned, audit = assign_instances_by_previous_pose(
        _objects(),
        candidates,
        minimum_margin_in_radii=0.25,
        maximum_displacement_in_radii=5.0,
    )
    assert audit["accepted"]
    assert assigned[0]["semantic_name"] == "dish"
    assert assigned[0]["candidate"] == "left"
    assert assigned[1]["semantic_name"] == "lid"
    assert assigned[1]["candidate"] == "right"


def test_ambiguous_instance_assignment_fails_closed():
    candidates = [
        {"center_scene_xyz_m": [0.100, 1.000, -0.566]},
        {"center_scene_xyz_m": [0.102, 1.000, -0.566]},
    ]
    with pytest.raises(ValueError, match="ambiguous SAM instance assignment"):
        assign_instances_by_previous_pose(
            _objects(),
            candidates,
            minimum_margin_in_radii=1.0,
            maximum_displacement_in_radii=5.0,
        )


def test_current_refresh_source_has_no_robot_control_imports():
    source = Path("src/update_current_semantic_objects.py").read_text()
    forbidden = ("PiperClient", "robot_rpc", "send_joint", "home_right_arm")
    assert not any(token in source for token in forbidden)


def test_jaw_aligned_home_is_not_a_plane_only_pca_result():
    from robot.arm.home import semantic_model_home_q

    expected = 1.355 - np.pi / 2.0
    assert semantic_model_home_q("left")[5] == pytest.approx(expected)
    assert semantic_model_home_q("right")[5] == pytest.approx(expected)

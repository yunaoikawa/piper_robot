import json
from pathlib import Path

import numpy as np
import pytest

from rollout.recorded_trajectory_replay import (
    _minimum_jerk,
    _segment_duration,
)
from src.run_pasteur_offline_replay import _validate_physical_arm_identity
from robot.arm.home import (
    physical_home_q,
    physical_to_semantic_model_q_offset,
    semantic_model_home_q,
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


def test_semantic_home_maps_both_nyu_grippers_to_common_jaw_aligned_roll():
    left = semantic_model_home_q("left")
    right = semantic_model_home_q("right")
    expected = 1.355 - np.pi / 2.0
    assert left[5] == pytest.approx(expected)
    assert right[5] == pytest.approx(expected)
    assert np.allclose(
        physical_home_q("right")
        + physical_to_semantic_model_q_offset("right"),
        right,
    )
    assert physical_to_semantic_model_q_offset("right")[5] == pytest.approx(
        expected - physical_home_q("right")[5]
    )


def test_offline_replay_source_does_not_import_hardware_control():
    source = __import__("pathlib").Path(
        "src/run_pasteur_offline_replay.py"
    ).read_text()
    forbidden = ("PiperClient", "robot_rpc", "teleop", "send_joint")
    assert not any(token in source for token in forbidden)


def _right_arm_identity_inputs():
    return {
        "production_calibration_mapping": {
            "left": "right_arm_",
            "right": "left_arm_",
        },
        "semantic_mapping": {"left": "left", "right": "right"},
        "carving": {
            "physical_right_model_branch": "right",
            "robot_body_prefixes": ["right/"],
        },
        "target_profile": {
            "kinematic_bridge": {
                "physical_arm": "right",
                "model_branch": "right",
            }
        },
        "replay_profile": {"physical_right_model_branch": "right"},
        "target_capture_manifests": [
            {"camera_label": "right"},
            {"camera_label": "right"},
        ],
        "replay_capture_manifests": [{"camera_label": "right"}],
    }


def test_right_wrist_evidence_drives_only_right_model_branch():
    result = _validate_physical_arm_identity(**_right_arm_identity_inputs())
    assert result["accepted"]
    assert result["physical_arm"] == "right"
    assert result["model_branch"] == "right"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("carving", "physical_right_model_branch", "left"),
        ("replay_profile", "physical_right_model_branch", "left"),
    ],
)
def test_offline_replay_rejects_cross_arm_mapping(section, field, value):
    inputs = _right_arm_identity_inputs()
    inputs[section][field] = value
    with pytest.raises(ValueError, match="physical-right branch mismatch"):
        _validate_physical_arm_identity(**inputs)


def test_offline_replay_rejects_left_camera_capture():
    inputs = _right_arm_identity_inputs()
    inputs["target_capture_manifests"][0]["camera_label"] = "left"
    with pytest.raises(ValueError, match="capture labels"):
        _validate_physical_arm_identity(**inputs)


def test_offline_replay_rejects_swapped_semantic_namespace():
    inputs = _right_arm_identity_inputs()
    inputs["semantic_mapping"] = {"left": "right", "right": "left"}
    with pytest.raises(ValueError, match="preserve physical identity"):
        _validate_physical_arm_identity(**inputs)


def test_production_cross_mapping_does_not_change_semantic_identity():
    inputs = _right_arm_identity_inputs()
    inputs["production_calibration_mapping"] = {
        "left": "right_arm_",
        "right": "left_arm_",
    }
    result = _validate_physical_arm_identity(**inputs)
    assert result["production_calibration_mapping"]["right"] == "left_arm_"
    assert result["semantic_mapping"]["right"] == "right"


def test_pasteur_profiles_keep_production_and_semantic_namespaces_separate():
    scene = json.loads(
        Path("src/configs/pasteur_semantic_scene.json").read_text()
    )
    replay = json.loads(
        Path(
            "src/configs/pasteur_recorded_lid_replay_20260730.json"
        ).read_text()
    )
    wrist = json.loads(
        Path("src/configs/pasteur_wrist_target_20260730.json").read_text()
    )
    offline = json.loads(
        Path("src/configs/pasteur_offline_replay_20260730.json").read_text()
    )
    assert scene["robot_calibration"][
        "physical_to_production_branch"
    ] == {"left": "right_arm_", "right": "left_arm_"}
    assert scene["semantic_robot"][
        "physical_to_semantic_branch"
    ] == {"left": "left", "right": "right"}
    assert replay["physical_right_model_branch"] == "right"
    assert wrist["kinematic_bridge"]["model_branch"] == "right"
    assert offline["collision_carving"]["robot_body_prefixes"] == [
        "right/"
    ]

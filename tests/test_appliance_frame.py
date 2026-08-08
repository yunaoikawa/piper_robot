import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rollout.appliance_frame import (
    appliance_pose_from_local_tag,
    appliance_pose_from_scene,
    enroll_local_tag,
    load_accepted_robot_scene_transform,
    matrix_to_pose7,
    registration_between_appliance_frames,
    registration_gate,
    retarget_appliance_trajectory,
    trajectory_to_appliance_frame,
)


def _transform(x=0.0, y=0.0, z=0.0, yaw_deg=0.0, pitch_deg=0.0):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler(
        "zy", [yaw_deg, pitch_deg], degrees=True
    ).as_matrix()
    result[:3, 3] = [x, y, z]
    return result


def _scene(confidence=0.96, fit=True):
    return {
        "objects": [
            {
                "instance_id": "incubator-1",
                "semantic_name": "incubator",
                "confidence": confidence,
                "geometry": {
                    "kind": "box",
                    "center_xyz_m": [0.4, 0.2, 0.3],
                    "size_xyz_m": [0.5, 0.3, 0.4],
                    "yaw_rad": 0.2,
                },
                "semantic_volume_fit": {"accepted": fit},
            }
        ]
    }


def test_scene_pose_uses_explicit_scene_to_robot_transform():
    T_robot_scene = _transform(1.0, -0.5, 0.2, 30.0)
    actual, report = appliance_pose_from_scene(
        _scene(), "incubator", T_robot_scene
    )
    local = _transform(0.4, 0.2, 0.3, np.degrees(0.2))
    np.testing.assert_allclose(actual, T_robot_scene @ local, atol=1e-9)
    assert report["semantic_volume_fit_accepted"] is True


@pytest.mark.parametrize(
    "scene",
    [_scene(confidence=0.2), _scene(fit=False)],
)
def test_scene_pose_fails_closed_on_weak_semantics(scene):
    with pytest.raises(ValueError):
        appliance_pose_from_scene(scene, "incubator", np.eye(4))


def test_different_tag_placements_in_two_labs_recover_each_appliance_pose():
    appliance_a = _transform(0.4, -0.3, 0.5, 8.0)
    appliance_b = _transform(-0.2, 0.7, 0.45, -24.0)
    # These tag placements and ids intentionally differ across installations.
    appliance_from_tag_a = _transform(0.21, 0.0, 0.12, 90.0)
    appliance_from_tag_b = _transform(-0.08, 0.14, 0.22, -35.0)
    tag_a = appliance_a @ appliance_from_tag_a
    tag_b = appliance_b @ appliance_from_tag_b
    enrollment_a = enroll_local_tag(
        appliance_a, tag_a, tag_id=13, appliance_semantic_name="incubator"
    )
    enrollment_b = enroll_local_tag(
        appliance_b, tag_b, tag_id=41, appliance_semantic_name="incubator"
    )
    assert enrollment_a["local_tag"]["id"] != enrollment_b["local_tag"]["id"]
    assert not np.allclose(
        enrollment_a["local_tag"]["T_appliance_tag"],
        enrollment_b["local_tag"]["T_appliance_tag"],
    )
    np.testing.assert_allclose(
        appliance_pose_from_local_tag(tag_a, enrollment_a), appliance_a, atol=1e-9
    )
    np.testing.assert_allclose(
        appliance_pose_from_local_tag(tag_b, enrollment_b), appliance_b, atol=1e-9
    )


def test_portable_trajectory_contains_no_tag_assumption_and_retargets():
    reference_appliance = _transform(0.4, -0.3, 0.5, 8.0)
    live_appliance = _transform(-0.2, 0.7, 0.45, -24.0)
    appliance_from_ee = _transform(0.25, -0.12, 0.05, 90.0)
    reference_ee = reference_appliance @ appliance_from_ee
    source = [
        {
            "frame": 7,
            "t_s": 0.1,
            "pose_wxyz_xyz": matrix_to_pose7(reference_ee).tolist(),
            "gripper": 1.0,
        }
    ]
    portable = trajectory_to_appliance_frame(source, reference_appliance)
    assert "tag" not in json.dumps(portable).lower()
    target = retarget_appliance_trajectory(portable, live_appliance)
    expected = matrix_to_pose7(live_appliance @ appliance_from_ee)
    np.testing.assert_allclose(target[0]["pose_wxyz_xyz"], expected, atol=1e-9)


def test_registration_maps_reference_appliance_and_rejects_large_or_tilted_move():
    reference = _transform(0.2, 0.3, 0.4, 5.0)
    live = _transform(0.3, 0.2, 0.4, 12.0)
    registration = registration_between_appliance_frames(reference, live)
    np.testing.assert_allclose(registration @ reference, live, atol=1e-9)
    assert registration_gate(reference, live)["accepted"] is True
    assert registration_gate(reference, _transform(1.0, 0, 0, 0))["accepted"] is False
    assert (
        registration_gate(reference, _transform(0.2, 0.3, 0.4, 5, 20))["accepted"]
        is False
    )


def test_scene_transform_requires_explicit_acceptance(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"accepted": False, "T_robot_scene": np.eye(4).tolist()}))
    with pytest.raises(ValueError, match="not accepted"):
        load_accepted_robot_scene_transform(path)
    path.write_text(
        json.dumps(
            {
                "accepted": True,
                "transform_convention": "p_robot = T_robot_scene @ p_scene",
                "T_robot_scene": np.eye(4).tolist(),
            }
        )
    )
    transform, _ = load_accepted_robot_scene_transform(path)
    np.testing.assert_allclose(transform, np.eye(4))

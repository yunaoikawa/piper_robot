from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from src.optimize_lid_grasp_trajectory import (
    CLOSED_TARGET_HALF_GAP_M,
    JOINT_LIMIT_MARGIN_RAD,
    OPEN_HALF_GAP_M,
    GraspKinematics,
    _rotation_for_rim,
    _trajectory_samples,
    build_articulated_grasp_model,
)


def _arm(prefix: str) -> str:
    nested = ""
    closing = ""
    ranges = (
        (-2.61, 2.61),
        (0.0, 3.13),
        (-2.965, 0.0),
        (-1.74, 1.74),
        (-1.2, 1.2),
        (-3.0, 3.0),
    )
    for index in range(1, 7):
        lower, upper = ranges[index - 1]
        nested += (
            f'<body name="{prefix}/link{index}">'
            f'<joint name="{prefix}/joint{index}" '
            f'range="{lower} {upper}"/>'
            '<geom type="sphere" size="0.01" density="100"/>'
        )
        closing += "</body>"
    return (
        f'<body name="{prefix}/base_link" pos="0 0 -0.2">'
        + nested
        + f'<body name="{prefix}/gripper_base">'
        f'<geom name="{prefix}/nyu_gripper_visual" type="box" '
        'size="0.02 0.03 0.05" contype="0" conaffinity="0"/>'
        f'<geom name="{prefix}/nyu_gripper_collision" type="box" '
        'size="0.02 0.03 0.05"/>'
        f'<site name="{prefix}/ee" pos="0 0 0.1"/>'
        "</body>"
        + closing
        + "</body>"
    )


def _write_minimal_scene(tmp_path: Path) -> tuple[Path, Path]:
    included = tmp_path / "robot.mjcf"
    actuators = "".join(
        f'<position name="{arm}/joint{i}_pos" joint="{arm}/joint{i}" kp="10"/>'
        for arm in ("left", "right")
        for i in range(1, 7)
    )
    included.write_text(
        "<mujoco><worldbody>"
        + _arm("left")
        + _arm("right")
        + "</worldbody><actuator>"
        + actuators
        + "</actuator></mujoco>"
    )
    scene = tmp_path / "scene.mjcf"
    scene.write_text(
        f'<mujoco><include file="{included}"/>'
        '<compiler angle="radian" autolimits="true"/>'
        '<worldbody>'
        '<body name="petri_dish-1" pos="0 0.1 0">'
        '<geom type="cylinder" size="0.045 0.007"/>'
        "</body>"
        '<body name="petri_lid-1" pos="0 0 0.01">'
        '<geom type="cylinder" size="0.047 0.003"/>'
        "</body>"
        '<body name="support-platform-1-cell-0" pos="0 0 0">'
        '<geom type="box" size="0.1 0.1 0.005"/>'
        "</body>"
        "</worldbody><keyframe><key name=\"home\"/></keyframe></mujoco>"
    )
    return scene, included


def test_derived_grasp_model_has_dynamic_lid_and_articulated_pads(tmp_path):
    scene_path, _ = _write_minimal_scene(tmp_path)
    object_scene = {
        "objects": [
            {
                "role": "target_lid",
                "body_name": "petri_lid-1",
                "pose_scene": np.eye(4).tolist(),
                "geometry": {"radius_m": 0.047},
            }
        ]
    }
    output = build_articulated_grasp_model(
        model_path=scene_path,
        object_scene=object_scene,
        output_path=tmp_path / "derived.mjcf",
    )
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.joint("grasp_search_lid_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("right/grasp_search_upper_joint").limited[0]
    assert model.joint("right/grasp_search_lower_joint").limited[0]
    assert model.geom("right/nyu_gripper_collision").contype[0] == 0
    assert model.geom("right/grasp_search_upper_pad").contype[0] == 1
    assert model.geom("right/grasp_search_lower_pad").contype[0] == 1
    kinematics = GraspKinematics(output)
    assert np.all(
        kinematics.lower
        <= kinematics.upper - 2.0 * JOINT_LIMIT_MARGIN_RAD
    )


def test_rim_rotation_is_level_and_trajectory_closes_before_lift():
    rotation = _rotation_for_rim(np.asarray([-0.4, -0.8]), 0.0)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-12)
    home = np.zeros(6)
    grasp = np.full(6, 0.1)
    samples = _trajectory_samples(
        [
            {
                "stage": "home",
                "q_model": home,
                "jaw_target_m": OPEN_HALF_GAP_M,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "close",
                "q_model": grasp,
                "jaw_target_m": CLOSED_TARGET_HALF_GAP_M,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "verification_lift",
                "q_model": grasp,
                "jaw_target_m": CLOSED_TARGET_HALF_GAP_M,
                "minimum_duration_s": 0.1,
            },
        ],
        hz=20,
    )
    stages = [item["stage"] for item in samples]
    assert stages.index("close") < stages.index("verification_lift")
    assert samples[-1]["jaw_target_m"] == CLOSED_TARGET_HALF_GAP_M


def test_grasp_search_source_has_no_robot_control_imports():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/optimize_lid_grasp_trajectory.py"
    ).read_text()
    forbidden = (
        "from robot.arm.motion",
        "import robot.arm.motion",
        "send_joint",
        "claim_motion_execution",
    )
    assert not any(value in source for value in forbidden)

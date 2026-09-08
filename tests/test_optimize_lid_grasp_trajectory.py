from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from src.optimize_lid_grasp_trajectory import (
    CLOSED_TARGET_HALF_GAP_M,
    JOINT_LIMIT_MARGIN_RAD,
    NYU_JAW_AXIS_LOCAL,
    OPEN_HALF_GAP_M,
    VERTICAL_LIFT_WAYPOINT_COUNT,
    GraspKinematics,
    MAXIMUM_PLANNED_JOINT_SPEED_RAD_S,
    _effective_trajectory_knots,
    _rotation_for_rim,
    _trajectory_samples,
    _vertical_lift_targets,
    build_articulated_grasp_model,
    geometry_closure_policy,
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
    target_pose = np.eye(4)
    target_pose[:3, 3] = [0.12, -0.08, 0.03]
    object_scene = {
        "objects": [
            {
                "role": "target_lid",
                "body_name": "petri_lid-1",
                "pose_scene": target_pose.tolist(),
                "geometry": {"radius_m": 0.047, "height_m": 0.006},
            }
        ]
    }
    output = build_articulated_grasp_model(
        model_path=scene_path,
        object_scene=object_scene,
        output_path=tmp_path / "derived.mjcf",
    )
    model = mujoco.MjModel.from_xml_path(str(output))
    assert np.allclose(model.body("petri_lid-1").pos, target_pose[:3, 3])
    assert model.joint("grasp_search_lid_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("right/grasp_search_upper_joint").limited[0]
    assert model.joint("right/grasp_search_lower_joint").limited[0]
    assert model.geom("right/grasp_search_housing_collision").contype[0] == 2
    assert (
        model.geom("right/grasp_search_upper_environment_collision").contype[0]
        == 2
    )
    assert model.geom("right/grasp_search_upper_pad").contype[0] == 3
    assert model.geom("right/grasp_search_lower_pad").contype[0] == 3
    assert model.geom("grasp_search_local_support_surface").contype[0] == 1
    assert model.geom("grasp_search_lid").contype[0] == 1
    assert model.actuator("right/grasp_search_close").id >= 0
    assert model.actuator("right/grasp_search_close_lower").id >= 0
    weld = model.equality("right/grasp_search_verified_grasp_weld")
    assert weld.id >= 0
    assert model.eq_active0[weld.id] == 0
    assert np.allclose(model.eq_solref[weld.id], [0.001, 1.0])
    assert (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "right/nyu_gripper_collision",
        )
        == -1
    )
    kinematics = GraspKinematics(output)
    assert np.all(
        kinematics.lower
        <= kinematics.upper - 2.0 * JOINT_LIMIT_MARGIN_RAD
    )


def test_rim_rotation_lays_fingers_flat_and_closes_tangent_to_rim():
    rotation = _rotation_for_rim(np.asarray([-0.4, -0.8]), 0.0)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-12)
    outward = np.asarray([-0.4, -0.8, 0.0])
    outward /= np.linalg.norm(outward)
    tangent = np.cross([0.0, 0.0, 1.0], outward)
    assert np.allclose(
        rotation @ NYU_JAW_AXIS_LOCAL, tangent, atol=1e-12
    )
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


def test_vertical_lift_targets_hold_xy_and_increase_z():
    start = np.asarray([0.1, -0.2, 0.3])
    targets = _vertical_lift_targets(start, height_m=0.04)
    assert len(targets) == VERTICAL_LIFT_WAYPOINT_COUNT == 16
    assert all(np.allclose(target[:2], start[:2]) for target in targets)
    assert np.all(np.diff([target[2] for target in targets]) > 0)
    assert np.isclose(targets[-1][2], 0.34)


def test_exported_knots_use_same_joint_speed_timing_as_simulation():
    q0 = np.zeros(6)
    q1 = q0.copy()
    q1[3] = 1.0
    effective = _effective_trajectory_knots(
        [
            {
                "stage": "start",
                "q_model": q0,
                "minimum_duration_s": 0.1,
            },
            {
                "stage": "move",
                "q_model": q1,
                "minimum_duration_s": 0.2,
            },
        ]
    )
    assert np.isclose(
        effective[1]["minimum_duration_s"],
        1.0 / MAXIMUM_PLANNED_JOINT_SPEED_RAD_S,
    )
    assert effective[1]["nominal_minimum_duration_s"] == 0.2


def test_closure_policy_is_geometry_only_and_closes_until_obstructed():
    policy = geometry_closure_policy()
    assert policy["mode"] == "close_until_obstructed"
    assert policy["demonstration_used"] is False
    assert policy["physical_gripper_target_open_ratio"] == 0.0
    assert policy["proxy_target_half_gap_m"] == CLOSED_TARGET_HALF_GAP_M


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
        "demonstration_config",
        "load_demonstrated_closure",
    )
    assert not any(value in source for value in forbidden)

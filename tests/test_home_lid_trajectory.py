import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot.arm.home import (
    PHYSICAL_LEFT_HOME_Q,
    PHYSICAL_RIGHT_HOME_Q,
    mujoco_home_qpos,
)
from rollout.home_lid_trajectory import (
    load_object_scene,
    plan_home_lid_trajectory,
)
from src.render_home_lid_trajectory import (
    _apply_visibility_policy,
    _arm_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
OBJECTS = (
    ROOT
    / "src/configs/pasteur_verified_lid_grasp_scene_20260730.json"
)
ROBOT_MODEL = (
    ROOT / "robot/pasteur-calibrated-scene/scene.mjcf"
)


def test_hardware_home_order_is_physical_right_then_left():
    expected = np.concatenate(
        [PHYSICAL_RIGHT_HOME_Q, PHYSICAL_LEFT_HOME_Q]
    )
    np.testing.assert_allclose(mujoco_home_qpos(), expected)

    model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL))
    # This SAM-aligned model has representation-specific home offsets. The
    # planner preserves this keyframe and applies only physical-home deltas.
    assert model.key("home").qpos.shape[0] >= 12


def test_verified_plan_starts_at_home_and_ends_at_recorded_lift():
    scene = load_object_scene(OBJECTS)
    plan = plan_home_lid_trajectory(
        scene, model_path=ROBOT_MODEL, now_s=123.0
    )
    q = np.asarray(plan.mujoco_q_waypoints)
    lid = next(
        item for item in scene["objects"] if item["role"] == "target_lid"
    )

    assert q.shape[0] == len(plan.plan.waypoints)
    assert q.shape[1] == 6
    assert np.max(np.abs(q[0] - PHYSICAL_RIGHT_HOME_Q)) < 2e-6
    np.testing.assert_allclose(
        q[-1], lid["verification_lift_right_q_rad"]
    )
    assert plan.plan.metadata["corridor"] == "direct"
    assert plan.plan.metadata["fixed_arm"] == "physical_left"
    assert (
        plan.plan.metadata["model_variant"]
        == "sam_reconstruction_upright_nyu"
    )
    assert plan.display_only
    assert not plan.to_dict()["motion_authorized"]
    assert plan.to_dict()["commands_sent"] is False
    assert "confirmed_target_container_missing" in plan.authority_reasons


def test_object_scene_rejects_unconfirmed_lid(tmp_path):
    payload = json.loads(OBJECTS.read_text())
    payload["objects"][0]["status"] = "unknown"
    path = tmp_path / "objects.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="confirmed target_lid"):
        load_object_scene(path)


def test_renderer_keeps_semantic_observed_geometry_visible():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL))
    mapping = _arm_mapping(model)
    _apply_visibility_policy(model, mapping)
    groups = {}
    for name in (
        "microscope-1-observed",
        "support-bench-observed",
        "support-platform-1-observed",
        "support-platform-2-observed",
        "measured-static-scene-observed",
    ):
        body = model.body(name)
        groups[name] = int(model.geom_group[body.geomadr[0]])
    assert groups["microscope-1-observed"] == 2
    assert groups["support-bench-observed"] == 2
    assert groups["support-platform-1-observed"] == 2
    assert groups["support-platform-2-observed"] == 2
    assert groups["measured-static-scene-observed"] == 5

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
from rollout.autonomous_mpc import AutonomousStop
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
    assert model.key("home").qpos.shape[0] >= 12


def test_semantic_model_maps_physical_right_to_model_left():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL))
    mapping = _arm_mapping(model)
    expected_right = [
        int(model.joint(f"left/joint{index}").qposadr[0])
        for index in range(1, 7)
    ]
    expected_left = [
        int(model.joint(f"right/joint{index}").qposadr[0])
        for index in range(1, 7)
    ]
    assert mapping["right_ids"] == expected_right
    assert mapping["left_ids"] == expected_left
    assert mapping["ee"] == "left/ee"


def test_legacy_uncarved_scene_is_not_silently_accepted_for_motion():
    scene = load_object_scene(OBJECTS)
    with pytest.raises(AutonomousStop, match="all home-to-lid corridors"):
        plan_home_lid_trajectory(
            scene,
            model_path=ROBOT_MODEL,
            now_s=123.0,
        )


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

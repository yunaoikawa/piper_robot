import xml.etree.ElementTree as ET

import numpy as np

from robot.arm.home import physical_home_q
from src.refine_scene_robot_alignment import _pin_canonical_physical_home


def test_derived_scene_home_is_physical_right_then_left(tmp_path):
    model = tmp_path / "positioned.mjcf"
    model.write_text(
        """<mujoco><keyframe><key name="home"
        qpos="0 0 0 0 0 0 0 0 0 0 0 0"
        ctrl="0 0 0 0 0 0 0 0 0 0 0 0"/></keyframe></mujoco>"""
    )

    _pin_canonical_physical_home(model)

    key = ET.parse(model).getroot().find("./keyframe/key[@name='home']")
    expected = np.r_[physical_home_q("right"), physical_home_q("left")]
    np.testing.assert_allclose(
        np.fromstring(key.get("qpos"), sep=" "),
        expected,
    )
    np.testing.assert_allclose(
        np.fromstring(key.get("ctrl"), sep=" "),
        expected,
    )

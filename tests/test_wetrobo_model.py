from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np


LAB_SCENE = (
    Path(__file__).resolve().parents[1]
    / "robot"
    / "piper-mujoco"
    / "xml"
    / "lab-scene.xml"
)


@lru_cache(maxsize=1)
def lab_model():
    # Loading from its checked-in path exercises include and mesh resolution in
    # exactly the way a clean checkout does.
    return mujoco.MjModel.from_xml_path(str(LAB_SCENE))


def test_wetrobo_lab_scene_compiles_from_repo_assets():
    model = lab_model()
    assert model.nbody > 1
    assert model.nmesh > 1
    assert model.nkey >= 1


def test_both_grippers_have_sponge_pad_sites():
    model = lab_model()
    site_pairs = (
        ("pad_upper", "pad_lower", "upper_jaw", "lower_jaw"),
        (
            "left_pad_upper",
            "left_pad_lower",
            "left_upper_jaw",
            "left_lower_jaw",
        ),
    )
    for upper, lower, upper_body, lower_body in site_pairs:
        upper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, upper
        )
        lower_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, lower
        )
        assert upper_id >= 0
        assert lower_id >= 0
        assert model.site_bodyid[upper_id] == model.body(upper_body).id
        assert model.site_bodyid[lower_id] == model.body(lower_body).id

    assert np.allclose(
        model.site("pad_upper").pos,
        model.site("left_pad_upper").pos,
    )
    assert np.allclose(
        model.site("pad_lower").pos,
        model.site("left_pad_lower").pos,
    )


if __name__ == "__main__":
    test_wetrobo_lab_scene_compiles_from_repo_assets()
    test_both_grippers_have_sponge_pad_sites()
    print("WetRobo MuJoCo model checks passed")

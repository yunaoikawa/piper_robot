import numpy as np
import pytest

from robot.arm.home import physical_home_q
from rollout.agent_home import agent_home_trajectory


def test_agent_home_is_synchronized_minimum_jerk_for_both_arms():
    starts = {
        "left": physical_home_q("left") + np.array([.1, 0, 0, 0, 0, 0]),
        "right": physical_home_q("right") + np.array([0, -.2, 0, 0, 0, 0]),
    }
    paths, duration = agent_home_trajectory(starts)
    assert paths["left"].shape == paths["right"].shape
    assert duration >= .2 / .18
    np.testing.assert_allclose(paths["left"][-1], physical_home_q("left"))
    np.testing.assert_allclose(paths["right"][-1], physical_home_q("right"))
    assert not np.array_equal(paths["left"][0], physical_home_q("left"))


@pytest.mark.parametrize(
    "starts",
    [
        {"right": np.zeros(6)},
        {"left": np.zeros(5), "right": np.zeros(6)},
        {"left": np.zeros(6), "right": np.full(6, np.nan)},
    ],
)
def test_agent_home_rejects_missing_or_invalid_joint_state(starts):
    with pytest.raises(ValueError):
        agent_home_trajectory(starts)

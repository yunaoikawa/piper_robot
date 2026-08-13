import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cloud_inference_clean-main"))

from act_arm_adapter import action_chunk_to_quat16, adapt_observation_for_active_arm


def test_right_single_arm_checkpoint_receives_only_right_state():
    observation = {"qpos": np.arange(20, dtype=np.float32), "images": {"cam_high": object()}}
    adapted = adapt_observation_for_active_arm(observation, 10, "right")
    np.testing.assert_array_equal(adapted["qpos"], np.arange(10, 20))
    assert adapted["images"] is observation["images"]
    np.testing.assert_array_equal(observation["qpos"], np.arange(20))


def test_left_single_arm_checkpoint_receives_only_left_state():
    observation = {"qpos": np.arange(20, dtype=np.float32)}
    adapted = adapt_observation_for_active_arm(observation, 10, "left")
    np.testing.assert_array_equal(adapted["qpos"], np.arange(10))


def test_single_right_action_is_placed_in_right_wire_slots_only():
    raw = np.array([[.1, .2, .3, 1, 0, 0, 0, 1, 0, .4]], dtype=np.float32)
    wire = action_chunk_to_quat16(raw, "right")
    assert wire.shape == (1, 16)
    assert np.all(np.isnan(wire[0, :7]))
    assert np.isnan(wire[0, 14])
    np.testing.assert_allclose(wire[0, 7:14], [1, 0, 0, 0, .1, .2, .3], atol=1e-6)
    assert wire[0, 15] == pytest.approx(.4)


def test_single_arm_checkpoint_rejects_both_mode():
    with pytest.raises(ValueError, match="requires one active arm"):
        adapt_observation_for_active_arm({"qpos": np.zeros(20)}, 10, "both")
    with pytest.raises(ValueError, match="requires one active arm"):
        action_chunk_to_quat16(np.zeros((2, 10)), "both")

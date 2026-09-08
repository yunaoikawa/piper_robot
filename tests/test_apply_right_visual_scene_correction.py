import numpy as np
import pytest

from src.apply_right_visual_scene_correction import apply_scene_correction


def _scene():
    return {
        "schema": "test",
        "objects": [
            {
                "semantic_name": "target",
                "pose_scene": np.eye(4).tolist(),
            },
            {
                "semantic_name": "obstacle",
                "pose_scene": np.eye(4).tolist(),
            },
        ],
    }


def _correction(delta):
    return {
        "schema": "piper_robot.right_visual_correction/v1",
        "world_delta_model_m": delta,
        "correction": {
            "metric_scale_source": "accepted_goal_reference_quantiles"
        },
    }


def test_applies_only_planar_target_translation():
    result = apply_scene_correction(
        _scene(),
        _correction([0.01, -0.02, 0.0]),
        semantic_name="target",
        maximum_correction_m=0.05,
    )
    target = result["objects"][0]["pose_scene"]
    obstacle = result["objects"][1]["pose_scene"]
    assert np.allclose(np.asarray(target)[:3, 3], [0.01, -0.02, 0.0])
    assert np.allclose(obstacle, np.eye(4))
    assert result["operator_confirmed"] is False


def test_rejects_vertical_or_excessive_correction():
    with pytest.raises(ValueError, match="planar"):
        apply_scene_correction(
            _scene(),
            _correction([0.0, 0.0, 0.001]),
            semantic_name="target",
            maximum_correction_m=0.05,
        )
    with pytest.raises(ValueError, match="exceeds"):
        apply_scene_correction(
            _scene(),
            _correction([0.1, 0.0, 0.0]),
            semantic_name="target",
            maximum_correction_m=0.05,
        )

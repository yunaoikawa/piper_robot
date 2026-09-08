import numpy as np

from rollout.grasp_window import calibrate_grasp_window
from rollout.tool_relative_visual_correction import (
    estimate_model_plane_correction,
)
from tests.test_grasp_window import _scene


def test_metric_visual_correction_is_resolution_independent_and_bounded():
    image, reference = _scene()
    template, frame = calibrate_grasp_window(image, reference)
    _, shifted = _scene(target_shift=(60, -20))
    correction = estimate_model_plane_correction(
        shifted,
        frame,
        template,
        target_diameter_m=0.094,
        maximum_step_m=0.01,
    )
    assert correction.raw_norm_m > correction.bounded_norm_m
    assert np.isclose(correction.bounded_norm_m, 0.01)
    assert correction.raw_error_uv[0] != 0.0
    assert (
        correction.metric_scale_source
        == "accepted_goal_reference_quantiles"
    )


def test_metric_scale_does_not_change_when_live_mask_is_cropped():
    image, reference = _scene()
    template, frame = calibrate_grasp_window(image, reference)
    _, shifted = _scene(target_shift=(60, -20))
    full = estimate_model_plane_correction(
        shifted,
        frame,
        template,
        target_diameter_m=0.094,
        maximum_step_m=1.0,
    )
    cropped = shifted.copy()
    cropped[:, : int(cropped.shape[1] * 0.35)] = False
    partial = estimate_model_plane_correction(
        cropped,
        frame,
        template,
        target_diameter_m=0.094,
        maximum_step_m=1.0,
    )
    assert np.isclose(
        full.metres_per_tool_unit,
        partial.metres_per_tool_unit,
    )

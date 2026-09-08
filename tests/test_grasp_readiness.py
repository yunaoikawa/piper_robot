from rollout.grasp_readiness import (
    PrecloseThresholds,
    assess_preclose,
)


THRESHOLDS = PrecloseThresholds(
    maximum_orientation_error_deg=3.0,
    maximum_tip_clearance_m=0.002,
    maximum_tip_penetration_m=0.001,
    maximum_normalized_image_gap=0.08,
)


def test_preclose_requires_every_independent_gate():
    result = assess_preclose(
        target_visible=True,
        target_in_window=True,
        normalized_image_gap=0.01,
        orientation_error_deg=1.0,
        tool_support_clearance_m=0.001,
        thresholds=THRESHOLDS,
    )
    assert result.allowed
    assert not result.reasons


def test_white_window_cannot_override_floating_tool():
    result = assess_preclose(
        target_visible=True,
        target_in_window=True,
        normalized_image_gap=0.01,
        orientation_error_deg=1.0,
        tool_support_clearance_m=0.020,
        thresholds=THRESHOLDS,
    )
    assert not result.allowed
    assert "tool_tip_not_at_support_plane" in result.reasons


def test_excess_plane_penetration_fails_closed():
    result = assess_preclose(
        target_visible=True,
        target_in_window=True,
        normalized_image_gap=0.01,
        orientation_error_deg=1.0,
        tool_support_clearance_m=-0.005,
        thresholds=THRESHOLDS,
    )
    assert not result.allowed
    assert "tool_tip_not_at_support_plane" in result.reasons


def test_negative_threshold_is_rejected():
    try:
        PrecloseThresholds(3.0, -0.001, 0.001, 0.08)
    except ValueError:
        pass
    else:
        raise AssertionError("negative preclose threshold was accepted")

import numpy as np

from rollout.right_grasp_geometry import assess_right_grasp


def _ellipse(cx, cy, rx, ry):
    yy, xx = np.ogrid[:480, :640]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def test_rejects_distant_lid_even_if_vertically_aligned():
    reference = _ellipse(187, 332, 121, 61)
    current = _ellipse(126, 319, 83, 39)
    result = assess_right_grasp(current, reference)
    assert not result.allowed_to_close
    assert not result.scale_ready
    assert result.insertion_error_px > 50


def test_accepts_demo_like_lid_in_jaw_corridor():
    reference = _ellipse(187, 332, 121, 61)
    current = _ellipse(190, 329, 118, 60)
    result = assess_right_grasp(current, reference)
    assert result.allowed_to_close
    assert result.area_ratio > 0.9

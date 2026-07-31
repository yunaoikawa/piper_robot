import cv2
import numpy as np
import pytest

from rollout.tapped_lid_target import (
    TappedTarget,
    associate_blue_component,
    frame_sha256,
    register_fixed_head,
    validate_tap_frame,
)


def _scene(shape=(400, 600, 3)):
    image = np.full(shape, 80, np.uint8)
    cv2.circle(image, (120, 220), 28, (255, 140, 0), -1)
    cv2.circle(image, (430, 210), 9, (255, 140, 0), -1)
    return image


def test_tap_selects_nearest_not_largest_blue_component():
    target = associate_blue_component(_scene(), (430 / 600, 210 / 400))
    assert np.allclose(target.center_px, (430, 210), atol=1)
    assert target.area_fraction < 0.01


def test_tap_rejects_distant_component():
    with pytest.raises(ValueError, match="too far"):
        associate_blue_component(_scene(), (0.5, 0.1))


def test_exact_fresh_lit_frame_is_required():
    image = _scene()
    tap = TappedTarget((0.2, 0.5), frame_sha256(image), 10.0)
    validate_tap_frame(image, tap, frame_timestamp=10.0, now=11.0)
    changed = image.copy()
    changed[0, 0] = 0
    with pytest.raises(ValueError, match="different"):
        validate_tap_frame(changed, tap, frame_timestamp=10.0, now=11.0)
    dark = np.zeros_like(image)
    dark_tap = TappedTarget((0.2, 0.5), frame_sha256(dark), 10.0)
    with pytest.raises(ValueError, match="dark"):
        validate_tap_frame(dark, dark_tap, frame_timestamp=10.0, now=11.0)
    with pytest.raises(ValueError, match="stale"):
        validate_tap_frame(image, tap, frame_timestamp=10.0, now=13.0)


def test_fixed_head_registration_accepts_same_textured_view_and_rejects_shape():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
    report = register_fixed_head(image, image.copy(), minimum_matches=20)
    assert report.accepted
    assert report.inlier_fraction > 0.9
    assert not register_fixed_head(image, image[:200]).accepted

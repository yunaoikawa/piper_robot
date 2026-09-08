import cv2
import numpy as np

from rollout.apriltag_retarget import TagDetection
from rollout.stage_aperture_visibility import (
    assess_aperture_visibility,
    project_tag_point,
    refine_tag_point_from_pixel,
)


def _image():
    return np.full((360, 480, 3), 92, dtype=np.uint8)


def test_out_of_frame_target_is_not_visible():
    result = assess_aperture_visibility(
        _image(), projected_uv=(-4, 120), anchor_perimeter_px=120
    )
    assert result.state == "target_out_of_frame"
    assert not result.visible


def test_blue_gripper_occlusion_fails_closed():
    image = _image()
    cv2.circle(image, (240, 180), 45, (220, 125, 20), -1)
    result = assess_aperture_visibility(
        image, projected_uv=(240, 180), anchor_perimeter_px=120
    )
    assert result.state == "target_predicted_but_occluded"
    assert result.blue_occlusion_fraction > 0.18


def test_dark_elongated_aperture_is_appearance_confirmed():
    image = _image()
    cv2.ellipse(image, (240, 180), (30, 10), 12, 0, 360, (8, 8, 8), -1)
    result = assess_aperture_visibility(
        image, projected_uv=(240, 180), anchor_perimeter_px=120
    )
    assert result.visible
    assert result.state == "target_visible"
    assert np.linalg.norm(np.asarray(result.observed_uv) - [240, 180]) < 4


def test_prediction_without_hole_is_not_confirmed():
    result = assess_aperture_visibility(
        _image(), projected_uv=(240, 180), anchor_perimeter_px=120
    )
    assert result.state == "target_predicted_not_confirmed"
    assert not result.visible


def test_uniformly_surrounded_hole_beats_nearby_high_contrast_shadow():
    image = np.full((360, 480, 3), 82, dtype=np.uint8)
    # A bright/dark boundary near the metric projection mimics the microscope
    # bridge shadow; its surrounding ring is not a uniform support surface.
    cv2.rectangle(image, (235, 145), (360, 215), (220, 220, 220), -1)
    cv2.rectangle(image, (235, 205), (360, 228), (8, 8, 8), -1)
    # The actual opening may be farther away because of reference-depth
    # parallax, but it is enclosed by one uniform stage surface.
    cv2.ellipse(image, (190, 245), (34, 11), 10, 0, 360, (8, 8, 8), -1)
    result = assess_aperture_visibility(
        image, projected_uv=(200, 240), anchor_perimeter_px=200,
        minimum_appearance_score=0.35,
    )
    assert result.visible
    assert np.linalg.norm(np.asarray(result.observed_uv) - [190, 245]) < 5


def test_confirmed_pixel_refines_tag_point_without_changing_depth():
    matrix = np.asarray([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    tag = TagDetection(
        12,
        np.asarray([[300.0, 220.0], [340.0, 220.0], [340.0, 260.0], [300.0, 260.0]]),
        "DICT_APRILTAG_36h11",
    )
    prior = np.asarray([0.10, 0.02, 0.40])
    old_uv, old_camera, _ = project_tag_point(tag, matrix, 0.04, prior)
    confirmed = old_uv + np.asarray([-25.0, 18.0])
    refined, retained_depth, _ = refine_tag_point_from_pixel(
        tag, matrix, 0.04, prior, confirmed
    )
    new_uv, new_camera, _ = project_tag_point(tag, matrix, 0.04, refined)
    assert np.allclose(new_uv, confirmed)
    assert np.isclose(retained_depth, old_camera[2])
    assert np.isclose(new_camera[2], old_camera[2])

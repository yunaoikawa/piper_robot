from pathlib import Path

import cv2
import numpy as np
import pytest

from rollout.grasp_window import detect_light_pad_tool_frame
from rollout.media_cap_target import (
    MediaCapTargetAdapter,
    detect_coloured_support_anchor,
    detect_media_cap,
    detect_open_jaw_center_head,
    fixed_target_in_jaw_segment,
)
from src.run_culture_media_cap_grasp import load_task_profile


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "src/configs/pasteur_culture_media_cap_grasp.json"
HEAD = Path("/tmp/pasteur_home_views/head_home.png")
RIGHT = (
    ROOT
    / "data/runs/pasteur/culture_media_cap_grasp_20260806_stable_plan/right_00.png"
)


def test_cap_task_cannot_import_thin_object_demo_goals():
    task, base = load_task_profile(TASK)
    assert task["target_adapter"] == "culture_media_cap"
    assert "canonical_hover_goal_uv" not in task
    assert "canonical_preclose_goal_uv" not in task
    assert base["target_identity"]["feature_adapter"] == "blue_cross"
    assert task["fiducials"]["included_in_collision_geometry"] is False
    assert task["scene_refresh"]["support_relation"] == (
        "recessed_between_two_equal-height_platforms"
    )
    assert task["head_coarse"]["fixed_camera_stationary_target"] is True


def test_tap_identity_disambiguates_fixed_head_cap_from_white_clutter():
    if not HEAD.exists():
        pytest.skip("Pasteur hardware observation is not installed")
    image = cv2.imread(str(HEAD))
    identity = __import__("json").loads(
        (ROOT / "data/runs/pasteur/culture_media_cap_tap_retry_20260806/tap_identity.json").read_text()
    )
    uv = np.asarray(identity["tap"]["uv"], dtype=float)
    anchor = uv * np.asarray([image.shape[1], image.shape[0]])
    mask, center, diagnostics = detect_media_cap(
        image, identity_anchor_px=anchor
    )
    assert np.linalg.norm(center - anchor) < 0.02 * np.hypot(
        image.shape[1], image.shape[0]
    )
    assert np.count_nonzero(mask) > 1000
    assert diagnostics["identity_anchor_px"] is not None


def test_fixed_head_open_jaw_center_uses_two_large_elongated_components():
    if not HEAD.exists():
        pytest.skip("Pasteur hardware observation is not installed")
    image = cv2.imread(str(HEAD))
    center, mask, diagnostics = detect_open_jaw_center_head(image)
    assert np.count_nonzero(mask) > 5000
    assert diagnostics["jaw_span_px"] > 50
    assert len(diagnostics["jaw_centers_px"]) == 2
    assert np.all(np.isfinite(center))


def test_right_adapter_reports_cap_in_dynamic_tool_coordinates():
    if not RIGHT.exists():
        pytest.skip("Pasteur hardware observation is not installed")
    image = cv2.imread(str(RIGHT))
    frame = detect_light_pad_tool_frame(image)
    observed = MediaCapTargetAdapter().observe(image, frame)
    assert observed.component_pixels > 1000
    assert observed.equivalent_diameter_tool_units > 0.1
    assert np.count_nonzero(observed.mask) == observed.component_pixels
    assert np.all(np.isfinite(observed.center_uv))


def test_tap_local_white_track_beats_distant_tag_when_neck_is_occluded():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(image, (100, 100), 14, (245, 245, 245), -1)
    cv2.rectangle(image, (190, 130), (270, 210), (255, 255, 255), -1)
    mask, center, diagnostics = detect_media_cap(
        image,
        identity_anchor_px=[102, 99],
        maximum_anchor_displacement_diagonal_fraction=0.25,
    )
    assert np.linalg.norm(center - [100, 100]) < 3
    assert np.count_nonzero(mask) > 400
    assert diagnostics["selection_method"] == (
        "tap_local_white_component_under_occlusion"
    )


def test_fixed_target_must_lie_between_closed_jaws():
    accepted = fixed_target_in_jaw_segment(
        [50, 52], [[20, 50], [80, 50]],
        maximum_perpendicular_span_fraction=0.10,
    )
    rejected = fixed_target_in_jaw_segment(
        [50, 70], [[20, 50], [80, 50]],
        maximum_perpendicular_span_fraction=0.10,
    )
    assert accepted["accepted"] is True
    assert rejected["accepted"] is False


def test_coloured_support_guard_uses_lower_body_under_occluded_neck():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (90, 115), (130, 145), (180, 40, 220), -1)
    cv2.rectangle(image, (85, 165), (135, 225), (180, 40, 220), -1)
    center, diagnostics = detect_coloured_support_anchor(
        image, target_anchor_px=[110, 90]
    )
    assert center[1] > 180
    assert diagnostics["component_diagonal_px"] > 50

import cv2
import numpy as np

from rollout.grasp_window import (
    ToolImageFrame,
    assess_grasp_window,
    calibrate_grasp_window,
    detect_light_pad_tool_frame,
    normalized_pad_target_gap,
)


def _scene(scale=1.0, offset=(0, 0), target_shift=(0, 0)):
    image = np.full((480, 640, 3), 25, dtype=np.uint8)
    cyan = np.array(
        [[180, 430], [250, 465], [420, 300], [385, 270]], dtype=float
    )
    pad = np.array(
        [[280, 350], [385, 275], [410, 295], [305, 370]], dtype=float
    )
    target_center = np.array([184, 322], dtype=float) + target_shift
    transform = lambda points: np.rint(
        np.asarray(points) * scale + np.asarray(offset)
    ).astype(np.int32)
    cv2.fillConvexPoly(image, transform(cyan), (230, 170, 20))
    cv2.fillConvexPoly(image, transform(pad), (150, 150, 150))
    center = tuple(transform(target_center.reshape(1, 2))[0])
    axes = tuple(np.maximum(1, np.rint(np.array([110, 55]) * scale).astype(int)))
    target = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(target, center, axes, -8, 0, 360, 255, -1)
    return image, target > 0


def test_white_pad_frame_is_resolution_and_translation_independent():
    image, _ = _scene()
    transformed, _ = _scene(scale=0.7, offset=(45, 30))
    first = detect_light_pad_tool_frame(image)
    second = detect_light_pad_tool_frame(transformed)
    assert first.light_pad_pixels > 50
    assert second.light_pad_pixels > 50
    assert np.dot(first.forward_xy, second.forward_xy) > 0.95


def test_grasp_window_accepts_equivalent_scaled_scene():
    reference_image, reference_target = _scene()
    template, _ = calibrate_grasp_window(reference_image, reference_target)
    live_image, live_target = _scene(scale=0.75, offset=(30, 22))
    result, _ = assess_grasp_window(live_image, live_target, template)
    assert result.allowed_to_close
    assert result.target_inside_fraction > 0.70


def test_grasp_window_rejects_target_outside_tool_relative_square():
    reference_image, reference_target = _scene()
    template, _ = calibrate_grasp_window(reference_image, reference_target)
    live_image, live_target = _scene(target_shift=(150, 0))
    result, _ = assess_grasp_window(live_image, live_target, template)
    assert not result.allowed_to_close


def test_tool_frame_round_trip():
    frame = ToolImageFrame((20, 30), (1, 0), (0, 1), 10, 100, 50)
    points = np.array([[30, 50], [5, 12]], dtype=float)
    assert np.allclose(frame.tool_to_image(frame.image_to_tool(points)), points)


def test_pad_target_gap_is_normalized_by_observed_tool_scale():
    touching_image, touching_target = _scene()
    scaled_image, scaled_target = _scene(scale=0.7, offset=(45, 30))
    first, _ = normalized_pad_target_gap(touching_image, touching_target)
    second, _ = normalized_pad_target_gap(scaled_image, scaled_target)
    assert first < 0.10
    assert second < 0.10
    assert abs(first - second) < 0.02


def test_pad_target_gap_increases_when_target_moves_away():
    image, target = _scene(target_shift=(-170, -100))
    gap, _ = normalized_pad_target_gap(image, target)
    assert gap > 0.20

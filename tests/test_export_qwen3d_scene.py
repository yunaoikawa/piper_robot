import cv2
import numpy as np

from src.export_qwen3d_scene import (
    _resize_rgbd_and_intrinsics,
    _stratified_indices,
)


def test_stratified_display_sampling_preserves_small_semantic_regions():
    labels = np.r_[np.full(10000, 2), np.full(20, 7), np.full(11, 8)]
    indices = _stratified_indices(labels, 500)
    selected = labels[indices]
    assert len(indices) == 500
    assert set(np.unique(selected)) == {2, 7, 8}
    assert np.sum(selected == 7) == 20
    assert np.sum(selected == 8) == 11


def test_stratified_sampling_is_deterministic():
    labels = np.repeat(np.arange(6), 1000)
    np.testing.assert_array_equal(
        _stratified_indices(labels, 700),
        _stratified_indices(labels, 700),
    )


def test_rgbd_resize_preserves_backprojected_ray_geometry():
    rgb = np.zeros((192, 256, 3), dtype=np.uint8)
    depth = np.ones((192, 256), dtype=np.float32)
    intrinsics = np.array(
        [[200.0, 0.0, 127.5], [0.0, 210.0, 95.5], [0.0, 0.0, 1.0]]
    )

    resized_rgb, resized_depth, resized_intrinsics = (
        _resize_rgbd_and_intrinsics(rgb, depth, intrinsics, (512, 512))
    )

    assert resized_rgb.shape == (512, 512, 3)
    assert resized_depth.shape == (512, 512)
    np.testing.assert_allclose(
        resized_intrinsics,
        np.array(
            [
                [400.0, 0.0, 255.0],
                [0.0, 560.0, 254.6666666667],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    assert cv2.INTER_NEAREST == 0

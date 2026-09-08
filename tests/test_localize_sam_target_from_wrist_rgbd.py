import numpy as np

from src.localize_sam_target_from_wrist_rgbd import (
    support_plane_target_point,
)


def test_transparent_mask_ray_intersects_surrounding_support_plane():
    height, width = 240, 320
    matrix = np.asarray(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    )
    depth = np.full((height, width), 0.4)
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - 120) ** 2 + (yy - 140) ** 2 <= 25**2
    depth[mask] = np.nan
    point, report = support_plane_target_point(
        depth,
        matrix,
        mask,
        ransac_iterations=80,
    )
    expected = np.asarray(
        [(120 - 160) * 0.4 / 300, (140 - 120) * 0.4 / 300, 0.4]
    )
    assert np.allclose(point, expected, atol=1e-6)
    assert report["target_mask_depth_valid_fraction"] == 0.0
    assert report["support_plane_inlier_ratio"] > 0.99

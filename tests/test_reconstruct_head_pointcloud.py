#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reconstruct_head_pointcloud import (
    align_depth,
    backproject,
    scaled_camera_matrix,
)


rgb_shape = (720, 960, 3)
portrait_depth = np.ones((256, 192), dtype=float)
aligned = align_depth(portrait_depth, rgb_shape)
assert aligned.shape == (192, 256)

full_matrix = np.array(
    [[720.0, 0.0, 480.0], [0.0, 720.0, 360.0], [0.0, 0.0, 1.0]]
)
small_matrix = scaled_camera_matrix(
    full_matrix, rgb_shape, aligned.shape
)
points = backproject(aligned, small_matrix)
center = points[96, 128]
assert np.allclose(center, [0.0, 0.0, 1.0])
assert points[96, 129, 0] > 0.0
assert points[97, 128, 1] > 0.0

print("head point-cloud reconstruction checks passed")

#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.fixed_head_scene_registration import (
    ransac_rigid_transform,
    rigid_transform,
    transform_points,
)


class FixedHeadSceneRegistrationTest(unittest.TestCase):
    def test_rigid_transform_recovers_known_pose(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(80, 3))
        truth = np.eye(4)
        truth[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
        truth[:3, 3] = [0.2, -0.1, 0.4]
        target = transform_points(source, truth)
        fitted = rigid_transform(source, target)
        self.assertTrue(np.allclose(fitted, truth, atol=1e-9))

    def test_ransac_rejects_metric_feature_outliers(self):
        rng = np.random.default_rng(8)
        source = rng.uniform(-0.4, 0.4, (100, 3))
        truth = np.eye(4)
        truth[:3, :3] = Rotation.from_euler("zyx", [0.2, 0.05, -0.03]).as_matrix()
        truth[:3, 3] = [0.15, 0.02, -0.08]
        target = transform_points(source, truth)
        target += rng.normal(0.0, 0.001, target.shape)
        target[:35] = rng.uniform(-1.0, 1.0, (35, 3))
        fitted, inliers, _ = ransac_rigid_transform(
            source, target, threshold_m=0.01, iterations=1500
        )
        self.assertGreaterEqual(np.count_nonzero(inliers), 60)
        self.assertLess(np.linalg.norm(fitted[:3, 3] - truth[:3, 3]), 0.004)
        angle = Rotation.from_matrix(
            fitted[:3, :3] @ truth[:3, :3].T
        ).magnitude()
        self.assertLess(np.degrees(angle), 0.5)


if __name__ == "__main__":
    unittest.main()

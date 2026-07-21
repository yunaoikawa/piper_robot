#!/usr/bin/env python3
"""Hardware-free AprilTag localization, retargeting and servo checks."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.apriltag_retarget import (
    TagDetection,
    classify_roles,
    detect_tags,
    detect_blue_cross,
    estimate_tag_camera_pose,
    fit_image_to_robot,
    fit_image_to_plane,
    lid_pose_robot,
    object_delta,
    retarget_pose,
    retarget_weight,
    servo_error,
    servo_step,
)


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
canvas = np.full((500, 700, 3), 255, np.uint8)
specs = [(1, 40, 40, 100), (2, 550, 40, 100), (3, 40, 350, 100), (9, 330, 230, 50)]
for tag_id, x, y, size in specs:
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, size)
    canvas[y:y+size, x:x+size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

tags = detect_tags(canvas)
assert [tag.tag_id for tag in tags] == [1, 2, 3, 9]
roles = classify_roles(tags)
assert roles[9] == "lid" and sum(v == "fixed" for v in roles.values()) == 3

blue = np.zeros((100, 100, 3), np.uint8)
cv2.rectangle(blue, (44, 35), (55, 64), (255, 0, 0), -1)
cv2.rectangle(blue, (35, 44), (64, 55), (255, 0, 0), -1)
cross = detect_blue_cross(
    blue, np.eye(3), np.eye(2), [50, 50],
    {"hsv_low": [100, 80, 50], "hsv_high": [125, 255, 255],
     "min_area": 30, "max_area": 1000, "max_distance_m": 1.0})
assert np.allclose(cross["center"], [49.5, 49.5])

# Verify that the declared 30 mm physical size produces a stable camera pose.
camera_matrix = np.array([[600.0, 0.0, 350.0],
                          [0.0, 600.0, 250.0],
                          [0.0, 0.0, 1.0]])
object_points = np.float32([
    [-0.015, 0.015, 0.0], [0.015, 0.015, 0.0],
    [0.015, -0.015, 0.0], [-0.015, -0.015, 0.0],
])
expected_rvec = np.array([0.08, -0.12, 0.03])
expected_tvec = np.array([0.02, -0.01, 0.42])
image_points, _ = cv2.projectPoints(
    object_points, expected_rvec, expected_tvec, camera_matrix, np.zeros(5))
synthetic = TagDetection(11, image_points.reshape(4, 2), "DICT_APRILTAG_36h11")
_, estimated_tvec, reprojection = estimate_tag_camera_pose(
    synthetic, camera_matrix, 0.03)
assert np.allclose(estimated_tvec, expected_tvec, atol=1e-5), estimated_tvec
assert reprojection < 1e-3, reprojection

fixed_xy = {"1": [0.0, 0.0], "2": [0.5, 0.0], "3": [0.0, 0.3]}
transform = fit_image_to_robot(tags, fixed_xy)
pose = lid_pose_robot(tags, 9, transform)
assert np.allclose(pose[:2], [0.259804, 0.159677], atol=0.003), pose

fixed_corners = {str(tag_id): next(tag for tag in tags if tag.tag_id == tag_id).corners.tolist()
                 for tag_id in (1, 2, 3)}
pixel_identity = fit_image_to_plane(tags, fixed_corners)
identity_pose = lid_pose_robot(tags, 9, pixel_identity, np.eye(2))
lid_pixels = next(tag for tag in tags if tag.tag_id == 9).center
assert np.allclose(identity_pose[:2], lid_pixels, atol=0.1)

delta = object_delta([0.20, 0.10, np.deg2rad(10)], [0.18, 0.12, 0.0])
assert np.allclose(delta[:2], [0.02, -0.02])
assert retarget_weight(0, {}) == 0.0
assert retarget_weight(60, {}) == 1.0
assert retarget_weight(140, {}) == 1.0
assert retarget_weight(190, {}) == 0.0

pose0 = np.array([1.0, 0.0, 0.0, 0.0, 0.30, 0.10, 0.80])
shifted = retarget_pose(pose0, [0.02, -0.01, 0.0], 1.0, [0.2, 0.1])
assert np.allclose(shifted[4:7], [0.32, 0.09, 0.80])

reference = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], float)
current = reference + [4, -3]
error = servo_error(current, reference)
assert np.allclose(error[:2], [-4, 3])
jacobian = np.diag([2000.0, 2000.0, 1.0])
step = servo_step(jacobian, error)
assert np.allclose(step[:2], [-0.002, 0.0015], atol=1e-5)

print("AprilTag retarget checks passed")

#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.realtime_sam_servo import (
    gripper_cyan_support_mask,
    gripper_cyan_tip_px,
    gripper_tip_px,
    scene_feature,
)
from rollout.sam_segmentation import MaskCandidate


def candidate(mask: np.ndarray) -> MaskCandidate:
    ys, xs = np.where(mask)
    return MaskCandidate(
        mask=np.asarray(mask, dtype=bool),
        box_xyxy=np.array(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
            dtype=float,
        ),
        score=0.95,
    )


shape = (240, 360)
clean_mask = np.zeros(shape, dtype=np.uint8)
jaw_polygon = cv2.boxPoints(((245, 150), (150, 34), -10)).astype(np.int32)
cv2.fillConvexPoly(clean_mask, jaw_polygon, 1)

# Model a SAM instability seen in the saved lab observations: one prediction
# contains a remote piece of another cyan object while an equivalent prediction
# does not.  The jaw itself and the RGB observation are unchanged.
merged_mask = clean_mask.copy()
cv2.circle(merged_mask, (130, 55), 18, 1, thickness=-1)

image = np.zeros((*shape, 3), dtype=np.uint8)
image[clean_mask.astype(bool)] = (255, 190, 15)  # cyan in BGR
image[merged_mask.astype(bool) & ~clean_mask.astype(bool)] = (255, 190, 15)

clean = candidate(clean_mask)
merged = candidate(merged_mask)
clean_tip = gripper_cyan_tip_px(clean, image)
merged_tip = gripper_cyan_tip_px(merged, image)
assert np.linalg.norm(clean_tip - merged_tip) <= 1.0

# Even when a false SAM extension touches the jaw and therefore belongs to the
# same semantic connected component, HSV support must keep the contact feature
# on the physical cyan tool.
connected_false_mask = clean_mask.copy()
cv2.line(connected_false_mask, (182, 143), (115, 75), 1, thickness=16)
connected_false = candidate(connected_false_mask)
connected_image = image.copy()
false_extension = connected_false_mask.astype(bool) & ~clean_mask.astype(bool)
connected_image[false_extension] = (40, 40, 40)
colour_tip = gripper_cyan_tip_px(connected_false, connected_image)
sam_only_tip = gripper_tip_px(connected_false)
assert np.linalg.norm(colour_tip - clean_tip) <= 1.0
assert np.linalg.norm(sam_only_tip - clean_tip) >= 5.0

support = gripper_cyan_support_mask(connected_false, connected_image)
assert np.count_nonzero(support & false_extension) == 0
assert np.count_nonzero(support) >= 4000

# The scene-level API must expose the exact support used for both the feature
# and its depth, so the live caller can preserve it as an auditable artifact.
lid_mask = np.zeros(shape, dtype=np.uint8)
cv2.circle(lid_mask, (70, 175), 25, 1, thickness=-1)
depth = np.ones(shape, dtype=float)
depth[lid_mask.astype(bool)] = 1.10
depth[support] = 0.82
scene = scene_feature(
    lid_candidates=[candidate(lid_mask)],
    gripper_candidates=[connected_false],
    depth_m=depth,
    image_bgr=connected_image,
)
assert scene.gripper_feature_support_mask is not None
assert np.array_equal(scene.gripper_feature_support_mask, support)
assert np.linalg.norm(scene.gripper_feature[:2] - clean_tip) <= 1.0
assert abs(scene.gripper_feature[2] - 820.0) < 1e-9

# Colour evidence is a safety observation, not an optional heuristic.  A wrong
# tool colour must stop feature extraction instead of falling back to SAM.
non_cyan_image = np.zeros_like(image)
non_cyan_image[clean_mask.astype(bool)] = (20, 200, 20)
try:
    gripper_cyan_tip_px(clean, non_cyan_image)
except ValueError as error:
    assert "cyan gripper" in str(error)
else:
    raise AssertionError("missing cyan support must fail closed")

print("gripper colour feature checks passed")

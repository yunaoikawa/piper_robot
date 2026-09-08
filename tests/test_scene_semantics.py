#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_semantics import (
    LABEL_BACKGROUND,
    LABEL_LID,
    LABEL_ROBOT,
    compose_surface_labels,
    largest_components,
    recover_blended_sam_mask,
)


def test_alpha_overlay_recovers_mask_without_annotation():
    source = np.full((80, 100, 3), 70, dtype=np.uint8)
    expected = np.zeros(source.shape[:2], dtype=bool)
    cv2.circle(expected.view(np.uint8), (55, 40), 18, 1, -1)
    tint = np.zeros_like(source)
    tint[:] = (0, 255, 255)
    overlay = source.copy()
    overlay[expected] = cv2.addWeighted(
        source[expected], 0.55, tint[expected], 0.45, 0
    )
    cv2.rectangle(overlay, (5, 5), (40, 25), (0, 255, 255), 2)
    recovered = recover_blended_sam_mask(
        source,
        overlay,
        source_weight=0.55,
        tint_bgr=(0, 255, 255),
    )
    assert np.array_equal(recovered, expected)


def test_semantic_priority_and_component_filter():
    robot = np.zeros((30, 40), dtype=np.uint8)
    robot[5:20, 5:20] = 1
    robot[25, 35] = 1
    robot = largest_components(robot, count=1)
    lid = np.zeros_like(robot)
    lid[15:25, 15:25] = 1
    labels = compose_surface_labels(
        robot.shape, robot_mask=robot, lid_mask=lid
    )
    assert labels[7, 7] == LABEL_ROBOT
    # The gripper is foreground.  Overlap must never turn a moving robot
    # pixel into a static target return.
    assert labels[17, 17] == LABEL_ROBOT
    assert labels[0, 0] == LABEL_BACKGROUND
    assert labels[25, 35] == LABEL_BACKGROUND


if __name__ == "__main__":
    test_alpha_overlay_recovers_mask_without_annotation()
    test_semantic_priority_and_component_filter()
    print("scene semantic checks passed")

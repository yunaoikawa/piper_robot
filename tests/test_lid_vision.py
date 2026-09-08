#!/usr/bin/env python3
"""Hardware-free tests for marker-anchored transparent-edge inspection."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.lid_vision import VisionProfile, detect_blue_marker, inspect_lid


def profile():
    return VisionProfile(
        goal_frame=82,
        marker_hsv_low=(106, 100, 60),
        marker_hsv_high=(115, 255, 255),
        marker_center=(180, 95),
        lid_ellipse=((160, 130), (105, 55), -8),
        search_band_px=16,
    )


p = profile()
image = np.full((260, 360, 3), 35, np.uint8)
blue = cv2.cvtColor(np.uint8([[[110, 230, 220]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
teal = cv2.cvtColor(np.uint8([[[95, 230, 220]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
cv2.rectangle(image, (270, 20), (350, 170), teal, -1)  # larger distractor
cv2.circle(image, (180, 95), 9, blue, -1)
cv2.ellipse(image, (160, 130), (105, 55), -8, 0, 360, (180, 180, 180), 2)

marker, _ = detect_blue_marker(image, p)
assert marker is not None and np.linalg.norm(marker - [180, 95]) < 2, marker
result = inspect_lid(image, p)
assert result["ok"], result.get("reason")
assert result["analysis_scale"] == 4

missing = image.copy()
missing[:] = 35
result = inspect_lid(missing, p)
assert not result["ok"] and result["reason"] == "blue marker not found"

print("lid vision checks passed")

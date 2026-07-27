#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from rollout.sam_segmentation import (
    MaskCandidate,
    choose_lid_candidate,
    decode_request,
    decode_response,
    enhance_low_light,
    encode_request,
    encode_response,
    map_image_point,
    mask_geometry,
)


image = np.zeros((120, 160, 3), np.uint8)
cv2.circle(image, (80, 60), 22, (180, 100, 30), -1)
request = encode_request(
    image, frame_id=7, timestamp=12.5, prompt="petri lid", jpeg_quality=100
)
metadata, decoded = decode_request(request)
assert metadata["frame_id"] == 7
assert decoded.shape == image.shape

mask = np.zeros(image.shape[:2], np.uint8)
cv2.circle(mask, (80, 60), 22, 1, -1)
candidate = MaskCandidate(mask.astype(bool), np.array([58, 38, 102, 82]), 0.91)
parts = encode_response(
    frame_id=7,
    source_timestamp=12.5,
    model="test",
    inference_ms=4.2,
    candidates=[candidate],
)
result = decode_response(parts)
assert result.frame_id == 7
assert len(result.candidates) == 1
assert np.array_equal(result.candidates[0].mask, candidate.mask)

geometry = mask_geometry(mask)
assert geometry is not None
assert np.linalg.norm(geometry.center_px - [80, 60]) < 1
assert 21 <= geometry.radius_px <= 23
assert geometry.circularity > 0.8

selected = choose_lid_candidate([candidate], image_bgr=image)
assert selected is not None
assert selected[0].score == 0.91
assert (
    choose_lid_candidate(
        [candidate], image_bgr=image, require_blue_cross=True
    )
    is None
)

# A stronger blue cross elsewhere in the full frame must not hide the smaller
# marker inside the SAM lid mask.
cross_image = np.zeros((240, 320, 3), np.uint8)
correct_mask = np.zeros(cross_image.shape[:2], np.uint8)
cv2.circle(correct_mask, (250, 170), 35, 1, -1)
wrong_mask = np.zeros(cross_image.shape[:2], np.uint8)
cv2.circle(wrong_mask, (70, 65), 38, 1, -1)
cv2.rectangle(cross_image, (35, 57), (105, 73), (255, 0, 0), -1)
cv2.rectangle(cross_image, (62, 30), (78, 100), (255, 0, 0), -1)
cv2.rectangle(cross_image, (228, 164), (272, 176), (255, 0, 0), -1)
cv2.rectangle(cross_image, (244, 148), (256, 192), (255, 0, 0), -1)
wrong_candidate = MaskCandidate(
    wrong_mask.astype(bool), np.array([32, 27, 108, 103]), 0.99
)
correct_candidate = MaskCandidate(
    correct_mask.astype(bool), np.array([215, 135, 285, 205]), 0.88
)
selected = choose_lid_candidate(
    [wrong_candidate, correct_candidate],
    image_bgr=cross_image,
    require_blue_cross=True,
)
assert selected is not None
assert selected[0] is correct_candidate

dark = np.tile(np.arange(2, 18, dtype=np.uint8), (32, 2))
dark = cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR)
enhanced = enhance_low_light(dark)
assert enhanced.shape == dark.shape
assert float(enhanced.mean()) > float(dark.mean()) * 3

H = np.array([[0.001, 0, -0.1], [0, 0.002, 0.2], [0, 0, 1]])
mapped = map_image_point(H, [100, 50])
assert np.allclose(mapped, [0.0, 0.3])

print("SAM segmentation protocol/geometry checks passed")

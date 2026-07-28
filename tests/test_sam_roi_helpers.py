#!/usr/bin/env python3

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.sam_segmentation import (
    MaskCandidate,
    SegmentationRoi,
    SegmentationResult,
    compute_candidate_roi,
    extract_enlarged_roi,
    remap_segmentation_result_from_roi,
)


def _assert_value_error(message_fragment, callback):
    try:
        callback()
    except ValueError as exc:
        assert message_fragment in str(exc), (message_fragment, str(exc))
    else:
        raise AssertionError(
            f"expected ValueError containing {message_fragment!r}"
        )


def _candidate(
    shape_hw=(8, 10),
    *,
    mask_slice=(slice(1, 5), slice(2, 7)),
    box=(2.0, 1.0, 7.0, 5.0),
    score=0.9,
):
    mask = np.zeros(shape_hw, dtype=bool)
    if mask_slice is not None:
        mask[mask_slice] = True
    return MaskCandidate(
        mask=mask,
        box_xyxy=np.asarray(box, dtype=float),
        score=score,
    )


def _result(candidate, *, timestamp=12.5, inference_ms=4.0):
    return SegmentationResult(
        frame_id=31,
        source_timestamp=timestamp,
        model="test-sam",
        inference_ms=inference_ms,
        candidates=(candidate,),
    )


def test_compute_candidate_roi_clips_and_reports_actual_scale():
    candidate = _candidate(
        (10, 12),
        mask_slice=(slice(1, 4), slice(0, 3)),
        box=(-2.0, 0.5, 5.2, 5.1),
    )
    roi = compute_candidate_roi(
        [candidate],
        full_shape_hw=(10, 12),
        padding_px=2.3,
        scale=1.6,
    )

    assert roi.crop_xyxy == (0, 0, 8, 8)
    assert roi.crop_shape_hw == (8, 8)
    assert roi.resized_shape_hw == (13, 13)
    assert roi.scale_xy == (13 / 8, 13 / 8)
    assert roi.coarse_bbox_clipped_count == 1
    assert roi.padding_clipped_to_frame is True
    assert roi.metadata() == {
        "schema": "sam_segmentation_roi/v1",
        "full_shape_hw": [10, 12],
        "crop_xyxy": [0, 0, 8, 8],
        "crop_shape_hw": [8, 8],
        "resized_shape_hw": [13, 13],
        "requested_scale": 1.6,
        "scale_xy": [13 / 8, 13 / 8],
        "padding_px": 2.3,
        "coarse_candidate_count": 1,
        "coarse_bbox_clipped_count": 1,
        "padding_clipped_to_frame": True,
    }


def test_roi_uses_union_of_boxes_and_masks_for_multiple_candidates():
    first = _candidate(
        (20, 30),
        mask_slice=(slice(3, 6), slice(4, 8)),
        box=(5.0, 4.0, 7.0, 6.0),
    )
    second = _candidate(
        (20, 30),
        mask_slice=(slice(14, 18), slice(22, 27)),
        box=(21.0, 13.0, 26.0, 17.0),
    )
    roi = compute_candidate_roi(
        [first, second],
        full_shape_hw=(20, 30),
        padding_px=1,
        scale=2,
    )
    assert roi.crop_xyxy == (3, 2, 28, 19)
    assert roi.coarse_candidate_count == 2


def test_extract_enlarged_roi_uses_declared_integer_geometry():
    coarse = _candidate()
    roi = compute_candidate_roi(
        [coarse],
        full_shape_hw=(8, 10),
        padding_px=0,
        scale=2,
    )
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    enlarged = extract_enlarged_roi(
        image,
        roi,
        interpolation=cv2.INTER_NEAREST,
    )
    expected = cv2.resize(
        image[1:5, 2:7],
        (10, 8),
        interpolation=cv2.INTER_NEAREST,
    )
    assert roi.resized_shape_hw == (8, 10)
    assert np.array_equal(enlarged, expected)


def test_remap_uses_nearest_mask_and_inverse_scale_for_bbox():
    coarse = _candidate()
    roi = compute_candidate_roi(
        [coarse],
        full_shape_hw=(8, 10),
        padding_px=0,
        scale=2,
    )
    local_mask = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    enlarged_mask = cv2.resize(
        local_mask,
        (10, 8),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    fine = MaskCandidate(
        mask=enlarged_mask,
        box_xyxy=np.array([2.0, 2.0, 8.0, 6.0]),
        score=0.87,
    )

    mapped, metadata = remap_segmentation_result_from_roi(
        _result(fine),
        roi,
    )

    expected_mask = np.zeros((8, 10), dtype=bool)
    expected_mask[1:5, 2:7] = local_mask.astype(bool)
    assert mapped.frame_id == 31
    assert mapped.model == "test-sam"
    assert len(mapped.candidates) == 1
    assert np.array_equal(mapped.candidates[0].mask, expected_mask)
    assert np.allclose(
        mapped.candidates[0].box_xyxy,
        [3.0, 2.0, 6.0, 4.0],
    )
    assert mapped.candidates[0].score == 0.87
    assert metadata["scale_xy"] == [2.0, 2.0]
    assert metadata["mask_remap_interpolation"] == "nearest"
    assert metadata["fine_candidate_count"] == 1
    assert metadata["fine_bbox_clipped_count"] == 0


def test_remap_clips_fine_bbox_to_roi_before_mapping():
    coarse = _candidate()
    roi = compute_candidate_roi(
        [coarse],
        full_shape_hw=(8, 10),
        padding_px=0,
        scale=2,
    )
    fine = MaskCandidate(
        mask=np.ones(roi.resized_shape_hw, dtype=bool),
        box_xyxy=np.array([-2.0, -3.0, 12.0, 9.0]),
        score=0.75,
    )
    mapped, metadata = remap_segmentation_result_from_roi(
        _result(fine),
        roi,
    )
    assert np.allclose(mapped.candidates[0].box_xyxy, [2.0, 1.0, 7.0, 5.0])
    assert metadata["fine_bbox_clipped_count"] == 1


def test_non_integer_resize_rounding_is_inverted_with_actual_xy_scales():
    coarse = _candidate(
        (9, 11),
        mask_slice=(slice(2, 7), slice(3, 10)),
        box=(3.0, 2.0, 10.0, 7.0),
    )
    roi = compute_candidate_roi(
        [coarse],
        full_shape_hw=(9, 11),
        padding_px=0,
        scale=1.5,
    )
    assert roi.crop_shape_hw == (5, 7)
    assert roi.resized_shape_hw == (8, 11)
    assert np.allclose(roi.scale_xy, [11 / 7, 8 / 5])
    fine = MaskCandidate(
        mask=np.ones(roi.resized_shape_hw, dtype=bool),
        box_xyxy=np.array([1.0, 2.0, 10.0, 7.0]),
        score=0.8,
    )
    mapped, _ = remap_segmentation_result_from_roi(_result(fine), roi)
    assert np.allclose(
        mapped.candidates[0].box_xyxy,
        [
            3.0 + 1.0 / (11 / 7),
            2.0 + 2.0 / (8 / 5),
            3.0 + 10.0 / (11 / 7),
            2.0 + 7.0 / (8 / 5),
        ],
    )


def test_compute_roi_rejects_invalid_geometry():
    cases = [
        ({"full_shape_hw": (0, 10)}, "positive"),
        ({"full_shape_hw": (8.5, 10)}, "integer"),
        ({"padding_px": -1}, "non-negative"),
        ({"padding_px": np.nan}, "finite"),
        ({"scale": 0}, "positive"),
        ({"scale": np.inf}, "finite"),
    ]
    for kwargs, message in cases:
        arguments = {
            "full_shape_hw": (8, 10),
            "padding_px": 0,
            "scale": 2,
        }
        arguments.update(kwargs)
        _assert_value_error(
            message,
            lambda arguments=arguments: compute_candidate_roi(
                [_candidate()], **arguments
            ),
        )


def test_compute_roi_rejects_empty_or_malformed_candidates():
    _assert_value_error(
        "at least one",
        lambda: compute_candidate_roi([], full_shape_hw=(8, 10)),
    )

    malformed = [
        MaskCandidate(
            np.zeros((7, 10), bool),
            np.array([2.0, 1.0, 7.0, 5.0]),
            0.9,
        ),
        MaskCandidate(
            np.zeros((8, 10), bool),
            np.array([2.0, 1.0, np.nan, 5.0]),
            0.9,
        ),
        MaskCandidate(
            np.zeros((8, 10), bool),
            np.array([2.0, 1.0, 2.0, 5.0]),
            0.9,
        ),
        MaskCandidate(
            np.zeros((8, 10), bool),
            np.array([20.0, 1.0, 21.0, 5.0]),
            0.9,
        ),
        MaskCandidate(
            np.zeros((8, 10), bool),
            np.array([2.0, 1.0, 7.0, 5.0]),
            np.nan,
        ),
    ]
    messages = ["shape", "finite", "positive", "intersect", "score"]
    for candidate, message in zip(malformed, messages):
        _assert_value_error(
            message,
            lambda candidate=candidate: compute_candidate_roi(
                [candidate],
                full_shape_hw=(8, 10),
                padding_px=0,
                scale=2,
            ),
        )

    nonfinite_mask = np.zeros((8, 10), dtype=float)
    nonfinite_mask[2, 3] = np.nan
    _assert_value_error(
        "non-finite",
        lambda: compute_candidate_roi(
            [
                MaskCandidate(
                    nonfinite_mask,
                    np.array([2.0, 1.0, 7.0, 5.0]),
                    0.9,
                )
            ],
            full_shape_hw=(8, 10),
        ),
    )


def test_extract_and_remap_reject_shape_and_finite_errors():
    roi = compute_candidate_roi(
        [_candidate()],
        full_shape_hw=(8, 10),
        padding_px=0,
        scale=2,
    )
    _assert_value_error(
        "does not match",
        lambda: extract_enlarged_roi(
            np.zeros((9, 10, 3), np.uint8), roi
        ),
    )

    wrong_mask = MaskCandidate(
        mask=np.zeros((7, 10), dtype=bool),
        box_xyxy=np.array([1.0, 1.0, 4.0, 4.0]),
        score=0.8,
    )
    _assert_value_error(
        "shape",
        lambda: remap_segmentation_result_from_roi(
            _result(wrong_mask), roi
        ),
    )

    outside_box = MaskCandidate(
        mask=np.zeros(roi.resized_shape_hw, dtype=bool),
        box_xyxy=np.array([20.0, 1.0, 21.0, 2.0]),
        score=0.8,
    )
    _assert_value_error(
        "intersect",
        lambda: remap_segmentation_result_from_roi(
            _result(outside_box), roi
        ),
    )

    valid = MaskCandidate(
        mask=np.zeros(roi.resized_shape_hw, dtype=bool),
        box_xyxy=np.array([1.0, 1.0, 4.0, 4.0]),
        score=0.8,
    )
    _assert_value_error(
        "source_timestamp",
        lambda: remap_segmentation_result_from_roi(
            _result(valid, timestamp=np.nan),
            roi,
        ),
    )
    _assert_value_error(
        "inference_ms",
        lambda: remap_segmentation_result_from_roi(
            _result(valid, inference_ms=-1),
            roi,
        ),
    )
    invalid_frame = SegmentationResult(
        frame_id=31.5,
        source_timestamp=12.5,
        model="test-sam",
        inference_ms=4.0,
        candidates=(valid,),
    )
    _assert_value_error(
        "frame_id",
        lambda: remap_segmentation_result_from_roi(invalid_frame, roi),
    )


def test_public_roi_dataclass_is_revalidated_at_helper_boundaries():
    roi = compute_candidate_roi(
        [_candidate()],
        full_shape_hw=(8, 10),
        padding_px=0,
        scale=2,
    )
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    invalid_rois: tuple[tuple[SegmentationRoi, str], ...] = (
        (replace(roi, full_shape_hw=(8.0, 10.0)), "two integers"),
        (replace(roi, resized_shape_hw=(8.0, 10.0)), "two integers"),
        (replace(roi, crop_xyxy=(-1, 1, 7, 5)), "outside"),
        (replace(roi, scale_xy=(2.1, 2.0)), "inconsistent"),
        (replace(roi, scale_xy=(np.nan, 2.0)), "inconsistent"),
    )
    for invalid_roi, message in invalid_rois:
        _assert_value_error(
            message,
            lambda invalid_roi=invalid_roi: extract_enlarged_roi(
                image, invalid_roi
            ),
        )


if __name__ == "__main__":
    test_compute_candidate_roi_clips_and_reports_actual_scale()
    test_roi_uses_union_of_boxes_and_masks_for_multiple_candidates()
    test_extract_enlarged_roi_uses_declared_integer_geometry()
    test_remap_uses_nearest_mask_and_inverse_scale_for_bbox()
    test_remap_clips_fine_bbox_to_roi_before_mapping()
    test_non_integer_resize_rounding_is_inverted_with_actual_xy_scales()
    test_compute_roi_rejects_invalid_geometry()
    test_compute_roi_rejects_empty_or_malformed_candidates()
    test_extract_and_remap_reject_shape_and_finite_errors()
    test_public_roi_dataclass_is_revalidated_at_helper_boundaries()
    print("SAM ROI helper checks passed")

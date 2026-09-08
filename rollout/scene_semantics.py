"""Semantic labels recovered from saved SAM overlays."""

from __future__ import annotations

import cv2
import numpy as np


LABEL_UNKNOWN = 0
LABEL_FREE = 1
LABEL_BACKGROUND = 2
LABEL_ROBOT = 3
LABEL_LID = 4

LABEL_NAMES = {
    LABEL_UNKNOWN: "unknown",
    LABEL_FREE: "observed_free",
    LABEL_BACKGROUND: "background_surface",
    LABEL_ROBOT: "robot_surface",
    LABEL_LID: "lid_surface",
}

LABEL_COLORS_RGB = {
    LABEL_UNKNOWN: (168, 85, 247),
    LABEL_FREE: (34, 197, 94),
    LABEL_BACKGROUND: (156, 163, 175),
    LABEL_ROBOT: (34, 211, 238),
    LABEL_LID: (59, 130, 246),
}


def recover_blended_sam_mask(
    source_bgr,
    overlay_bgr,
    *,
    source_weight: float,
    tint_bgr,
    tolerance: int = 1,
) -> np.ndarray:
    """Recover a lossless SAM mask from an alpha-blended diagnostic overlay.

    Bounding boxes and text are rejected because they do not satisfy the same
    alpha-blend equation as mask pixels.
    """

    source = np.asarray(source_bgr, dtype=np.uint8)
    overlay = np.asarray(overlay_bgr, dtype=np.uint8)
    if source.shape != overlay.shape or source.ndim != 3:
        raise ValueError("source and SAM overlay image shapes differ")
    tint = np.empty_like(source)
    tint[:] = np.asarray(tint_bgr, dtype=np.uint8)
    predicted = cv2.addWeighted(
        source, float(source_weight), tint, 1.0 - float(source_weight), 0
    )
    residual = np.max(
        np.abs(overlay.astype(np.int16) - predicted.astype(np.int16)), axis=2
    )
    return (residual <= int(tolerance)) & np.any(overlay != source, axis=2)


def largest_components(mask, *, count: int = 1, minimum_area: int = 20):
    binary = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    candidates = [
        label
        for label in range(1, component_count)
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area
    ]
    candidates.sort(
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True
    )
    return np.isin(labels, candidates[: int(count)])


def estimate_image_homography(source_bgr, target_bgr) -> tuple[np.ndarray, dict]:
    """Estimate source-to-target registration from static-scene ORB features."""

    source = cv2.cvtColor(np.asarray(source_bgr), cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(np.asarray(target_bgr), cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=5000, fastThreshold=8)
    source_keypoints, source_desc = detector.detectAndCompute(source, None)
    target_keypoints, target_desc = detector.detectAndCompute(target, None)
    if source_desc is None or target_desc is None:
        raise ValueError("insufficient features for saved SAM registration")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(source_desc, target_desc, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
    ]
    if len(good) < 20:
        raise ValueError("too few matches for saved SAM registration")
    source_points = np.float32(
        [source_keypoints[match.queryIdx].pt for match in good]
    )
    target_points = np.float32(
        [target_keypoints[match.trainIdx].pt for match in good]
    )
    homography, inliers = cv2.findHomography(
        source_points, target_points, cv2.RANSAC, 1.5
    )
    if homography is None or inliers is None:
        raise ValueError("saved SAM homography fit failed")
    inlier_mask = inliers.ravel().astype(bool)
    projected = cv2.perspectiveTransform(
        source_points.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    residual = np.linalg.norm(projected - target_points, axis=1)
    report = {
        "matches": int(len(good)),
        "inliers": int(np.count_nonzero(inlier_mask)),
        "median_inlier_residual_px": float(np.median(residual[inlier_mask])),
    }
    return homography, report


def warp_mask(mask, homography, target_shape) -> np.ndarray:
    height, width = target_shape[:2]
    return (
        cv2.warpPerspective(
            np.asarray(mask, dtype=np.uint8),
            np.asarray(homography, dtype=float),
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        > 0
    )


def compose_surface_labels(
    shape, *, robot_mask=None, lid_mask=None
) -> np.ndarray:
    """Compose mutually-exclusive surface labels.

    The robot is foreground and therefore wins an overlap with a target mask.
    This is deliberately conservative: an occluded target pixel must not bake a
    moving gripper into the static scene map.
    """

    labels = np.full(shape[:2], LABEL_BACKGROUND, dtype=np.uint8)
    if lid_mask is not None:
        labels[np.asarray(lid_mask, dtype=bool)] = LABEL_LID
    if robot_mask is not None:
        labels[np.asarray(robot_mask, dtype=bool)] = LABEL_ROBOT
    return labels


def render_semantic_labels(image_bgr, labels) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.shape != image.shape[:2]:
        raise ValueError("semantic labels and image shapes differ")
    tint = np.zeros_like(image)
    for label, color_rgb in LABEL_COLORS_RGB.items():
        tint[labels == label] = color_rgb[::-1]
    out = cv2.addWeighted(image, 0.45, tint, 0.55, 0)
    legend = (
        (LABEL_BACKGROUND, "background"),
        (LABEL_ROBOT, "robot (SAM)"),
        (LABEL_LID, "lid (SAM)"),
    )
    x = 16
    for label, name in legend:
        color = LABEL_COLORS_RGB[label][::-1]
        cv2.rectangle(out, (x, 14), (x + 22, 36), color, -1)
        cv2.putText(
            out,
            name,
            (x + 28, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        x += 155
    return out

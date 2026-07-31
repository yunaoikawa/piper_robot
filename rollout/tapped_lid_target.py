"""Operator-tapped lid identity and fixed-head camera validation.

The tap is the only semantic input needed by the fast grasp path.  It selects
the connected blue-marker component nearest the tap; component area never
selects identity.  Coordinates and all gates are resolution independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import cv2
import numpy as np


@dataclass(frozen=True)
class TappedTarget:
    uv: tuple[float, float]
    frame_sha256: str
    frame_timestamp: float

    def __post_init__(self):
        if len(self.uv) != 2 or not all(0.0 <= float(v) <= 1.0 for v in self.uv):
            raise ValueError("tap uv must lie in [0, 1]")
        if len(self.frame_sha256) != 64:
            raise ValueError("tap must identify an exact frame sha256")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TargetAssociation:
    center_px: tuple[float, float]
    center_uv: tuple[float, float]
    area_fraction: float
    tap_distance_diagonal_fraction: float
    component_mask: np.ndarray


def frame_sha256(image_bgr: np.ndarray) -> str:
    image = np.ascontiguousarray(image_bgr)
    return hashlib.sha256(image.tobytes()).hexdigest()


def validate_tap_frame(
    image_bgr: np.ndarray,
    tap: TappedTarget,
    *,
    frame_timestamp: float,
    now: float,
    maximum_age_s: float = 2.0,
    minimum_p99_brightness: float = 20.0,
) -> None:
    if frame_sha256(image_bgr) != tap.frame_sha256:
        raise ValueError("tap refers to a different camera frame")
    if abs(float(frame_timestamp) - float(tap.frame_timestamp)) > 1e-3:
        raise ValueError("tap timestamp does not match camera frame")
    age = float(now) - float(frame_timestamp)
    if age < -0.1 or age > maximum_age_s:
        raise ValueError(f"camera frame is stale: age={age:.3f}s")
    if float(np.percentile(image_bgr, 99)) < minimum_p99_brightness:
        raise ValueError("camera frame is too dark")


def associate_blue_component(
    image_bgr: np.ndarray,
    tap_uv,
    *,
    hsv_low=(100, 70, 35),
    hsv_high=(125, 255, 255),
    maximum_tap_distance_diagonal_fraction: float = 0.045,
    minimum_area_fraction: float = 2e-5,
    maximum_area_fraction: float = 0.03,
) -> TargetAssociation:
    """Select the component containing/nearest the tap, never the largest one."""

    image = np.asarray(image_bgr)
    height, width = image.shape[:2]
    tap = np.asarray(tap_uv, dtype=float).reshape(2)
    if not np.all((0 <= tap) & (tap <= 1)):
        raise ValueError("tap uv must lie in [0, 1]")
    tap_px = tap * np.array([width, height], dtype=float)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(hsv_low, np.uint8), np.asarray(hsv_high, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    diagonal = float(np.hypot(width, height))
    candidates = []
    for label in range(1, count):
        area_fraction = float(stats[label, cv2.CC_STAT_AREA]) / float(width * height)
        if not minimum_area_fraction <= area_fraction <= maximum_area_fraction:
            continue
        distance = float(np.linalg.norm(centers[label] - tap_px)) / diagonal
        candidates.append((distance, label, area_fraction))
    if not candidates:
        raise ValueError("no blue marker component exists near the tap")
    distance, label, area_fraction = min(candidates, key=lambda item: item[0])
    if distance > maximum_tap_distance_diagonal_fraction:
        raise ValueError(
            "nearest blue marker is too far from tap: "
            f"distance={distance:.4f} image diagonal"
        )
    center = np.asarray(centers[label], dtype=float)
    return TargetAssociation(
        center_px=(float(center[0]), float(center[1])),
        center_uv=(float(center[0] / width), float(center[1] / height)),
        area_fraction=area_fraction,
        tap_distance_diagonal_fraction=distance,
        component_mask=labels == label,
    )


@dataclass(frozen=True)
class HeadRegistration:
    accepted: bool
    matches: int
    inlier_fraction: float
    median_residual_diagonal_fraction: float
    homography: np.ndarray | None


def register_fixed_head(
    reference_bgr: np.ndarray,
    current_bgr: np.ndarray,
    *,
    minimum_matches: int = 35,
    minimum_inlier_fraction: float = 0.45,
    maximum_median_residual_diagonal_fraction: float = 0.004,
) -> HeadRegistration:
    """Check that the head camera has not moved using background ORB/RANSAC."""

    if reference_bgr.shape[:2] != current_bgr.shape[:2]:
        return HeadRegistration(False, 0, 0.0, float("inf"), None)
    gray_a = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2500)
    key_a, des_a = orb.detectAndCompute(gray_a, None)
    key_b, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None:
        return HeadRegistration(False, 0, 0.0, float("inf"), None)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(good) < 4:
        return HeadRegistration(False, len(good), 0.0, float("inf"), None)
    source = np.float32([key_a[m.queryIdx].pt for m in good])
    target = np.float32([key_b[m.trainIdx].pt for m in good])
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or inliers is None:
        return HeadRegistration(False, len(good), 0.0, float("inf"), None)
    selected = inliers.ravel().astype(bool)
    projected = cv2.perspectiveTransform(source[:, None, :], homography)[:, 0, :]
    residual = np.linalg.norm(projected[selected] - target[selected], axis=1)
    diagonal = float(np.hypot(*reference_bgr.shape[:2]))
    fraction = float(np.mean(selected))
    median = float(np.median(residual) / diagonal) if residual.size else float("inf")
    accepted = bool(
        len(good) >= minimum_matches
        and fraction >= minimum_inlier_fraction
        and median <= maximum_median_residual_diagonal_fraction
    )
    return HeadRegistration(accepted, len(good), fraction, median, homography)

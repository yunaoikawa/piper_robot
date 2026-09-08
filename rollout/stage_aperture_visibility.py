"""Scale-aware visibility gate for a user-selected workspace aperture.

The reference observation establishes the target in a nearby fixed AprilTag
frame once.  At runtime the tag is only a geometric anchor: the target is
declared visible only when its projected neighborhood also contains a dark,
elongated aperture and is not covered by the blue gripper.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rollout.apriltag_retarget import estimate_tag_camera_pose


@dataclass(frozen=True)
class ApertureVisibility:
    state: str
    projected_uv: tuple[float, float] | None
    observed_uv: tuple[float, float] | None
    anchor_rms_px: float | None
    blue_occlusion_fraction: float
    appearance_score: float
    reason: str

    @property
    def visible(self) -> bool:
        return self.state == "target_visible"


def project_tag_point(tag, camera_matrix, tag_size_m, point_tag_xyz):
    """Project a calibrated tag-frame point into the current camera image."""

    rvec, tvec, rms = estimate_tag_camera_pose(
        tag, camera_matrix, tag_size_m
    )
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    camera_xyz = rotation @ np.asarray(point_tag_xyz, dtype=float) + tvec
    if not np.all(np.isfinite(camera_xyz)) or float(camera_xyz[2]) <= 0.0:
        raise ValueError("calibrated aperture lies behind the camera")
    homogeneous = np.asarray(camera_matrix, dtype=float) @ camera_xyz
    uv = homogeneous[:2] / homogeneous[2]
    return uv, camera_xyz, float(rms)


def refine_tag_point_from_pixel(
    tag,
    camera_matrix,
    tag_size_m,
    prior_point_tag_xyz,
    confirmed_uv,
):
    """Refine a target ray from a confirmed pixel while retaining metric depth.

    A wrist RGB camera has no depth.  The prior 3D estimate supplies only the
    camera-Z distance; the operator-confirmed pixel replaces its uncertain ray.
    Subsequent frames use the resulting tag-frame point without a fixed pixel.
    """

    matrix = np.asarray(camera_matrix, dtype=float)
    rvec, tvec, rms = estimate_tag_camera_pose(tag, matrix, tag_size_m)
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    prior_camera = rotation @ np.asarray(prior_point_tag_xyz, dtype=float) + tvec
    depth = float(prior_camera[2])
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("prior target depth is invalid")
    uv = np.asarray(confirmed_uv, dtype=float).reshape(2)
    if not np.all(np.isfinite(uv)):
        raise ValueError("confirmed target pixel must be finite")
    camera_point = np.asarray(
        [
            (uv[0] - matrix[0, 2]) * depth / matrix[0, 0],
            (uv[1] - matrix[1, 2]) * depth / matrix[1, 1],
            depth,
        ],
        dtype=float,
    )
    refined_tag_point = rotation.T @ (camera_point - tvec)
    return refined_tag_point, depth, float(rms)


def _blue_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # The physical gripper ranges from cyan to saturated blue as exposure
    # changes.  This mask is used only to reject occlusion, never as a target.
    first = cv2.inRange(hsv, np.asarray([78, 55, 35]), np.asarray([112, 255, 255]))
    second = cv2.inRange(hsv, np.asarray([113, 75, 30]), np.asarray([132, 255, 255]))
    return cv2.bitwise_or(first, second)


def assess_aperture_visibility(
    image_bgr: np.ndarray,
    *,
    projected_uv,
    anchor_perimeter_px: float,
    anchor_rms_px: float | None = None,
    minimum_border_tag_ratio: float = 0.35,
    maximum_blue_occlusion: float = 0.18,
    minimum_appearance_score: float = 0.42,
    search_radius_tag_ratio: float = 0.42,
) -> ApertureVisibility:
    """Confirm an elongated dark opening near a metric projection.

    All neighborhood sizes derive from the observed tag scale.  This avoids a
    camera-resolution-specific pixel threshold and transfers to another lab
    after one target click plus one known-size local anchor.
    """

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must have shape HxWx3")
    uv = np.asarray(projected_uv, dtype=float).reshape(2)
    height, width = image.shape[:2]
    border = max(2.0, float(anchor_perimeter_px) * minimum_border_tag_ratio)
    if (
        uv[0] < border
        or uv[0] >= width - border
        or uv[1] < border
        or uv[1] >= height - border
    ):
        return ApertureVisibility(
            "target_out_of_frame", tuple(uv), None, anchor_rms_px, 0.0, 0.0,
            "projected aperture lacks a full verification neighborhood",
        )

    # After the one-time target click has been refined into the tag frame, the
    # ROI can stay local.  The ratio remains configurable for a coarse first
    # pass, while the image shape (not the projection) chooses the final center.
    if not 0.2 <= search_radius_tag_ratio <= 1.0:
        raise ValueError("search_radius_tag_ratio must be within [0.2, 1.0]")
    radius = int(
        round(
            np.clip(
                float(anchor_perimeter_px) * search_radius_tag_ratio,
                16.0,
                140.0,
            )
        )
    )
    cx, cy = np.rint(uv).astype(int)
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    patch = image[y0:y1, x0:x1]
    blue = _blue_mask(patch)
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    disk = (xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2 <= radius ** 2
    blue_fraction = float(np.count_nonzero((blue > 0) & disk) / max(1, np.count_nonzero(disk)))
    if blue_fraction > maximum_blue_occlusion:
        return ApertureVisibility(
            "target_predicted_but_occluded", tuple(uv), None, anchor_rms_px,
            blue_fraction, 0.0, "blue gripper covers the projected aperture",
        )

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), max(1.0, radius / 24.0))
    local_values = smooth[disk]
    threshold = min(
        float(np.percentile(local_values, 10.0)),
        float(np.median(local_values) - max(4.0, 0.35 * np.std(local_values))),
    )
    dark = np.zeros_like(gray, dtype=np.uint8)
    dark[(smooth <= threshold) & disk] = 255
    kernel_size = max(3, int(round(radius / 18.0)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    best = None
    for contour in cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        if area <= 0.004 * np.pi * radius**2 or area >= 0.48 * np.pi * radius**2:
            continue
        rect = cv2.minAreaRect(contour)
        short, long = sorted(float(value) for value in rect[1])
        if short <= 1e-6:
            continue
        aspect = long / short
        if not 1.25 <= aspect <= 7.0:
            continue
        center = np.asarray(rect[0], dtype=float)
        distance = float(np.linalg.norm(center - np.asarray([cx - x0, cy - y0])))
        contour_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        ring_width = max(5, int(round(radius * 0.06)) | 1)
        inner_ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ring_width, ring_width)
        )
        outer_width = ring_width * 3
        outer_ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (outer_width, outer_width)
        )
        inner_ring = cv2.dilate(contour_mask, inner_ring_kernel) > 0
        outer_ring = cv2.dilate(contour_mask, outer_ring_kernel) > 0
        ring = outer_ring & ~inner_ring
        inside_values = gray[contour_mask > 0]
        ring_values = gray[ring]
        if inside_values.size == 0 or ring_values.size < 8:
            continue
        contrast = float(np.median(ring_values) - np.median(inside_values))
        if contrast < max(3.0, 0.04 * float(np.median(ring_values))):
            continue
        # A true stage aperture is surrounded by one locally uniform stage
        # surface.  A shadow beneath the microscope bridge has a high-variance
        # ring containing both the bright bridge and the dark background.
        ring_scale = max(12.0, 0.50 * float(np.median(ring_values)))
        uniform_ring_score = max(
            0.0, 1.0 - float(np.std(ring_values)) / ring_scale
        )
        distance_score = max(0.0, 1.0 - distance / max(radius * 0.92, 1.0))
        aspect_score = min(1.0, (aspect - 1.0) / 1.5)
        area_score = min(1.0, area / max(0.08 * np.pi * radius**2, 1.0))
        score = (
            0.25 * distance_score
            + 0.15 * aspect_score
            + 0.15 * area_score
            + 0.45 * uniform_ring_score
        )
        candidate = (score, center, contour)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None or float(best[0]) < minimum_appearance_score:
        return ApertureVisibility(
            "target_predicted_not_confirmed", tuple(uv), None, anchor_rms_px,
            blue_fraction, 0.0 if best is None else float(best[0]),
            "no scale-compatible elongated dark opening near the projection",
        )

    observed = best[1] + np.asarray([x0, y0], dtype=float)
    return ApertureVisibility(
        "target_visible", tuple(uv), tuple(observed), anchor_rms_px,
        blue_fraction, float(best[0]), "projected and appearance-confirmed aperture",
    )


def render_aperture_visibility(image_bgr: np.ndarray, result: ApertureVisibility):
    out = np.asarray(image_bgr).copy()
    color = (0, 255, 255) if result.visible else (0, 0, 255)
    if result.projected_uv is not None:
        center = tuple(np.rint(result.projected_uv).astype(int))
        if 0 <= center[0] < out.shape[1] and 0 <= center[1] < out.shape[0]:
            cv2.circle(out, center, 18, color, 3)
            cv2.drawMarker(out, center, color, cv2.MARKER_CROSS, 30, 3)
    if result.observed_uv is not None:
        observed = tuple(np.rint(result.observed_uv).astype(int))
        cv2.circle(out, observed, 8, (0, 255, 0), 3)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 54), (0, 0, 0), -1)
    cv2.putText(
        out, result.state, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
    )
    cv2.putText(
        out,
        f"blue={result.blue_occlusion_fraction:.2f} shape={result.appearance_score:.2f}",
        (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
    )
    return out

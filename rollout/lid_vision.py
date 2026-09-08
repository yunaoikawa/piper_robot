"""Marker-anchored transparent lid-edge inspection for replay checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VisionProfile:
    goal_frame: int
    marker_hsv_low: tuple[int, int, int]
    marker_hsv_high: tuple[int, int, int]
    marker_center: tuple[float, float]
    lid_ellipse: tuple[tuple[float, float], tuple[float, float], float]
    roi_padding: int = 24
    search_band_px: int = 18

    @classmethod
    def load(cls, path: str | Path) -> "VisionProfile":
        cfg = json.loads(Path(path).read_text())
        ellipse = cfg["right_goal"]["lid_ellipse"]
        return cls(
            goal_frame=int(cfg["goal_frame"]),
            marker_hsv_low=tuple(cfg["marker_hsv"]["low"]),
            marker_hsv_high=tuple(cfg["marker_hsv"]["high"]),
            marker_center=tuple(cfg["right_goal"]["marker_center"]),
            lid_ellipse=(tuple(ellipse["center"]), tuple(ellipse["axes"]), float(ellipse["angle_deg"])),
            roi_padding=int(cfg.get("roi_padding", 24)),
            search_band_px=int(cfg.get("search_band_px", 18)),
        )


def detect_blue_marker(image_bgr: np.ndarray, profile: VisionProfile):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(profile.marker_hsv_low), np.array(profile.marker_hsv_high))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    expected = np.asarray(profile.marker_center)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if 8 <= area <= 2500:
            center = centers[label]
            candidates.append((float(np.linalg.norm(center - expected)), -area, center))
    if not candidates:
        return None, mask
    return min(candidates, key=lambda item: (item[0], item[1]))[2], mask


def expected_ellipse(profile: VisionProfile, marker_center):
    delta = np.asarray(marker_center) - np.asarray(profile.marker_center)
    center, axes, angle = profile.lid_ellipse
    shifted = tuple((np.asarray(center) + delta).tolist())
    return shifted, axes, angle


def _ellipse_band(shape, ellipse, width):
    outer = np.zeros(shape[:2], np.uint8)
    inner = np.zeros(shape[:2], np.uint8)
    center, axes, angle = ellipse
    center_i = tuple(np.rint(center).astype(int))
    axes_i = tuple(np.rint(axes).astype(int))
    cv2.ellipse(outer, center_i, axes_i, angle, 0, 360, 255, width * 2 + 1)
    cv2.ellipse(inner, center_i, axes_i, angle, 0, 360, 255, max(1, width // 2))
    return cv2.bitwise_or(outer, inner)


def inspect_lid(image_bgr: np.ndarray, profile: VisionProfile) -> dict:
    """Detect the marker, then fit edges only near the registered teacher ellipse."""
    marker, marker_mask = detect_blue_marker(image_bgr, profile)
    if marker is None:
        return {"ok": False, "reason": "blue marker not found", "marker_mask": marker_mask}
    expected = expected_ellipse(profile, marker)
    center, axes, angle = expected
    pad = profile.roi_padding + profile.search_band_px
    height, width = image_bgr.shape[:2]
    x0 = max(0, int(center[0] - axes[0] - pad))
    y0 = max(0, int(center[1] - axes[1] - pad))
    x1 = min(width, int(center[0] + axes[0] + pad + 1))
    y1 = min(height, int(center[1] + axes[1] + pad + 1))
    roi = image_bgr[y0:y1, x0:x1]
    scale = 4
    enlarged = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    local_expected = (
        ((center[0] - x0) * scale, (center[1] - y0) * scale),
        (axes[0] * scale, axes[1] * scale),
        angle,
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(gray, 35, 100)
    band = _ellipse_band(enlarged.shape, local_expected, profile.search_band_px * scale)
    teal = cv2.inRange(cv2.cvtColor(enlarged, cv2.COLOR_BGR2HSV), (85, 60, 40), (105, 255, 255))
    usable = cv2.bitwise_and(edges, band)
    usable[cv2.dilate(teal, np.ones((15, 15), np.uint8)) > 0] = 0
    points = np.column_stack(np.nonzero(usable))[:, ::-1].astype(np.float32)
    fitted = None
    reason = "not enough edge points"
    if len(points) >= 20:
        candidate = cv2.fitEllipse(points.reshape(-1, 1, 2))
        (cx, cy), (diam_a, diam_b), angle = candidate
        if diam_b > diam_a:
            angle += 90.0
        angle = ((angle + 90.0) % 180.0) - 90.0
        candidate_axes = np.sort([diam_a / (2.0 * scale), diam_b / (2.0 * scale)])[::-1]
        expected_axes = np.sort(np.asarray(axes))[::-1]
        fitted_center = np.array([cx / scale + x0, cy / scale + y0])
        center_error = float(np.linalg.norm(fitted_center - np.asarray(center)))
        axis_error = float(np.max(np.abs(candidate_axes - expected_axes) / expected_axes))
        if center_error <= profile.search_band_px and axis_error <= 0.30:
            fitted = (tuple(fitted_center), tuple(candidate_axes), angle)
            reason = None
        else:
            reason = f"ellipse mismatch center={center_error:.1f}px axes={axis_error:.0%}"
    return {
        "ok": fitted is not None,
        "reason": reason,
        "marker": marker,
        "expected": expected,
        "fitted": fitted,
        "edge_points": int(len(points)),
        "marker_mask": marker_mask,
        "edges": usable,
        "roi": [x0, y0, x1, y1],
        "analysis_scale": scale,
    }


def render_inspection(image_bgr: np.ndarray, result: dict, title: str = "RIGHT") -> np.ndarray:
    out = image_bgr.copy()
    if result.get("marker") is not None:
        cv2.circle(out, tuple(np.rint(result["marker"]).astype(int)), 18, (0, 255, 255), 3)
    if result.get("expected") is not None:
        center, axes, angle = result["expected"]
        cv2.ellipse(out, tuple(np.rint(center).astype(int)), tuple(np.rint(axes).astype(int)),
                    angle, 0, 360, (0, 0, 255), 4)
    if result.get("fitted") is not None:
        center, axes, angle = result["fitted"]
        cv2.ellipse(out, tuple(np.rint(center).astype(int)), tuple(np.rint(axes).astype(int)),
                    angle, 0, 360, (0, 255, 0), 2)
    status = "EDGE OK" if result.get("ok") else result.get("reason", "EDGE UNCERTAIN")
    cv2.putText(out, f"{title}: {status}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 3)
    cv2.putText(out, f"{title}: {status}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 1)
    return out

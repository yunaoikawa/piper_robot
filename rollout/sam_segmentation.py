"""Remote SAM segmentation protocol and lid-mask geometry helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class MaskCandidate:
    mask: np.ndarray
    box_xyxy: np.ndarray
    score: float


@dataclass(frozen=True)
class SegmentationResult:
    frame_id: int
    source_timestamp: float
    model: str
    inference_ms: float
    candidates: tuple[MaskCandidate, ...]


@dataclass(frozen=True)
class LidMaskGeometry:
    center_px: np.ndarray
    radius_px: float
    contour: np.ndarray
    area_px: float
    circularity: float


def encode_request(
    image_bgr: np.ndarray,
    *,
    frame_id: int,
    timestamp: float,
    prompt: str,
    confidence_threshold: float = 0.25,
    jpeg_quality: int = 90,
) -> list[bytes]:
    ok, jpeg = cv2.imencode(
        ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    )
    if not ok:
        raise ValueError("could not JPEG-encode segmentation frame")
    metadata = {
        "version": PROTOCOL_VERSION,
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "prompt": str(prompt),
        "confidence_threshold": float(confidence_threshold),
    }
    return [json.dumps(metadata).encode("utf-8"), jpeg.tobytes()]


def decode_request(parts: list[bytes]) -> tuple[dict, np.ndarray]:
    if len(parts) != 2:
        raise ValueError(f"expected 2 request parts, got {len(parts)}")
    metadata = json.loads(parts[0])
    if metadata.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {metadata.get('version')}")
    image = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid JPEG request image")
    return metadata, image


def encode_response(
    *,
    frame_id: int,
    source_timestamp: float,
    model: str,
    inference_ms: float,
    candidates: Iterable[MaskCandidate],
) -> list[bytes]:
    encoded_masks = []
    candidate_metadata = []
    for candidate in candidates:
        mask_u8 = (np.asarray(candidate.mask, dtype=bool) * 255).astype(np.uint8)
        ok, png = cv2.imencode(".png", mask_u8)
        if not ok:
            raise ValueError("could not PNG-encode segmentation mask")
        encoded_masks.append(png.tobytes())
        candidate_metadata.append(
            {
                "box_xyxy": np.asarray(candidate.box_xyxy, dtype=float).tolist(),
                "score": float(candidate.score),
            }
        )
    metadata = {
        "version": PROTOCOL_VERSION,
        "status": "ok",
        "frame_id": int(frame_id),
        "source_timestamp": float(source_timestamp),
        "model": str(model),
        "inference_ms": float(inference_ms),
        "candidates": candidate_metadata,
    }
    return [json.dumps(metadata).encode("utf-8"), *encoded_masks]


def encode_error(frame_id: int, message: str) -> list[bytes]:
    metadata = {
        "version": PROTOCOL_VERSION,
        "status": "error",
        "frame_id": int(frame_id),
        "message": str(message),
    }
    return [json.dumps(metadata).encode("utf-8")]


def decode_response(parts: list[bytes]) -> SegmentationResult:
    if not parts:
        raise ValueError("empty segmentation response")
    metadata = json.loads(parts[0])
    if metadata.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {metadata.get('version')}")
    if metadata.get("status") != "ok":
        raise RuntimeError(metadata.get("message", "segmentation server error"))
    descriptions = metadata.get("candidates", [])
    if len(parts) != len(descriptions) + 1:
        raise ValueError("candidate metadata/mask count mismatch")
    candidates = []
    for description, encoded_mask in zip(descriptions, parts[1:]):
        mask = cv2.imdecode(
            np.frombuffer(encoded_mask, np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise ValueError("invalid PNG response mask")
        candidates.append(
            MaskCandidate(
                mask=mask > 0,
                box_xyxy=np.asarray(description["box_xyxy"], dtype=float),
                score=float(description["score"]),
            )
        )
    return SegmentationResult(
        frame_id=int(metadata["frame_id"]),
        source_timestamp=float(metadata["source_timestamp"]),
        model=str(metadata["model"]),
        inference_ms=float(metadata["inference_ms"]),
        candidates=tuple(candidates),
    )


class SamSegmentationClient:
    """Synchronous latest-frame client; callers decide when the robot may move."""

    def __init__(self, endpoint: str, timeout_ms: int = 5000):
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("pyzmq is required for SAM segmentation") from exc
        self._zmq = zmq
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.connect(endpoint)
        self.last_frame_id = -1

    def close(self):
        self.socket.close(linger=0)

    def segment(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: int,
        timestamp: float | None = None,
        prompt: str = "petri dish lid",
        confidence_threshold: float = 0.25,
    ) -> SegmentationResult:
        timestamp = time.time() if timestamp is None else float(timestamp)
        if frame_id <= self.last_frame_id:
            raise ValueError(
                f"frame_id must increase: {frame_id} <= {self.last_frame_id}"
            )
        self.socket.send_multipart(
            encode_request(
                image_bgr,
                frame_id=frame_id,
                timestamp=timestamp,
                prompt=prompt,
                confidence_threshold=confidence_threshold,
            )
        )
        result = decode_response(self.socket.recv_multipart())
        if result.frame_id != frame_id:
            raise RuntimeError(
                f"stale segmentation response {result.frame_id}, expected {frame_id}"
            )
        self.last_frame_id = frame_id
        return result


def mask_geometry(mask: np.ndarray, min_area_px: int = 200) -> LidMaskGeometry | None:
    binary = (np.asarray(mask, dtype=bool) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area_px:
        return None
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return None
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    return LidMaskGeometry(
        center_px=np.array([center_x, center_y], dtype=float),
        radius_px=float(radius),
        contour=contour.reshape(-1, 2),
        area_px=area,
        circularity=circularity,
    )


def detect_blue_cross_center(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return the most cross-shaped blue component, rejecting teal gripper blobs."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 80, 50]), np.array([125, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not 12 <= area <= 1800:
            continue
        x, y, width, height = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        aspect = width / max(height, 1)
        fill = area / max(width * height, 1)
        if not 0.55 <= aspect <= 1.8 or not 0.18 <= fill <= 0.78:
            continue
        component = labels[y : y + height, x : x + width] == label
        cx = int(np.clip(round(centers[label][0] - x), 0, width - 1))
        cy = int(np.clip(round(centers[label][1] - y), 0, height - 1))
        band_x = max(1, width // 5)
        band_y = max(1, height // 5)
        hspan = np.count_nonzero(
            np.any(
                component[
                    max(0, cy - band_y) : min(height, cy + band_y + 1), :
                ],
                axis=0,
            )
        ) / width
        vspan = np.count_nonzero(
            np.any(
                component[
                    :, max(0, cx - band_x) : min(width, cx + band_x + 1)
                ],
                axis=1,
            )
        ) / height
        score = min(float(hspan), float(vspan))
        if score >= 0.72:
            candidates.append((score, area, np.asarray(centers[label], dtype=float)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def choose_lid_candidate(
    candidates: Iterable[MaskCandidate],
    *,
    image_bgr: np.ndarray,
    previous_center_px: np.ndarray | None = None,
) -> tuple[MaskCandidate, LidMaskGeometry] | None:
    blue = detect_blue_cross_center(image_bgr)
    ranked = []
    for candidate in candidates:
        geometry = mask_geometry(candidate.mask)
        if geometry is None or geometry.circularity < 0.20:
            continue
        height, width = candidate.mask.shape[:2]
        area_fraction = geometry.area_px / float(height * width)
        if not 0.002 <= area_fraction <= 0.35:
            continue
        cross_penalty = 0.0
        contains_cross = False
        if blue is not None:
            bx, by = np.rint(blue).astype(int)
            if 0 <= by < height and 0 <= bx < width:
                contains_cross = bool(candidate.mask[by, bx])
            cross_penalty = 0.0 if contains_cross else 500.0
        temporal_distance = (
            0.0
            if previous_center_px is None
            else float(np.linalg.norm(geometry.center_px - previous_center_px))
        )
        rank = (
            cross_penalty,
            temporal_distance,
            -float(candidate.score),
            -geometry.circularity,
        )
        ranked.append((rank, candidate, geometry))
    if not ranked:
        return None
    _, candidate, geometry = min(ranked, key=lambda item: item[0])
    return candidate, geometry


def map_image_point(homography: np.ndarray, point_px: np.ndarray) -> np.ndarray:
    point = np.asarray(point_px, dtype=float).reshape(1, 1, 2)
    return cv2.perspectiveTransform(point, np.asarray(homography, dtype=float))[0, 0]


def render_segmentation(
    image_bgr: np.ndarray,
    candidate: MaskCandidate | None,
    geometry: LidMaskGeometry | None,
    *,
    label: str,
) -> np.ndarray:
    out = image_bgr.copy()
    if candidate is not None:
        color = np.zeros_like(out)
        color[:, :] = (0, 180, 255)
        mask = np.asarray(candidate.mask, dtype=bool)
        out[mask] = cv2.addWeighted(out[mask], 0.45, color[mask], 0.55, 0)
    if geometry is not None:
        center = tuple(np.rint(geometry.center_px).astype(int))
        cv2.circle(out, center, int(round(geometry.radius_px)), (0, 255, 0), 3)
        cv2.drawMarker(out, center, (0, 255, 255), cv2.MARKER_CROSS, 28, 3)
    cv2.putText(
        out,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        3,
    )
    cv2.putText(
        out,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        1,
    )
    return out

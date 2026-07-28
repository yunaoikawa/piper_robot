"""Remote SAM segmentation protocol and lid-mask geometry helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np


PROTOCOL_VERSION = 1
MAX_SEGMENTATION_ROI_CANDIDATES = 64
MAX_SEGMENTATION_ROI_RESIZED_DIMENSION = 4096
MAX_SEGMENTATION_ROI_RESIZED_PIXELS = 16_777_216


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
class SegmentationRoi:
    """An integer full-frame crop and its exact resized-image transform.

    ``crop_xyxy`` is a half-open ``[x0, y0, x1, y1]`` interval in the full
    frame.  ``scale_xy`` records the *actual* resize ratios after rounding the
    requested output size, so the same transform can be inverted exactly for
    both masks and floating-point boxes.
    """

    full_shape_hw: tuple[int, int]
    crop_xyxy: tuple[int, int, int, int]
    resized_shape_hw: tuple[int, int]
    requested_scale: float
    scale_xy: tuple[float, float]
    padding_px: float
    coarse_candidate_count: int
    coarse_bbox_clipped_count: int
    padding_clipped_to_frame: bool

    @property
    def crop_shape_hw(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.crop_xyxy
        return y1 - y0, x1 - x0

    def metadata(self) -> dict[str, object]:
        """Return a JSON-serializable description of the ROI transform."""

        _validate_segmentation_roi(self)
        return {
            "schema": "sam_segmentation_roi/v1",
            "full_shape_hw": list(self.full_shape_hw),
            "crop_xyxy": list(self.crop_xyxy),
            "crop_shape_hw": list(self.crop_shape_hw),
            "resized_shape_hw": list(self.resized_shape_hw),
            "requested_scale": self.requested_scale,
            "scale_xy": list(self.scale_xy),
            "padding_px": self.padding_px,
            "coarse_candidate_count": self.coarse_candidate_count,
            "coarse_bbox_clipped_count": self.coarse_bbox_clipped_count,
            "padding_clipped_to_frame": self.padding_clipped_to_frame,
        }


@dataclass(frozen=True)
class LidMaskGeometry:
    center_px: np.ndarray
    radius_px: float
    contour: np.ndarray
    area_px: float
    circularity: float


def enhance_low_light(image_bgr: np.ndarray) -> np.ndarray:
    """Contrast-stretch a dark Record3D frame without discarding its colour."""

    image = np.asarray(image_bgr, dtype=np.uint8)
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(luminance, [1.0, 99.5])
    if high <= low + 2:
        return image.copy()
    scale = 255.0 / float(high - low)
    stretched = np.clip(
        (image.astype(np.float32) - float(low)) * scale, 0, 255
    ).astype(np.uint8)
    lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _validated_shape_hw(
    shape_hw: tuple[int, int] | list[int] | np.ndarray,
    *,
    name: str,
) -> tuple[int, int]:
    shape = np.asarray(shape_hw)
    if shape.shape != (2,):
        raise ValueError(f"{name} must contain exactly [height, width]")
    if shape.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric")
    values = shape.astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{name} must contain integer dimensions")
    height, width = (int(value) for value in values)
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return height, width


def _validated_mask(
    mask: np.ndarray,
    *,
    expected_shape_hw: tuple[int, int],
    name: str,
) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    if array.shape != expected_shape_hw:
        raise ValueError(
            f"{name} shape {array.shape} does not match {expected_shape_hw}"
        )
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if array.dtype.kind not in "buif":
        raise ValueError(f"{name} must have a boolean or real numeric dtype")
    return array.astype(bool, copy=False)


def _validated_box_xyxy(
    box_xyxy: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    raw_box = np.asarray(box_xyxy)
    if raw_box.shape != (4,):
        raise ValueError(f"{name} must contain exactly [x0, y0, x1, y1]")
    if raw_box.dtype.kind not in "uif":
        raise ValueError(f"{name} must have a real numeric dtype")
    box = raw_box.astype(float)
    if not np.all(np.isfinite(box)):
        raise ValueError(f"{name} must be finite")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{name} must have positive width and height")
    return box


def _clip_box_to_shape(
    box_xyxy: np.ndarray,
    shape_hw: tuple[int, int],
    *,
    name: str,
) -> tuple[np.ndarray, bool]:
    height, width = shape_hw
    clipped = np.array(
        [
            np.clip(box_xyxy[0], 0.0, float(width)),
            np.clip(box_xyxy[1], 0.0, float(height)),
            np.clip(box_xyxy[2], 0.0, float(width)),
            np.clip(box_xyxy[3], 0.0, float(height)),
        ],
        dtype=float,
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"{name} does not intersect the image bounds")
    return clipped, not np.array_equal(clipped, box_xyxy)


def _validate_segmentation_roi(roi: SegmentationRoi) -> None:
    if not isinstance(roi, SegmentationRoi):
        raise TypeError("roi must be a SegmentationRoi")
    for name, value in (
        ("full_shape_hw", roi.full_shape_hw),
        ("resized_shape_hw", roi.resized_shape_hw),
    ):
        raw_shape = np.asarray(value)
        if raw_shape.shape != (2,) or raw_shape.dtype.kind not in "iu":
            raise ValueError(f"roi.{name} must contain two integers")
    full_height, full_width = _validated_shape_hw(
        roi.full_shape_hw,
        name="roi.full_shape_hw",
    )
    crop = np.asarray(roi.crop_xyxy)
    if crop.shape != (4,) or crop.dtype.kind not in "iu":
        raise ValueError("roi.crop_xyxy must contain four integers")
    x0, y0, x1, y1 = (int(value) for value in crop)
    if not (0 <= x0 < x1 <= full_width and 0 <= y0 < y1 <= full_height):
        raise ValueError("roi.crop_xyxy is outside roi.full_shape_hw")
    resized_height, resized_width = _validated_shape_hw(
        roi.resized_shape_hw,
        name="roi.resized_shape_hw",
    )
    if max(resized_height, resized_width) > MAX_SEGMENTATION_ROI_RESIZED_DIMENSION:
        raise ValueError("roi.resized_shape_hw exceeds the practical dimension limit")
    if (
        resized_height * resized_width
        > MAX_SEGMENTATION_ROI_RESIZED_PIXELS
    ):
        raise ValueError("roi.resized_shape_hw exceeds the practical pixel limit")
    requested_scale = float(roi.requested_scale)
    padding = float(roi.padding_px)
    if not np.isfinite(requested_scale) or requested_scale <= 0.0:
        raise ValueError("roi.requested_scale must be finite and positive")
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("roi.padding_px must be finite and non-negative")
    scale_xy = np.asarray(roi.scale_xy)
    if scale_xy.shape != (2,) or scale_xy.dtype.kind not in "uif":
        raise ValueError("roi.scale_xy must contain two real numbers")
    scale_xy = scale_xy.astype(float)
    expected_scale = np.array(
        [
            resized_width / float(x1 - x0),
            resized_height / float(y1 - y0),
        ]
    )
    if (
        not np.all(np.isfinite(scale_xy))
        or np.any(scale_xy <= 0.0)
        or not np.allclose(
            scale_xy,
            expected_scale,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "roi.scale_xy is inconsistent with crop and resized shapes"
        )
    for name, value in (
        ("coarse_candidate_count", roi.coarse_candidate_count),
        ("coarse_bbox_clipped_count", roi.coarse_bbox_clipped_count),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"roi.{name} must be a non-negative integer")
    if roi.coarse_bbox_clipped_count > roi.coarse_candidate_count:
        raise ValueError(
            "roi.coarse_bbox_clipped_count exceeds candidate count"
        )
    if not isinstance(roi.padding_clipped_to_frame, (bool, np.bool_)):
        raise ValueError("roi.padding_clipped_to_frame must be boolean")


def compute_candidate_roi(
    candidates: Iterable[MaskCandidate],
    *,
    full_shape_hw: tuple[int, int] | list[int] | np.ndarray,
    padding_px: float = 24.0,
    scale: float = 4.0,
) -> SegmentationRoi:
    """Compute one clipped, enlarged ROI covering coarse SAM candidates.

    Candidate masks must be expressed in the full-frame image.  The returned
    crop covers the union of every candidate's clipped box and non-empty mask,
    then applies ``padding_px`` and clips to the full-frame bounds.
    """

    full_shape = _validated_shape_hw(full_shape_hw, name="full_shape_hw")
    padding = float(padding_px)
    requested_scale = float(scale)
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("padding_px must be finite and non-negative")
    if not np.isfinite(requested_scale) or requested_scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    coarse_candidates = tuple(candidates)
    if not coarse_candidates:
        raise ValueError("at least one coarse candidate is required")
    if len(coarse_candidates) > MAX_SEGMENTATION_ROI_CANDIDATES:
        raise ValueError(
            "coarse candidate count exceeds the segmentation ROI limit "
            f"of {MAX_SEGMENTATION_ROI_CANDIDATES}"
        )

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    clipped_count = 0
    for index, candidate in enumerate(coarse_candidates):
        prefix = f"candidates[{index}]"
        mask = _validated_mask(
            candidate.mask,
            expected_shape_hw=full_shape,
            name=f"{prefix}.mask",
        )
        box = _validated_box_xyxy(
            candidate.box_xyxy,
            name=f"{prefix}.box_xyxy",
        )
        clipped_box, was_clipped = _clip_box_to_shape(
            box,
            full_shape,
            name=f"{prefix}.box_xyxy",
        )
        clipped_count += int(was_clipped)
        score = float(candidate.score)
        if not np.isfinite(score):
            raise ValueError(f"{prefix}.score must be finite")

        min_x = min(min_x, float(clipped_box[0]))
        min_y = min(min_y, float(clipped_box[1]))
        max_x = max(max_x, float(clipped_box[2]))
        max_y = max(max_y, float(clipped_box[3]))
        rows, columns = np.nonzero(mask)
        if rows.size:
            min_x = min(min_x, float(columns.min()))
            min_y = min(min_y, float(rows.min()))
            max_x = max(max_x, float(columns.max() + 1))
            max_y = max(max_y, float(rows.max() + 1))

    full_height, full_width = full_shape
    padded_x0 = min_x - padding
    padded_y0 = min_y - padding
    padded_x1 = max_x + padding
    padded_y1 = max_y + padding
    padding_clipped = (
        padded_x0 < 0.0
        or padded_y0 < 0.0
        or padded_x1 > full_width
        or padded_y1 > full_height
    )
    x0 = 0 if padded_x0 <= 0.0 else int(np.floor(padded_x0))
    y0 = 0 if padded_y0 <= 0.0 else int(np.floor(padded_y0))
    x1 = (
        full_width
        if padded_x1 >= full_width
        else int(np.ceil(padded_x1))
    )
    y1 = (
        full_height
        if padded_y1 >= full_height
        else int(np.ceil(padded_y1))
    )
    if x1 <= x0 or y1 <= y0:  # Defensive: candidates already intersect.
        raise ValueError("computed ROI is empty after clipping")

    crop_width = x1 - x0
    crop_height = y1 - y0
    scaled_width = crop_width * requested_scale
    scaled_height = crop_height * requested_scale
    if (
        not np.isfinite(scaled_width)
        or not np.isfinite(scaled_height)
        or scaled_width + 0.5 > MAX_SEGMENTATION_ROI_RESIZED_DIMENSION
        or scaled_height + 0.5 > MAX_SEGMENTATION_ROI_RESIZED_DIMENSION
    ):
        raise ValueError(
            "scaled ROI dimensions exceed the practical limit of "
            f"{MAX_SEGMENTATION_ROI_RESIZED_DIMENSION}"
        )
    resized_width = max(1, int(np.floor(scaled_width + 0.5)))
    resized_height = max(1, int(np.floor(scaled_height + 0.5)))
    resized_pixels = resized_width * resized_height
    if resized_pixels > MAX_SEGMENTATION_ROI_RESIZED_PIXELS:
        raise ValueError(
            "scaled ROI pixel count exceeds the practical limit of "
            f"{MAX_SEGMENTATION_ROI_RESIZED_PIXELS}"
        )
    scale_x = resized_width / float(crop_width)
    scale_y = resized_height / float(crop_height)
    roi = SegmentationRoi(
        full_shape_hw=full_shape,
        crop_xyxy=(x0, y0, x1, y1),
        resized_shape_hw=(resized_height, resized_width),
        requested_scale=requested_scale,
        scale_xy=(scale_x, scale_y),
        padding_px=padding,
        coarse_candidate_count=len(coarse_candidates),
        coarse_bbox_clipped_count=clipped_count,
        padding_clipped_to_frame=padding_clipped,
    )
    _validate_segmentation_roi(roi)
    return roi


def extract_enlarged_roi(
    image: np.ndarray,
    roi: SegmentationRoi,
    *,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Crop a full-frame image and resize it exactly as ``roi`` specifies."""

    _validate_segmentation_roi(roi)
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be a two- or three-dimensional array")
    if array.shape[:2] != roi.full_shape_hw:
        raise ValueError(
            f"image shape {array.shape[:2]} does not match "
            f"{roi.full_shape_hw}"
        )
    x0, y0, x1, y1 = roi.crop_xyxy
    resized_height, resized_width = roi.resized_shape_hw
    return cv2.resize(
        array[y0:y1, x0:x1],
        (resized_width, resized_height),
        interpolation=int(interpolation),
    )


def remap_segmentation_result_from_roi(
    result: SegmentationResult,
    roi: SegmentationRoi,
) -> tuple[SegmentationResult, dict[str, object]]:
    """Map enlarged-ROI SAM candidates back into the original full frame.

    Masks are reduced using nearest-neighbour interpolation, then pasted into
    full-frame masks.  Boxes are clipped to the enlarged ROI bounds before
    applying the inverse of ``roi.scale_xy`` and the crop translation.
    """

    _validate_segmentation_roi(roi)
    fine_candidates = tuple(result.candidates)
    if len(fine_candidates) > MAX_SEGMENTATION_ROI_CANDIDATES:
        raise ValueError(
            "fine candidate count exceeds the segmentation ROI limit "
            f"of {MAX_SEGMENTATION_ROI_CANDIDATES}"
        )
    frame_id_raw = np.asarray(result.frame_id)
    if (
        frame_id_raw.shape != ()
        or frame_id_raw.dtype.kind not in "uif"
        or not np.isfinite(float(frame_id_raw))
        or float(frame_id_raw) != np.floor(float(frame_id_raw))
        or float(frame_id_raw) < 0.0
    ):
        raise ValueError("result.frame_id must be a non-negative integer")
    frame_id = int(frame_id_raw)
    source_timestamp = float(result.source_timestamp)
    inference_ms = float(result.inference_ms)
    if not np.isfinite(source_timestamp):
        raise ValueError("result.source_timestamp must be finite")
    if not np.isfinite(inference_ms) or inference_ms < 0.0:
        raise ValueError("result.inference_ms must be finite and non-negative")

    crop_height, crop_width = roi.crop_shape_hw
    resized_height, resized_width = roi.resized_shape_hw
    full_height, full_width = roi.full_shape_hw
    scale_x, scale_y = roi.scale_xy
    x0, y0, x1, y1 = roi.crop_xyxy
    remapped_candidates = []
    fine_bbox_clipped_count = 0
    for index, candidate in enumerate(fine_candidates):
        prefix = f"result.candidates[{index}]"
        mask = _validated_mask(
            candidate.mask,
            expected_shape_hw=roi.resized_shape_hw,
            name=f"{prefix}.mask",
        )
        box = _validated_box_xyxy(
            candidate.box_xyxy,
            name=f"{prefix}.box_xyxy",
        )
        clipped_box, was_clipped = _clip_box_to_shape(
            box,
            roi.resized_shape_hw,
            name=f"{prefix}.box_xyxy",
        )
        fine_bbox_clipped_count += int(was_clipped)
        score = float(candidate.score)
        if not np.isfinite(score):
            raise ValueError(f"{prefix}.score must be finite")

        local_mask = cv2.resize(
            mask.astype(np.uint8),
            (crop_width, crop_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        full_mask = np.zeros((full_height, full_width), dtype=bool)
        full_mask[y0:y1, x0:x1] = local_mask

        remapped_box = np.array(
            [
                x0 + clipped_box[0] / scale_x,
                y0 + clipped_box[1] / scale_y,
                x0 + clipped_box[2] / scale_x,
                y0 + clipped_box[3] / scale_y,
            ],
            dtype=float,
        )
        remapped_box[[0, 2]] = np.clip(
            remapped_box[[0, 2]], 0.0, float(full_width)
        )
        remapped_box[[1, 3]] = np.clip(
            remapped_box[[1, 3]], 0.0, float(full_height)
        )
        remapped_candidates.append(
            MaskCandidate(
                mask=full_mask,
                box_xyxy=remapped_box,
                score=score,
            )
        )

    metadata = roi.metadata()
    metadata.update(
        {
            "mask_remap_interpolation": "nearest",
            "fine_candidate_count": len(remapped_candidates),
            "fine_bbox_clipped_count": fine_bbox_clipped_count,
        }
    )
    return (
        SegmentationResult(
            frame_id=frame_id,
            source_timestamp=source_timestamp,
            model=str(result.model),
            inference_ms=inference_ms,
            candidates=tuple(remapped_candidates),
        ),
        metadata,
    )


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
        jpeg_quality: int = 90,
        request_observer: (
            Callable[[tuple[bytes, ...]], None] | None
        ) = None,
    ) -> SegmentationResult:
        """Segment an image, optionally recording its exact wire request.

        ``request_observer`` is called synchronously with immutable copies of
        the multipart request after encoding and before the socket send.  If
        the observer raises, no request is sent.  This lets callers durably
        journal the exact JPEG and metadata even when send/receive later fails.
        """

        timestamp = time.time() if timestamp is None else float(timestamp)
        if frame_id <= self.last_frame_id:
            raise ValueError(
                f"frame_id must increase: {frame_id} <= {self.last_frame_id}"
            )
        request = tuple(
            bytes(part)
            for part in encode_request(
                image_bgr,
                frame_id=frame_id,
                timestamp=timestamp,
                prompt=prompt,
                confidence_threshold=confidence_threshold,
                jpeg_quality=jpeg_quality,
            )
        )
        if request_observer is not None:
            request_observer(request)
        self.socket.send_multipart(request)
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


def detect_blue_cross_centers(image_bgr: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return all cross-shaped blue components, strongest candidate first.

    Keeping every plausible component lets a semantic mask disambiguate the
    lid marker from unrelated blue hardware elsewhere in a full-frame view.
    """
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
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return tuple(item[2] for item in candidates)


def detect_blue_cross_center(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return the strongest cross-shaped blue component, if one exists."""

    centers = detect_blue_cross_centers(image_bgr)
    return centers[0] if centers else None


def choose_lid_candidate(
    candidates: Iterable[MaskCandidate],
    *,
    image_bgr: np.ndarray,
    previous_center_px: np.ndarray | None = None,
    require_blue_cross: bool = False,
) -> tuple[MaskCandidate, LidMaskGeometry] | None:
    blue_centers = detect_blue_cross_centers(image_bgr)
    if require_blue_cross and not blue_centers:
        return None
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
        for blue in blue_centers:
            bx, by = np.rint(blue).astype(int)
            if 0 <= by < height and 0 <= bx < width:
                contains_cross = bool(candidate.mask[by, bx])
            if contains_cross:
                break
        if blue_centers:
            cross_penalty = 0.0 if contains_cross else 500.0
        if require_blue_cross and not contains_cross:
            continue
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

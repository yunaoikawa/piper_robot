#!/usr/bin/env python3
"""Real-time SAM+depth grasp servo with no demo or endpoint calibration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import socket
import struct
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.rpc import RPCClient
from rollout.realtime_sam_servo import (
    GRIPPER_COLOR_MINIMUM_MASK_FRACTION,
    GRIPPER_COLOR_MINIMUM_PIXELS,
    GRIPPER_CYAN_HSV_LOWER,
    GRIPPER_CYAN_HSV_UPPER,
    bounded_reachable_servo_step,
    choose_right_gripper,
    estimate_reachable_feature_model,
    gripper_mask_center,
    lid_left_grasp_px,
    render_scene,
    scene_feature,
)
from rollout.sam_head_camera import SamHeadlessRecord3DManager
from rollout.sam_segmentation import SamSegmentationClient
from rollout.sam_segmentation import (
    choose_lid_candidate,
    compute_candidate_roi,
    enhance_low_light,
    extract_enlarged_roi,
    remap_segmentation_result_from_roi,
)
from rollout.scene_3d import (
    assess_target_geometry,
    backproject,
    estimate_target_on_support_plane,
    nearest_scene_distance,
    register_point_clouds,
    scaled_camera_matrix,
    temporal_median_depth,
)


class TorqueStop(RuntimeError):
    pass


PIPER_MIT_MODE_CAN_ID = 0x151
PIPER_MIT_MODE_PAYLOAD = bytes.fromhex("010400AD00000000")
DEFAULT_HOLDING_KP = np.full(6, 2.5)
DEFAULT_HOLDING_KD = np.full(6, 0.2)
DEFAULT_MOTION_KP = np.array([7.0, 7.0, 7.0, 5.0, 5.0, 5.0])
DEFAULT_MOTION_KD = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3])
MINIMUM_SEGMENT_IMAGE_MARGIN_PX = 10
SAM_REQUEST_JPEG_QUALITY = 90
MAX_DEPTH_BURST_SPAN_S = 2.50
MAX_DEPTH_BURST_COLLECTION_WAIT_S = 2.75
MAX_POST_SAM_ROI_MEAN_ABS_DIFF = 0.04
MAX_POST_SAM_ROI_P95_ABS_DIFF = 0.12
MIN_FINE_GRIPPER_COARSE_IOU = 0.10
MAX_FINE_GRIPPER_CENTER_DISTANCE_PX = 100.0


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rgb_frame_sha256(rgb) -> str:
    frame = np.ascontiguousarray(np.asarray(rgb))
    if frame.ndim != 3 or frame.shape[2] not in (3, 4) or frame.size == 0:
        raise RuntimeError("head RGB frame has an invalid shape")
    return _sha256_bytes(memoryview(frame).cast("B"))


def _depth_frame_sha256(depth) -> str:
    frame = np.ascontiguousarray(np.asarray(depth))
    if frame.ndim != 2 or frame.size == 0:
        raise RuntimeError("head depth frame has an invalid shape")
    return _sha256_bytes(memoryview(frame).cast("B"))


def _canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(directory):
    descriptor = os.open(Path(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes):
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError(
                f"short write: {offset}/{len(payload)} bytes"
            )
        offset += written


def _reserve_new_file(path, payload: bytes):
    """Create a durable reservation with O_EXCL."""

    path = Path(path)
    payload = bytes(payload)
    descriptor = None
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created = True
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _fsync_directory(path.parent)
    except FileExistsError:
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                path.unlink(missing_ok=True)
                _fsync_directory(path.parent)
            except OSError:
                pass
        raise RuntimeError(
            f"could not reserve artifact prefix {path}: {exc}"
        ) from exc


def _publish_new_bytes(path, payload: bytes):
    """Atomically publish bytes without ever replacing an existing target."""

    path = Path(path)
    payload = bytes(payload)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            written = output.write(payload)
            if written != len(payload):
                raise OSError(
                    f"short write: {written}/{len(payload)} bytes"
                )
            output.flush()
            os.fsync(output.fileno())
        # A same-filesystem hard link is an atomic no-replace publication.
        os.link(temporary_path, path)
        _fsync_directory(path.parent)
    except FileExistsError:
        raise
    except OSError as exc:
        raise RuntimeError(f"could not save artifact {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
                _fsync_directory(path.parent)
            except OSError:
                pass


def _encode_image(extension: str, image, parameters=()) -> bytes:
    ok, encoded = cv2.imencode(
        str(extension),
        np.asarray(image),
        list(parameters),
    )
    if not ok:
        raise RuntimeError(f"could not encode {extension} artifact")
    return encoded.tobytes()


def _npz_bytes(**arrays) -> bytes:
    output = io.BytesIO()
    try:
        np.savez_compressed(output, **arrays)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"could not encode depth artifact: {exc}") from exc
    return output.getvalue()


_OBSERVATION_PREFIX = re.compile(r"^([0-9]+)(?:[_.]|$)")


class _ObservationArtifactWriter:
    """Write one immutable observation or an explicit failure journal."""

    def __init__(
        self,
        output_dir,
        sequence: int,
        *,
        run_id: str,
        attempt_id: str,
        reservation_path,
    ):
        self.output_dir = Path(output_dir)
        self.sequence = int(sequence)
        self.run_id = str(run_id)
        self.attempt_id = str(attempt_id)
        self.reservation_path = Path(reservation_path)
        self.files = {}
        self.paths = {}
        self.failure_context = {}
        self.finished = False
        self.failed = False

    @classmethod
    def reserve(
        cls,
        output_dir,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ):
        """Reserve a collision-free numeric prefix across concurrent writers."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = []
        for path in output_dir.iterdir():
            match = _OBSERVATION_PREFIX.match(path.name)
            if match is not None:
                existing.append(int(match.group(1)))
        sequence = max(existing, default=-1) + 1
        while True:
            current_attempt_id = (
                uuid.uuid4().hex
                if attempt_id is None
                else str(attempt_id)
            )
            reservation_path = (
                output_dir
                / f"{sequence:03d}_head_observation.reserved.json"
            )
            reservation = {
                "schema": "sam_head_observation_reservation/v1",
                "sequence": sequence,
                "run_id": str(run_id),
                "attempt_id": current_attempt_id,
                "reserved_at_unix_s": time.time(),
            }
            try:
                _reserve_new_file(
                    reservation_path,
                    _canonical_json_bytes(reservation),
                )
            except FileExistsError:
                sequence += 1
                continue
            return cls(
                output_dir,
                sequence,
                run_id=str(run_id),
                attempt_id=current_attempt_id,
                reservation_path=reservation_path,
            )

    def _artifact_path(self, filename: str) -> Path:
        filename = str(filename)
        if Path(filename).name != filename:
            raise ValueError("observation artifact filename must be local")
        if _OBSERVATION_PREFIX.match(filename) is None:
            raise ValueError("observation artifact filename lacks a prefix")
        prefix = int(_OBSERVATION_PREFIX.match(filename).group(1))
        if prefix != self.sequence:
            raise ValueError(
                "observation artifact filename does not use its reservation"
            )
        return self.output_dir / filename

    def _release_reservation(self):
        try:
            self.reservation_path.unlink(missing_ok=True)
            _fsync_directory(self.output_dir)
        except OSError as exc:
            print(
                "WARNING: could not remove observation reservation "
                f"{self.reservation_path}: {exc}",
                file=sys.stderr,
            )

    def add_bytes(
        self,
        role: str,
        filename: str,
        payload: bytes,
        *,
        media_type: str,
    ) -> Path:
        if role in self.files:
            raise ValueError(f"duplicate observation artifact role {role}")
        if self.finished or self.failed:
            raise RuntimeError("observation artifact set is already closed")
        path = self._artifact_path(filename)
        payload = bytes(payload)
        _publish_new_bytes(path, payload)
        self.paths[role] = path
        self.files[role] = {
            "path": path.name,
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
            "media_type": str(media_type),
        }
        return path

    def add_image(
        self,
        role: str,
        filename: str,
        image,
        *,
        extension: str = ".png",
        parameters=(),
    ) -> Path:
        payload = _encode_image(extension, image, parameters)
        media_type = (
            "image/jpeg"
            if extension.lower() in (".jpg", ".jpeg")
            else "image/png"
        )
        return self.add_bytes(
            role,
            filename,
            payload,
            media_type=media_type,
        )

    def add_npz(self, role: str, filename: str, **arrays) -> Path:
        return self.add_bytes(
            role,
            filename,
            _npz_bytes(**arrays),
            media_type="application/x-npz",
        )

    def finish(self, metadata: dict):
        if self.finished or self.failed:
            raise RuntimeError("observation artifact set is already closed")
        document = {
            **metadata,
            "schema": "sam_head_observation/v2",
            "sequence": self.sequence,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "files": self.files,
        }
        payload = _canonical_json_bytes(document)
        manifest_path = (
            self.output_dir
            / f"{self.sequence:03d}_head_observation.json"
        )
        _publish_new_bytes(manifest_path, payload)
        self.finished = True
        self._release_reservation()
        return document, manifest_path, _sha256_bytes(payload)

    def fail(self, error: BaseException):
        """Publish a failure journal; never publish a success manifest."""

        if self.finished or self.failed:
            return None
        document = {
            "schema": "sam_head_observation_failure/v1",
            "status": "failed",
            "sequence": self.sequence,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "files": self.files,
            "context": self.failure_context,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        path = (
            self.output_dir
            / f"{self.sequence:03d}_head_observation.failed.json"
        )
        _publish_new_bytes(path, _canonical_json_bytes(document))
        self.failed = True
        self._release_reservation()
        return path


def _require_candidate_image_margin(
    candidate,
    image_shape,
    *,
    label: str,
    minimum_margin_px: int = MINIMUM_SEGMENT_IMAGE_MARGIN_PX,
):
    """Reject a selected mask whose contour may continue outside the image."""

    height, width = (int(image_shape[0]), int(image_shape[1]))
    label = str(label).strip()
    if not label:
        raise ValueError("candidate label must not be empty")
    minimum_margin_px = int(minimum_margin_px)
    if height <= 0 or width <= 0 or minimum_margin_px < 0:
        raise ValueError("invalid image shape or lid image margin")
    mask = np.asarray(candidate.mask, dtype=bool)
    if mask.shape != (height, width):
        raise RuntimeError(
            f"selected {label} mask shape does not match the SAM input"
        )
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise RuntimeError(f"selected {label} mask is empty")
    mask_margin = int(
        min(
            int(xx.min()),
            int(yy.min()),
            width - 1 - int(xx.max()),
            height - 1 - int(yy.max()),
        )
    )
    box = np.asarray(candidate.box_xyxy, dtype=float).reshape(-1)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise RuntimeError(f"selected {label} bounding box is invalid")
    x1, y1, x2, y2 = box
    bbox_margin = float(min(x1, y1, width - x2, height - y2))
    if (
        mask_margin < minimum_margin_px
        or bbox_margin < float(minimum_margin_px)
    ):
        raise RuntimeError(
            f"selected {label} is clipped by the head image boundary: "
            f"mask_margin={mask_margin}px, "
            f"bbox_margin={bbox_margin:.1f}px, "
            f"required={minimum_margin_px}px"
        )
    return {
        "required_margin_px": minimum_margin_px,
        "mask_margin_px": mask_margin,
        "bbox_margin_px": bbox_margin,
    }


def _require_candidate_roi_margin(
    candidate,
    roi,
    *,
    label: str,
    minimum_margin_px: int = MINIMUM_SEGMENT_IMAGE_MARGIN_PX,
):
    """Reject a remapped fine mask that may have been truncated by its ROI."""

    minimum_margin_px = int(minimum_margin_px)
    if minimum_margin_px < 0:
        raise ValueError("ROI margin must be non-negative")
    mask = np.asarray(candidate.mask, dtype=bool)
    if mask.shape != tuple(roi.full_shape_hw):
        raise RuntimeError(
            f"selected {label} mask shape does not match the ROI full frame"
        )
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise RuntimeError(f"selected {label} mask is empty")
    x0, y0, x1, y1 = (float(value) for value in roi.crop_xyxy)
    mask_margin = float(
        min(
            float(xx.min()) - x0,
            float(yy.min()) - y0,
            (x1 - 1.0) - float(xx.max()),
            (y1 - 1.0) - float(yy.max()),
        )
    )
    box = np.asarray(candidate.box_xyxy, dtype=float).reshape(-1)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise RuntimeError(f"selected {label} bounding box is invalid")
    bbox_margin = float(
        min(
            box[0] - x0,
            box[1] - y0,
            x1 - box[2],
            y1 - box[3],
        )
    )
    if (
        mask_margin < float(minimum_margin_px)
        or bbox_margin < float(minimum_margin_px)
    ):
        raise RuntimeError(
            f"selected {label} is clipped by the SAM ROI boundary: "
            f"mask_margin={mask_margin:.1f}px, "
            f"bbox_margin={bbox_margin:.1f}px, "
            f"required={minimum_margin_px}px"
        )
    return {
        "required_margin_px": minimum_margin_px,
        "mask_margin_px": mask_margin,
        "bbox_margin_px": bbox_margin,
    }


def _candidate_mask_center(candidate) -> np.ndarray:
    yy, xx = np.nonzero(np.asarray(candidate.mask, dtype=bool))
    if len(xx) < 50:
        raise ValueError("candidate mask is too small for stable association")
    return np.array([np.median(xx), np.median(yy)], dtype=float)


def _identity_index(candidates, selected) -> int:
    for index, candidate in enumerate(candidates):
        if candidate is selected:
            return index
    raise RuntimeError("selected SAM candidate is not in its response")


def _associate_fine_gripper(
    coarse_candidate,
    fine_candidates,
    *,
    minimum_iou: float = MIN_FINE_GRIPPER_COARSE_IOU,
    maximum_center_distance_px: float = MAX_FINE_GRIPPER_CENTER_DISTANCE_PX,
):
    """Associate a fine mask to the coarse gripper instead of reselecting it."""

    coarse_mask = np.asarray(coarse_candidate.mask, dtype=bool)
    coarse_center = _candidate_mask_center(coarse_candidate)
    minimum_iou = float(minimum_iou)
    maximum_center_distance_px = float(maximum_center_distance_px)
    if (
        not np.isfinite(minimum_iou)
        or not 0.0 <= minimum_iou <= 1.0
        or not np.isfinite(maximum_center_distance_px)
        or maximum_center_distance_px < 0.0
    ):
        raise ValueError("invalid coarse/fine gripper association limits")
    ranked = []
    for index, candidate in enumerate(tuple(fine_candidates)):
        fine_mask = np.asarray(candidate.mask, dtype=bool)
        if fine_mask.shape != coarse_mask.shape:
            raise ValueError("coarse and fine gripper masks have different shapes")
        try:
            fine_center = _candidate_mask_center(candidate)
        except ValueError:
            continue
        intersection = int(np.count_nonzero(coarse_mask & fine_mask))
        union = int(np.count_nonzero(coarse_mask | fine_mask))
        iou = 0.0 if union == 0 else intersection / float(union)
        center_distance = float(np.linalg.norm(fine_center - coarse_center))
        if (
            iou < minimum_iou
            or center_distance > maximum_center_distance_px
        ):
            continue
        rank = (
            -iou,
            center_distance,
            -float(candidate.score),
        )
        ranked.append((rank, index, candidate, iou, center_distance))
    if not ranked:
        raise RuntimeError(
            "ROI-refined SAM did not preserve the coarse gripper instance"
        )
    _, index, candidate, iou, center_distance = min(
        ranked, key=lambda item: item[0]
    )
    return candidate, {
        "schema": "sam_coarse_fine_association/v1",
        "fine_candidate_index": int(index),
        "mask_iou": float(iou),
        "center_distance_px": center_distance,
        "minimum_iou": minimum_iou,
        "maximum_center_distance_px": maximum_center_distance_px,
    }


def _compare_roi_images(before, after):
    """Return fail-closed normalized change metrics for equal-size ROI images."""

    before_array = np.asarray(before)
    after_array = np.asarray(after)
    if (
        before_array.shape != after_array.shape
        or before_array.ndim not in (2, 3)
        or before_array.size == 0
    ):
        raise RuntimeError("post-SAM ROI image shape does not match its source")
    difference = cv2.absdiff(
        before_array.astype(np.uint8, copy=False),
        after_array.astype(np.uint8, copy=False),
    ).astype(np.float32)
    normalized = difference / 255.0
    mean_abs = float(np.mean(normalized))
    p95_abs = float(np.percentile(normalized, 95.0))
    accepted = (
        np.isfinite(mean_abs)
        and np.isfinite(p95_abs)
        and mean_abs <= MAX_POST_SAM_ROI_MEAN_ABS_DIFF
        and p95_abs <= MAX_POST_SAM_ROI_P95_ABS_DIFF
    )
    return {
        "schema": "sam_post_inference_roi_consistency/v1",
        "mean_abs_difference": mean_abs,
        "p95_abs_difference": p95_abs,
        "maximum_mean_abs_difference": MAX_POST_SAM_ROI_MEAN_ABS_DIFF,
        "maximum_p95_abs_difference": MAX_POST_SAM_ROI_P95_ABS_DIFF,
        "accepted": bool(accepted),
    }


def refresh_right_mit_mode(
    can_interface: str = "can_right", socket_factory=socket.socket
):
    """Reassert Piper's right-arm MIT mode without reset, enable, or homing."""

    if can_interface != "can_right":
        raise ValueError("MIT mode refresh is restricted to can_right")
    frame = struct.pack(
        "=IB3x8s",
        PIPER_MIT_MODE_CAN_ID,
        len(PIPER_MIT_MODE_PAYLOAD),
        PIPER_MIT_MODE_PAYLOAD,
    )
    can_socket = socket_factory(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        can_socket.bind((can_interface,))
        sent = can_socket.send(frame)
    finally:
        can_socket.close()
    if sent != len(frame):
        raise RuntimeError(
            f"short right MIT mode CAN write: {sent}/{len(frame)} bytes"
        )


def _gain_vector(values, name: str, maximum) -> np.ndarray:
    gain = np.asarray(values, dtype=float)
    if gain.shape != (6,) or not np.all(np.isfinite(gain)):
        raise ValueError(f"{name} must contain six finite values")
    if np.any(gain < 0.0) or np.any(gain > np.asarray(maximum, dtype=float)):
        raise ValueError(f"{name} is outside the tested safe range")
    return gain


def _note_exception(error: BaseException, note: str):
    """Attach cleanup context without replacing the primary failure."""

    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:
        print(f"{error!r}; NOTE: {note}", file=sys.stderr)


class LiveSamGrasp:
    def __init__(self, args):
        self.args = args
        self.stop_event = threading.Event()
        self.rpc = None
        self.camera = None
        self.sam = None
        torque_cfg = json.loads(Path(args.torque_config).read_text())
        self.torque_limit = np.asarray(
            torque_cfg["thresholds"]["right"], dtype=float
        )
        self.torque_samples = int(torque_cfg["consecutive_samples"])
        self.holding_kp = _gain_vector(
            getattr(args, "holding_kp", DEFAULT_HOLDING_KP),
            "holding kp",
            DEFAULT_MOTION_KP,
        )
        self.holding_kd = _gain_vector(
            getattr(args, "holding_kd", DEFAULT_HOLDING_KD),
            "holding kd",
            DEFAULT_MOTION_KD,
        )
        self.motion_kp = _gain_vector(
            getattr(args, "motion_kp", DEFAULT_MOTION_KP),
            "motion kp",
            DEFAULT_MOTION_KP,
        )
        self.motion_kd = _gain_vector(
            getattr(args, "motion_kd", DEFAULT_MOTION_KD),
            "motion kd",
            DEFAULT_MOTION_KD,
        )
        self.gain_ramp_s = float(getattr(args, "gain_ramp_s", 1.0))
        self.mode_settle_s = float(getattr(args, "mode_settle_s", 0.5))
        self.hold_settle_s = float(getattr(args, "hold_settle_s", 0.25))
        preparation_times = np.array(
            [self.gain_ramp_s, self.mode_settle_s, self.hold_settle_s]
        )
        if (
            not np.all(np.isfinite(preparation_times))
            or np.any(preparation_times < 0.0)
            or self.gain_ramp_s > 5.0
            or max(self.mode_settle_s, self.hold_settle_s) > 2.0
        ):
            raise ValueError("motion preparation times are outside safe bounds")
        right_can_interface = str(
            getattr(args, "right_can_interface", "can_right")
        )
        self.motion_mode_refresher = lambda: refresh_right_mit_mode(
            right_can_interface
        )
        self.run_id = uuid.uuid4().hex
        self.frame_id = 1000
        self.sequence = 0
        self.previous_lid_center = None
        self.previous_gripper_center = None
        self.orientation = None
        self.joint_command = None
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        profile = json.loads(Path(args.scene_config).read_text())
        self.head_camera_matrix = np.asarray(
            profile["head_camera_matrix_rotated"], dtype=float
        )
        reference_shape = profile.get("head_camera_reference_shape_hw")
        if (
            reference_shape is None
            or len(reference_shape) != 2
            or any(int(value) <= 0 for value in reference_shape)
        ):
            raise ValueError(
                "scene profile requires head_camera_reference_shape_hw"
            )
        self.head_camera_reference_shape = tuple(
            int(value) for value in reference_shape
        )
        self.support_plane_config = profile.get("support_plane", {})
        self.registration_config = profile.get("registration", {})
        self.geometry_quality_config = profile.get("geometry_quality", {})
        self.proximity_warning_m = float(
            profile.get("proximity_warning_m", 0.02)
        )
        self.last_target_3d = None
        self.last_geometry_quality = None
        self.last_depth = None
        self.last_camera_matrix = None
        self.last_proximity_m = None
        self.last_observation_artifacts = None
        self.last_head_timestamp = None
        self.last_head_rgb_sha256 = None
        try:
            self.rpc = RPCClient("localhost", 8081, timeout_ms=10000)
            self.camera = SamHeadlessRecord3DManager(self.stop_event)
            self.sam = SamSegmentationClient(
                args.sam_endpoint, timeout_ms=20000
            )
        except BaseException as init_error:
            try:
                self.stop()
            except BaseException as cleanup_error:
                _note_exception(
                    init_error,
                    "partially initialized runner cleanup also failed: "
                    f"{cleanup_error!r}",
                )
            raise

    def start(self):
        self.camera.start()
        self._await_fresh_head_frame(timeout_s=15.0)

    def _await_fresh_head_frame(self, *, timeout_s=3.0):
        """Return an RGB-D callback received after this observation began.

        Record3D's capture loop can expose an old buffer with a newly assigned
        wall-clock timestamp.  Requiring both a post-barrier timestamp and new
        RGB bytes prevents a pre-motion image from being used as post-motion
        feedback.
        """

        not_before = time.time()
        deadline = time.monotonic() + float(timeout_s)
        newest_timestamp = None
        repeated_digest = False
        while time.monotonic() < deadline:
            rgb, timestamp, depth = self.camera.get_latest_frame()
            if rgb is None or timestamp is None or depth is None:
                time.sleep(0.01)
                continue
            timestamp = float(timestamp)
            if not np.isfinite(timestamp):
                raise RuntimeError("head frame timestamp is not finite")
            newest_timestamp = timestamp
            if timestamp + 1e-6 < not_before:
                time.sleep(0.01)
                continue
            digest = _rgb_frame_sha256(rgb)
            if (
                self.last_head_timestamp is not None
                and timestamp <= self.last_head_timestamp
            ):
                time.sleep(0.01)
                continue
            if (
                self.last_head_rgb_sha256 is not None
                and digest == self.last_head_rgb_sha256
            ):
                repeated_digest = True
                time.sleep(0.01)
                continue
            depth_array = np.asarray(depth)
            if depth_array.ndim != 2 or depth_array.size == 0:
                time.sleep(0.01)
                continue
            self.last_head_timestamp = timestamp
            self.last_head_rgb_sha256 = digest
            return np.asarray(rgb), timestamp, depth_array
        reason = (
            "head RGB bytes repeated despite newer timestamps"
            if repeated_digest
            else "no post-barrier RGB-D callback arrived"
        )
        raise RuntimeError(
            f"fresh head RGB-D frame unavailable: {reason}; "
            f"latest_timestamp={newest_timestamp}"
        )

    def _collect_depth_burst(
        self,
        rgb,
        timestamp,
        depth_raw,
        *,
        diagnostics=None,
    ):
        """Collect a short RGB-D burst before any SAM network round trip."""

        if diagnostics is None:
            diagnostics = {}
        else:
            diagnostics.clear()
        requested = int(self.args.depth_frames)
        if not 3 <= requested <= 120:
            raise ValueError("depth frame count must be between 3 and 120")
        minimum_required = max(3, requested // 2)
        timestamp = float(timestamp)
        maximum_timestamp = timestamp + MAX_DEPTH_BURST_SPAN_S
        depth_frames = [np.asarray(depth_raw)]
        depth_timestamps = [timestamp]
        depth_rgb_sha256 = [_rgb_frame_sha256(rgb)]
        depth_frame_sha256 = [_depth_frame_sha256(depth_raw)]
        maximum_rgb_mean_abs_difference = 0.0
        maximum_rgb_p95_abs_difference = 0.0
        native_shape = depth_frames[0].shape
        diagnostics.update(
            {
                "schema": "sam_pre_sam_depth_burst/v1",
                "frames_requested": requested,
                "minimum_frames_required": minimum_required,
                "frames_accepted": 1,
                "poll_count": 0,
                "timestamp_advanced_count": 0,
                "timestamp_not_advanced_count": 0,
                "timestamp_beyond_span_count": 0,
                "invalid_timestamp_count": 0,
                "missing_rgb_count": 0,
                "missing_depth_count": 0,
                "depth_shape_mismatch_count": 0,
                "repeated_rgb_digest_count": 0,
                "repeated_depth_digest_count": 0,
                "scene_change_rejection_count": 0,
                "source_timestamp": timestamp,
                "latest_observed_timestamp": timestamp,
                "accepted_timestamps": list(depth_timestamps),
                "accepted_rgb_sha256": list(depth_rgb_sha256),
                "accepted_depth_sha256": list(depth_frame_sha256),
                "maximum_timestamp_span_s": MAX_DEPTH_BURST_SPAN_S,
                "collection_wait_limit_s": (
                    MAX_DEPTH_BURST_COLLECTION_WAIT_S
                ),
            }
        )
        started = time.monotonic()
        deadline = (
            started + MAX_DEPTH_BURST_COLLECTION_WAIT_S
        )
        while len(depth_frames) < requested and time.monotonic() < deadline:
            next_rgb, next_timestamp, next_depth = self.camera.get_latest_frame()
            diagnostics["poll_count"] += 1
            try:
                next_timestamp = float(next_timestamp)
            except (TypeError, ValueError):
                next_timestamp = float("nan")
            if np.isfinite(next_timestamp):
                diagnostics["latest_observed_timestamp"] = next_timestamp
            else:
                diagnostics["invalid_timestamp_count"] += 1
                time.sleep(0.01)
                continue
            if np.isfinite(next_timestamp) and next_timestamp > maximum_timestamp:
                diagnostics["timestamp_beyond_span_count"] += 1
                break
            if next_timestamp <= depth_timestamps[-1]:
                diagnostics["timestamp_not_advanced_count"] += 1
                time.sleep(0.01)
                continue
            diagnostics["timestamp_advanced_count"] += 1
            if next_rgb is None:
                diagnostics["missing_rgb_count"] += 1
                time.sleep(0.01)
                continue
            if next_depth is None:
                diagnostics["missing_depth_count"] += 1
                time.sleep(0.01)
                continue
            next_depth_array = np.asarray(next_depth)
            if next_depth_array.shape != native_shape:
                diagnostics["depth_shape_mismatch_count"] += 1
                time.sleep(0.01)
                continue
            next_rgb_digest = _rgb_frame_sha256(next_rgb)
            if next_rgb_digest in depth_rgb_sha256:
                diagnostics["repeated_rgb_digest_count"] += 1
            next_depth_digest = _depth_frame_sha256(next_depth_array)
            if next_depth_digest in depth_frame_sha256:
                diagnostics["repeated_depth_digest_count"] += 1
                time.sleep(0.01)
                continue
            consistency = _compare_roi_images(rgb, next_rgb)
            if not consistency["accepted"]:
                diagnostics["scene_change_rejection_count"] += 1
                diagnostics["wall_elapsed_s"] = (
                    time.monotonic() - started
                )
                raise RuntimeError(
                    "head-camera scene changed during pre-SAM depth "
                    "burst: "
                    f"mean={consistency['mean_abs_difference']:.4f}, "
                    f"p95={consistency['p95_abs_difference']:.4f}"
                )
            maximum_rgb_mean_abs_difference = max(
                maximum_rgb_mean_abs_difference,
                consistency["mean_abs_difference"],
            )
            maximum_rgb_p95_abs_difference = max(
                maximum_rgb_p95_abs_difference,
                consistency["p95_abs_difference"],
            )
            depth_frames.append(next_depth_array)
            depth_timestamps.append(next_timestamp)
            depth_rgb_sha256.append(next_rgb_digest)
            depth_frame_sha256.append(next_depth_digest)
            diagnostics["frames_accepted"] = len(depth_frames)
            diagnostics["accepted_timestamps"] = list(depth_timestamps)
            diagnostics["accepted_rgb_sha256"] = list(depth_rgb_sha256)
            diagnostics["accepted_depth_sha256"] = list(
                depth_frame_sha256
            )
        diagnostics["wall_elapsed_s"] = time.monotonic() - started
        if len(depth_frames) < minimum_required:
            raise RuntimeError(
                "insufficient fresh pre-SAM depth frames: "
                f"accepted={len(depth_frames)}/{minimum_required}, "
                f"requested={requested}, polls={diagnostics['poll_count']}, "
                "timestamp_advanced="
                f"{diagnostics['timestamp_advanced_count']}, "
                "timestamp_not_advanced="
                f"{diagnostics['timestamp_not_advanced_count']}, "
                "repeated_rgb="
                f"{diagnostics['repeated_rgb_digest_count']}, "
                "repeated_depth="
                f"{diagnostics['repeated_depth_digest_count']}, "
                f"accepted_timestamps={depth_timestamps}"
            )
        timestamp_span = depth_timestamps[-1] - depth_timestamps[0]
        if (
            not np.isfinite(timestamp_span)
            or timestamp_span < 0.0
            or timestamp_span > MAX_DEPTH_BURST_SPAN_S
        ):
            raise RuntimeError(
                "pre-SAM depth burst exceeded its timestamp span limit"
            )
        return (
            depth_frames,
            depth_timestamps,
            depth_rgb_sha256,
            {
                **diagnostics,
                "timestamp_span_s": float(timestamp_span),
                "captured_before_sam": True,
                "source_depth_sha256": list(depth_frame_sha256),
                "rgb_consistency": {
                    "maximum_mean_abs_difference": (
                        maximum_rgb_mean_abs_difference
                    ),
                    "maximum_p95_abs_difference": (
                        maximum_rgb_p95_abs_difference
                    ),
                    "allowed_mean_abs_difference": (
                        MAX_POST_SAM_ROI_MEAN_ABS_DIFF
                    ),
                    "allowed_p95_abs_difference": (
                        MAX_POST_SAM_ROI_P95_ABS_DIFF
                    ),
                    "accepted": True,
                },
            },
        )

    def _verify_post_sam_roi_consistency(
        self,
        *,
        roi,
        reference_roi_image,
        enhanced: bool,
    ):
        """Require a new post-inference frame whose ROI still matches."""

        post_rgb, post_timestamp, _ = self._await_fresh_head_frame()
        post_image = cv2.cvtColor(
            np.rot90(post_rgb, k=3), cv2.COLOR_RGB2BGR
        )
        post_sam_image = (
            enhance_low_light(post_image) if enhanced else post_image
        )
        post_roi_image = extract_enlarged_roi(post_sam_image, roi)
        report = _compare_roi_images(reference_roi_image, post_roi_image)
        report.update(
            {
                "source_timestamp": float(post_timestamp),
                "source_rgb_sha256": _rgb_frame_sha256(post_rgb),
            }
        )
        if not report["accepted"]:
            raise RuntimeError(
                "head-camera ROI changed during SAM inference: "
                f"mean={report['mean_abs_difference']:.4f}, "
                f"p95={report['p95_abs_difference']:.4f}"
            )
        return post_roi_image, report

    def stop(self):
        failures = []

        def attempt(label, callback):
            try:
                callback()
            except BaseException as exc:
                failures.append((str(label), exc))

        stop_event = getattr(self, "stop_event", None)
        if stop_event is not None:
            attempt("stop event", stop_event.set)
        camera = getattr(self, "camera", None)
        if camera is not None:
            attempt("head camera", camera.stop)
        sam = getattr(self, "sam", None)
        if sam is not None:
            attempt("SAM client", sam.close)
        rpc = getattr(self, "rpc", None)
        rpc_state = vars(rpc) if rpc is not None else {}
        rpc_socket = rpc_state.get("socket")
        rpc_context = rpc_state.get("context")
        if rpc_socket is not None:
            attempt(
                "RPC socket",
                lambda: rpc_socket.close(linger=0),
            )
        if rpc_context is not None:
            attempt("RPC context", rpc_context.term)
        if failures:
            primary_label, primary = failures[0]
            for label, error in failures[1:]:
                _note_exception(
                    primary,
                    f"{label} cleanup also failed: {error!r}",
                )
            _note_exception(
                primary,
                f"cleanup failure originated in {primary_label}",
            )
            raise primary

    def observe(self, clearance_m: float):
        self.last_observation_artifacts = None
        artifacts = _ObservationArtifactWriter.reserve(
            self.output_dir, self.run_id
        )
        self.sequence = artifacts.sequence
        try:
            return self._observe_reserved(clearance_m, artifacts)
        except BaseException as exc:
            try:
                artifacts.fail(exc)
            except BaseException as journal_error:
                _note_exception(
                    exc,
                    "observation failure journal also failed: "
                    f"{journal_error!r}",
                )
            raise
        finally:
            self.sequence = max(
                self.sequence, artifacts.sequence + 1
            )

    def _observe_reserved(
        self,
        clearance_m: float,
        artifacts: _ObservationArtifactWriter,
    ):
        rgb, timestamp, depth_raw = self._await_fresh_head_frame()
        image = cv2.cvtColor(np.rot90(rgb, k=3), cv2.COLOR_RGB2BGR)
        artifacts.failure_context.update(
            {
                "source_timestamp": float(timestamp),
                "image_shape_hw": list(image.shape[:2]),
            }
        )
        raw_path = artifacts.add_image(
            "raw_image",
            f"{self.sequence:03d}_head_raw.png",
            image,
        )
        sam_image = (
            enhance_low_light(image) if float(image.mean()) < 35.0 else image
        )
        enhanced = sam_image is not image
        enhanced_path = artifacts.add_image(
            "sam_input_png",
            f"{self.sequence:03d}_head_sam_input.png",
            sam_image,
        )
        depth_burst_diagnostics = {}
        artifacts.failure_context["depth_burst"] = depth_burst_diagnostics
        (
            depth_frames,
            depth_timestamps,
            depth_rgb_sha256,
            depth_timing,
        ) = self._collect_depth_burst(
            rgb,
            timestamp,
            depth_raw,
            diagnostics=depth_burst_diagnostics,
        )
        artifacts.failure_context["depth"] = {
            **depth_timing,
            "frames_requested": int(self.args.depth_frames),
            "frames_used": len(depth_frames),
            "source_timestamps": list(depth_timestamps),
            "source_rgb_sha256": list(depth_rgb_sha256),
        }
        request_path = None
        request_jpeg = None
        request_groups = {}
        sam_requests = []
        artifacts.failure_context["sam_transport"] = {
            "request_image_format": "jpeg",
            "jpeg_quality": SAM_REQUEST_JPEG_QUALITY,
            "requests": sam_requests,
        }

        def segment_and_record(
            role,
            prompt,
            confidence_threshold,
            *,
            request_image=None,
            request_group="full_frame",
            input_artifact_role="sam_input_png",
            roi_metadata=None,
        ):
            nonlocal request_path, request_jpeg
            if request_image is None:
                request_image = sam_image
            description = None

            def observe_request(wire_request):
                nonlocal request_path, request_jpeg, description
                if len(wire_request) != 2:
                    raise RuntimeError(
                        "SAM request observer expected two multipart frames"
                    )
                wire_metadata, wire_jpeg = (
                    bytes(part) for part in wire_request
                )
                captured = request_groups.get(request_group)
                if captured is None:
                    artifact_role = (
                        "sam_request_jpeg_q90"
                        if request_group == "full_frame"
                        else f"sam_{request_group}_request_jpeg_q90"
                    )
                    captured_path = artifacts.add_bytes(
                        artifact_role,
                        (
                            f"{self.sequence:03d}_head_"
                            f"{request_group}_request_q90.jpg"
                        ),
                        wire_jpeg,
                        media_type="image/jpeg",
                    )
                    request_groups[request_group] = (
                        wire_jpeg,
                        captured_path,
                        artifact_role,
                    )
                    if request_group == "full_frame":
                        request_jpeg = wire_jpeg
                        request_path = captured_path
                elif wire_jpeg != captured[0]:
                    raise RuntimeError(
                        f"SAM {request_group} request JPEG changed within "
                        "one observation"
                    )
                _, captured_path, artifact_role = request_groups[
                    request_group
                ]
                request_index = len(sam_requests)
                metadata_artifact_role = (
                    f"sam_request_{request_index:03d}_wire_metadata"
                )
                metadata_path = artifacts.add_bytes(
                    metadata_artifact_role,
                    (
                        f"{self.sequence:03d}_head_sam_request_"
                        f"{request_index:03d}_metadata.json"
                    ),
                    wire_metadata,
                    media_type="application/json",
                )
                description = {
                    "role": str(role),
                    "request_group": str(request_group),
                    "input_artifact_role": str(input_artifact_role),
                    "request_artifact_role": artifact_role,
                    "request_artifact_path": captured_path.name,
                    "wire_metadata_artifact_role": metadata_artifact_role,
                    "wire_metadata_artifact_path": metadata_path.name,
                    "wire_metadata": json.loads(wire_metadata),
                    "wire_metadata_sha256": _sha256_bytes(
                        wire_metadata
                    ),
                    "jpeg_sha256": _sha256_bytes(wire_jpeg),
                    "outcome": "prepared_before_send",
                    "response": None,
                }
                if roi_metadata is not None:
                    description["roi"] = copy.deepcopy(roi_metadata)
                sam_requests.append(description)

            try:
                result = self.sam.segment(
                    request_image,
                    frame_id=self.frame_id,
                    timestamp=timestamp,
                    prompt=prompt,
                    confidence_threshold=confidence_threshold,
                    jpeg_quality=SAM_REQUEST_JPEG_QUALITY,
                    request_observer=observe_request,
                )
                if (
                    not np.isfinite(result.source_timestamp)
                    or abs(
                        float(result.source_timestamp) - timestamp
                    )
                    > 1e-6
                ):
                    raise RuntimeError(
                        "segmentation response source timestamp does not "
                        "match request"
                    )
            except BaseException as exc:
                if description is not None:
                    self.frame_id += 1
                    description["outcome"] = "segment_failed"
                    description["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                raise
            if description is None:
                raise RuntimeError(
                    "SAM client returned without observing its request"
                )
            self.frame_id += 1
            description["outcome"] = "response_ok"
            description["response"] = {
                "frame_id": int(result.frame_id),
                "source_timestamp": float(result.source_timestamp),
                "model": str(result.model),
                "inference_ms": float(result.inference_ms),
                "candidate_count": len(result.candidates),
            }
            return result

        lid = None
        selected_lid = None
        selected_lid_prompt = None
        lid_attempts = []
        for prompt in (
            "transparent round petri dish lid with blue cross",
            "petri dish lid",
            "round transparent plastic dish",
        ):
            lid = segment_and_record(
                "lid",
                prompt,
                0.05,
            )
            lid_attempts.append((prompt, len(lid.candidates)))
            selected_lid = choose_lid_candidate(
                lid.candidates,
                image_bgr=sam_image,
                previous_center_px=self.previous_lid_center,
                require_blue_cross=True,
            )
            if selected_lid is not None:
                selected_lid_prompt = prompt
                break
        if selected_lid is None:
            raise ValueError(
                "SAM did not identify the blue-cross lid; "
                f"attempts={lid_attempts}, raw={raw_path}, "
                f"input={enhanced_path}"
            )
        lid_candidate, lid_geometry = selected_lid
        coarse_lid_image_margin = _require_candidate_image_margin(
            lid_candidate,
            image.shape,
            label="lid",
        )
        gripper = segment_and_record(
            "gripper",
            "blue clamp",
            0.10,
        )
        coarse_gripper_candidate = choose_right_gripper(
            gripper.candidates,
            image_width=image.shape[1],
            previous_center_px=self.previous_gripper_center,
        )
        coarse_gripper_image_margin = _require_candidate_image_margin(
            coarse_gripper_candidate,
            image.shape,
            label="coarse gripper",
        )
        roi = compute_candidate_roi(
            (lid_candidate, coarse_gripper_candidate),
            full_shape_hw=image.shape[:2],
            padding_px=72.0,
            scale=1.0,
        )
        roi_scale = min(
            4.0,
            1536.0 / float(max(roi.crop_shape_hw)),
        )
        if roi_scale < 1.5:
            raise RuntimeError(
                "coarse SAM instances span too much of the frame for "
                "meaningful ROI refinement"
            )
        roi = compute_candidate_roi(
            (lid_candidate, coarse_gripper_candidate),
            full_shape_hw=image.shape[:2],
            padding_px=72.0,
            scale=roi_scale,
        )
        roi_image = extract_enlarged_roi(sam_image, roi)
        roi_input_path = artifacts.add_image(
            "sam_roi_input_png",
            f"{self.sequence:03d}_head_sam_roi_input.png",
            roi_image,
        )
        roi_request_metadata = roi.metadata()
        fine_lid = segment_and_record(
            "lid_roi_refined",
            selected_lid_prompt,
            0.05,
            request_image=roi_image,
            request_group="roi",
            input_artifact_role="sam_roi_input_png",
            roi_metadata=roi_request_metadata,
        )
        lid, fine_lid_remap = remap_segmentation_result_from_roi(
            fine_lid, roi
        )
        selected_lid = choose_lid_candidate(
            lid.candidates,
            image_bgr=sam_image,
            previous_center_px=self.previous_lid_center,
            require_blue_cross=True,
        )
        if selected_lid is None:
            raise RuntimeError(
                "ROI-refined SAM did not preserve the blue-cross lid"
            )
        lid_candidate, lid_geometry = selected_lid
        fine_lid_candidate_index = _identity_index(
            lid.candidates, lid_candidate
        )
        lid_image_margin = _require_candidate_image_margin(
            lid_candidate,
            image.shape,
            label="ROI-refined lid",
        )
        lid_roi_margin = _require_candidate_roi_margin(
            lid_candidate,
            roi,
            label="ROI-refined lid",
        )
        fine_lid_raw_mask_path = artifacts.add_image(
            "sam_lid_roi_selected_raw_mask",
            f"{self.sequence:03d}_head_sam_lid_roi_selected_raw_mask.png",
            np.asarray(
                fine_lid.candidates[fine_lid_candidate_index].mask,
                dtype=np.uint8,
            )
            * 255,
        )
        fine_gripper = segment_and_record(
            "gripper_roi_refined",
            "blue clamp",
            0.10,
            request_image=roi_image,
            request_group="roi",
            input_artifact_role="sam_roi_input_png",
            roi_metadata=roi_request_metadata,
        )
        gripper, fine_gripper_remap = remap_segmentation_result_from_roi(
            fine_gripper, roi
        )
        (
            fine_gripper_candidate,
            gripper_association,
        ) = _associate_fine_gripper(
            coarse_gripper_candidate,
            gripper.candidates,
        )
        selected_fine_gripper = choose_right_gripper(
            (fine_gripper_candidate,),
            image_width=image.shape[1],
            previous_center_px=self.previous_gripper_center,
        )
        if selected_fine_gripper is not fine_gripper_candidate:
            raise RuntimeError("fine gripper association changed unexpectedly")
        fine_gripper_candidate_index = int(
            gripper_association["fine_candidate_index"]
        )
        gripper_image_margin = _require_candidate_image_margin(
            fine_gripper_candidate,
            image.shape,
            label="ROI-refined gripper",
        )
        gripper_roi_margin = _require_candidate_roi_margin(
            fine_gripper_candidate,
            roi,
            label="ROI-refined gripper",
        )
        fine_gripper_raw_mask_path = artifacts.add_image(
            "sam_gripper_roi_selected_raw_mask",
            (
                f"{self.sequence:03d}_head_"
                "sam_gripper_roi_selected_raw_mask.png"
            ),
            np.asarray(
                fine_gripper.candidates[fine_gripper_candidate_index].mask,
                dtype=np.uint8,
            )
            * 255,
        )
        post_sam_roi_image, scene_consistency = (
            self._verify_post_sam_roi_consistency(
                roi=roi,
                reference_roi_image=roi_image,
                enhanced=enhanced,
            )
        )
        post_sam_roi_path = artifacts.add_image(
            "sam_post_inference_roi_input_png",
            f"{self.sequence:03d}_head_sam_post_inference_roi_input.png",
            post_sam_roi_image,
        )
        roi_refinement = {
            "schema": "sam_roi_refinement/v1",
            "input_artifact_role": "sam_roi_input_png",
            "input_artifact_path": roi_input_path.name,
            "coarse_lid_image_margin": coarse_lid_image_margin,
            "coarse_gripper_image_margin": coarse_gripper_image_margin,
            "transform": roi_request_metadata,
            "lid_remap": fine_lid_remap,
            "gripper_remap": fine_gripper_remap,
            "lid_selected_candidate": {
                "response_frame_id": int(fine_lid.frame_id),
                "candidate_index": fine_lid_candidate_index,
                "raw_roi_mask_artifact_role": (
                    "sam_lid_roi_selected_raw_mask"
                ),
                "raw_roi_mask_artifact_path": fine_lid_raw_mask_path.name,
                "remapped_mask_artifact_role": "lid_mask",
                "roi_margin": lid_roi_margin,
            },
            "gripper_selected_candidate": {
                "response_frame_id": int(fine_gripper.frame_id),
                "candidate_index": fine_gripper_candidate_index,
                "raw_roi_mask_artifact_role": (
                    "sam_gripper_roi_selected_raw_mask"
                ),
                "raw_roi_mask_artifact_path": (
                    fine_gripper_raw_mask_path.name
                ),
                "remapped_mask_artifact_role": "gripper_mask",
                "roi_margin": gripper_roi_margin,
                "coarse_association": gripper_association,
            },
            "post_inference_scene_consistency": {
                **scene_consistency,
                "artifact_role": "sam_post_inference_roi_input_png",
                "artifact_path": post_sam_roi_path.name,
            },
        }
        # The temporal depth burst was captured around the source RGB timestamp
        # before any network inference.  The post-SAM RGB gate above ensures
        # that the ROI did not change while those requests were in flight.
        depth = temporal_median_depth(
            depth_frames, rotate_clockwise=True
        )
        native_depth_shape = depth.shape
        depth = cv2.resize(
            depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        grasp_px = lid_left_grasp_px(lid_candidate, lid_geometry)
        camera_matrix = scaled_camera_matrix(
            self.head_camera_matrix,
            self.head_camera_reference_shape,
            depth.shape,
        )
        self.last_depth = depth
        self.last_camera_matrix = camera_matrix
        target_3d = estimate_target_on_support_plane(
            depth,
            camera_matrix,
            lid_candidate.mask,
            grasp_px,
            ring_margin_px=int(
                self.support_plane_config.get("ring_margin_px", 70)
            ),
            plane_threshold_m=float(
                self.support_plane_config.get(
                    "ransac_threshold_m", 0.006
                )
            ),
        )
        minimum_plane_confidence = float(
            self.support_plane_config.get("minimum_confidence", 0.20)
        )
        if target_3d.confidence < minimum_plane_confidence:
            raise RuntimeError(
                "low-confidence local support plane: "
                f"{target_3d.confidence:.3f}"
            )
        self.last_target_3d = target_3d
        self.last_geometry_quality = assess_target_geometry(
            target_3d,
            camera_matrix,
            native_pixel_stride_xy=(
                image.shape[1] / native_depth_shape[1],
                image.shape[0] / native_depth_shape[0],
            ),
            maximum_view_angle_deg=float(
                self.geometry_quality_config.get(
                    "maximum_view_angle_deg", 40.0
                )
            ),
            maximum_native_footprint_m=float(
                self.geometry_quality_config.get(
                    "maximum_native_depth_pixel_footprint_m", 0.007
                )
            ),
        )
        feature = scene_feature(
            lid_candidates=lid.candidates,
            gripper_candidates=(fine_gripper_candidate,),
            depth_m=depth,
            previous_lid_center_px=self.previous_lid_center,
            previous_gripper_center_px=self.previous_gripper_center,
            clearance_m=clearance_m,
            lid_support_depth_m=float(
                target_3d.point_camera_xyz_m[2]
            ),
            selected_lid=selected_lid,
            image_bgr=image,
        )
        if feature.gripper_candidate is not fine_gripper_candidate:
            raise RuntimeError("scene feature changed the associated gripper")
        lid_mask_path = artifacts.add_image(
            "lid_mask",
            f"{self.sequence:03d}_head_lid_mask.png",
            np.asarray(feature.lid_candidate.mask, dtype=np.uint8) * 255,
        )
        gripper_mask_path = artifacts.add_image(
            "gripper_mask",
            f"{self.sequence:03d}_head_gripper_mask.png",
            np.asarray(feature.gripper_candidate.mask, dtype=np.uint8)
            * 255,
        )
        gripper_feature_support_path = None
        gripper_feature_support = getattr(
            feature, "gripper_feature_support_mask", None
        )
        if gripper_feature_support is not None:
            gripper_feature_support = np.asarray(
                gripper_feature_support, dtype=bool
            )
            if (
                gripper_feature_support.shape != image.shape[:2]
                or np.count_nonzero(gripper_feature_support)
                < GRIPPER_COLOR_MINIMUM_PIXELS
            ):
                raise RuntimeError(
                    "cyan gripper feature support is malformed"
                )
            gripper_feature_support_path = artifacts.add_image(
                "gripper_feature_support_mask",
                (
                    f"{self.sequence:03d}_head_"
                    "gripper_feature_support_mask.png"
                ),
                gripper_feature_support.astype(np.uint8) * 255,
            )
        depth_path = artifacts.add_npz(
            "depth_npz",
            f"{self.sequence:03d}_head_depth.npz",
            depth_m=np.asarray(depth).copy(),
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            source_timestamps=np.asarray(depth_timestamps, dtype=np.float64),
            source_rgb_sha256=np.asarray(depth_rgb_sha256),
            source_depth_sha256=np.asarray(
                depth_timing["source_depth_sha256"]
            ),
            image_timestamp=np.asarray(timestamp, dtype=np.float64),
            timestamp_span_s=np.asarray(
                depth_timing["timestamp_span_s"], dtype=np.float64
            ),
            maximum_timestamp_span_s=np.asarray(
                depth_timing["maximum_timestamp_span_s"],
                dtype=np.float64,
            ),
            native_depth_shape_hw=np.asarray(
                native_depth_shape, dtype=np.int64
            ),
        )
        self.previous_lid_center = feature.lid_geometry.center_px.copy()
        self.previous_gripper_center = gripper_mask_center(
            feature.gripper_candidate
        )
        xyz = backproject(depth, camera_matrix)
        exclusion = (
            np.asarray(feature.lid_candidate.mask, dtype=np.uint8)
            | np.asarray(feature.gripper_candidate.mask, dtype=np.uint8)
        )
        exclusion = cv2.dilate(
            exclusion,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        ).astype(bool)
        scene_valid = (
            ~exclusion
            & np.isfinite(depth)
            & (depth > 0.05)
            & (depth < 5.0)
        )
        tip = np.rint(feature.gripper_feature[:2]).astype(int)
        tip[0] = np.clip(tip[0], 0, depth.shape[1] - 1)
        tip[1] = np.clip(tip[1], 0, depth.shape[0] - 1)
        tool_point = xyz[tip[1], tip[0]]
        self.last_proximity_m = nearest_scene_distance(
            tool_point, xyz[scene_valid]
        )
        error = feature.lid_grasp_feature - feature.gripper_feature
        proximity_label = (
            "WARN"
            if self.last_proximity_m < self.proximity_warning_m
            else "ok"
        )
        geometry_label = (
            "ok" if self.last_geometry_quality.accepted else "BAD"
        )
        label = (
            f"live SAM seq={self.sequence} error="
            f"({error[0]:.1f}px,{error[1]:.1f}px,{error[2]:.1f}mm) "
            f"plane={target_3d.confidence:.2f} "
            f"view={self.last_geometry_quality.view_angle_deg:.0f}deg"
            f"({geometry_label}) "
            f"near={self.last_proximity_m*1000:.0f}mm({proximity_label})"
        )
        overlay = render_scene(image, feature, label)
        path = artifacts.add_image(
            "overlay_image",
            f"{self.sequence:03d}.png",
            overlay,
        )
        if request_path is None or request_jpeg is None:
            raise RuntimeError("observation has no captured SAM request")
        metadata = {
            "source_timestamp": timestamp,
            "image_shape_hw": list(image.shape[:2]),
            "image_dtype": str(image.dtype),
            "image_color_order": "BGR",
            "head_rotation": "clockwise_90",
            "preprocess": (
                "enhance_low_light" if enhanced else "identity"
            ),
            "sam_transport": {
                "request_image_format": "jpeg",
                "jpeg_quality": SAM_REQUEST_JPEG_QUALITY,
                "requests": sam_requests,
            },
            "roi_refinement": roi_refinement,
            "depth": {
                "representation": (
                    "temporal_median_rotate_clockwise_aligned_nearest"
                ),
                "frames_requested": int(self.args.depth_frames),
                "frames_used": len(depth_frames),
                "source_timestamps": depth_timestamps,
                "source_rgb_sha256": depth_rgb_sha256,
                **depth_timing,
                "native_shape_hw": list(native_depth_shape),
                "aligned_shape_hw": list(depth.shape),
                "dtype": str(depth.dtype),
                "camera_matrix_dtype": "float64",
                "camera_matrix": np.asarray(
                    camera_matrix, dtype=float
                ).tolist(),
            },
            "target_3d": {
                "pixel_xy": np.asarray(
                    target_3d.pixel_xy, dtype=float
                ).tolist(),
                "point_camera_xyz_m": np.asarray(
                    target_3d.point_camera_xyz_m, dtype=float
                ).tolist(),
                "support_plane": {
                    "normal_camera_xyz": np.asarray(
                        target_3d.plane.normal, dtype=float
                    ).tolist(),
                    "offset_m": float(target_3d.plane.offset),
                    "inlier_fraction": float(
                        target_3d.plane.inlier_fraction
                    ),
                    "rms_m": float(target_3d.plane.rms_m),
                },
                "support_sample_count": int(
                    target_3d.support_sample_count
                ),
                "confidence": float(target_3d.confidence),
                "minimum_confidence": minimum_plane_confidence,
            },
            "geometry_quality": {
                "accepted": bool(self.last_geometry_quality.accepted),
                "view_angle_deg": float(
                    self.last_geometry_quality.view_angle_deg
                ),
                "native_depth_pixel_footprint_m": [
                    float(
                        self.last_geometry_quality
                        .native_pixel_footprint_x_m
                    ),
                    float(
                        self.last_geometry_quality
                        .native_pixel_footprint_y_m
                    ),
                ],
                "maximum_view_angle_deg": float(
                    self.geometry_quality_config.get(
                        "maximum_view_angle_deg", 40.0
                    )
                ),
                "maximum_native_depth_pixel_footprint_m": float(
                    self.geometry_quality_config.get(
                        "maximum_native_depth_pixel_footprint_m", 0.007
                    )
                ),
                "reasons": list(self.last_geometry_quality.reasons),
            },
            "proximity": {
                "nearest_scene_distance_m": (
                    float(self.last_proximity_m)
                    if np.isfinite(self.last_proximity_m)
                    else None
                ),
                "distance_is_finite": bool(
                    np.isfinite(self.last_proximity_m)
                ),
                "warning_threshold_m": float(self.proximity_warning_m),
                "warning": bool(
                    np.isfinite(self.last_proximity_m)
                    and self.last_proximity_m < self.proximity_warning_m
                ),
                "policy": "warning_only",
            },
            "lid_attempts": [
                {"prompt": prompt, "candidate_count": int(count)}
                for prompt, count in lid_attempts
            ],
            "lid": {
                "prompt": selected_lid_prompt,
                "confidence_threshold": 0.05,
                "model": lid.model,
                "frame_id": int(lid.frame_id),
                "score": float(feature.lid_candidate.score),
                "box_xyxy": np.asarray(
                    feature.lid_candidate.box_xyxy, dtype=float
                ).tolist(),
                "image_margin": lid_image_margin,
            },
            "gripper": {
                "prompt": "blue clamp",
                "confidence_threshold": 0.10,
                "model": gripper.model,
                "frame_id": int(gripper.frame_id),
                "score": float(feature.gripper_candidate.score),
                "box_xyxy": np.asarray(
                    feature.gripper_candidate.box_xyxy, dtype=float
                ).tolist(),
                "image_margin": gripper_image_margin,
                "feature_extractor": {
                    "schema": "sam_hsv_gripper_tip/v1",
                    "semantic_source": "roi_refined_sam_mask",
                    "colour_space": "HSV",
                    "hsv_lower": list(GRIPPER_CYAN_HSV_LOWER),
                    "hsv_upper": list(GRIPPER_CYAN_HSV_UPPER),
                    "minimum_pixels": GRIPPER_COLOR_MINIMUM_PIXELS,
                    "minimum_sam_mask_fraction": (
                        GRIPPER_COLOR_MINIMUM_MASK_FRACTION
                    ),
                    "connected_component": "largest_8_connected",
                    "tip": "longitudinal_right_terminal_percentile_99",
                    "depth_support": "same_colour_component",
                    "support_pixel_count": (
                        int(np.count_nonzero(gripper_feature_support))
                        if gripper_feature_support is not None
                        else None
                    ),
                    "support_artifact_role": (
                        "gripper_feature_support_mask"
                        if gripper_feature_support_path is not None
                        else None
                    ),
                    "support_artifact_path": (
                        gripper_feature_support_path.name
                        if gripper_feature_support_path is not None
                        else None
                    ),
                },
            },
            "feature": {
                "clearance_m": float(clearance_m),
                "lid_center_px": np.asarray(
                    feature.lid_geometry.center_px, dtype=float
                ).tolist(),
                "lid_grasp_feature": np.asarray(
                    feature.lid_grasp_feature, dtype=float
                ).tolist(),
                "gripper_feature": np.asarray(
                    feature.gripper_feature, dtype=float
                ).tolist(),
                "error": np.asarray(error, dtype=float).tolist(),
            },
        }
        (
            artifact_document,
            manifest_path,
            manifest_sha256,
        ) = artifacts.finish(metadata)
        self.last_observation_artifacts = {
            **artifact_document,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            # Absolute compatibility paths for probe-record writers.
            "raw_image": str(raw_path),
            "sam_input_png": str(enhanced_path),
            "sam_request_jpeg_q90": str(request_path),
            "overlay_image": str(path),
            "lid_mask": str(lid_mask_path),
            "gripper_mask": str(gripper_mask_path),
            "depth_npz": str(depth_path),
        }
        return feature, error, str(path), float(timestamp)

    def check_scene_registration(self, reference_points_path: str | None):
        """Validate head-camera placement without detecting an AprilTag."""

        if not reference_points_path:
            return None
        if self.last_depth is None or self.last_camera_matrix is None:
            raise RuntimeError("observe a depth frame before registration")
        xyz = backproject(self.last_depth, self.last_camera_matrix)
        valid = (
            np.all(np.isfinite(xyz), axis=2)
            & (self.last_depth >= 0.20)
            & (self.last_depth <= 2.00)
        )
        config = self.registration_config
        result = register_point_clouds(
            xyz[valid],
            np.load(reference_points_path),
            max_correspondence_m=float(
                config.get("max_correspondence_m", 0.035)
            ),
            acceptance_rmse_m=float(
                config.get("maximum_rmse_m", 0.012)
            ),
            acceptance_inlier_fraction=float(
                config.get("minimum_inlier_fraction", 0.55)
            ),
        )
        report = {
            "accepted": result.accepted,
            "rmse_m": result.rmse_m,
            "inlier_fraction": result.inlier_fraction,
            "iterations": result.iterations,
            "live_to_reference": result.live_to_reference.tolist(),
        }
        (self.output_dir / "scene_registration.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if not result.accepted:
            raise RuntimeError(
                "head scene registration rejected; do not use old alignment"
            )
        return report

    def hold_measured(self):
        q = np.asarray(self.rpc.get_right_joint_positions(), dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise RuntimeError(
                "refusing to hold invalid measured right joint state"
            )
        self.rpc.set_right_joint_target(
            q, gripper_target=None, preview_time=0.2
        )
        self.joint_command = None

    def monitor_settle(self, duration_s: float):
        strikes = 0
        deadline = time.time() + duration_s
        while time.time() < deadline:
            torque = np.abs(
                np.asarray(self.rpc.get_right_joint_torque(), dtype=float)
            )
            strikes = strikes + 1 if np.any(torque > self.torque_limit) else 0
            if strikes >= self.torque_samples:
                self.hold_measured()
                raise TorqueStop(
                    f"right torque stop: {np.round(torque, 3).tolist()}"
                )
            time.sleep(0.05)

    @staticmethod
    def _wait_with_torque(
        duration_s, check_torque, stage, check_state=None
    ):
        if duration_s <= 0.0:
            return
        deadline = time.monotonic() + duration_s
        while True:
            check_torque(stage)
            if check_state is not None:
                check_state(stage)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(0.05, remaining))

    def _prepare_cartesian_motion(self, check_torque):
        # First latch the measured joints. Reasserting MIT mode can otherwise
        # activate a stale target left by an earlier client.
        def measured_state():
            q = np.asarray(
                self.rpc.get_right_joint_positions(), dtype=float
            )
            xyz = np.asarray(
                self.rpc.get_right_ee_pose().translation(), dtype=float
            )
            if (
                q.shape != (6,)
                or xyz.shape != (3,)
                or not np.all(np.isfinite(q))
                or not np.all(np.isfinite(xyz))
            ):
                raise RuntimeError("invalid measured right-arm state")
            return q, xyz

        start_q, start_xyz = measured_state()

        def check_preparation_drift(stage):
            q, xyz = measured_state()
            if (
                np.max(np.abs(q - start_q)) > 0.02
                or np.linalg.norm(xyz - start_xyz) > 0.002
            ):
                raise RuntimeError(
                    f"unexpected right-arm motion {stage}"
                )

        self.hold_measured()
        self.rpc.set_right_gain(self.holding_kp, self.holding_kd)
        self._wait_with_torque(
            self.hold_settle_s,
            check_torque,
            "while latching measured right pose",
            check_preparation_drift,
        )
        check_preparation_drift("while latching measured pose")
        self.motion_mode_refresher()
        self._wait_with_torque(
            self.mode_settle_s,
            check_torque,
            "after right MIT mode refresh",
            check_preparation_drift,
        )
        check_preparation_drift("after MIT mode refresh")

        if self.gain_ramp_s <= 0.0:
            self.rpc.set_right_gain(self.motion_kp, self.motion_kd)
            check_torque("after right motion gain")
            check_preparation_drift("after right motion gain")
        else:
            steps = max(1, int(np.ceil(self.gain_ramp_s / 0.1)))
            step_duration = self.gain_ramp_s / steps
            for index in range(steps):
                # Re-latch the measured pose before every stiffness increase.
                # If the arm yielded under low gain, this avoids pulling it
                # back toward a stale target as gain rises.
                self.hold_measured()
                alpha = (index + 1) / steps
                kp = (
                    self.holding_kp * (1.0 - alpha)
                    + self.motion_kp * alpha
                )
                kd = (
                    self.holding_kd * (1.0 - alpha)
                    + self.motion_kd * alpha
                )
                self.rpc.set_right_gain(kp, kd)
                self._wait_with_torque(
                    step_duration,
                    check_torque,
                    f"during right gain ramp {index + 1}/{steps}",
                    check_preparation_drift,
                )
                check_preparation_drift(
                    f"during gain ramp {index + 1}/{steps}"
                )

        # The Piper V2 MIT examples reassert 0x151 immediately before each
        # motion command. The first refresh above verifies a stable hold; this
        # second one removes any ambiguity before the first non-zero target.
        self.motion_mode_refresher()
        check_torque("after final right MIT mode refresh")
        check_preparation_drift("after final MIT mode refresh")

    def _finish_cartesian_motion(self):
        # Hold before lowering stiffness so a completed or interrupted move
        # cannot resume toward an old interpolator waypoint.
        hold_error = None
        try:
            self.hold_measured()
        except BaseException as exc:
            hold_error = exc
        try:
            self.rpc.set_right_gain(self.holding_kp, self.holding_kd)
        except BaseException as gain_error:
            if hold_error is not None:
                _note_exception(
                    hold_error,
                    f"holding-gain restoration also failed: {gain_error!r}"
                )
            else:
                raise
        if hold_error is not None:
            raise hold_error

    def move_cartesian_delta(
        self, delta_xyz, preview_time=None, minimum_progress=None
    ):
        if preview_time is None:
            preview_time = self.args.preview_time
        if minimum_progress is None:
            minimum_progress = self.args.minimum_progress
        preview_time = float(preview_time)
        if not np.isfinite(preview_time) or not 0.0 < preview_time <= 10.0:
            raise ValueError("preview_time must be within (0, 10] seconds")
        requested = np.asarray(delta_xyz, dtype=float).reshape(3)
        if not np.all(np.isfinite(requested)):
            raise ValueError("Cartesian delta must contain three finite values")

        # ConeE teleoperation succeeds by refreshing 0.05-second targets at
        # 30 Hz.  A single long-horizon target can stall under the deliberately
        # low position gains, so stream the same straight path with exactly
        # those proven timing parameters while checking torque every tick.
        command_rate_hz = 30.0
        steps = max(1, int(np.ceil(preview_time * command_rate_hz)))
        period_s = preview_time / steps
        command_preview_s = 0.05
        strikes = 0

        def check_torque(stage):
            nonlocal strikes
            torque = np.abs(
                np.asarray(self.rpc.get_right_joint_torque(), dtype=float)
            )
            if (
                torque.shape != self.torque_limit.shape
                or not np.all(np.isfinite(torque))
            ):
                raise TorqueStop(
                    f"invalid right torque sample {stage}: "
                    f"shape={torque.shape}, values={torque.tolist()}"
                )
            strikes = (
                strikes + 1
                if np.any(torque > self.torque_limit)
                else 0
            )
            if strikes >= self.torque_samples:
                raise TorqueStop(
                    f"right torque stop {stage}: "
                    f"{np.round(torque, 3).tolist()}"
                )

        motion_error = None
        try:
            self._prepare_cartesian_motion(check_torque)
            before = np.asarray(
                self.rpc.get_right_ee_pose().parameters(), dtype=float
            )
            if before.shape != (7,) or not np.all(np.isfinite(before)):
                raise RuntimeError(
                    "refusing to move from invalid measured right pose"
                )
            # Preparation deliberately takes longer than a short probe.  Start
            # the streaming deadline only after the arm is stably in MIT mode
            # with the motion gains applied.
            started = time.monotonic()
            for index in range(steps):
                check_torque("during streamed move")

                target = before.copy()
                target[4:7] += requested * ((index + 1) / steps)
                accepted = self.rpc.set_right_ee_target(
                    mink.SE3(target),
                    gripper_target=None,
                    preview_time=command_preview_s,
                )
                if accepted is not True:
                    raise RuntimeError(
                        f"right Cartesian setpoint {index + 1}/{steps} rejected"
                    )

                deadline = started + (index + 1) * period_s
                remaining = deadline - time.monotonic()
                if remaining < -period_s:
                    raise RuntimeError(
                        "right Cartesian streaming missed its control "
                        f"deadline at setpoint {index + 1}/{steps}"
                    )
                if remaining > 0.0:
                    time.sleep(remaining)

            settle_deadline = time.monotonic() + command_preview_s + 0.15
            while True:
                check_torque("while settling after streamed move")
                remaining = settle_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.05, remaining))
            check_torque("after streamed move settled")
        except BaseException as exc:
            motion_error = exc
            raise
        finally:
            try:
                self._finish_cartesian_motion()
            except BaseException as cleanup_error:
                if motion_error is None:
                    raise
                _note_exception(
                    motion_error,
                    f"right-arm cleanup also failed: {cleanup_error!r}"
                )

        after = np.asarray(
            self.rpc.get_right_ee_pose().parameters(), dtype=float
        )
        if after.shape != (7,) or not np.all(np.isfinite(after)):
            raise RuntimeError("invalid right pose after Cartesian move")
        actual = after[4:7] - before[4:7]
        if np.linalg.norm(requested) > 0.001:
            progress = float(
                np.dot(actual, requested)
                / np.dot(requested, requested)
            )
            if progress < minimum_progress:
                raise RuntimeError(
                    f"right arm did not follow request: {progress:.2f}"
                )
        return actual

    def move_joint_delta(
        self,
        delta_joints,
        preview_time=None,
        minimum_progress=None,
        *,
        accumulate=False,
    ):
        if preview_time is None:
            preview_time = self.args.preview_time
        if minimum_progress is None:
            minimum_progress = self.args.joint_minimum_progress
        before = np.asarray(
            self.rpc.get_right_joint_positions(), dtype=float
        )
        if accumulate:
            if self.joint_command is None:
                self.joint_command = before.copy()
            target = self.joint_command.copy()
            # The wrist is never accumulated or commanded away from its
            # measured state by this position-only visual servo.
            target[3:] = before[3:]
        else:
            target = before.copy()
        target[:3] += np.asarray(delta_joints, dtype=float)
        self.rpc.set_right_joint_target(
            target,
            gripper_target=None,
            preview_time=preview_time,
        )
        self.monitor_settle(preview_time + 0.15)
        if accumulate:
            self.joint_command = target.copy()
        after = np.asarray(
            self.rpc.get_right_joint_positions(), dtype=float
        )
        actual = after[:3] - before[:3]
        requested = np.asarray(delta_joints, dtype=float)
        if np.linalg.norm(requested) > 0.003:
            progress = np.linalg.norm(actual) / np.linalg.norm(requested)
            if progress < minimum_progress:
                self.hold_measured()
                raise RuntimeError(
                    f"right joints did not follow request: {progress:.2f}"
                )
        return actual

    def move_control_delta(
        self, delta, *, minimum_progress=None, accumulate=False
    ):
        if self.args.control_space == "joint":
            kwargs = {}
            if minimum_progress is not None:
                kwargs["minimum_progress"] = minimum_progress
            return self.move_joint_delta(
                delta, accumulate=accumulate, **kwargs
            )
        return self.move_cartesian_delta(
            delta, minimum_progress=minimum_progress
        )

    def calibrate(self, clearance_m: float):
        if self.args.control_space == "joint":
            origin = np.asarray(
                self.rpc.get_right_joint_positions(), dtype=float
            )
            probe_sizes = np.array(
                [
                    self.args.joint_probe_rad,
                    self.args.joint_probe2_rad,
                    self.args.joint_probe3_rad,
                ],
                dtype=float,
            )
            calibration_min_progress = (
                self.args.joint_calibration_min_progress
            )
        else:
            origin = np.asarray(
                self.rpc.get_right_ee_pose().parameters(), dtype=float
            )
            self.orientation = origin[:4].copy()
            probe_sizes = np.full(3, self.args.probe_m, dtype=float)
            calibration_min_progress = 0.25
        robot_deltas = []
        feature_deltas = []
        for axis in range(3):
            probe_size = float(probe_sizes[axis])
            baseline = observed = None
            actual = None
            path = None
            direction = None
            for sign in (1.0, -1.0):
                baseline, _, _, _ = self.observe(clearance_m)
                probe = np.zeros(3)
                probe[axis] = sign * probe_size
                actual = self.move_control_delta(
                    probe, minimum_progress=0.0
                )
                progress = float(np.linalg.norm(actual) / probe_size)
                if progress >= calibration_min_progress:
                    observed, _, path, _ = self.observe(clearance_m)
                    direction = sign
                    break
                self.hold_measured()
            if direction is None:
                raise RuntimeError(
                    f"right arm could not produce probe motion on axis {axis}"
                )
            robot_deltas.append(actual)
            feature_deltas.append(
                observed.gripper_feature - baseline.gripper_feature
            )
            print(
                json.dumps(
                    {
                        "control_space": self.args.control_space,
                        "probe_axis": axis,
                        "probe_sign": direction,
                        "actual_m": actual.round(5).tolist(),
                        "feature_delta": feature_deltas[-1].round(2).tolist(),
                        "image": path,
                    }
                ),
                flush=True,
            )
            # Do not force a return to the exact origin between probes. Near a
            # joint branch boundary the reverse IK can fail even though each
            # small forward probe is valid. Differential measurements from
            # consecutive fresh observations are sufficient for the Jacobian.
        model = estimate_reachable_feature_model(robot_deltas, feature_deltas)
        print(
            json.dumps(
                {
                    "reachable_rank": model.rank,
                    "reachable_basis": model.basis_xyz.round(4).tolist(),
                    "feature_matrix": model.feature_matrix.round(2).tolist(),
                    "condition": model.condition,
                }
            ),
            flush=True,
        )
        return robot_deltas, feature_deltas, origin

    def approach(
        self, robot_deltas, feature_deltas, origin, clearance_m: float
    ):
        last_error_norm = None
        worse = 0
        tolerances = np.array(
            [
                self.args.xy_tolerance_px,
                self.args.xy_tolerance_px,
                self.args.depth_tolerance_mm,
            ],
            dtype=float,
        )
        for iteration in range(self.args.max_iters):
            feature, error, path, timestamp = self.observe(clearance_m)
            xy_ok = float(np.linalg.norm(error[:2])) <= self.args.xy_tolerance_px
            depth_ok = abs(float(error[2])) <= self.args.depth_tolerance_mm
            report = {
                "iteration": iteration,
                "error": error.round(2).tolist(),
                "right_pose": np.asarray(
                    self.rpc.get_right_ee_pose().parameters()
                ).round(6).tolist(),
                "image": path,
                "timestamp": timestamp,
            }
            print(json.dumps(report), flush=True)
            if xy_ok and depth_ok:
                return feature, path
            error_norm = float(
                np.linalg.norm(error / tolerances)
            )
            if last_error_norm is not None and error_norm > 1.25 * last_error_norm:
                worse += 1
                if worse >= 2:
                    self.hold_measured()
                    raise RuntimeError("real-time SAM error diverged")
            else:
                worse = 0
            model = estimate_reachable_feature_model(
                robot_deltas, feature_deltas
            )
            step = bounded_reachable_servo_step(
                model,
                error,
                tolerances=tolerances,
                max_norm_m=(
                    self.args.joint_max_step_rad
                    if self.args.control_space == "joint"
                    else self.args.max_step_m
                ),
                max_axis_m=(
                    self.args.joint_max_axis_rad
                    if self.args.control_space == "joint"
                    else self.args.max_axis_m
                ),
            )
            report = {
                "reachable_rank": model.rank,
                "condition": model.condition,
                "step_m": step.round(5).tolist(),
            }
            print(json.dumps(report), flush=True)
            if self.args.control_space == "cartesian":
                current = np.asarray(
                    self.rpc.get_right_ee_pose().parameters(), dtype=float
                )
                if (
                    np.linalg.norm(
                        current[4:7] + step - origin[4:7]
                    )
                    > 0.20
                ):
                    self.hold_measured()
                    raise RuntimeError(
                        "right-arm SAM servo excursion exceeded 200mm"
                    )
            before_gripper = feature.gripper_feature.copy()
            actual = self.move_control_delta(step, accumulate=True)
            after, _, _, _ = self.observe(clearance_m)
            delta_feature = after.gripper_feature - before_gripper
            minimum_update = (
                5e-4
                if self.args.control_space == "joint"
                else 3e-4
            )
            if np.linalg.norm(actual) >= minimum_update:
                robot_deltas.append(actual)
                feature_deltas.append(delta_feature)
                # Refit from recent real-time observations.  This adapts to
                # local geometry while zero-motion depth noise cannot erase
                # the last useful excitation.
                robot_deltas[:] = robot_deltas[-6:]
                feature_deltas[:] = feature_deltas[-6:]
            last_error_norm = error_norm
        self.hold_measured()
        raise RuntimeError("real-time SAM pregrasp did not converge")


def _stop_runner_without_masking(runner, primary_error):
    """Stop a runner while preserving an exception already in flight."""

    if runner is None:
        return
    try:
        runner.stop()
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        _note_exception(
            primary_error,
            f"runner cleanup also failed: {cleanup_error!r}",
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:15563")
    parser.add_argument("--output-dir", default="/tmp/realtime_sam_grasp")
    parser.add_argument(
        "--torque-config", default="src/configs/pasteur_lid_torque.json"
    )
    parser.add_argument(
        "--scene-config",
        default="src/configs/pasteur_lid_scene3d.json",
    )
    parser.add_argument("--reference-points")
    parser.add_argument("--probe-m", type=float, default=0.008)
    parser.add_argument(
        "--control-space",
        choices=("cartesian", "joint"),
        default="cartesian",
    )
    parser.add_argument("--joint-probe-rad", type=float, default=0.025)
    parser.add_argument("--joint-probe2-rad", type=float, default=0.025)
    parser.add_argument("--joint-probe3-rad", type=float, default=0.025)
    parser.add_argument("--joint-max-step-rad", type=float, default=0.035)
    parser.add_argument("--joint-max-axis-rad", type=float, default=0.025)
    parser.add_argument(
        "--joint-minimum-progress", type=float, default=0.25
    )
    parser.add_argument(
        "--joint-calibration-min-progress", type=float, default=0.05
    )
    parser.add_argument("--clearance-m", type=float, default=0.040)
    parser.add_argument("--max-step-m", type=float, default=0.012)
    parser.add_argument("--max-axis-m", type=float, default=0.008)
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument("--minimum-progress", type=float, default=0.15)
    parser.add_argument("--depth-frames", type=int, default=15)
    parser.add_argument("--preview-time", type=float, default=1.2)
    parser.add_argument("--xy-tolerance-px", type=float, default=6.0)
    parser.add_argument("--depth-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--right-can-interface", default="can_right")
    parser.add_argument(
        "--holding-kp", type=float, nargs=6, default=DEFAULT_HOLDING_KP
    )
    parser.add_argument(
        "--holding-kd", type=float, nargs=6, default=DEFAULT_HOLDING_KD
    )
    parser.add_argument(
        "--motion-kp", type=float, nargs=6, default=DEFAULT_MOTION_KP
    )
    parser.add_argument(
        "--motion-kd", type=float, nargs=6, default=DEFAULT_MOTION_KD
    )
    parser.add_argument("--gain-ramp-s", type=float, default=1.0)
    parser.add_argument("--mode-settle-s", type=float, default=0.5)
    parser.add_argument("--hold-settle-s", type=float, default=0.25)
    parser.add_argument("--execute-pregrasp", action="store_true")
    args = parser.parse_args(argv)
    if args.execute_pregrasp:
        parser.error(
            "standalone 3D pregrasp execution is disabled; use "
            "run_staged_sam_pregrasp.py with --execute-horizontal and a "
            "new one-shot --motion-token"
        )

    runner = None
    primary_error = None
    try:
        runner = LiveSamGrasp(args)
        runner.start()
        # The first Record3D depth burst after stream startup is often an
        # outlier; warm the live temporal filter before reporting or moving.
        runner.observe(args.clearance_m)
        feature, error, path, timestamp = runner.observe(args.clearance_m)
        registration = runner.check_scene_registration(
            args.reference_points
        )
        initial = {
            "mode": "dry_run",
            "error": error.round(2).tolist(),
            "image": path,
            "timestamp": timestamp,
            "lid_center": feature.lid_geometry.center_px.round(2).tolist(),
            "lid_score": feature.lid_candidate.score,
            "gripper_score": feature.gripper_candidate.score,
            "proximity_warning_only_m": runner.last_proximity_m,
            "geometry_quality": {
                "accepted": runner.last_geometry_quality.accepted,
                "view_angle_deg": (
                    runner.last_geometry_quality.view_angle_deg
                ),
                "native_depth_pixel_footprint_mm": [
                    runner.last_geometry_quality.native_pixel_footprint_x_m
                    * 1000.0,
                    runner.last_geometry_quality.native_pixel_footprint_y_m
                    * 1000.0,
                ],
                "reasons": list(runner.last_geometry_quality.reasons),
            },
            "registration": registration,
        }
        print(json.dumps(initial), flush=True)
        return
    except BaseException as exc:
        primary_error = exc
        # This entry point is camera-only.  An observation failure must not
        # turn a dry run into a right-arm hold command.  Physical horizontal
        # execution lives in run_staged_sam_pregrasp.py, which tracks whether
        # a one-shot motion was actually attempted before emergency holding.
        raise
    finally:
        _stop_runner_without_masking(runner, primary_error)


if __name__ == "__main__":
    main()

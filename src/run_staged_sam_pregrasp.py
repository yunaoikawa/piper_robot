#!/usr/bin/env python3
"""SAM-only staged pregrasp: horizontal alignment before any descent."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_realtime_sam_grasp import LiveSamGrasp
from robot.camera_id import load_camera_map
from rollout.camera import USBWristCameraFeedManager
from rollout.local_feature_calibration import register_fixed_camera_view
from rollout.sam_segmentation import (
    PROTOCOL_VERSION,
    choose_lid_candidate,
    enhance_low_light,
)


def _atomic_write_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    encoded = (
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _exclusive_write_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class MotionExecutionClaim:
    """One-shot, process-independent authorization for physical motion."""

    def __init__(self, path: Path, payload):
        self.path = Path(path)
        self.payload = dict(payload)
        self.finalized = False
        self.result = None

    def set_result(self, result):
        self.result = result

    def mark_motion_attempt(self):
        if self.payload.get("motion_attempted") is True:
            return
        previous = copy.deepcopy(self.payload)
        attempted = datetime.now(timezone.utc).isoformat()
        self.payload["motion_attempted"] = True
        self.payload["motion_attempted_at_utc"] = attempted
        try:
            _atomic_write_json(self.path, self.payload)
        except BaseException:
            self.payload = previous
            raise

    def finalize(self, error: BaseException | None = None):
        if self.finalized:
            return
        payload = copy.deepcopy(self.payload)
        payload["finished_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        if error is None:
            payload["status"] = "completed"
            if self.result is not None:
                payload["result"] = copy.deepcopy(self.result)
        else:
            payload["status"] = "failed"
            payload["failure_timing"] = (
                "during_or_after_motion_attempt"
                if payload.get("motion_attempted") is True
                else "before_motion_attempt"
            )
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        _atomic_write_json(self.path, payload)
        self.payload = payload
        self.finalized = True


def claim_motion_execution(
    motion_token: str,
    claim_dir: str | Path,
    output_dir: str | Path,
    intent,
):
    """Atomically consume a token and reserve a never-overwritten run dir."""

    token = str(motion_token)
    if not token or len(token) > 256 or any(char.isspace() for char in token):
        raise ValueError("motion token must be non-empty and contain no spaces")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    claim_root = Path(claim_dir)
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_path = claim_root / f"{token_hash}.json"
    payload = {
        "schema": "piper_motion_claim/v1",
        "token_sha256": token_hash,
        "status": "claimed",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "output_dir": str(Path(output_dir).resolve()),
        "intent": intent,
        "motion_attempted": False,
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    try:
        descriptor = os.open(
            claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "motion token was already consumed; refusing duplicate execution"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(claim_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

    claim = MotionExecutionClaim(claim_path, payload)
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=False)
    except BaseException as exc:
        claim.finalize(exc)
        raise RuntimeError(
            "physical-run output directory already exists; refusing overwrite"
        ) from exc
    return claim


class ProbeExecutionJournal:
    """Durable phase journal written before a single physical probe."""

    def __init__(
        self,
        path,
        *,
        axis,
        requested_xyz_m,
        motion_token_sha256=None,
    ):
        self.path = Path(path).resolve() if path is not None else None
        self.axis = str(axis)
        started = datetime.now(timezone.utc).isoformat()
        self.payload = {
            "schema": "sam_horizontal_probe/v2",
            "record_id": uuid.uuid4().hex,
            "created_at_utc": started,
            "status": f"SINGLE_{axis.upper()}_PROBE_PREPARING",
            "execution": {
                "stage": "capturing_pre_motion_observation",
                "motion_attempted": False,
                "motion_command_completed": False,
                "immediate_motion_validated": False,
                "settled_motion_validated": False,
                "failure_timing": None,
                "started_at_utc": started,
                "updated_at_utc": started,
            },
            "motion": {
                "requested_xyz_m": np.asarray(
                    requested_xyz_m, dtype=float
                ).tolist(),
            },
        }
        if motion_token_sha256 is not None:
            token_hash = str(motion_token_sha256)
            if (
                len(token_hash) != 64
                or any(char not in "0123456789abcdef" for char in token_hash)
            ):
                raise ValueError(
                    "motion token SHA-256 must be a lowercase hex digest"
                )
            self.payload["authorization"] = {
                "motion_token_sha256": token_hash
            }
        if self.path is not None:
            _exclusive_write_json(self.path, self.payload)

    @property
    def motion_attempted(self):
        return bool(self.payload["execution"]["motion_attempted"])

    def update(self, stage, **sections):
        self.payload["execution"]["stage"] = str(stage)
        self.payload["execution"]["updated_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        self.payload.update(copy.deepcopy(sections))
        if self.path is not None:
            _atomic_write_json(self.path, self.payload)

    def mark_motion_attempt(self):
        execution = self.payload["execution"]
        previous = copy.deepcopy(execution)
        execution["motion_attempted"] = True
        execution["motion_attempted_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        try:
            self.update("motion_command_in_progress")
        except BaseException:
            self.payload["execution"] = previous
            raise

    def mark_motion_command_completed(self, motion):
        execution = self.payload["execution"]
        execution["motion_command_completed"] = True
        execution["motion_command_completed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        self.update("validating_immediate_motion", motion=motion)

    def mark_immediate_motion_validated(self, motion):
        self.payload["execution"]["immediate_motion_validated"] = True
        self.update("post_motion_settle", motion=motion)

    def mark_settled_motion_validated(self, motion):
        self.payload["execution"]["settled_motion_validated"] = True
        self.update("validating_post_motion_observation", motion=motion)

    def fail(self, error):
        finished = datetime.now(timezone.utc).isoformat()
        execution = self.payload["execution"]
        execution["failure_timing"] = (
            "during_or_after_motion_attempt"
            if execution["motion_attempted"]
            else "before_motion_attempt"
        )
        execution["failed_stage"] = execution["stage"]
        execution["finished_at_utc"] = finished
        execution["updated_at_utc"] = finished
        self.payload["status"] = (
            f"SINGLE_{self.axis.upper()}_PROBE_FAILED"
        )
        self.payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        if self.path is not None:
            _atomic_write_json(self.path, self.payload)

    def commit(self, report):
        finished = datetime.now(timezone.utc).isoformat()
        execution = self.payload["execution"]
        execution["stage"] = "committed"
        execution["failure_timing"] = None
        execution["finished_at_utc"] = finished
        execution["updated_at_utc"] = finished
        if (
            not execution["motion_attempted"]
            or not execution["motion_command_completed"]
            or not execution["immediate_motion_validated"]
            or not execution["settled_motion_validated"]
        ):
            raise RuntimeError(
                "cannot commit a probe without validated measured motion"
            )
        committed = copy.deepcopy(report)
        committed["schema"] = self.payload["schema"]
        committed["record_id"] = self.payload["record_id"]
        committed["created_at_utc"] = self.payload["created_at_utc"]
        committed["execution"] = copy.deepcopy(execution)
        if "authorization" in self.payload:
            committed["authorization"] = copy.deepcopy(
                self.payload["authorization"]
            )
        if self.path is not None:
            committed["record_path"] = str(self.path)
            _atomic_write_json(self.path, committed)
        self.payload = committed
        return committed


def _right_state(rpc):
    pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    torque = np.asarray(rpc.get_right_joint_torque(), dtype=float)
    if (
        pose.shape != (7,)
        or q.shape != (6,)
        or torque.shape != (6,)
        or not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(torque))
    ):
        raise RuntimeError("invalid right-arm state in calibration journal")
    return {
        "monotonic_s": time.monotonic(),
        "pose_wxyz_xyz": pose.tolist(),
        "joint_position_rad": q.tolist(),
        "joint_torque_nm": torque.tolist(),
    }


def verify_right_stationary(
    rpc,
    duration_s=0.5,
    sample_period_s=0.05,
    torque_limit=None,
):
    duration_s = float(duration_s)
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("stationary verification duration must be non-negative")
    if torque_limit is not None:
        torque_limit = np.asarray(torque_limit, dtype=float)
        if (
            torque_limit.shape != (6,)
            or not np.all(np.isfinite(torque_limit))
            or np.any(torque_limit <= 0.0)
        ):
            raise ValueError("invalid stationary torque limit")
    samples = [_right_state(rpc)]
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        if torque_limit is not None and np.any(
            np.abs(samples[-1]["joint_torque_nm"]) > torque_limit
        ):
            break
        time.sleep(min(sample_period_s, max(0.0, deadline - time.monotonic())))
        samples.append(_right_state(rpc))
    poses = np.asarray([sample["pose_wxyz_xyz"] for sample in samples])
    joints = np.asarray(
        [sample["joint_position_rad"] for sample in samples]
    )
    torques = np.abs(
        np.asarray([sample["joint_torque_nm"] for sample in samples])
    )
    xyz_span = np.ptp(poses[:, 4:7], axis=0)
    joint_span = np.ptp(joints, axis=0)
    max_abs_torque = np.max(torques, axis=0)
    torque_ok = True
    if torque_limit is not None:
        torque_ok = bool(np.all(max_abs_torque <= torque_limit))
    return {
        "duration_s": duration_s,
        "sample_count": len(samples),
        "xyz_span_m": xyz_span.tolist(),
        "joint_span_rad": joint_span.tolist(),
        "max_abs_torque_nm": max_abs_torque.tolist(),
        "torque_limit_nm": (
            None if torque_limit is None else torque_limit.tolist()
        ),
        "torque_within_limit": torque_ok,
        "verified": bool(
            np.max(xyz_span) <= 0.0005
            and np.max(joint_span) <= 0.01
            and torque_ok
        ),
        "final_state": samples[-1],
    }


def _image_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sha256(path, description):
    digest = _image_sha256(path)
    if digest is None:
        raise RuntimeError(f"{description} is missing: {Path(path)}")
    return digest


def _head_raw_path(overlay_path):
    overlay = Path(overlay_path)
    return overlay.with_name(f"{overlay.stem}_head_raw.png")


_HEAD_OBSERVATION_ARTIFACT_FIELDS = (
    "raw_image",
    "sam_input_png",
    "sam_request_jpeg_q90",
    "overlay_image",
    "lid_mask",
    "gripper_mask",
    "depth_npz",
    "manifest",
)


def _snapshot_artifacts(source, *, schema, fields, run_root, overlay_path):
    artifacts = copy.deepcopy(source)
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("schema") != schema
    ):
        raise RuntimeError(f"invalid {schema} observation provenance")
    root = Path(run_root).resolve()
    hashes = {}
    byte_counts = {}
    image_shapes = {}
    for field in fields:
        value = artifacts.get(field)
        if not isinstance(value, str):
            raise RuntimeError(f"missing observation artifact: {field}")
        path = Path(value).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"observation artifact escapes run directory: {path}"
            ) from exc
        hashes[field] = _required_sha256(path, f"observation artifact {field}")
        byte_counts[field] = path.stat().st_size
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim not in (2, 3):
                raise RuntimeError(f"unreadable image artifact: {path}")
            image_shapes[field] = list(image.shape)
        artifacts[field] = str(path)
    if Path(artifacts["overlay_image"]) != Path(overlay_path).resolve():
        raise RuntimeError("observation artifact metadata/overlay mismatch")
    artifacts["sha256"] = hashes
    artifacts["byte_count"] = byte_counts
    artifacts["image_shape"] = image_shapes
    return artifacts


def _snapshot_head_observation_artifacts(runner, overlay_path):
    """Freeze the exact SAM/depth inputs before the next observation."""

    artifacts = _snapshot_artifacts(
        getattr(runner, "last_observation_artifacts", None),
        schema="sam_head_observation/v2",
        fields=_HEAD_OBSERVATION_ARTIFACT_FIELDS,
        run_root=runner.output_dir,
        overlay_path=overlay_path,
    )
    if (
        artifacts.get("manifest_sha256")
        != artifacts["sha256"]["manifest"]
    ):
        raise RuntimeError("head observation manifest hash mismatch")
    return artifacts


def _snapshot_right_observation_artifacts(observer, overlay_path):
    source = getattr(observer, "last_observation_artifacts", None)
    fields = [
        "raw_image",
        "sam_input_png",
        "sam_request_jpeg_q90",
        "overlay_image",
    ]
    if isinstance(source, dict) and source.get("lid_mask") is not None:
        fields.append("lid_mask")
    return _snapshot_artifacts(
        source,
        schema="sam_right_observation/v1",
        fields=tuple(fields),
        run_root=observer.output_dir,
        overlay_path=overlay_path,
    )


def _probe_context(runner, anchor_head_path, anchor_artifacts=None):
    anchor_raw = (
        _head_raw_path(anchor_head_path)
        if anchor_artifacts is None
        else Path(anchor_artifacts["raw_image"])
    )
    anchor_image = cv2.imread(str(anchor_raw), cv2.IMREAD_COLOR)
    if anchor_image is None:
        raise RuntimeError(f"probe anchor image is unreadable: {anchor_raw}")
    scene_path = Path(runner.args.scene_config).resolve()
    scene_profile = json.loads(scene_path.read_text())
    placement = scene_profile.get("head_camera_placement_reference")
    if not isinstance(placement, dict):
        raise RuntimeError("scene profile lacks head camera placement reference")
    reference_path = Path(placement.get("rgb_path", ""))
    if not reference_path.is_absolute():
        reference_path = scene_path.parents[2] / reference_path
    reference_hash = _required_sha256(
        reference_path, "head camera placement reference"
    )
    if reference_hash != placement.get("rgb_sha256"):
        raise RuntimeError("head camera placement reference hash mismatch")
    registration = register_fixed_camera_view(reference_path, anchor_image)
    if not registration.accepted:
        raise RuntimeError(
            "head camera placement no longer matches its reference: "
            + registration.reason
        )

    stable = {
        "schema": "sam_probe_context/v2",
        "scene_config_sha256": _required_sha256(
            runner.args.scene_config, "scene config"
        ),
        "torque_config_sha256": _required_sha256(
            runner.args.torque_config, "torque config"
        ),
        "head_camera_udid": placement.get("record3d_udid"),
        "head_camera_matrix_rotated": np.asarray(
            runner.head_camera_matrix, dtype=float
        ).tolist(),
        "head_camera_reference_shape_hw": list(
            runner.head_camera_reference_shape
        ),
        "head_image_shape_hw": list(anchor_image.shape[:2]),
        "head_rotation": "clockwise_90",
        "feature_definition": (
            "lid-left-ellipse/gripper-sam-hsv-terminal-roi-v4"
        ),
        "placement_reference_capture_id": placement.get("capture_id"),
        "placement_reference_rgb_sha256": reference_hash,
        "registration_gate": placement.get("registration_gate"),
        "sam_protocol_version": PROTOCOL_VERSION,
        "sam_models": {
            role: (
                anchor_artifacts.get(role, {}).get("model")
                if anchor_artifacts is not None
                else None
            )
            for role in ("lid", "gripper")
        },
        "sam_prompts": {
            role: (
                anchor_artifacts.get(role, {}).get("prompt")
                if anchor_artifacts is not None
                else None
            )
            for role in ("lid", "gripper")
        },
        "sam_policy": {
            "lid_prompt_sequence": [
                "transparent round petri dish lid with blue cross",
                "petri dish lid",
                "round transparent plastic dish",
            ],
            "lid_confidence_threshold": 0.05,
            "gripper_prompt": "blue clamp",
            "gripper_confidence_threshold": 0.10,
            "jpeg_quality": 90,
            "preprocess": (
                anchor_artifacts.get("preprocess")
                if anchor_artifacts is not None
                else None
            ),
        },
        "depth_frames_requested": int(
            (
                anchor_artifacts.get("depth", {}).get(
                    "frames_requested"
                )
                if anchor_artifacts is not None
                else None
            )
            or runner.args.depth_frames
        ),
        "feature_layout": ["image_u_px", "image_v_px", "camera_depth_mm"],
        "temporal_depth_method": (
            "fresh-frame-median/rotate-clockwise/nearest-resize-v1"
        ),
        "control_frame": "piper_right_base_xyz_m",
        "observation_pipeline_sha256": _required_sha256(
            Path(__file__).resolve().parents[1]
            / "src"
            / "run_realtime_sam_grasp.py",
            "SAM observation pipeline source",
        ),
        "feature_extractor_sha256": _required_sha256(
            Path(__file__).resolve().parents[1]
            / "rollout"
            / "realtime_sam_servo.py",
            "feature extractor source",
        ),
        "segmentation_selector_sha256": _required_sha256(
            Path(__file__).resolve().parents[1]
            / "rollout"
            / "sam_segmentation.py",
            "segmentation selector source",
        ),
    }
    canonical = json.dumps(
        stable, sort_keys=True, separators=(",", ":")
    ).encode()
    context = {
        "stable": stable,
        "context_id_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    context["anchor_head_raw_image"] = str(anchor_raw.resolve())
    context["anchor_head_raw_sha256"] = _required_sha256(
        anchor_raw, "probe anchor image"
    )
    context["placement_registration"] = {
        "accepted": registration.accepted,
        "matches": registration.matches,
        "inliers": registration.inliers,
        "inlier_fraction": registration.inlier_fraction,
        "median_inlier_error_px": registration.median_inlier_error_px,
        "maximum_corner_motion_px": registration.maximum_corner_motion_px,
    }
    context["depth_geometry_quality"] = _geometry_quality_record(
        runner.last_geometry_quality
    )
    try:
        context["host_boot_id"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        )
    except OSError:
        context["host_boot_id"] = None
    return context


def fit_horizontal_jacobian(robot_deltas, feature_deltas, rcond=0.12):
    """Return image XY change per robot XY metre from horizontal probes."""

    robot = np.asarray(robot_deltas, dtype=float)
    feature = np.asarray(feature_deltas, dtype=float)
    if (
        robot.ndim != 2
        or feature.ndim != 2
        or robot.shape != feature.shape
        or robot.shape[1] != 3
        or robot.shape[0] < 2
    ):
        raise ValueError("at least two paired XYZ observations are required")
    horizontal_robot = robot[:, :2]
    horizontal_feature = feature[:, :2]
    if np.linalg.matrix_rank(horizontal_robot, tol=5e-4) < 2:
        raise ValueError("live probes span fewer than two horizontal directions")
    jacobian = horizontal_feature.T @ np.linalg.pinv(
        horizontal_robot.T, rcond=rcond
    )
    return jacobian


def estimate_horizontal_displacement(jacobian, feature_error):
    """Map SAM pixel error into local robot XY without requesting descent."""

    jacobian = np.asarray(jacobian, dtype=float).reshape(2, 2)
    error = np.asarray(feature_error, dtype=float).reshape(3)[:2]
    horizontal = np.linalg.lstsq(jacobian, error, rcond=0.12)[0]
    return np.array([horizontal[0], horizontal[1], 0.0])


def bound_horizontal_step(displacement, max_norm_m, max_axis_m):
    step = np.zeros(3, dtype=float)
    step[:2] = np.asarray(displacement, dtype=float).reshape(3)[:2]
    peak = float(np.max(np.abs(step[:2])))
    if peak > max_axis_m:
        step[:2] *= max_axis_m / peak
    norm = float(np.linalg.norm(step[:2]))
    if norm > max_norm_m:
        step[:2] *= max_norm_m / norm
    return step


def _geometry_quality_record(geometry_quality):
    """Record future 3D/descent quality without gating a UV-only probe."""

    def finite_nonnegative(value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(result) or result < 0.0:
            return None
        return result

    if geometry_quality is None:
        view_angle = None
        footprint = [None, None]
        reasons = ["depth geometry quality observation is unavailable"]
        accepted = False
    else:
        view_angle = finite_nonnegative(
            getattr(geometry_quality, "view_angle_deg", None)
        )
        footprint = [
            finite_nonnegative(
                getattr(
                    geometry_quality,
                    "native_pixel_footprint_x_m",
                    None,
                )
            ),
            finite_nonnegative(
                getattr(
                    geometry_quality,
                    "native_pixel_footprint_y_m",
                    None,
                )
            ),
        ]
        reasons = [
            str(reason)
            for reason in getattr(geometry_quality, "reasons", ())
        ]
        if view_angle is None or any(value is None for value in footprint):
            reasons.append("depth geometry quality values are invalid")
        accepted = bool(
            getattr(geometry_quality, "accepted", False)
            and view_angle is not None
            and all(value is not None for value in footprint)
        )
    return {
        "accepted_for_future_descent_or_3d": accepted,
        "view_angle_deg": view_angle,
        "native_depth_pixel_footprint_m": footprint,
        "reasons": reasons,
        "horizontal_uv_probe_policy": "record_only_not_motion_gate",
    }


def _probe_motion_quality(actual_xyz, requested_xyz):
    actual = np.asarray(actual_xyz, dtype=float)
    requested = np.asarray(requested_xyz, dtype=float)
    if actual.shape != (3,) or requested.shape != (3,):
        raise ValueError("probe motion vectors must have shape (3,)")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(requested)):
        raise ValueError("probe motion vectors must be finite")
    actual_norm = float(np.linalg.norm(actual[:2]))
    requested_norm = float(np.linalg.norm(requested[:2]))
    if requested_norm <= 0.0:
        raise ValueError("probe request must contain horizontal motion")
    direction_cosine = (
        float(np.dot(actual[:2], requested[:2]))
        / (actual_norm * requested_norm)
        if actual_norm > 0.0
        else None
    )
    progress_ratio = float(
        np.dot(actual[:2], requested[:2])
        / np.dot(requested[:2], requested[:2])
    )
    minimum_norm = 0.0005
    minimum_direction_cosine = 0.80
    accepted = bool(
        actual_norm >= minimum_norm
        and direction_cosine is not None
        and direction_cosine >= minimum_direction_cosine
    )
    return actual, {
        "horizontal_norm_m": actual_norm,
        "direction_cosine": direction_cosine,
        "signed_progress_ratio": progress_ratio,
        "minimum_horizontal_norm_m": minimum_norm,
        "minimum_direction_cosine": minimum_direction_cosine,
        "accepted": accepted,
    }


def _require_valid_probe_motion(quality, stage):
    if (
        float(quality["horizontal_norm_m"])
        < float(quality["minimum_horizontal_norm_m"])
    ):
        raise RuntimeError(
            f"{stage} horizontal motion was below 0.5 mm"
        )
    direction_cosine = quality["direction_cosine"]
    if (
        direction_cosine is None
        or float(direction_cosine)
        < float(quality["minimum_direction_cosine"])
    ):
        raise RuntimeError(
            f"{stage} motion opposed or diverged from signed request: "
            f"direction cosine={direction_cosine}"
        )


def _probe_observation_record(
    feature,
    error,
    head_timestamp,
    right_geometry,
    right_candidate,
    right_timestamp,
    head_artifacts,
    right_artifacts,
):
    head_overlay = head_artifacts["overlay_image"]
    head_raw = head_artifacts["raw_image"]
    right_overlay = right_artifacts["overlay_image"]
    return {
        "feature_error": np.asarray(error, dtype=float).tolist(),
        "gripper_feature": np.asarray(
            feature.gripper_feature, dtype=float
        ).tolist(),
        "lid_grasp_feature": np.asarray(
            feature.lid_grasp_feature, dtype=float
        ).tolist(),
        "head_image": head_overlay,
        "head_image_sha256": head_artifacts["sha256"]["overlay_image"],
        "head_raw_image": head_raw,
        "head_raw_image_sha256": head_artifacts["sha256"]["raw_image"],
        "head_timestamp": float(head_timestamp),
        "right_lid_center_px": (
            right_geometry.center_px.tolist()
            if right_geometry is not None
            else None
        ),
        "right_lid_score": (
            float(right_candidate.score)
            if right_candidate is not None
            else None
        ),
        "right_image": right_overlay,
        "right_image_sha256": right_artifacts["sha256"]["overlay_image"],
        "right_timestamp": float(right_timestamp),
        "head_artifacts": head_artifacts,
        "right_artifacts": right_artifacts,
    }


def execute_single_horizontal_probe(
    runner,
    right_observer,
    axis: str,
    distance_m: float,
    *,
    hold_window_s: float = 0.5,
    probe_context=None,
    journal_path=None,
    motion_token_sha256=None,
    motion_attempt_callback=None,
):
    """Execute one horizontal probe and return an auditable atomic sample."""

    if axis not in ("x", "y"):
        raise ValueError("single probe axis must be x or y")
    distance_m = float(distance_m)
    if (
        not np.isfinite(distance_m)
        or not 0.0 < abs(distance_m) <= 0.008
    ):
        raise ValueError(
            "single probe distance magnitude must be within (0, 0.008] m"
        )
    request = np.zeros(3, dtype=float)
    request[0 if axis == "x" else 1] = distance_m
    journal = ProbeExecutionJournal(
        journal_path,
        axis=axis,
        requested_xyz_m=request,
        motion_token_sha256=motion_token_sha256,
    )
    try:
        return _execute_single_horizontal_probe_body(
            runner,
            right_observer,
            axis,
            request,
            hold_window_s=hold_window_s,
            probe_context=probe_context,
            journal=journal,
            motion_attempt_callback=motion_attempt_callback,
        )
    except BaseException as error:
        try:
            setattr(
                error,
                "probe_motion_attempted",
                journal.motion_attempted,
            )
        except Exception:
            pass
        try:
            journal.fail(error)
        except BaseException as journal_error:
            add_note = getattr(error, "add_note", None)
            if add_note is not None:
                add_note(
                    "probe failure journal update also failed: "
                    f"{journal_error!r}"
                )
        raise


def _execute_single_horizontal_probe_body(
    runner,
    right_observer,
    axis,
    request,
    *,
    hold_window_s,
    probe_context,
    journal,
    motion_attempt_callback,
):
    torque_limit = np.asarray(
        getattr(runner, "torque_limit", None), dtype=float
    )
    if (
        torque_limit.shape != (6,)
        or not np.all(np.isfinite(torque_limit))
        or np.any(torque_limit <= 0.0)
    ):
        raise RuntimeError(
            "single probe requires a valid right-arm torque limit"
        )

    before_feature, before_error, before_head_path, before_head_timestamp = (
        runner.observe(0.0)
    )
    before_head_artifacts = _snapshot_head_observation_artifacts(
        runner, before_head_path
    )
    context = (
        _probe_context(
            runner, before_head_path, before_head_artifacts
        )
        if probe_context is None
        else dict(probe_context)
    )
    if not context.get("context_id_sha256"):
        raise ValueError("probe context requires context_id_sha256")
    (
        before_right_geometry,
        before_right_candidate,
        before_right_path,
        before_right_timestamp,
    ) = right_observer.observe(require_lid=False)
    before_right_artifacts = _snapshot_right_observation_artifacts(
        right_observer, before_right_path
    )
    before_state = _right_state(runner.rpc)
    before_observation = _probe_observation_record(
        before_feature,
        before_error,
        before_head_timestamp,
        before_right_geometry,
        before_right_candidate,
        before_right_timestamp,
        before_head_artifacts,
        before_right_artifacts,
    )
    journal.update(
        "pre_motion_ready",
        context=context,
        motion={
            "requested_xyz_m": request.tolist(),
            "before_state": before_state,
        },
        observation={"before": before_observation},
    )
    journal.mark_motion_attempt()
    if motion_attempt_callback is not None:
        motion_attempt_callback()
    actual_immediate = runner.move_cartesian_delta(
        request, minimum_progress=0.0
    )
    post_motion_state = _right_state(runner.rpc)
    actual_immediate, immediate_motion_quality = _probe_motion_quality(
        actual_immediate, request
    )
    motion = {
        "requested_xyz_m": request.tolist(),
        "actual_immediate_xyz_m": actual_immediate.tolist(),
        "actual_immediate_quality": immediate_motion_quality,
        "before_state": before_state,
        "post_motion_state": post_motion_state,
    }
    journal.mark_motion_command_completed(motion)
    _require_valid_probe_motion(
        immediate_motion_quality, "immediate"
    )
    journal.mark_immediate_motion_validated(motion)
    initial_hold = verify_right_stationary(
        runner.rpc,
        duration_s=hold_window_s,
        torque_limit=torque_limit,
    )
    if not initial_hold["verified"]:
        runner.hold_measured()
        initial_hold = verify_right_stationary(
            runner.rpc,
            duration_s=hold_window_s,
            torque_limit=torque_limit,
        )
        if not initial_hold["verified"]:
            raise RuntimeError(
                "right arm did not become stationary after single probe"
            )
    after_feature, after_error, after_head_path, after_head_timestamp = (
        runner.observe(0.0)
    )
    after_head_artifacts = _snapshot_head_observation_artifacts(
        runner, after_head_path
    )
    (
        after_right_geometry,
        after_right_candidate,
        after_right_path,
        after_right_timestamp,
    ) = right_observer.observe(require_lid=False)
    after_right_artifacts = _snapshot_right_observation_artifacts(
        right_observer, after_right_path
    )
    hold = verify_right_stationary(
        runner.rpc,
        duration_s=hold_window_s,
        torque_limit=torque_limit,
    )
    if not hold["verified"]:
        runner.hold_measured()
        hold = verify_right_stationary(
            runner.rpc,
            duration_s=hold_window_s,
            torque_limit=torque_limit,
        )
        if not hold["verified"]:
            raise RuntimeError(
                "right arm moved during post-probe SAM observation"
            )
    initial_hold_pose = np.asarray(
        initial_hold["final_state"]["pose_wxyz_xyz"], dtype=float
    )
    initial_hold_q = np.asarray(
        initial_hold["final_state"]["joint_position_rad"], dtype=float
    )
    final_hold_pose = np.asarray(
        hold["final_state"]["pose_wxyz_xyz"], dtype=float
    )
    final_hold_q = np.asarray(
        hold["final_state"]["joint_position_rad"], dtype=float
    )
    observation_xyz_shift = (
        final_hold_pose[4:7] - initial_hold_pose[4:7]
    )
    observation_joint_shift = final_hold_q - initial_hold_q
    if (
        np.max(np.abs(observation_xyz_shift)) > 0.0005
        or np.max(np.abs(observation_joint_shift)) > 0.01
    ):
        raise RuntimeError(
            "right arm changed pose while post-probe SAM was running"
        )
    final_pose = np.asarray(
        hold["final_state"]["pose_wxyz_xyz"], dtype=float
    )
    before_pose = np.asarray(before_state["pose_wxyz_xyz"], dtype=float)
    actual_settled = final_pose[4:7] - before_pose[4:7]
    actual_settled, settled_motion_quality = _probe_motion_quality(
        actual_settled, request
    )
    motion.update(
        {
            "actual_settled_xyz_m": actual_settled.tolist(),
            "actual_settled_quality": settled_motion_quality,
            "initial_hold": initial_hold,
            "hold": hold,
            "observation_xyz_shift_m": observation_xyz_shift.tolist(),
            "observation_joint_shift_rad": (
                observation_joint_shift.tolist()
            ),
        }
    )
    journal.update("validating_settled_motion", motion=motion)
    _require_valid_probe_motion(settled_motion_quality, "settled")
    journal.mark_settled_motion_validated(motion)
    gripper_delta = (
        after_feature.gripper_feature - before_feature.gripper_feature
    )
    lid_delta = (
        after_feature.lid_grasp_feature - before_feature.lid_grasp_feature
    )
    relative_delta = gripper_delta - lid_delta
    pixel_signal = float(np.linalg.norm(relative_delta[:2]))
    pixel_noise = max(0.5, float(np.linalg.norm(lid_delta[:2])))
    signal_to_noise = pixel_signal / pixel_noise
    usable_for_fit = bool(
        pixel_signal >= 2.0 and signal_to_noise >= 3.0
    )
    after_observation = _probe_observation_record(
        after_feature,
        after_error,
        after_head_timestamp,
        after_right_geometry,
        after_right_candidate,
        after_right_timestamp,
        after_head_artifacts,
        after_right_artifacts,
    )
    fixed_view = register_fixed_camera_view(
        before_observation["head_raw_image"],
        after_observation["head_raw_image"],
    )
    if not fixed_view.accepted:
        raise RuntimeError(
            "head camera moved during the probe: " + fixed_view.reason
        )
    stable = context.get("stable", {})
    expected_models = stable.get("sam_models")
    if isinstance(expected_models, dict):
        after_models = {
            role: after_head_artifacts.get(role, {}).get("model")
            for role in ("lid", "gripper")
        }
        if after_models != expected_models:
            raise RuntimeError("SAM model changed during the probe")

    observation = {
        "before": before_observation,
        "after": after_observation,
        "gripper_feature_delta": gripper_delta.tolist(),
        "lid_feature_delta": lid_delta.tolist(),
        "relative_feature_delta": relative_delta.tolist(),
        "fixed_camera_during_probe": {
            "accepted": fixed_view.accepted,
            "matches": fixed_view.matches,
            "inliers": fixed_view.inliers,
            "inlier_fraction": fixed_view.inlier_fraction,
            "median_inlier_error_px": fixed_view.median_inlier_error_px,
            "maximum_corner_motion_px": fixed_view.maximum_corner_motion_px,
        },
    }
    journal.update(
        "validating_post_motion_observation",
        context=context,
        motion=motion,
        observation=observation,
    )
    report = {
        "context": context,
        "status": f"SINGLE_{axis.upper()}_PROBE_COMMITTED",
        "motion": motion,
        "observation": observation,
        "quality": {
            "pixel_signal_norm": pixel_signal,
            "pixel_noise_norm": pixel_noise,
            "signal_to_noise": signal_to_noise,
            "usable_for_fit": usable_for_fit,
            "reasons": (
                []
                if usable_for_fit
                else [
                    reason
                    for reason, rejected in (
                        ("image motion was below 2 px", pixel_signal < 2.0),
                        (
                            "relative image motion SNR was below 3",
                            signal_to_noise < 3.0,
                        ),
                    )
                    if rejected
                ]
            ),
        },
    }
    return journal.commit(report)


class RightLidObserver:
    """Independent live SAM check from the right wrist camera."""

    def __init__(self, runner: LiveSamGrasp, output_dir: Path):
        camera_map = load_camera_map()
        self.runner = runner
        self.camera = USBWristCameraFeedManager(
            runner.stop_event,
            device_index=camera_map.get("right", 2),
            label="right wrist",
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.previous_center = None
        self.sequence = 0
        self.last_observation_artifacts = None
        self.last_timestamp = None
        self.last_rgb_sha256 = None

    def start(self):
        self.camera.start()
        self._await_fresh_frame(timeout_s=12.0)

    def stop(self):
        self.camera.stop()

    def _await_fresh_frame(self, *, timeout_s=3.0):
        not_before = time.time()
        deadline = time.monotonic() + float(timeout_s)
        repeated_digest = False
        while time.monotonic() < deadline:
            rgb, timestamp, _ = self.camera.get_latest_frame()
            if rgb is None or timestamp is None:
                time.sleep(0.01)
                continue
            timestamp = float(timestamp)
            if not np.isfinite(timestamp):
                raise RuntimeError(
                    "right wrist frame timestamp is not finite"
                )
            if timestamp + 1e-6 < not_before:
                time.sleep(0.01)
                continue
            frame = np.ascontiguousarray(np.asarray(rgb))
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                raise RuntimeError("right wrist RGB frame shape is invalid")
            digest = hashlib.sha256(
                memoryview(frame).cast("B")
            ).hexdigest()
            if (
                self.last_timestamp is not None
                and timestamp <= self.last_timestamp
            ):
                time.sleep(0.01)
                continue
            if (
                self.last_rgb_sha256 is not None
                and digest == self.last_rgb_sha256
            ):
                repeated_digest = True
                time.sleep(0.01)
                continue
            self.last_timestamp = timestamp
            self.last_rgb_sha256 = digest
            return frame, timestamp
        reason = (
            "RGB bytes repeated despite newer timestamps"
            if repeated_digest
            else "no post-barrier frame arrived"
        )
        raise RuntimeError(
            f"fresh right wrist frame unavailable: {reason}"
        )

    def observe(self, *, require_lid=True):
        rgb, timestamp = self._await_fresh_frame()
        image = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sequence = self.sequence
        self.sequence += 1
        raw_path = self.output_dir / f"{sequence:03d}_raw.png"
        if not cv2.imwrite(str(raw_path), image):
            raise RuntimeError(f"could not save right raw image {raw_path}")
        sam_image = (
            enhance_low_light(image) if float(image.mean()) < 35.0 else image
        )
        enhanced = sam_image is not image
        input_path = self.output_dir / f"{sequence:03d}_sam_input.png"
        if not cv2.imwrite(str(input_path), sam_image):
            raise RuntimeError(f"could not save right SAM input {input_path}")
        request_path = (
            self.output_dir / f"{sequence:03d}_sam_request_q90.jpg"
        )
        if not cv2.imwrite(
            str(request_path),
            sam_image,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        ):
            raise RuntimeError(f"could not save right SAM request {request_path}")
        selected = None
        attempts = []
        selected_prompt = None
        result = None
        for prompt in (
            "transparent round petri dish lid with blue cross",
            "petri dish lid",
            "round transparent plastic dish",
        ):
            result = self.runner.sam.segment(
                sam_image,
                frame_id=self.runner.frame_id,
                timestamp=timestamp,
                prompt=prompt,
                confidence_threshold=0.05,
            )
            self.runner.frame_id += 1
            attempts.append((prompt, len(result.candidates)))
            selected = choose_lid_candidate(
                result.candidates,
                image_bgr=sam_image,
                previous_center_px=self.previous_center,
                require_blue_cross=True,
            )
            if selected is not None:
                selected_prompt = prompt
                break
        if selected is None:
            if require_lid:
                raise RuntimeError(
                    "right SAM did not identify the blue-cross lid; "
                    f"attempts={attempts}, raw={raw_path}, input={input_path}"
                )
            overlay = sam_image.copy()
            label = f"RIGHT lid outside view; attempts={attempts}"
            cv2.putText(
                overlay,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                overlay,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            path = self.output_dir / f"{sequence:03d}.png"
            if not cv2.imwrite(str(path), overlay):
                raise RuntimeError(f"could not save right overlay {path}")
            self.last_observation_artifacts = {
                "schema": "sam_right_observation/v1",
                "sequence": int(sequence),
                "raw_image": str(raw_path),
                "sam_input_png": str(input_path),
                "sam_request_jpeg_q90": str(request_path),
                "overlay_image": str(path),
                "lid_mask": None,
                "image_shape_hw": list(image.shape[:2]),
                "preprocess": (
                    "enhance_low_light" if enhanced else "identity"
                ),
                "attempts": attempts,
                "lid": None,
            }
            return None, None, str(path), float(timestamp)
        candidate, geometry = selected
        self.previous_center = geometry.center_px.copy()
        overlay = sam_image.copy()
        mask = np.asarray(candidate.mask, dtype=bool)
        tint = np.zeros_like(overlay)
        tint[:] = (0, 255, 0)
        overlay[mask] = cv2.addWeighted(
            overlay[mask], 0.55, tint[mask], 0.45, 0
        )
        center = tuple(np.rint(geometry.center_px).astype(int))
        cv2.drawMarker(
            overlay, center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3
        )
        label = (
            f"RIGHT live SAM score={candidate.score:.3f} "
            f"center=({center[0]},{center[1]})"
        )
        cv2.putText(
            overlay,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            overlay,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        path = self.output_dir / f"{sequence:03d}.png"
        if not cv2.imwrite(str(path), overlay):
            raise RuntimeError(f"could not save right overlay {path}")
        mask_path = self.output_dir / f"{sequence:03d}_lid_mask.png"
        if not cv2.imwrite(
            str(mask_path), np.asarray(candidate.mask, dtype=np.uint8) * 255
        ):
            raise RuntimeError(f"could not save right lid mask {mask_path}")
        self.last_observation_artifacts = {
            "schema": "sam_right_observation/v1",
            "sequence": int(sequence),
            "raw_image": str(raw_path),
            "sam_input_png": str(input_path),
            "sam_request_jpeg_q90": str(request_path),
            "overlay_image": str(path),
            "lid_mask": str(mask_path),
            "image_shape_hw": list(image.shape[:2]),
            "preprocess": (
                "enhance_low_light" if enhanced else "identity"
            ),
            "attempts": attempts,
            "lid": {
                "prompt": selected_prompt,
                "confidence_threshold": 0.05,
                "model": result.model,
                "frame_id": int(result.frame_id),
                "score": float(candidate.score),
                "box_xyxy": np.asarray(
                    candidate.box_xyxy, dtype=float
                ).tolist(),
            },
        }
        return geometry, candidate, str(path), float(timestamp)


def _add_exception_note(error, message):
    message = str(message)
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        try:
            add_note(message)
            return
        except BaseException:
            pass
    try:
        notes = list(getattr(error, "__notes__", ()))
        notes.append(message)
        setattr(error, "__notes__", notes)
    except BaseException:
        pass


def _cleanup_staged_runtime(
    runner,
    right,
    execution_claim,
    execution_error,
    prior_cleanup_failures=(),
):
    """Stop both observers and finalize a claim without masking the primary."""

    cleanup_failures = list(prior_cleanup_failures)
    for label, callback in (
        ("right.stop", right.stop),
        ("runner.stop", runner.stop),
    ):
        try:
            callback()
        except BaseException as error:
            cleanup_failures.append((label, error))

    if execution_error is not None:
        for label, error in cleanup_failures:
            _add_exception_note(
                execution_error,
                f"{label} also failed during cleanup: {error!r}",
            )
        if execution_claim is not None:
            try:
                execution_claim.finalize(execution_error)
            except BaseException as error:
                _add_exception_note(
                    execution_error,
                    "execution claim finalization also failed: "
                    f"{error!r}",
                )
        return None

    cleanup_error = None
    if cleanup_failures:
        primary_label, cleanup_error = cleanup_failures[0]
        _add_exception_note(
            cleanup_error,
            f"cleanup operation failed: {primary_label}",
        )
        for label, error in cleanup_failures[1:]:
            _add_exception_note(
                cleanup_error,
                f"{label} also failed during cleanup: {error!r}",
            )

    if execution_claim is not None:
        try:
            execution_claim.finalize(cleanup_error)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            else:
                _add_exception_note(
                    cleanup_error,
                    "execution claim finalization also failed: "
                    f"{error!r}",
                )
    return cleanup_error


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:15563")
    parser.add_argument(
        "--torque-config", default="src/configs/pasteur_lid_torque.json"
    )
    parser.add_argument(
        "--scene-config",
        default="src/configs/pasteur_lid_scene3d.json",
    )
    parser.add_argument("--reference-points")
    parser.add_argument(
        "--output-dir", default="/tmp/realtime_sam_horizontal"
    )
    parser.add_argument("--depth-frames", type=int, default=15)
    parser.add_argument("--preview-time", type=float, default=2.0)
    parser.add_argument(
        "--single-probe-axis",
        choices=("x", "y"),
        help="execute exactly one horizontal probe, capture both views, stop",
    )
    parser.add_argument("--single-probe-m", type=float, default=0.006)
    parser.add_argument(
        "--motion-token",
        help="one-shot user-approval token required for any physical motion",
    )
    parser.add_argument(
        "--motion-claim-dir",
        default="/var/tmp/piper_robot_motion_claims",
    )
    parser.add_argument(
        "--execute-horizontal",
        action="store_true",
        help="allow right-arm horizontal motion; default is camera-only dry-run",
    )
    args = parser.parse_args(argv)
    if args.single_probe_axis and not args.execute_horizontal:
        parser.error("--single-probe-axis requires --execute-horizontal")
    if args.execute_horizontal and not args.single_probe_axis:
        parser.error(
            "--execute-horizontal requires --single-probe-axis; "
            "legacy multi-command alignment is disabled"
        )
    if args.execute_horizontal and not args.motion_token:
        parser.error("--execute-horizontal requires --motion-token")

    execution_claim = None
    if args.execute_horizontal:
        execution_claim = claim_motion_execution(
            args.motion_token,
            args.motion_claim_dir,
            args.output_dir,
            {
                "mode": f"single_probe_{args.single_probe_axis}",
                "axis": args.single_probe_axis,
                "distance_m": args.single_probe_m,
                "left_arm": False,
                "gripper": False,
                "descent": False,
                "home": False,
            },
        )

    # LiveSamGrasp owns camera, SAM, torque monitoring, and right-only RPC.
    args.control_space = "cartesian"
    args.minimum_progress = 0.0
    args.joint_minimum_progress = 0.0
    try:
        runner = LiveSamGrasp(args)
        right = RightLidObserver(
            runner, Path(args.output_dir) / "right"
        )
    except BaseException as exc:
        if execution_claim is not None:
            try:
                execution_claim.finalize(exc)
            except BaseException as finalize_error:
                _add_exception_note(
                    exc,
                    "execution claim finalization also failed: "
                    f"{finalize_error!r}",
                )
        raise
    moved = False
    execution_error = None
    cleanup_failures = []
    try:
        runner.start()
        right.start()
        runner.observe(0.0)  # warm Record3D temporal depth
        feature, raw_error, head_path, timestamp = runner.observe(0.0)
        registration = runner.check_scene_registration(
            args.reference_points
        )
        right_geometry, right_candidate, right_path, right_timestamp = (
            right.observe(require_lid=False)
        )
        depth_geometry_quality = _geometry_quality_record(
            runner.last_geometry_quality
        )
        print(
            json.dumps(
                {
                    "mode": (
                        f"SINGLE_PROBE_{args.single_probe_axis.upper()}"
                        if args.single_probe_axis
                        else (
                            "HORIZONTAL_ONLY"
                            if args.execute_horizontal
                            else "DRY_RUN_NO_MOTION"
                        )
                    ),
                    "feature_error": raw_error.round(2).tolist(),
                    "target_camera_xyz_m": (
                        runner.last_target_3d.point_camera_xyz_m.round(4).tolist()
                    ),
                    "support_plane_confidence": (
                        runner.last_target_3d.confidence
                    ),
                    "depth_geometry_quality": depth_geometry_quality,
                    "proximity_warning_only_m": (
                        runner.last_proximity_m
                    ),
                    "registration": registration,
                    "head_image": head_path,
                    "head_timestamp": timestamp,
                    "right_lid_center_px": (
                        right_geometry.center_px.round(2).tolist()
                        if right_geometry is not None
                        else None
                    ),
                    "right_lid_score": (
                        float(right_candidate.score)
                        if right_candidate is not None
                        else None
                    ),
                    "right_image": right_path,
                    "right_timestamp": right_timestamp,
                }
            ),
            flush=True,
        )
        if not args.execute_horizontal:
            return
        record_path = (
            Path(args.output_dir) / "probe_record.json"
        ).resolve()
        report = execute_single_horizontal_probe(
            runner,
            right,
            args.single_probe_axis,
            args.single_probe_m,
            journal_path=record_path,
            motion_token_sha256=execution_claim.payload["token_sha256"],
            motion_attempt_callback=execution_claim.mark_motion_attempt,
        )
        execution_claim.set_result(
            {
                "status": report["status"],
                "record_path": str(record_path),
                "record_sha256": _required_sha256(
                    record_path, "probe record"
                ),
                "usable_for_fit": report["quality"]["usable_for_fit"],
            }
        )
        print(json.dumps(report), flush=True)
        return
    except BaseException as exc:
        execution_error = exc
        moved = moved or bool(
            getattr(exc, "probe_motion_attempted", False)
        )
        if moved:
            try:
                runner.hold_measured()
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    ("runner.hold_measured", cleanup_error)
                )
        raise
    finally:
        cleanup_error = _cleanup_staged_runtime(
            runner,
            right,
            execution_claim,
            execution_error,
            cleanup_failures,
        )
        if execution_error is None and cleanup_error is not None:
            raise cleanup_error


if __name__ == "__main__":
    main()

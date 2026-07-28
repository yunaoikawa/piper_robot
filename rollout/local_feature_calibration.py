"""Reusable local visual-feature calibration from auditable motion probes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from rollout.realtime_sam_servo import (
    ReachableFeatureModel,
    bounded_reachable_servo_step,
    estimate_reachable_feature_model,
)
from rollout.sam_segmentation import PROTOCOL_VERSION
from rollout.scene_semantics import estimate_image_homography


PROBE_SCHEMA = "sam_horizontal_probe/v2"
CONTEXT_SCHEMA = "sam_probe_context/v2"
FEATURE_DEFINITION = "lid-left-ellipse/gripper-pca-terminal-roi-v3"
MODEL_SCHEMA = "sam_local_xy_feature_model/v1"
# Record3D gripper depth has shown 14--22 mm stationary drift in this setup.
# Horizontal motion eligibility therefore uses UV by default.  Depth remains
# available as an explicitly selected, low-weight diagnostic feature.
DEFAULT_FEATURE_SCALE = np.array([2.0, 2.0, 20.0])


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return result


def _finite_scalar(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _step_limits(max_norm_m, max_axis_m) -> tuple[float, float]:
    max_norm = _finite_scalar(max_norm_m, "maximum step norm")
    max_axis = _finite_scalar(max_axis_m, "maximum step axis")
    if max_norm <= 0.0 or max_axis <= 0.0:
        raise ValueError("step limits must be positive")
    return max_norm, max_axis


def _hex_digest(value, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _unit_quaternion(value, name: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if not 0.9 <= norm <= 1.1:
        raise ValueError(f"{name} is not a unit quaternion")
    return quaternion / norm


def _quaternion_error_deg(first, second) -> float:
    first_q = _unit_quaternion(first, "first quaternion")
    second_q = _unit_quaternion(second, "second quaternion")
    cosine = float(np.clip(abs(first_q @ second_q), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(cosine)))


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(
            path.read_text(), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid probe JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: probe JSON must contain one object")
    _require_finite_json(payload, str(path))
    return payload


def _require_finite_json(value, name: str):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json(item, name)
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item, name)
        return
    raise ValueError(f"{name} contains an unsupported value")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedViewRegistration:
    accepted: bool
    matches: int
    inliers: int
    inlier_fraction: float
    median_inlier_error_px: float
    maximum_corner_motion_px: float
    homography: np.ndarray | None
    reason: str = ""
    median_grid_motion_px: float = float("inf")


def _bgr_image(image_or_path) -> np.ndarray:
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image {image_or_path}")
        return image
    image = np.asarray(image_or_path)
    if image.ndim == 2:
        return cv2.cvtColor(
            image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera-view image must be grayscale or BGR")
    return image.astype(np.uint8, copy=False)


def register_fixed_camera_view(
    reference,
    current,
    *,
    minimum_matches: int = 80,
    minimum_inlier_fraction: float = 0.60,
    maximum_median_error_px: float = 1.5,
    maximum_corner_motion_px: float = 2.0,
) -> FixedViewRegistration:
    """Verify that two head frames came from the same fixed camera pose.

    RANSAC may explain a moved camera with a valid homography, so acceptance
    additionally requires that the fitted transform remain near identity.
    """

    reference_bgr = _bgr_image(reference)
    current_bgr = _bgr_image(current)
    if reference_bgr.shape != current_bgr.shape:
        return FixedViewRegistration(
            False, 0, 0, 0.0, float("inf"), float("inf"), None,
            "image shape changed",
        )
    try:
        homography, report = estimate_image_homography(
            reference_bgr, current_bgr
        )
    except ValueError as exc:
        return FixedViewRegistration(
            False, 0, 0, 0.0, float("inf"), float("inf"), None,
            str(exc),
        )
    if (
        np.asarray(homography).shape != (3, 3)
        or not np.all(np.isfinite(homography))
        or abs(float(homography[2, 2])) < 1e-12
    ):
        return FixedViewRegistration(
            False,
            int(report.get("matches", 0)),
            int(report.get("inliers", 0)),
            0.0,
            float("inf"),
            float("inf"),
            None,
            "invalid camera-view homography",
        )
    homography = homography / homography[2, 2]
    matches = int(report["matches"])
    inlier_count = int(report["inliers"])
    inlier_fraction = inlier_count / matches
    median_error = float(report["median_inlier_residual_px"])
    height, width = reference_bgr.shape[:2]
    corners = np.float32(
        [[[0, 0]], [[width - 1, 0]], [[width - 1, height - 1]], [[0, height - 1]]]
    )
    moved_corners = cv2.perspectiveTransform(corners, homography)
    corner_motion = np.linalg.norm(moved_corners - corners, axis=2)
    maximum_motion = float(np.max(corner_motion))
    grid_x = np.linspace(0.0, width - 1.0, 5)
    grid_y = np.linspace(0.0, height - 1.0, 5)
    grid = np.float32(
        [[x, y] for y in grid_y for x in grid_x]
    ).reshape(-1, 1, 2)
    moved_grid = cv2.perspectiveTransform(grid, homography)
    grid_motion = np.linalg.norm(moved_grid - grid, axis=2).reshape(-1)
    median_grid_motion = float(np.median(grid_motion))
    reasons = []
    if matches < minimum_matches:
        reasons.append("insufficient feature matches")
    if inlier_fraction < minimum_inlier_fraction:
        reasons.append("low inlier fraction")
    if median_error > maximum_median_error_px:
        reasons.append("high registration residual")
    if maximum_motion > maximum_corner_motion_px:
        reasons.append("camera pose changed")
    if median_grid_motion > maximum_corner_motion_px:
        reasons.append("camera view is not near identity")
    return FixedViewRegistration(
        accepted=not reasons,
        matches=matches,
        inliers=inlier_count,
        inlier_fraction=float(inlier_fraction),
        median_inlier_error_px=median_error,
        maximum_corner_motion_px=maximum_motion,
        homography=homography,
        reason="; ".join(reasons),
        median_grid_motion_px=median_grid_motion,
    )


@dataclass(frozen=True)
class ProbeCalibrationSample:
    record_id: str
    content_sha256: str
    sample_fingerprint: str
    path: Path
    context: dict
    context_fingerprint: str
    actual_xyz_m: np.ndarray
    requested_xyz_m: np.ndarray
    feature_delta: np.ndarray
    lid_feature_delta: np.ndarray
    midpoint_xyz_m: np.ndarray
    orientation_wxyz: np.ndarray
    midpoint_feature: np.ndarray
    before_head_image: Path
    after_head_image: Path


def _verified_artifact(
    record_path: Path,
    description: dict,
    path_field: str,
    hash_field: str,
) -> Path:
    if not isinstance(description, dict):
        raise ValueError(f"{record_path}: artifact description is missing")
    raw_path = description.get(path_field)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{record_path}: artifact path is missing")
    artifact = Path(raw_path).resolve()
    try:
        artifact.relative_to(record_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{record_path}: artifact escapes the run directory"
        ) from exc
    expected_hash = _hex_digest(
        description.get(hash_field),
        f"{record_path}: {hash_field}",
    )
    if not artifact.is_file() or _sha256(artifact) != expected_hash:
        raise ValueError(f"{record_path}: image artifact hash mismatch")
    image = cv2.imread(str(artifact), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3):
        raise ValueError(f"{record_path}: image artifact is unreadable")
    return artifact


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


def _verify_head_observation_artifacts(
    record_path: Path, description: dict
) -> dict:
    artifacts = description.get("head_artifacts")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("schema") != "sam_head_observation/v2"
    ):
        raise ValueError(
            f"{record_path}: exact SAM observation provenance is missing"
        )
    hashes = artifacts.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError(f"{record_path}: artifact hashes are missing")
    verified = {}
    for field in _HEAD_OBSERVATION_ARTIFACT_FIELDS:
        artifact = Path(artifacts.get(field, "")).resolve()
        try:
            artifact.relative_to(record_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                f"{record_path}: observation artifact escapes the run directory"
            ) from exc
        expected_hash = _hex_digest(
            hashes.get(field), f"{record_path}: {field} SHA-256"
        )
        if (
            not artifact.is_file()
            or _sha256(artifact) != expected_hash
        ):
            raise ValueError(
                f"{record_path}: {field} artifact hash mismatch"
            )
        verified[field] = artifact

    if artifacts.get("manifest_sha256") != hashes["manifest"]:
        raise ValueError(f"{record_path}: manifest digest is inconsistent")
    try:
        manifest = _load_json(verified["manifest"])
    except ValueError as exc:
        raise ValueError(
            f"{record_path}: observation manifest is invalid"
        ) from exc
    if (
        manifest.get("schema") != "sam_head_observation/v2"
        or manifest.get("sequence") != artifacts.get("sequence")
        or manifest.get("run_id") != artifacts.get("run_id")
        or manifest.get("attempt_id") != artifacts.get("attempt_id")
        or not isinstance(manifest.get("run_id"), str)
        or not manifest.get("run_id")
        or not isinstance(manifest.get("attempt_id"), str)
        or not manifest.get("attempt_id")
        or not isinstance(manifest.get("files"), dict)
        or artifacts.get("files") != manifest.get("files")
        or not isinstance(manifest.get("feature"), dict)
        or artifacts.get("feature") != manifest.get("feature")
    ):
        raise ValueError(
            f"{record_path}: observation manifest metadata is inconsistent"
        )
    required_manifest_roles = {
        "raw_image",
        "sam_input_png",
        "sam_request_jpeg_q90",
        "sam_roi_input_png",
        "sam_roi_request_jpeg_q90",
        "overlay_image",
        "lid_mask",
        "gripper_mask",
        "depth_npz",
    }
    if not required_manifest_roles.issubset(manifest["files"]):
        raise ValueError(
            f"{record_path}: ROI observation artifacts are incomplete"
        )
    for role, entry in manifest["files"].items():
        if not isinstance(role, str) or not role or not isinstance(entry, dict):
            raise ValueError(
                f"{record_path}: observation manifest file entry is invalid"
            )
        relative_path = entry.get("path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).name != relative_path
        ):
            raise ValueError(
                f"{record_path}: observation manifest path is invalid"
            )
        artifact = (verified["manifest"].parent / relative_path).resolve()
        try:
            artifact.relative_to(record_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                f"{record_path}: manifest artifact escapes the run directory"
            ) from exc
        expected_hash = _hex_digest(
            entry.get("sha256"),
            f"{record_path}: manifest {role} SHA-256",
        )
        byte_count = entry.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, Integral)
            or int(byte_count) < 0
            or not isinstance(entry.get("media_type"), str)
            or not entry["media_type"]
            or not artifact.is_file()
            or artifact.stat().st_size != int(byte_count)
            or _sha256(artifact) != expected_hash
        ):
            raise ValueError(
                f"{record_path}: manifest {role} artifact is inconsistent"
            )
    for field in _HEAD_OBSERVATION_ARTIFACT_FIELDS:
        if field == "manifest":
            continue
        entry = manifest["files"].get(field)
        if (
            not isinstance(entry, dict)
            or entry.get("path") != verified[field].name
            or entry.get("sha256") != hashes[field]
            or entry.get("bytes") != verified[field].stat().st_size
        ):
            raise ValueError(
                f"{record_path}: {field} manifest entry is inconsistent"
            )

    for field in (
        "raw_image",
        "sam_input_png",
        "overlay_image",
        "lid_mask",
        "gripper_mask",
    ):
        image = cv2.imread(str(verified[field]), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim not in (2, 3):
            raise ValueError(
                f"{record_path}: {field} artifact is unreadable"
            )
    try:
        with np.load(verified["depth_npz"], allow_pickle=False) as depth:
            depth_m = np.asarray(depth["depth_m"])
            camera_matrix = np.asarray(depth["camera_matrix"])
            timestamps = np.asarray(depth["source_timestamps"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            f"{record_path}: depth artifact is unreadable"
        ) from exc
    if (
        depth_m.ndim != 2
        or depth_m.size == 0
        or camera_matrix.shape != (3, 3)
        or timestamps.ndim != 1
        or timestamps.size == 0
        or not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(timestamps))
    ):
        raise ValueError(f"{record_path}: depth artifact is malformed")
    return manifest


def _context_fingerprint(context: dict, record_path: Path) -> str:
    stable = context.get("stable")
    if not isinstance(stable, dict):
        raise ValueError(f"{record_path}: stable context is missing")
    _require_finite_json(stable, "stable context")
    required = {
        "schema",
        "scene_config_sha256",
        "torque_config_sha256",
        "head_camera_udid",
        "head_camera_matrix_rotated",
        "head_camera_reference_shape_hw",
        "head_image_shape_hw",
        "head_rotation",
        "feature_definition",
        "placement_reference_capture_id",
        "placement_reference_rgb_sha256",
        "registration_gate",
        "sam_protocol_version",
        "sam_models",
        "sam_prompts",
        "sam_policy",
        "depth_frames_requested",
        "feature_layout",
        "temporal_depth_method",
        "control_frame",
        "observation_pipeline_sha256",
        "feature_extractor_sha256",
        "segmentation_selector_sha256",
    }
    if not required.issubset(stable):
        raise ValueError(f"{record_path}: calibration context is incomplete")
    if stable["schema"] != CONTEXT_SCHEMA:
        raise ValueError(f"{record_path}: unsupported context schema")
    for field in (
        "scene_config_sha256",
        "torque_config_sha256",
        "placement_reference_rgb_sha256",
        "observation_pipeline_sha256",
        "feature_extractor_sha256",
        "segmentation_selector_sha256",
    ):
        _hex_digest(stable[field], f"{record_path}: {field}")
    if (
        not isinstance(stable["head_camera_udid"], str)
        or not stable["head_camera_udid"]
        or not isinstance(stable["placement_reference_capture_id"], str)
        or not stable["placement_reference_capture_id"]
    ):
        raise ValueError(f"{record_path}: camera identity is incomplete")
    matrix = np.asarray(stable["head_camera_matrix_rotated"], dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or abs(float(matrix[2, 2]) - 1.0) > 1e-6
    ):
        raise ValueError(f"{record_path}: camera matrix is invalid")
    for field in (
        "head_camera_reference_shape_hw",
        "head_image_shape_hw",
    ):
        shape = np.asarray(stable[field])
        if (
            shape.shape != (2,)
            or not np.issubdtype(shape.dtype, np.integer)
            or np.any(shape <= 0)
        ):
            raise ValueError(f"{record_path}: {field} is invalid")
    if stable["head_rotation"] != "clockwise_90":
        raise ValueError(f"{record_path}: head image rotation is unsupported")
    if stable["feature_definition"] != FEATURE_DEFINITION:
        raise ValueError(f"{record_path}: feature definition is unsupported")
    if stable["registration_gate"] != "orb_identity_v1":
        raise ValueError(f"{record_path}: registration gate is unsupported")
    if (
        isinstance(stable["sam_protocol_version"], bool)
        or stable["sam_protocol_version"] != PROTOCOL_VERSION
    ):
        raise ValueError(f"{record_path}: SAM protocol version is unsupported")
    for field in ("sam_models", "sam_prompts"):
        values = stable[field]
        if (
            not isinstance(values, dict)
            or set(values) != {"lid", "gripper"}
            or any(
                not isinstance(values[role], str) or not values[role]
                for role in ("lid", "gripper")
            )
        ):
            raise ValueError(f"{record_path}: {field} is incomplete")
    policy = stable["sam_policy"]
    if (
        not isinstance(policy, dict)
        or policy.get("lid_prompt_sequence")
        != [
            "transparent round petri dish lid with blue cross",
            "petri dish lid",
            "round transparent plastic dish",
        ]
        or policy.get("lid_confidence_threshold") != 0.05
        or policy.get("gripper_prompt") != "blue clamp"
        or policy.get("gripper_confidence_threshold") != 0.10
        or policy.get("jpeg_quality") != 90
        or policy.get("preprocess") not in ("identity", "enhance_low_light")
    ):
        raise ValueError(f"{record_path}: SAM policy is unsupported")
    depth_frames_requested = stable["depth_frames_requested"]
    if (
        isinstance(depth_frames_requested, bool)
        or not isinstance(depth_frames_requested, Integral)
        or int(depth_frames_requested) < 3
    ):
        raise ValueError(
            f"{record_path}: depth frame request count is invalid"
        )
    if stable["feature_layout"] != [
        "image_u_px",
        "image_v_px",
        "camera_depth_mm",
    ]:
        raise ValueError(f"{record_path}: feature layout is unsupported")
    if (
        stable["temporal_depth_method"]
        != "fresh-frame-median/rotate-clockwise/nearest-resize-v1"
    ):
        raise ValueError(
            f"{record_path}: temporal depth method is unsupported"
        )
    if stable["control_frame"] != "piper_right_base_xyz_m":
        raise ValueError(f"{record_path}: control frame is unsupported")
    canonical = json.dumps(
        stable, sort_keys=True, separators=(",", ":")
    ).encode()
    calculated = hashlib.sha256(canonical).hexdigest()
    _hex_digest(
        context.get("context_id_sha256"),
        f"{record_path}: context_id_sha256",
    )
    if context["context_id_sha256"] != calculated:
        raise ValueError(f"{record_path}: context fingerprint is invalid")
    registration = context.get("placement_registration")
    if (
        not isinstance(registration, dict)
        or registration.get("accepted") is not True
    ):
        raise ValueError(f"{record_path}: camera placement was not accepted")
    for field in ("matches", "inliers"):
        value = registration.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{record_path}: invalid placement registration {field}"
            )
    for field in (
        "inlier_fraction",
        "median_inlier_error_px",
        "maximum_corner_motion_px",
    ):
        _finite_scalar(
            registration.get(field),
            f"{record_path}: placement registration {field}",
        )
    matches = registration["matches"]
    inliers = registration["inliers"]
    if (
        matches < 80
        or not 0 <= inliers <= matches
        or float(registration["inlier_fraction"]) < 0.60
        or not np.isclose(
            float(registration["inlier_fraction"]),
            inliers / matches,
            atol=1e-6,
            rtol=0.0,
        )
        or float(registration["median_inlier_error_px"]) > 1.5
        or float(registration["maximum_corner_motion_px"]) > 2.0
    ):
        raise ValueError(
            f"{record_path}: camera placement registration is below gate"
        )
    return calculated


def _validated_state(value, name: str, *, require_complete: bool) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} is missing")
    pose = _finite_vector(value.get("pose_wxyz_xyz"), 7, f"{name} pose")
    _unit_quaternion(pose[:4], f"{name} orientation")
    if require_complete:
        _finite_vector(
            value.get("joint_position_rad"), 6, f"{name} joint position"
        )
        _finite_vector(
            value.get("joint_torque_nm"), 6, f"{name} joint torque"
        )
        monotonic = value.get("monotonic_s")
        if (
            isinstance(monotonic, bool)
            or not isinstance(monotonic, (int, float))
            or not np.isfinite(monotonic)
        ):
            raise ValueError(f"{name} monotonic timestamp is invalid")
    return value


def load_probe_record(
    path: str | Path,
    *,
    expected_record_sha256: str | None = None,
) -> ProbeCalibrationSample:
    """Load one probe and independently recheck all motion-derived fields."""

    path = Path(path).resolve()
    content_sha256 = _sha256(path)
    if expected_record_sha256 is not None:
        expected_digest = _hex_digest(
            expected_record_sha256, "expected probe record SHA-256"
        )
        if content_sha256 != expected_digest:
            raise ValueError(f"{path}: probe record SHA-256 mismatch")
    record = _load_json(path)
    if _sha256(path) != content_sha256:
        raise ValueError(f"{path}: probe record changed while loading")
    if record.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"{path}: unsupported probe schema")
    record_id = record.get("record_id")
    if (
        not isinstance(record_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", record_id) is None
    ):
        raise ValueError(f"{path}: probe record id is missing")
    status = record.get("status")
    if status not in (
        "SINGLE_X_PROBE_COMMITTED",
        "SINGLE_Y_PROBE_COMMITTED",
    ):
        raise ValueError(f"{path}: probe was not committed")
    execution = record.get("execution")
    required_execution_true = (
        "motion_attempted",
        "motion_command_completed",
        "immediate_motion_validated",
        "settled_motion_validated",
    )
    if (
        not isinstance(execution, dict)
        or execution.get("stage") != "committed"
        or execution.get("failure_timing") is not None
        or any(
            execution.get(field) is not True
            for field in required_execution_true
        )
    ):
        raise ValueError(f"{path}: probe execution was not fully committed")
    authorization = record.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError(f"{path}: probe motion authorization is missing")
    _hex_digest(
        authorization.get("motion_token_sha256"),
        f"{path}: motion token SHA-256",
    )
    quality = record.get("quality")
    if not isinstance(quality, dict):
        raise ValueError(f"{path}: probe quality is missing")
    context = record.get("context")
    if not isinstance(context, dict):
        raise ValueError(f"{path}: calibration context is missing")
    context_fingerprint = _context_fingerprint(context, path)

    motion = record.get("motion")
    if not isinstance(motion, dict):
        raise ValueError(f"{path}: probe motion is missing")
    requested = _finite_vector(
        motion["requested_xyz_m"], 3, "requested xyz"
    )
    actual = _finite_vector(
        motion["actual_settled_xyz_m"], 3, "actual settled xyz"
    )
    before_state = _validated_state(
        motion.get("before_state"),
        f"{path}: before state",
        require_complete=True,
    )
    before_pose = _finite_vector(
        before_state["pose_wxyz_xyz"], 7, "before pose"
    )
    final_hold = motion.get("hold")
    if not isinstance(final_hold, dict):
        raise ValueError(f"{path}: hold is missing")
    final_state = _validated_state(
        final_hold.get("final_state"),
        f"{path}: final hold state",
        require_complete=True,
    )
    final_pose = _finite_vector(
        final_state["pose_wxyz_xyz"], 7, "final pose"
    )
    derived_actual = final_pose[4:7] - before_pose[4:7]
    if not np.allclose(actual, derived_actual, atol=1e-7, rtol=0.0):
        raise ValueError(f"{path}: recorded actual motion is inconsistent")
    if _quaternion_error_deg(before_pose[:4], final_pose[:4]) > 0.5:
        raise ValueError(f"{path}: probe changed tool orientation")
    if abs(float(requested[2])) > 1e-9:
        raise ValueError(f"{path}: horizontal probe requested Z motion")
    expected_axis = 0 if status == "SINGLE_X_PROBE_COMMITTED" else 1
    other_axis = 1 - expected_axis
    if (
        abs(float(requested[expected_axis])) <= 1e-9
        or abs(float(requested[other_axis])) > 1e-9
    ):
        raise ValueError(f"{path}: status and requested axis disagree")
    requested_norm = float(np.linalg.norm(requested[:2]))
    actual_norm = float(np.linalg.norm(actual[:2]))
    if (
        not 0.0 < requested_norm <= 0.008001
        or not 0.0005 <= actual_norm <= 0.010
    ):
        raise ValueError(f"{path}: horizontal motion magnitude is invalid")
    if abs(float(actual[2])) > 0.0005:
        raise ValueError(f"{path}: probe had excessive vertical drift")
    direction_cosine = float(
        np.dot(actual[:2], requested[:2])
        / (actual_norm * requested_norm)
    )
    if direction_cosine < 0.80:
        raise ValueError(f"{path}: probe did not follow requested direction")
    hold_states = {}
    for name in ("initial_hold", "hold"):
        hold = motion.get(name)
        if not isinstance(hold, dict):
            raise ValueError(f"{path}: {name} is missing")
        xyz_span = _finite_vector(
            hold.get("xyz_span_m"), 3, f"{name} xyz span"
        )
        joint_span = _finite_vector(
            hold.get("joint_span_rad"), 6, f"{name} joint span"
        )
        maximum_torque = _finite_vector(
            hold.get("max_abs_torque_nm"), 6, f"{name} maximum torque"
        )
        torque_limit = _finite_vector(
            hold.get("torque_limit_nm"), 6, f"{name} torque limit"
        )
        hold_state = _validated_state(
            hold.get("final_state"),
            f"{path}: {name} final state",
            require_complete=True,
        )
        if (
            np.any(xyz_span < 0.0)
            or np.any(joint_span < 0.0)
            or np.any(maximum_torque < 0.0)
            or np.any(torque_limit <= 0.0)
        ):
            raise ValueError(f"{path}: {name} quality metrics are invalid")
        stationary = bool(
            np.max(np.abs(xyz_span)) <= 0.0005
            and np.max(np.abs(joint_span)) <= 0.01
        )
        torque_within_limit = bool(
            np.all(maximum_torque <= torque_limit + 1e-7)
        )
        expected_verified = stationary and torque_within_limit
        if (
            (hold.get("torque_within_limit") is True)
            != torque_within_limit
        ):
            raise ValueError(
                f"{path}: {name} torque limit decision is inconsistent"
            )
        if (hold.get("verified") is True) != expected_verified:
            raise ValueError(
                f"{path}: {name} verification decision is inconsistent"
            )
        if not stationary:
            raise ValueError(f"{path}: {name} was not stationary")
        if not torque_within_limit:
            raise ValueError(f"{path}: {name} torque was outside its limit")
        final_torque = _finite_vector(
            hold_state["joint_torque_nm"],
            6,
            f"{path}: {name} final torque",
        )
        if np.any(np.abs(final_torque) > maximum_torque + 1e-7):
            raise ValueError(
                f"{path}: {name} maximum torque is inconsistent"
            )
        hold_states[name] = hold_state
    observation_xyz_shift = _finite_vector(
        motion.get("observation_xyz_shift_m"),
        3,
        "observation xyz shift",
    )
    observation_joint_shift = _finite_vector(
        motion.get("observation_joint_shift_rad"),
        6,
        "observation joint shift",
    )
    initial_hold_pose = _finite_vector(
        hold_states["initial_hold"]["pose_wxyz_xyz"],
        7,
        "initial hold pose",
    )
    initial_hold_joints = _finite_vector(
        hold_states["initial_hold"]["joint_position_rad"],
        6,
        "initial hold joints",
    )
    final_hold_joints = _finite_vector(
        hold_states["hold"]["joint_position_rad"],
        6,
        "final hold joints",
    )
    derived_observation_xyz_shift = (
        final_pose[4:7] - initial_hold_pose[4:7]
    )
    derived_observation_joint_shift = (
        final_hold_joints - initial_hold_joints
    )
    if not np.allclose(
        observation_xyz_shift,
        derived_observation_xyz_shift,
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError(
            f"{path}: observation Cartesian shift is inconsistent"
        )
    if not np.allclose(
        observation_joint_shift,
        derived_observation_joint_shift,
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError(
            f"{path}: observation joint shift is inconsistent"
        )
    if np.max(np.abs(observation_xyz_shift)) > 0.0005:
        raise ValueError(f"{path}: arm moved during SAM observation")
    if np.max(np.abs(observation_joint_shift)) > 0.01:
        raise ValueError(f"{path}: joints moved during SAM observation")
    immediate = _finite_vector(
        motion["actual_immediate_xyz_m"], 3, "actual immediate xyz"
    )
    post_motion_state = _validated_state(
        motion.get("post_motion_state"),
        f"{path}: post-motion state",
        require_complete=True,
    )
    post_motion_pose = _finite_vector(
        post_motion_state["pose_wxyz_xyz"], 7, "post-motion pose"
    )
    derived_immediate = post_motion_pose[4:7] - before_pose[4:7]
    if not np.allclose(immediate, derived_immediate, atol=0.0005, rtol=0.0):
        raise ValueError(f"{path}: immediate motion is inconsistent")
    if np.linalg.norm(immediate - actual) > 0.00075:
        raise ValueError(f"{path}: immediate and settled motion disagree")
    state_times = [
        float(before_state["monotonic_s"]),
        float(post_motion_state["monotonic_s"]),
        float(motion["initial_hold"]["final_state"]["monotonic_s"]),
        float(final_state["monotonic_s"]),
    ]
    if any(
        later < earlier for earlier, later in zip(state_times, state_times[1:])
    ):
        raise ValueError(f"{path}: arm state timestamps are not ordered")

    observation = record.get("observation")
    if (
        not isinstance(observation, dict)
        or not isinstance(observation.get("before"), dict)
        or not isinstance(observation.get("after"), dict)
    ):
        raise ValueError(f"{path}: probe observations are missing")
    before_gripper = _finite_vector(
        observation["before"]["gripper_feature"], 3, "before gripper feature"
    )
    after_gripper = _finite_vector(
        observation["after"]["gripper_feature"], 3, "after gripper feature"
    )
    gripper_delta = _finite_vector(
        observation["gripper_feature_delta"], 3, "gripper feature delta"
    )
    if not np.allclose(
        gripper_delta, after_gripper - before_gripper, atol=1e-7, rtol=0.0
    ):
        raise ValueError(f"{path}: gripper feature delta is inconsistent")
    before_lid = _finite_vector(
        observation["before"]["lid_grasp_feature"], 3, "before lid feature"
    )
    after_lid = _finite_vector(
        observation["after"]["lid_grasp_feature"], 3, "after lid feature"
    )
    lid_delta = _finite_vector(
        observation["lid_feature_delta"], 3, "lid feature delta"
    )
    if not np.allclose(
        lid_delta, after_lid - before_lid, atol=1e-7, rtol=0.0
    ):
        raise ValueError(f"{path}: lid feature delta is inconsistent")
    if (
        np.linalg.norm(lid_delta[:2]) > 3.0
        or abs(float(lid_delta[2])) > 5.0
    ):
        raise ValueError(f"{path}: target moved during probe")
    feature_delta = gripper_delta - lid_delta
    recorded_relative = _finite_vector(
        observation["relative_feature_delta"],
        3,
        "relative feature delta",
    )
    if not np.allclose(
        recorded_relative, feature_delta, atol=1e-7, rtol=0.0
    ):
        raise ValueError(f"{path}: relative feature delta is inconsistent")
    for phase in ("before", "after"):
        expected_error = (
            _finite_vector(
                observation[phase]["lid_grasp_feature"],
                3,
                f"{phase} lid feature",
            )
            - _finite_vector(
                observation[phase]["gripper_feature"],
                3,
                f"{phase} gripper feature",
            )
        )
        recorded_error = _finite_vector(
            observation[phase]["feature_error"],
            3,
            f"{phase} feature error",
        )
        if not np.allclose(
            expected_error, recorded_error, atol=1e-7, rtol=0.0
        ):
            raise ValueError(f"{path}: feature error is inconsistent")
    before_timestamp = _finite_scalar(
        observation["before"].get("head_timestamp"),
        f"{path}: before head timestamp",
    )
    after_timestamp = _finite_scalar(
        observation["after"].get("head_timestamp"),
        f"{path}: after head timestamp",
    )
    if after_timestamp <= before_timestamp:
        raise ValueError(f"{path}: observation timestamps are not ordered")

    for phase in ("before", "after"):
        description = observation[phase]
        manifest = _verify_head_observation_artifacts(path, description)
        stable = context["stable"]
        if manifest.get("preprocess") != stable["sam_policy"]["preprocess"]:
            raise ValueError(
                f"{path}: manifest {phase} preprocessing changed"
            )
        for role in ("lid", "gripper"):
            manifest_role = manifest.get(role)
            if (
                not isinstance(manifest_role, dict)
                or manifest_role.get("model")
                != stable["sam_models"][role]
                or manifest_role.get("prompt")
                != stable["sam_prompts"][role]
            ):
                raise ValueError(
                    f"{path}: manifest {phase} {role} policy changed"
                )
        manifest_feature = manifest["feature"]
        for manifest_field, record_field in (
            ("lid_grasp_feature", "lid_grasp_feature"),
            ("gripper_feature", "gripper_feature"),
            ("error", "feature_error"),
        ):
            saved_feature = _finite_vector(
                manifest_feature.get(manifest_field),
                3,
                f"{path}: manifest {phase} {manifest_field}",
            )
            record_feature = _finite_vector(
                description.get(record_field),
                3,
                f"{path}: record {phase} {record_field}",
            )
            if not np.allclose(
                saved_feature, record_feature, atol=1e-7, rtol=0.0
            ):
                raise ValueError(
                    f"{path}: manifest and record {phase} "
                    f"{record_field} disagree"
                )
        _verified_artifact(
            path, description, "head_image", "head_image_sha256"
        )
        _verified_artifact(
            path,
            description,
            "right_image",
            "right_image_sha256",
        )
    before_image = _verified_artifact(
        path,
        observation["before"],
        "head_raw_image",
        "head_raw_image_sha256",
    )
    after_image = _verified_artifact(
        path,
        observation["after"],
        "head_raw_image",
        "head_raw_image_sha256",
    )
    before_after_view = register_fixed_camera_view(
        before_image, after_image
    )
    if not before_after_view.accepted:
        raise ValueError(
            f"{path}: head camera moved during probe: "
            f"{before_after_view.reason}"
        )
    anchor_path = context.get("anchor_head_raw_image")
    if not isinstance(anchor_path, str) or not anchor_path:
        raise ValueError(f"{path}: context anchor is missing")
    anchor = Path(anchor_path).resolve()
    anchor_hash = _hex_digest(
        context.get("anchor_head_raw_sha256"),
        f"{path}: anchor_head_raw_sha256",
    )
    if (
        anchor != before_image
        or anchor_hash != _sha256(anchor)
    ):
        raise ValueError(f"{path}: context anchor is inconsistent")
    image_shape = cv2.imread(str(before_image), cv2.IMREAD_COLOR).shape[:2]
    if list(image_shape) != list(context["stable"]["head_image_shape_hw"]):
        raise ValueError(f"{path}: head image shape is inconsistent")
    pixel_signal = float(np.linalg.norm(feature_delta[:2]))
    pixel_noise = max(0.5, float(np.linalg.norm(lid_delta[:2])))
    signal_to_noise = pixel_signal / pixel_noise
    expected_usable = pixel_signal >= 2.0 and signal_to_noise >= 3.0
    recorded_pixel_signal = _finite_scalar(
        quality.get("pixel_signal_norm"),
        f"{path}: pixel signal metric",
    )
    recorded_pixel_noise = _finite_scalar(
        quality.get("pixel_noise_norm"),
        f"{path}: pixel noise metric",
    )
    recorded_signal_to_noise = _finite_scalar(
        quality.get("signal_to_noise"),
        f"{path}: pixel SNR metric",
    )
    if not np.isclose(
        recorded_pixel_signal,
        pixel_signal,
        atol=1e-7,
    ):
        raise ValueError(f"{path}: pixel signal metric is inconsistent")
    if not np.isclose(
        recorded_pixel_noise,
        pixel_noise,
        atol=1e-7,
    ):
        raise ValueError(f"{path}: pixel noise metric is inconsistent")
    if not np.isclose(
        recorded_signal_to_noise,
        signal_to_noise,
        atol=1e-7,
    ):
        raise ValueError(f"{path}: pixel SNR metric is inconsistent")
    if (quality.get("usable_for_fit") is True) != expected_usable:
        raise ValueError(f"{path}: fit eligibility is inconsistent")
    expected_reasons = []
    if pixel_signal < 2.0:
        expected_reasons.append("image motion was below 2 px")
    if signal_to_noise < 3.0:
        expected_reasons.append("relative image motion SNR was below 3")
    if quality.get("reasons") != expected_reasons:
        raise ValueError(f"{path}: fit eligibility reasons are inconsistent")
    if not expected_usable:
        raise ValueError(f"{path}: probe is not marked usable for fit")
    midpoint_feature = 0.5 * (
        (before_gripper - before_lid) + (after_gripper - after_lid)
    )
    sample_payload = {
        "context": context_fingerprint,
        "actual_xyz_m": actual.tolist(),
        "feature_delta": feature_delta.tolist(),
        "before_raw_sha256": _sha256(before_image),
        "after_raw_sha256": _sha256(after_image),
        "before_pose": before_pose.tolist(),
        "final_pose": final_pose.tolist(),
    }
    sample_fingerprint = hashlib.sha256(
        json.dumps(
            sample_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if _sha256(path) != content_sha256:
        raise ValueError(f"{path}: probe record changed while loading")
    return ProbeCalibrationSample(
        record_id=record_id,
        content_sha256=content_sha256,
        sample_fingerprint=sample_fingerprint,
        path=path,
        context=context,
        context_fingerprint=context_fingerprint,
        actual_xyz_m=actual,
        requested_xyz_m=requested,
        feature_delta=feature_delta,
        lid_feature_delta=lid_delta,
        midpoint_xyz_m=0.5 * (before_pose[4:7] + final_pose[4:7]),
        orientation_wxyz=_unit_quaternion(
            before_pose[:4], "probe orientation"
        ),
        midpoint_feature=midpoint_feature,
        before_head_image=before_image,
        after_head_image=after_image,
    )


@dataclass(frozen=True)
class HeldOutVerification:
    accepted: bool
    sample_count: int
    horizontal_rank: int
    horizontal_condition: float | None
    combined_signed_axis_coverage: tuple[
        tuple[bool, bool], tuple[bool, bool]
    ]
    normalized_residual_rms: float
    maximum_normalized_residual: float
    minimum_direction_cosine: float
    minimum_gain_ratio: float
    maximum_gain_ratio: float
    record_ids: tuple[str, ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApplicabilityReport:
    accepted: bool
    fixed_view: FixedViewRegistration
    horizontal_distance_m: float
    vertical_distance_m: float
    orientation_error_deg: float
    feature_uv_distance_px: float
    feature_depth_distance_mm: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class HorizontalFeatureModel:
    """A local image-feature model that remains inert until held-out validation."""

    matrix: np.ndarray
    feature_scale: np.ndarray
    feature_components: tuple[int, ...]
    reachable_model: ReachableFeatureModel
    motion_condition: float
    feature_condition: float
    residual_rms: float
    maximum_residual: float
    sample_count: int
    inlier_mask: np.ndarray
    verified: bool = False
    verification: HeldOutVerification | None = None
    context_fingerprint: str | None = None
    source_record_ids: tuple[str, ...] = ()
    source_record_sha256s: tuple[str, ...] = ()
    source_sample_fingerprints: tuple[str, ...] = ()
    source_actual_xyz_m: np.ndarray | None = None
    reference_head_image: Path | None = None
    reference_head_image_sha256: str | None = None
    reference_midpoint_xyz_m: np.ndarray | None = None
    reference_orientation_wxyz: np.ndarray | None = None
    reference_midpoint_feature: np.ndarray | None = None
    applicability_xy_radius_m: float | None = None
    applicability_z_radius_m: float | None = None
    applicability_uv_radius_px: float | None = None
    applicability_depth_radius_mm: float | None = None
    artifact_sha256: str = ""

    def solve(
        self,
        feature_error,
        *,
        max_norm_m: float = 0.008,
        max_axis_m: float = 0.006,
        allow_provisional: bool = False,
        allow_unchecked_context: bool = False,
    ) -> np.ndarray:
        max_norm_m, max_axis_m = _step_limits(
            max_norm_m, max_axis_m
        )
        if (
            not self.artifact_sha256
            or self.artifact_sha256 != _model_artifact_sha256(self)
        ):
            raise RuntimeError("calibration model artifact hash mismatch")
        if self.context_fingerprint is not None:
            raise RuntimeError(
                "record-backed calibration requires solve_for_observation"
            )
        if not allow_unchecked_context:
            raise RuntimeError(
                "unchecked solve is disabled; use solve_for_observation"
            )
        if not self.verified and not allow_provisional:
            raise RuntimeError(
                "horizontal calibration is provisional; "
                "held-out validation probe required"
            )
        return self._solve_math(feature_error, max_norm_m, max_axis_m)

    def _solve_math(
        self, feature_error, max_norm_m: float, max_axis_m: float
    ) -> np.ndarray:
        error = _finite_vector(
            feature_error, self.matrix.shape[0], "feature error"
        )
        padded_error, padded_scale = _pad_features(
            error[None, :], self.feature_scale
        )
        step = bounded_reachable_servo_step(
            self.reachable_model,
            padded_error[0],
            tolerances=padded_scale,
            max_norm_m=max_norm_m,
            max_axis_m=max_axis_m,
        )
        if abs(float(step[2])) > 1e-8:
            raise RuntimeError("horizontal model unexpectedly requested Z motion")
        step[2] = 0.0
        return step

    def validate_applicability(
        self,
        *,
        context_fingerprint: str,
        current_raw_head_image,
        current_ee_xyz_m,
        current_orientation_wxyz,
        current_relative_feature,
    ) -> ApplicabilityReport:
        """Check that a current observation remains inside this local model.

        ``current_relative_feature`` is the current gripper-minus-lid UVD
        feature from the same SAM observation used for control.
        """

        if (
            not self.artifact_sha256
            or self.artifact_sha256 != _model_artifact_sha256(self)
        ):
            raise RuntimeError("calibration model artifact hash mismatch")
        if (
            self.context_fingerprint is None
            or self.reference_head_image is None
            or self.reference_head_image_sha256 is None
            or self.reference_midpoint_xyz_m is None
            or self.reference_orientation_wxyz is None
            or self.reference_midpoint_feature is None
            or self.applicability_xy_radius_m is None
            or self.applicability_z_radius_m is None
            or self.applicability_uv_radius_px is None
            or self.applicability_depth_radius_mm is None
        ):
            raise RuntimeError(
                "calibration model has no applicability provenance"
            )
        supplied_context = _hex_digest(
            context_fingerprint, "current context fingerprint"
        )
        if (
            not self.reference_head_image.is_file()
            or _sha256(self.reference_head_image)
            != self.reference_head_image_sha256
        ):
            raise RuntimeError("calibration reference image hash mismatch")

        current_xyz = _finite_vector(
            current_ee_xyz_m, 3, "current EE xyz"
        )
        current_orientation = _unit_quaternion(
            current_orientation_wxyz, "current EE orientation"
        )
        current_feature = _finite_vector(
            current_relative_feature, 3, "current relative feature"
        )
        reference_xyz = _finite_vector(
            self.reference_midpoint_xyz_m, 3, "reference midpoint xyz"
        )
        reference_feature = _finite_vector(
            self.reference_midpoint_feature,
            3,
            "reference midpoint feature",
        )
        fixed_view = register_fixed_camera_view(
            self.reference_head_image, current_raw_head_image
        )
        horizontal_distance = float(
            np.linalg.norm(current_xyz[:2] - reference_xyz[:2])
        )
        vertical_distance = float(
            abs(current_xyz[2] - reference_xyz[2])
        )
        orientation_error = _quaternion_error_deg(
            self.reference_orientation_wxyz, current_orientation
        )
        feature_uv_distance = float(
            np.linalg.norm(
                current_feature[:2] - reference_feature[:2]
            )
        )
        feature_depth_distance = float(
            abs(current_feature[2] - reference_feature[2])
        )
        reasons = []
        if not self.verified:
            reasons.append("calibration model is not held-out verified")
        if supplied_context != self.context_fingerprint:
            reasons.append("calibration context fingerprint changed")
        if not fixed_view.accepted:
            reasons.append("head camera view changed: " + fixed_view.reason)
        if horizontal_distance > self.applicability_xy_radius_m:
            reasons.append("current EE position is outside local XY range")
        if vertical_distance > self.applicability_z_radius_m:
            reasons.append("current EE position is outside local Z range")
        if orientation_error > 2.0:
            reasons.append("current EE orientation is outside local range")
        if feature_uv_distance > self.applicability_uv_radius_px:
            reasons.append("current feature is outside local UV range")
        if feature_depth_distance > self.applicability_depth_radius_mm:
            reasons.append("current feature is outside local depth range")
        return ApplicabilityReport(
            accepted=not reasons,
            fixed_view=fixed_view,
            horizontal_distance_m=horizontal_distance,
            vertical_distance_m=vertical_distance,
            orientation_error_deg=orientation_error,
            feature_uv_distance_px=feature_uv_distance,
            feature_depth_distance_mm=feature_depth_distance,
            reasons=tuple(reasons),
        )

    def solve_for_observation(
        self,
        feature_error,
        *,
        context_fingerprint: str,
        current_raw_head_image,
        current_ee_xyz_m,
        current_orientation_wxyz,
        current_relative_feature,
        max_norm_m: float = 0.008,
        max_axis_m: float = 0.006,
    ) -> np.ndarray:
        """Solve after checking view, context, pose, and feature locality.

        ``feature_error`` is lid-minus-gripper in the model's selected
        components.  ``current_relative_feature`` is the opposite-sign,
        three-component UVD value from that exact observation.
        """

        error = _finite_vector(
            feature_error, self.matrix.shape[0], "feature error"
        )
        relative = _finite_vector(
            current_relative_feature, 3, "current relative feature"
        )
        max_norm_m, max_axis_m = _step_limits(
            max_norm_m, max_axis_m
        )
        selected_relative = relative[list(self.feature_components)]
        if not np.allclose(
            selected_relative, -error, atol=1e-7, rtol=0.0
        ):
            raise ValueError(
                "feature error and current relative feature disagree"
            )
        report = self.validate_applicability(
            context_fingerprint=context_fingerprint,
            current_raw_head_image=current_raw_head_image,
            current_ee_xyz_m=current_ee_xyz_m,
            current_orientation_wxyz=current_orientation_wxyz,
            current_relative_feature=relative,
        )
        if not report.accepted:
            raise RuntimeError(
                "local calibration is not applicable: "
                + "; ".join(report.reasons)
            )
        return self._solve_math(error, max_norm_m, max_axis_m)


def _pad_features(feature: np.ndarray, scale: np.ndarray):
    """Adapt UV or UVD observations to the shared three-feature model."""

    feature = np.asarray(feature, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if feature.ndim != 2 or feature.shape[1] not in (2, 3):
        raise ValueError("feature observations must contain UV or UVD")
    if scale.shape != (feature.shape[1],):
        raise ValueError("feature scale does not match observations")
    if feature.shape[1] == 3:
        return feature, scale
    return (
        np.column_stack((feature, np.zeros(feature.shape[0]))),
        np.r_[scale, 1.0],
    )


def _reachable_fit(horizontal, feature, scale):
    robot_xyz = np.column_stack(
        (horizontal, np.zeros(np.asarray(horizontal).shape[0]))
    )
    padded_feature, padded_scale = _pad_features(feature, scale)
    reachable = estimate_reachable_feature_model(
        robot_xyz, padded_feature, rcond=0.15
    )
    if reachable.rank != 2:
        raise ValueError("horizontal probes do not span two directions")
    full_matrix = (
        reachable.feature_matrix @ reachable.basis_xyz[:2, :].T
    )
    return reachable, full_matrix[: feature.shape[1]], padded_scale


def _model_artifact_sha256(model: HorizontalFeatureModel) -> str:
    verification = None
    if model.verification is not None:
        verification = {
            "accepted": model.verification.accepted,
            "sample_count": model.verification.sample_count,
            "horizontal_rank": model.verification.horizontal_rank,
            "horizontal_condition": (
                model.verification.horizontal_condition
            ),
            "combined_signed_axis_coverage": [
                list(axis_coverage)
                for axis_coverage in (
                    model.verification.combined_signed_axis_coverage
                )
            ],
            "normalized_residual_rms": (
                model.verification.normalized_residual_rms
            ),
            "maximum_normalized_residual": (
                model.verification.maximum_normalized_residual
            ),
            "minimum_direction_cosine": (
                model.verification.minimum_direction_cosine
            ),
            "minimum_gain_ratio": model.verification.minimum_gain_ratio,
            "maximum_gain_ratio": model.verification.maximum_gain_ratio,
            "record_ids": list(model.verification.record_ids),
            "reasons": list(model.verification.reasons),
        }
    payload = {
        "schema": MODEL_SCHEMA,
        "matrix": np.asarray(model.matrix, dtype=float).tolist(),
        "feature_scale": np.asarray(
            model.feature_scale, dtype=float
        ).tolist(),
        "feature_components": list(model.feature_components),
        "motion_condition": model.motion_condition,
        "feature_condition": model.feature_condition,
        "residual_rms": model.residual_rms,
        "maximum_residual": model.maximum_residual,
        "sample_count": model.sample_count,
        "inlier_mask": np.asarray(model.inlier_mask, dtype=bool).tolist(),
        "reachable_model": {
            "basis_xyz": np.asarray(
                model.reachable_model.basis_xyz, dtype=float
            ).tolist(),
            "feature_matrix": np.asarray(
                model.reachable_model.feature_matrix, dtype=float
            ).tolist(),
            "rank": model.reachable_model.rank,
            "condition": model.reachable_model.condition,
        },
        "verified": model.verified,
        "verification": verification,
        "context_fingerprint": model.context_fingerprint,
        "source_record_ids": list(model.source_record_ids),
        "source_record_sha256s": list(model.source_record_sha256s),
        "source_sample_fingerprints": list(
            model.source_sample_fingerprints
        ),
        "source_actual_xyz_m": (
            None
            if model.source_actual_xyz_m is None
            else np.asarray(
                model.source_actual_xyz_m, dtype=float
            ).tolist()
        ),
        "reference_head_image_sha256": model.reference_head_image_sha256,
        "reference_midpoint_xyz_m": (
            None
            if model.reference_midpoint_xyz_m is None
            else np.asarray(
                model.reference_midpoint_xyz_m, dtype=float
            ).tolist()
        ),
        "reference_orientation_wxyz": (
            None
            if model.reference_orientation_wxyz is None
            else np.asarray(
                model.reference_orientation_wxyz, dtype=float
            ).tolist()
        ),
        "reference_midpoint_feature": (
            None
            if model.reference_midpoint_feature is None
            else np.asarray(
                model.reference_midpoint_feature, dtype=float
            ).tolist()
        ),
        "applicability_envelope": {
            "xy_radius_m": model.applicability_xy_radius_m,
            "z_radius_m": model.applicability_z_radius_m,
            "uv_radius_px": model.applicability_uv_radius_px,
            "depth_radius_mm": model.applicability_depth_radius_mm,
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def fit_horizontal_feature_model(
    actual_deltas_xyz,
    feature_deltas,
    *,
    feature_scale=None,
    feature_components=None,
    minimum_probe_count: int = 3,
    maximum_motion_condition: float = 5.0,
    maximum_feature_condition: float = 50.0,
    maximum_residual_rms: float = 1.50,
) -> HorizontalFeatureModel:
    """Fit a robust local UV/UVD-to-robot-XY model.

    Core least-squares and bounded inversion are delegated to
    :mod:`rollout.realtime_sam_servo`; this layer adds consensus and safety
    gates for durable probe records.
    """

    if (
        isinstance(minimum_probe_count, bool)
        or not isinstance(minimum_probe_count, Integral)
        or minimum_probe_count < 3
    ):
        raise ValueError("minimum probe count must be an integer of at least 3")
    minimum_probe_count = int(minimum_probe_count)
    maximum_motion_condition = _finite_scalar(
        maximum_motion_condition, "maximum motion condition"
    )
    maximum_feature_condition = _finite_scalar(
        maximum_feature_condition, "maximum feature condition"
    )
    maximum_residual_rms = _finite_scalar(
        maximum_residual_rms, "maximum residual RMS"
    )
    if (
        maximum_motion_condition < 1.0
        or maximum_feature_condition < 1.0
        or maximum_residual_rms <= 0.0
    ):
        raise ValueError("calibration fit thresholds are invalid")
    robot = np.asarray(actual_deltas_xyz, dtype=float)
    feature = np.asarray(feature_deltas, dtype=float)
    if (
        robot.ndim != 2
        or robot.shape[1] != 3
        or feature.ndim != 2
        or feature.shape[0] != robot.shape[0]
        or feature.shape[1] not in (2, 3)
        or robot.shape[0] < minimum_probe_count
        or not np.all(np.isfinite(robot))
        or not np.all(np.isfinite(feature))
    ):
        raise ValueError(
            "at least three finite paired UV or UVD probes are required"
        )
    components = (
        tuple(range(feature.shape[1]))
        if feature_components is None
        else tuple(int(index) for index in feature_components)
    )
    if components not in ((0, 1), (0, 1, 2)):
        raise ValueError("feature components must be UV or UVD in that order")
    scale = (
        DEFAULT_FEATURE_SCALE[: feature.shape[1]].copy()
        if feature_scale is None
        else _finite_vector(
            feature_scale, feature.shape[1], "feature scale"
        )
    )
    if np.any(scale <= 0.0):
        raise ValueError("feature scale must be positive")
    horizontal = robot[:, :2]
    singular = np.linalg.svd(horizontal, compute_uv=False)
    if singular.size < 2 or singular[1] < 5e-4:
        raise ValueError("horizontal probes do not span two directions")
    motion_condition = float(singular[0] / singular[1])
    if motion_condition > maximum_motion_condition:
        raise ValueError(
            f"horizontal probe condition is too high: {motion_condition:.1f}"
        )

    # Score every independent pair with the shared reachable-model fitter.
    candidates = []
    count = horizontal.shape[0]
    for first in range(count - 1):
        for second in range(first + 1, count):
            pair = horizontal[[first, second]]
            pair_singular = np.linalg.svd(pair, compute_uv=False)
            if (
                pair_singular[1] < 5e-4
                or pair_singular[0] / pair_singular[1]
                > maximum_motion_condition
            ):
                continue
            try:
                _, candidate, _ = _reachable_fit(
                    pair, feature[[first, second]], scale
                )
            except ValueError:
                continue
            normalized = np.linalg.norm(
                (horizontal @ candidate.T - feature) / scale, axis=1
            )
            inliers = normalized <= 3.0
            inlier_residual = normalized[inliers]
            candidates.append(
                (
                    -int(np.count_nonzero(inliers)),
                    (
                        float(np.median(inlier_residual))
                        if inlier_residual.size
                        else float("inf")
                    ),
                    first,
                    second,
                    inliers,
                )
            )
    if not candidates:
        raise ValueError("no independent horizontal probe pair")
    *_, inlier_mask = min(candidates, key=lambda item: item[:4])
    minimum_inliers = max(3, int(np.ceil(0.67 * count)))
    if np.count_nonzero(inlier_mask) < minimum_inliers:
        raise ValueError("horizontal probes have no visual consensus")

    fit_x = horizontal[inlier_mask]
    fit_y = feature[inlier_mask]
    reachable, matrix, _ = _reachable_fit(fit_x, fit_y, scale)
    weighted_matrix = matrix / scale[:, None]
    weighted_singular = np.linalg.svd(
        weighted_matrix, compute_uv=False
    )
    if weighted_singular.size < 2 or weighted_singular[1] <= 1e-8:
        raise ValueError("visual features do not observe both horizontal axes")
    feature_condition = float(
        weighted_singular[0] / weighted_singular[1]
    )
    if feature_condition > maximum_feature_condition:
        raise ValueError(
            f"visual feature condition is too high: {feature_condition:.1f}"
        )
    predicted = horizontal @ matrix.T
    normalized_residual = np.linalg.norm(
        (feature - predicted) / scale, axis=1
    )
    residual_rms = float(
        np.sqrt(np.mean(np.square(normalized_residual[inlier_mask])))
    )
    maximum_residual = float(np.max(normalized_residual[inlier_mask]))
    if (
        residual_rms > maximum_residual_rms
        or maximum_residual > 3.0
    ):
        raise ValueError(
            "horizontal feature residual exceeds its calibration gate"
        )
    model = HorizontalFeatureModel(
        matrix=matrix,
        feature_scale=scale,
        feature_components=components,
        reachable_model=reachable,
        motion_condition=motion_condition,
        feature_condition=feature_condition,
        residual_rms=residual_rms,
        maximum_residual=maximum_residual,
        sample_count=robot.shape[0],
        inlier_mask=inlier_mask,
    )
    return replace(model, artifact_sha256=_model_artifact_sha256(model))


def _validate_local_sample_set(
    samples: tuple[ProbeCalibrationSample, ...],
    *,
    minimum_count: int,
    expected_context_fingerprint: str | None = None,
    reference_image: Path | None = None,
    reference_image_sha256: str | None = None,
    reference_midpoint_xyz_m: np.ndarray | None = None,
    reference_orientation_wxyz: np.ndarray | None = None,
    reference_midpoint_feature: np.ndarray | None = None,
    maximum_xy_distance_m: float = 0.020,
    maximum_z_distance_m: float = 0.003,
    maximum_feature_uv_distance_px: float = 40.0,
    maximum_feature_depth_distance_mm: float = 25.0,
):
    locality_limits = tuple(
        _finite_scalar(value, name)
        for value, name in (
            (maximum_xy_distance_m, "maximum local XY distance"),
            (maximum_z_distance_m, "maximum local Z distance"),
            (
                maximum_feature_uv_distance_px,
                "maximum local feature UV distance",
            ),
            (
                maximum_feature_depth_distance_mm,
                "maximum local feature depth distance",
            ),
        )
    )
    if any(limit <= 0.0 for limit in locality_limits):
        raise ValueError("locality limits must be positive")
    (
        maximum_xy_distance_m,
        maximum_z_distance_m,
        maximum_feature_uv_distance_px,
        maximum_feature_depth_distance_mm,
    ) = locality_limits
    if len(samples) < minimum_count:
        raise ValueError(
            f"at least {minimum_count} probe records are required"
        )
    if len({sample.record_id for sample in samples}) != len(samples):
        raise ValueError("duplicate probe record id")
    if len({sample.content_sha256 for sample in samples}) != len(samples):
        raise ValueError("duplicate probe record content")
    if len({sample.sample_fingerprint for sample in samples}) != len(samples):
        raise ValueError("duplicate physical probe sample")
    context_fingerprint = (
        samples[0].context_fingerprint
        if expected_context_fingerprint is None
        else expected_context_fingerprint
    )
    if reference_image is None:
        reference_image = samples[0].before_head_image
        reference_image_sha256 = _sha256(reference_image)
    elif (
        reference_image_sha256 is None
        or _sha256(reference_image) != reference_image_sha256
    ):
        raise ValueError("calibration reference image hash mismatch")
    for sample in samples:
        if sample.context_fingerprint != context_fingerprint:
            raise ValueError("probe calibration contexts do not match")
        registration = register_fixed_camera_view(
            reference_image, sample.before_head_image
        )
        if not registration.accepted:
            raise ValueError(
                "head camera moved between probes: " + registration.reason
            )
    midpoint = np.stack([sample.midpoint_xyz_m for sample in samples])
    center = (
        np.median(midpoint, axis=0)
        if reference_midpoint_xyz_m is None
        else _finite_vector(
            reference_midpoint_xyz_m, 3, "reference midpoint xyz"
        )
    )
    if (
        np.max(np.linalg.norm(midpoint[:, :2] - center[:2], axis=1))
        > maximum_xy_distance_m
        or np.max(np.abs(midpoint[:, 2] - center[2]))
        > maximum_z_distance_m
    ):
        raise ValueError("probe samples are outside one local workspace")
    orientation = (
        samples[0].orientation_wxyz
        if reference_orientation_wxyz is None
        else _unit_quaternion(
            reference_orientation_wxyz, "reference orientation"
        )
    )
    if any(
        _quaternion_error_deg(orientation, sample.orientation_wxyz) > 2.0
        for sample in samples
    ):
        raise ValueError("probe samples have incompatible tool orientations")
    feature_midpoints = np.stack(
        [sample.midpoint_feature for sample in samples]
    )
    feature_center = (
        np.median(feature_midpoints, axis=0)
        if reference_midpoint_feature is None
        else _finite_vector(
            reference_midpoint_feature, 3, "reference midpoint feature"
        )
    )
    if (
        np.max(
            np.linalg.norm(
                feature_midpoints[:, :2] - feature_center[:2], axis=1
            )
        )
        > maximum_feature_uv_distance_px
        or np.max(
            np.abs(feature_midpoints[:, 2] - feature_center[2])
        )
        > maximum_feature_depth_distance_mm
    ):
        raise ValueError("probe visual features are outside one local region")
    return (
        context_fingerprint,
        Path(reference_image),
        center,
        orientation,
        feature_center,
    )


def _selected_feature_data(samples, components):
    indices = tuple(int(index) for index in components)
    if indices not in ((0, 1), (0, 1, 2)):
        raise ValueError("feature components must be UV or UVD in that order")
    feature = np.stack([sample.feature_delta for sample in samples])
    return feature[:, indices], indices


def fit_probe_records(
    record_paths: Iterable[str | Path],
    *,
    feature_components=(0, 1),
    feature_scale=None,
) -> tuple[HorizontalFeatureModel, tuple[ProbeCalibrationSample, ...]]:
    """Fit a provisional local model from at least three v2 probe records.

    UV is the motion-eligibility default.  Depth may be selected explicitly
    for diagnostics, but its stationary Record3D noise makes it unsuitable as
    a default horizontal control feature.
    """

    samples = tuple(load_probe_record(path) for path in record_paths)
    (
        context_fingerprint,
        reference_image,
        center,
        orientation,
        feature_center,
    ) = _validate_local_sample_set(samples, minimum_count=3)
    selected_feature, components = _selected_feature_data(
        samples, feature_components
    )
    if feature_scale is None:
        scale = DEFAULT_FEATURE_SCALE[list(components)]
    else:
        scale = _finite_vector(
            feature_scale, len(components), "feature scale"
        )
    model = fit_horizontal_feature_model(
        [sample.actual_xyz_m for sample in samples],
        selected_feature,
        feature_scale=scale,
        feature_components=components,
    )
    midpoint_xyz = np.stack(
        [sample.midpoint_xyz_m for sample in samples]
    )
    midpoint_feature = np.stack(
        [sample.midpoint_feature for sample in samples]
    )
    xy_spread = float(
        np.max(np.linalg.norm(midpoint_xyz[:, :2] - center[:2], axis=1))
    )
    z_spread = float(
        np.max(np.abs(midpoint_xyz[:, 2] - center[2]))
    )
    uv_spread = float(
        np.max(
            np.linalg.norm(
                midpoint_feature[:, :2] - feature_center[:2],
                axis=1,
            )
        )
    )
    depth_spread = float(
        np.max(np.abs(midpoint_feature[:, 2] - feature_center[2]))
    )
    model = replace(
        model,
        context_fingerprint=context_fingerprint,
        source_record_ids=tuple(sample.record_id for sample in samples),
        source_record_sha256s=tuple(
            sample.content_sha256 for sample in samples
        ),
        source_sample_fingerprints=tuple(
            sample.sample_fingerprint for sample in samples
        ),
        source_actual_xyz_m=np.stack(
            [sample.actual_xyz_m for sample in samples]
        ),
        reference_head_image=reference_image,
        reference_head_image_sha256=_sha256(reference_image),
        reference_midpoint_xyz_m=center,
        reference_orientation_wxyz=orientation,
        reference_midpoint_feature=feature_center,
        applicability_xy_radius_m=min(0.020, xy_spread + 0.008),
        applicability_z_radius_m=min(0.003, z_spread + 0.001),
        applicability_uv_radius_px=min(40.0, uv_spread + 20.0),
        applicability_depth_radius_mm=min(
            25.0, depth_spread + 20.0
        ),
    )
    model = replace(model, artifact_sha256=_model_artifact_sha256(model))
    return model, samples


def verify_probe_records(
    model: HorizontalFeatureModel,
    record_paths: Iterable[str | Path],
    *,
    maximum_residual_rms: float = 1.0,
    maximum_sample_residual: float = 2.0,
    minimum_direction_cosine: float = 0.90,
    minimum_gain_ratio: float = 0.70,
    maximum_gain_ratio: float = 1.30,
    maximum_motion_condition: float = 5.0,
) -> tuple[
    HorizontalFeatureModel,
    HeldOutVerification,
    tuple[ProbeCalibrationSample, ...],
]:
    """Gate a fitted model with one or more genuinely held-out probes.

    This function only evaluates already-recorded artifacts.  A successful
    report changes ``model.verified`` and thereby enables :meth:`solve`; it
    never commands a robot.
    """

    thresholds = (
        _finite_scalar(maximum_residual_rms, "maximum residual RMS"),
        _finite_scalar(
            maximum_sample_residual, "maximum sample residual"
        ),
        _finite_scalar(
            minimum_direction_cosine, "minimum direction cosine"
        ),
        _finite_scalar(minimum_gain_ratio, "minimum gain ratio"),
        _finite_scalar(maximum_gain_ratio, "maximum gain ratio"),
        _finite_scalar(
            maximum_motion_condition,
            "maximum held-out motion condition",
        ),
    )
    (
        maximum_residual_rms,
        maximum_sample_residual,
        minimum_direction_cosine,
        minimum_gain_ratio,
        maximum_gain_ratio,
        maximum_motion_condition,
    ) = thresholds
    if (
        maximum_residual_rms <= 0.0
        or maximum_sample_residual <= 0.0
        or not -1.0 <= minimum_direction_cosine <= 1.0
        or minimum_gain_ratio <= 0.0
        or maximum_gain_ratio < minimum_gain_ratio
        or maximum_motion_condition < 1.0
    ):
        raise ValueError("held-out verification thresholds are invalid")
    if (
        not model.artifact_sha256
        or model.artifact_sha256 != _model_artifact_sha256(model)
    ):
        raise ValueError("calibration model artifact hash mismatch")
    if (
        model.context_fingerprint is None
        or model.reference_head_image is None
        or model.reference_head_image_sha256 is None
        or model.reference_midpoint_xyz_m is None
        or model.reference_orientation_wxyz is None
        or model.reference_midpoint_feature is None
        or model.applicability_xy_radius_m is None
        or model.applicability_z_radius_m is None
        or model.applicability_uv_radius_px is None
        or model.applicability_depth_radius_mm is None
        or not model.source_record_ids
        or not model.source_record_sha256s
        or not model.source_sample_fingerprints
        or model.source_actual_xyz_m is None
    ):
        raise ValueError(
            "held-out verification requires a model fitted from probe records"
        )

    samples = tuple(load_probe_record(path) for path in record_paths)
    _validate_local_sample_set(
        samples,
        minimum_count=1,
        expected_context_fingerprint=model.context_fingerprint,
        reference_image=model.reference_head_image,
        reference_image_sha256=model.reference_head_image_sha256,
        reference_midpoint_xyz_m=model.reference_midpoint_xyz_m,
        reference_orientation_wxyz=model.reference_orientation_wxyz,
        reference_midpoint_feature=model.reference_midpoint_feature,
        maximum_xy_distance_m=model.applicability_xy_radius_m,
        maximum_z_distance_m=model.applicability_z_radius_m,
        maximum_feature_uv_distance_px=(
            model.applicability_uv_radius_px
        ),
        maximum_feature_depth_distance_mm=(
            model.applicability_depth_radius_mm
        ),
    )
    if set(model.source_record_ids) & {
        sample.record_id for sample in samples
    }:
        raise ValueError("held-out probe record was used for model fitting")
    if set(model.source_record_sha256s) & {
        sample.content_sha256 for sample in samples
    }:
        raise ValueError("held-out probe content was used for model fitting")
    if set(model.source_sample_fingerprints) & {
        sample.sample_fingerprint for sample in samples
    }:
        raise ValueError("held-out physical probe was used for model fitting")

    observed, components = _selected_feature_data(
        samples, model.feature_components
    )
    if components != model.feature_components:
        raise ValueError("held-out feature components do not match the model")
    horizontal = np.stack(
        [sample.actual_xyz_m[:2] for sample in samples]
    )
    source_actual = np.asarray(model.source_actual_xyz_m, dtype=float)
    if (
        source_actual.shape != (model.sample_count, 3)
        or not np.all(np.isfinite(source_actual))
    ):
        raise ValueError("calibration source motion provenance is malformed")
    held_out_singular = np.linalg.svd(horizontal, compute_uv=False)
    held_out_rank = int(
        np.count_nonzero(held_out_singular > 5e-4)
    )
    held_out_condition = (
        float(held_out_singular[0] / held_out_singular[1])
        if held_out_rank == 2
        else None
    )
    combined_horizontal = np.vstack((source_actual[:, :2], horizontal))
    signed_axis_coverage = tuple(
        (
            bool(np.any(combined_horizontal[:, axis] >= 5e-4)),
            bool(np.any(combined_horizontal[:, axis] <= -5e-4)),
        )
        for axis in range(2)
    )
    predicted = horizontal @ model.matrix.T
    normalized_error = (observed - predicted) / model.feature_scale
    sample_residual = np.linalg.norm(normalized_error, axis=1)
    residual_rms = float(np.sqrt(np.mean(np.square(sample_residual))))
    maximum_residual = float(np.max(sample_residual))

    weighted_observed = observed / model.feature_scale
    weighted_predicted = predicted / model.feature_scale
    observed_norm = np.linalg.norm(weighted_observed, axis=1)
    predicted_norm = np.linalg.norm(weighted_predicted, axis=1)
    valid_direction = (observed_norm > 1e-12) & (
        predicted_norm > 1e-12
    )
    direction_cosines = np.full(len(samples), -1.0)
    direction_cosines[valid_direction] = np.sum(
        weighted_observed[valid_direction]
        * weighted_predicted[valid_direction],
        axis=1,
    ) / (
        observed_norm[valid_direction] * predicted_norm[valid_direction]
    )
    direction_cosines = np.clip(direction_cosines, -1.0, 1.0)
    gain_ratios = np.full(len(samples), np.finfo(float).max)
    gain_ratios[predicted_norm > 1e-12] = (
        observed_norm[predicted_norm > 1e-12]
        / predicted_norm[predicted_norm > 1e-12]
    )
    minimum_cosine = float(np.min(direction_cosines))
    minimum_gain = float(np.min(gain_ratios))
    maximum_gain = float(np.max(gain_ratios))

    reasons = []
    if len(samples) < 2:
        reasons.append("at least two held-out probes are required")
    if held_out_rank < 2:
        reasons.append(
            "held-out probes do not span two horizontal directions"
        )
    elif held_out_condition > maximum_motion_condition:
        reasons.append("held-out motion condition is too high")
    if not all(
        positive and negative
        for positive, negative in signed_axis_coverage
    ):
        reasons.append(
            "combined probes lack positive and negative X/Y excitation"
        )
    if residual_rms > maximum_residual_rms:
        reasons.append("held-out normalized residual RMS is too high")
    if maximum_residual > maximum_sample_residual:
        reasons.append("held-out sample residual is too high")
    if minimum_cosine < minimum_direction_cosine:
        reasons.append("held-out feature direction is inconsistent")
    if (
        minimum_gain < minimum_gain_ratio
        or maximum_gain > maximum_gain_ratio
    ):
        reasons.append("held-out feature gain is inconsistent")
    report = HeldOutVerification(
        accepted=not reasons,
        sample_count=len(samples),
        horizontal_rank=held_out_rank,
        horizontal_condition=held_out_condition,
        combined_signed_axis_coverage=signed_axis_coverage,
        normalized_residual_rms=residual_rms,
        maximum_normalized_residual=maximum_residual,
        minimum_direction_cosine=minimum_cosine,
        minimum_gain_ratio=minimum_gain,
        maximum_gain_ratio=maximum_gain,
        record_ids=tuple(sample.record_id for sample in samples),
        reasons=tuple(reasons),
    )
    verified_model = replace(
        model,
        verified=report.accepted,
        verification=report,
        artifact_sha256="",
    )
    verified_model = replace(
        verified_model,
        artifact_sha256=_model_artifact_sha256(verified_model),
    )
    return verified_model, report, samples


# Explicit alias for callers that prefer the purpose in the API name.
verify_held_out_probe_records = verify_probe_records


def recommend_next_probe(
    record_paths: Iterable[str | Path],
    *,
    distance_m: float = 0.006,
) -> dict:
    """Recommend an information-gaining XY probe without commanding motion."""

    distance_m = float(distance_m)
    if not np.isfinite(distance_m) or not 0.0 < distance_m <= 0.008:
        raise ValueError("recommended probe distance is outside (0, 8] mm")
    attempts = []
    usable = []
    balance_motion = []
    for path in record_paths:
        raw = _load_json(Path(path).resolve())
        motion = raw.get("motion")
        if not isinstance(motion, dict):
            raise ValueError(f"{path}: probe motion is missing")
        requested = _finite_vector(
            motion.get("requested_xyz_m"), 3, "requested xyz"
        )
        norm = float(np.linalg.norm(requested[:2]))
        if norm > 0.0:
            # Keep every design row in metres.  Normalizing attempted rows
            # while candidates remain millimetre-scale corrupts rank scoring.
            attempts.append(requested[:2])
        try:
            sample = load_probe_record(path)
        except ValueError:
            if norm > 0.0:
                balance_motion.append(requested[:2])
            continue
        usable.append(sample.actual_xyz_m[:2])
        balance_motion.append(sample.actual_xyz_m[:2])

    candidates = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
        ]
    )
    candidates /= np.linalg.norm(candidates, axis=1)[:, None]
    design = (
        np.asarray(usable, dtype=float)
        if usable
        else np.asarray(attempts, dtype=float)
    )
    scores = []
    for index, direction in enumerate(candidates):
        candidate_motion = distance_m * direction
        augmented = (
            candidate_motion[None, :]
            if design.size == 0
            else np.vstack((design, candidate_motion))
        )
        singular = np.linalg.svd(augmented, compute_uv=False)
        rank = int(np.count_nonzero(singular > 5e-4))
        weakest = float(singular[-1]) if singular.size >= 2 else 0.0
        # Once rank is established, prefer a locally balanced +/- design so
        # linearity is checked without drifting away from the working pose.
        balance_augmented = (
            candidate_motion[None, :]
            if not balance_motion
            else np.vstack((balance_motion, candidate_motion))
        )
        imbalance = float(
            np.linalg.norm(np.sum(balance_augmented, axis=0))
        )
        scores.append((rank, weakest, -imbalance, -index, direction))
    *_, direction = max(scores, key=lambda item: item[:4])
    return {
        "requested_xyz_m": [
            float(distance_m * direction[0]),
            float(distance_m * direction[1]),
            0.0,
        ],
        "rationale": (
            "maximize the weakest measured horizontal excitation; "
            "physical execution still requires a new one-shot token"
        ),
        "usable_sample_count": len(usable),
        "attempted_direction_count": len(attempts),
    }

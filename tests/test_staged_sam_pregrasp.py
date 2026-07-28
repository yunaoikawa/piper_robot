#!/usr/bin/env python3

import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.local_feature_calibration import load_probe_record
from rollout.realtime_sam_servo import (
    GRIPPER_COLOR_MINIMUM_MASK_FRACTION,
    GRIPPER_COLOR_MINIMUM_PIXELS,
    GRIPPER_CYAN_HSV_LOWER,
    GRIPPER_CYAN_HSV_UPPER,
)
import src.run_staged_sam_pregrasp as staged_pregrasp
from src.run_staged_sam_pregrasp import (
    _cleanup_staged_runtime,
    bound_horizontal_step,
    claim_motion_execution,
    estimate_horizontal_displacement,
    execute_single_horizontal_probe,
    fit_horizontal_jacobian,
    main,
    verify_right_stationary,
)


true_jacobian = np.array(
    [[420.0, -90.0, 110.0], [60.0, 310.0, -520.0], [80.0, 640.0, -220.0]]
)
robot = np.array(
    [[0.006, 0.0, 0.0], [0.0, 0.006, 0.0], [0.0, 0.0, 0.006]]
)
feature = robot @ true_jacobian.T
estimated = fit_horizontal_jacobian(robot[:2], feature[:2])
assert np.allclose(estimated, true_jacobian[:2, :2])

expected_displacement = np.array([0.045, -0.070, -0.090])
error = np.r_[estimated @ expected_displacement[:2], 999.0]
displacement = estimate_horizontal_displacement(estimated, error)
assert np.allclose(displacement, [*expected_displacement[:2], 0.0])

step = bound_horizontal_step(
    displacement, max_norm_m=0.008, max_axis_m=0.006
)
assert step[2] == 0.0
assert np.linalg.norm(step[:2]) <= 0.008001
assert np.max(np.abs(step[:2])) <= 0.006001
assert np.dot(step[:2], expected_displacement[:2]) > 0.0

legacy_cli_stderr = io.StringIO()
try:
    with redirect_stderr(legacy_cli_stderr):
        main(
            [
                "--execute-horizontal",
                "--motion-token",
                "legacy-multi-motion-is-forbidden",
            ]
        )
    raise AssertionError("legacy multi-command CLI path was accepted")
except SystemExit as exc:
    assert exc.code == 2
assert "--execute-horizontal requires --single-probe-axis" in (
    legacy_cli_stderr.getvalue()
)


class FakeCleanupTarget:
    def __init__(self, stop_error=None):
        self.stop_error = stop_error
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error


class FakeCleanupClaim:
    def __init__(self, finalize_error=None):
        self.finalize_error = finalize_error
        self.finalize_calls = []

    def finalize(self, error):
        self.finalize_calls.append(error)
        if self.finalize_error is not None:
            raise self.finalize_error


primary_error = RuntimeError("primary execution failure")
hold_error = RuntimeError("hold cleanup failure")
right_stop_error = RuntimeError("right stop failure")
runner_stop_error = RuntimeError("runner stop failure")
claim_finalize_error = RuntimeError("claim finalize failure")
cleanup_runner = FakeCleanupTarget(runner_stop_error)
cleanup_right = FakeCleanupTarget(right_stop_error)
cleanup_claim = FakeCleanupClaim(claim_finalize_error)
assert (
    _cleanup_staged_runtime(
        cleanup_runner,
        cleanup_right,
        cleanup_claim,
        primary_error,
        [("runner.hold_measured", hold_error)],
    )
    is None
)
assert cleanup_runner.stop_calls == 1
assert cleanup_right.stop_calls == 1
assert cleanup_claim.finalize_calls == [primary_error]
primary_notes = "\n".join(getattr(primary_error, "__notes__", ()))
assert "runner.hold_measured" in primary_notes
assert "right.stop" in primary_notes
assert "runner.stop" in primary_notes
assert "execution claim finalization" in primary_notes

finalize_only_error = RuntimeError("finalize-only failure")
assert (
    _cleanup_staged_runtime(
        FakeCleanupTarget(),
        FakeCleanupTarget(),
        FakeCleanupClaim(finalize_only_error),
        None,
    )
    is finalize_only_error
)

main_calls = {}


class FakeRejectedGeometryMainRunner(FakeCleanupTarget):
    last_instance = None

    def __init__(self, args):
        super().__init__()
        type(self).last_instance = self
        self.args = args
        self.start_calls = 0
        self.last_geometry_quality = SimpleNamespace(
            accepted=False,
            view_angle_deg=68.0,
            native_pixel_footprint_x_m=0.010,
            native_pixel_footprint_y_m=0.012,
            reasons=("future descent geometry is too oblique",),
        )
        self.last_target_3d = SimpleNamespace(
            point_camera_xyz_m=np.array([0.1, 0.2, 0.8]),
            confidence=0.25,
        )
        self.last_proximity_m = 0.20

    def start(self):
        self.start_calls += 1

    def observe(self, clearance_m):
        feature = SimpleNamespace()
        return feature, np.array([2.0, 3.0, 4.0]), "head.png", 123.0

    def check_scene_registration(self, reference_points):
        return {"accepted": True}


class FakeMainRightObserver(FakeCleanupTarget):
    last_instance = None

    def __init__(self, runner, output_dir):
        super().__init__()
        type(self).last_instance = self
        self.start_calls = 0

    def start(self):
        self.start_calls += 1

    def observe(self, require_lid):
        return None, None, "right.png", 124.0


class FakeMainClaim:
    def __init__(self):
        self.payload = {"token_sha256": "a" * 64}
        self.result = None
        self.attempts = 0
        self.finalize_calls = []

    def mark_motion_attempt(self):
        self.attempts += 1

    def set_result(self, result):
        self.result = result

    def finalize(self, error):
        self.finalize_calls.append(error)


fake_main_claim = FakeMainClaim()


def fake_execute_single_probe(runner, right, axis, distance_m, **kwargs):
    kwargs["motion_attempt_callback"]()
    main_calls["execute"] = {
        "axis": axis,
        "distance_m": distance_m,
        "geometry_accepted": runner.last_geometry_quality.accepted,
    }
    return {
        "status": "SINGLE_X_PROBE_COMMITTED",
        "quality": {"usable_for_fit": True},
    }


patched_main_attributes = {
    "LiveSamGrasp": staged_pregrasp.LiveSamGrasp,
    "RightLidObserver": staged_pregrasp.RightLidObserver,
    "claim_motion_execution": staged_pregrasp.claim_motion_execution,
    "execute_single_horizontal_probe": (
        staged_pregrasp.execute_single_horizontal_probe
    ),
    "_required_sha256": staged_pregrasp._required_sha256,
}
try:
    staged_pregrasp.LiveSamGrasp = FakeRejectedGeometryMainRunner
    staged_pregrasp.RightLidObserver = FakeMainRightObserver
    staged_pregrasp.claim_motion_execution = (
        lambda *args, **kwargs: fake_main_claim
    )
    staged_pregrasp.execute_single_horizontal_probe = (
        fake_execute_single_probe
    )
    staged_pregrasp._required_sha256 = lambda *args, **kwargs: "b" * 64
    with redirect_stdout(io.StringIO()):
        main(
            [
                "--execute-horizontal",
                "--single-probe-axis",
                "x",
                "--single-probe-m",
                "+0.006",
                "--motion-token",
                "advisory-depth-geometry",
                "--output-dir",
                "/tmp/pure-rejected-geometry-probe",
            ]
        )
finally:
    for name, value in patched_main_attributes.items():
        setattr(staged_pregrasp, name, value)
assert main_calls["execute"]["geometry_accepted"] is False
assert main_calls["execute"]["axis"] == "x"
assert fake_main_claim.attempts == 1
assert fake_main_claim.finalize_calls == [None]
assert FakeRejectedGeometryMainRunner.last_instance.stop_calls == 1
assert FakeMainRightObserver.last_instance.stop_calls == 1
invalid_geometry_record = staged_pregrasp._geometry_quality_record(
    SimpleNamespace(
        accepted=False,
        view_angle_deg=float("nan"),
        native_pixel_footprint_x_m=float("nan"),
        native_pixel_footprint_y_m=-1.0,
        reasons=("invalid depth-only observation",),
    )
)
assert invalid_geometry_record["view_angle_deg"] is None
assert invalid_geometry_record["native_depth_pixel_footprint_m"] == [
    None,
    None,
]
assert (
    invalid_geometry_record["horizontal_uv_probe_policy"]
    == "record_only_not_motion_gate"
)


_artifact_tmp = tempfile.TemporaryDirectory()
_artifact_root = Path(_artifact_tmp.name)
_artifact_image = np.zeros((240, 320, 3), dtype=np.uint8)
for y in range(20, 230, 30):
    for x in range(20, 310, 30):
        cv2.circle(_artifact_image, (x, y), 4, (x % 255, y % 255, 200), -1)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _head_artifacts(output_dir, sequence, *, feature):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{sequence:03d}_head"
    observation_image = _artifact_image.copy()
    gripper_mask = np.zeros(observation_image.shape[:2], dtype=np.uint8)
    gripper_left = 50 + 4 * sequence
    cv2.rectangle(
        observation_image,
        (gripper_left, 75),
        (gripper_left + 50, 85),
        (255, 255, 0),
        -1,
    )
    cv2.rectangle(
        gripper_mask,
        (gripper_left, 75),
        (gripper_left + 50, 85),
        255,
        -1,
    )
    image_paths = {
        "raw_image": output_dir / f"{prefix}_raw.png",
        "sam_input_png": output_dir / f"{prefix}_sam_input.png",
        "overlay_image": output_dir / f"{sequence:03d}.png",
        "lid_mask": output_dir / f"{prefix}_lid_mask.png",
        "gripper_mask": output_dir / f"{prefix}_gripper_mask.png",
        "gripper_feature_support_mask": (
            output_dir / f"{prefix}_gripper_feature_support_mask.png"
        ),
    }
    for role in ("raw_image", "sam_input_png", "overlay_image"):
        assert cv2.imwrite(
            str(image_paths[role]), observation_image
        )
    assert cv2.imwrite(str(image_paths["lid_mask"]), np.zeros_like(gripper_mask))
    assert cv2.imwrite(str(image_paths["gripper_mask"]), gripper_mask)
    assert cv2.imwrite(
        str(image_paths["gripper_feature_support_mask"]),
        gripper_mask,
    )
    request_path = output_dir / f"{prefix}_sam_request_q90.jpg"
    assert cv2.imwrite(
        str(request_path),
        observation_image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    roi_input_path = output_dir / f"{prefix}_sam_roi_input.png"
    assert cv2.imwrite(str(roi_input_path), observation_image)
    roi_request_path = output_dir / f"{prefix}_sam_roi_request_q90.jpg"
    assert cv2.imwrite(
        str(roi_request_path),
        observation_image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    depth_path = output_dir / f"{prefix}_depth.npz"
    np.savez_compressed(
        depth_path,
        depth_m=np.full((240, 320), 0.9, dtype=np.float64),
        camera_matrix=np.eye(3),
        source_timestamps=np.array([123.0 + sequence]),
    )
    paths = {
        **{key: str(value) for key, value in image_paths.items()},
        "sam_request_jpeg_q90": str(request_path),
        "sam_roi_input_png": str(roi_input_path),
        "sam_roi_request_jpeg_q90": str(roi_request_path),
        "depth_npz": str(depth_path),
    }
    files = {
        role: {
            "path": Path(path).name,
            "sha256": _sha256(path),
            "bytes": Path(path).stat().st_size,
            "media_type": (
                "application/x-npz"
                if role == "depth_npz"
                else (
                    "image/jpeg"
                    if role
                    in (
                        "sam_request_jpeg_q90",
                        "sam_roi_request_jpeg_q90",
                    )
                    else "image/png"
                )
            ),
        }
        for role, path in paths.items()
    }
    document = {
        "schema": "sam_head_observation/v2",
        "sequence": sequence,
        "run_id": "test-run",
        "attempt_id": f"test-attempt-{sequence}",
        "files": files,
        "image_shape_hw": [240, 320],
        "preprocess": "identity",
        "depth_frames_requested": 3,
        "depth_frames_used": 1,
        "lid": {
            "prompt": "transparent round petri dish lid with blue cross",
            "model": "fake-sam",
        },
        "gripper": {
            "prompt": "blue clamp",
            "model": "fake-sam",
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
                "support_pixel_count": int(np.count_nonzero(gripper_mask)),
                "support_artifact_role": "gripper_feature_support_mask",
                "support_artifact_path": Path(
                    image_paths["gripper_feature_support_mask"]
                ).name,
            },
        },
        "feature": feature,
    }
    manifest = output_dir / f"{prefix}_observation.json"
    manifest.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return {
        **document,
        **paths,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
    }


def _right_artifacts(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "right.png"
    cv2.imwrite(str(image_path), _artifact_image)
    return {
        "schema": "sam_right_observation/v1",
        "sequence": 0,
        "raw_image": str(image_path),
        "sam_input_png": str(image_path),
        "sam_request_jpeg_q90": str(image_path),
        "overlay_image": str(image_path),
        "lid_mask": str(image_path),
    }


class FakeSingleProbeRunner:
    def __init__(self):
        self.moves = []
        self.holds = 0
        self.observations = 0
        self.rpc = SimpleNamespace()
        self.pose = np.array([1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.8])
        self.output_dir = _artifact_root / "head"
        self.last_observation_artifacts = None
        self.last_geometry_quality = SimpleNamespace(
            accepted=False,
            view_angle_deg=65.0,
            native_pixel_footprint_x_m=0.009,
            native_pixel_footprint_y_m=0.011,
            reasons=(
                "view angle 65.0deg > 40.0deg",
                "depth-pixel footprint 11.0mm > 7.0mm",
            ),
        )
        config_dir = _artifact_root / "config"
        config_dir.mkdir(exist_ok=True)
        reference = _artifact_root / "placement_reference.png"
        assert cv2.imwrite(str(reference), _artifact_image)
        scene_config = config_dir / "scene.json"
        scene_config.write_text(
            json.dumps(
                {
                    "head_camera_placement_reference": {
                        "capture_id": "test-placement",
                        "record3d_udid": "test-head-camera",
                        "rgb_path": str(reference),
                        "rgb_sha256": _sha256(reference),
                        "registration_gate": "orb_identity_v1",
                    }
                }
            )
        )
        torque_config = config_dir / "torque.json"
        torque_config.write_text('{"version": 1}\n')
        self.args = SimpleNamespace(
            scene_config=str(scene_config),
            torque_config=str(torque_config),
            depth_frames=3,
        )
        self.head_camera_matrix = np.eye(3)
        self.head_camera_reference_shape = (240, 320)
        self.torque_limit = np.ones(6)
        self.rpc.get_right_ee_pose = lambda: SimpleNamespace(
            parameters=lambda: self.pose.copy()
        )
        self.rpc.get_right_joint_positions = lambda: np.zeros(6)
        self.rpc.get_right_joint_torque = lambda: np.zeros(6)

    def hold_measured(self):
        self.holds += 1

    def move_cartesian_delta(self, request, minimum_progress):
        self.moves.append(
            (np.asarray(request, dtype=float).copy(), minimum_progress)
        )
        request = np.asarray(request, dtype=float)
        actual = 0.9666666666666667 * request
        if request[0] != 0.0:
            actual[1] = -np.sign(request[0]) * 0.0001
        self.pose[4:7] += actual
        return actual

    def observe(self, clearance_m):
        gripper = np.array(
            [100.0 + 4.0 * self.observations, 80.0, 900.0]
        )
        lid = np.array([20.0, 260.0, 840.0])
        feature = SimpleNamespace(
            gripper_feature=gripper,
            lid_grasp_feature=lid,
        )
        error = lid - gripper
        timestamp = 123.0 + self.observations
        self.observations += 1
        self.last_observation_artifacts = _head_artifacts(
            self.output_dir,
            self.observations - 1,
            feature={
                "lid_grasp_feature": lid.tolist(),
                "gripper_feature": gripper.tolist(),
                "error": error.tolist(),
            },
        )
        return (
            feature,
            error,
            self.last_observation_artifacts["overlay_image"],
            timestamp,
        )


class FakeSingleProbeRight:
    def __init__(self):
        self.observations = 0
        self.output_dir = _artifact_root / "right"
        self.last_observation_artifacts = None

    def observe(self, require_lid):
        assert require_lid is False
        self.observations += 1
        geometry = SimpleNamespace(center_px=np.array([120.0, 90.0]))
        candidate = SimpleNamespace(score=0.8)
        self.last_observation_artifacts = _right_artifacts(self.output_dir)
        return (
            geometry,
            candidate,
            self.last_observation_artifacts["overlay_image"],
            124.0,
        )


class MovingDuringSamRunner(FakeSingleProbeRunner):
    def observe(self, clearance_m):
        result = super().observe(clearance_m)
        if self.observations == 2:
            self.pose[4] += 0.001
        return result


class MissingPreMotionEvidenceRunner(FakeSingleProbeRunner):
    def observe(self, clearance_m):
        result = super().observe(clearance_m)
        self.last_observation_artifacts = None
        return result


class ZeroMotionRunner(FakeSingleProbeRunner):
    def move_cartesian_delta(self, request, minimum_progress):
        self.moves.append(
            (np.asarray(request, dtype=float).copy(), minimum_progress)
        )
        return np.zeros(3)


class ReverseMotionRunner(FakeSingleProbeRunner):
    def move_cartesian_delta(self, request, minimum_progress):
        self.moves.append(
            (np.asarray(request, dtype=float).copy(), minimum_progress)
        )
        actual = -0.5 * np.asarray(request, dtype=float)
        self.pose[4:7] += actual
        return actual


def _orientation_wxyz(angle_deg):
    half_angle = np.radians(float(angle_deg)) / 2.0
    return np.array(
        [np.cos(half_angle), np.sin(half_angle), 0.0, 0.0]
    )


class ExcessImmediateHorizontalRunner(FakeSingleProbeRunner):
    def move_cartesian_delta(self, request, minimum_progress):
        self.moves.append(
            (np.asarray(request, dtype=float).copy(), minimum_progress)
        )
        actual = np.array([0.0101, 0.0, 0.0])
        self.pose[4:7] += actual
        return actual


class ImmediateVerticalDriftRunner(FakeSingleProbeRunner):
    def move_cartesian_delta(self, request, minimum_progress):
        self.moves.append(
            (np.asarray(request, dtype=float).copy(), minimum_progress)
        )
        actual = np.asarray(request, dtype=float).copy()
        actual[2] = 0.00051
        self.pose[4:7] += actual
        return actual


class ImmediateOrientationDriftRunner(FakeSingleProbeRunner):
    def move_cartesian_delta(self, request, minimum_progress):
        actual = super().move_cartesian_delta(request, minimum_progress)
        self.pose[:4] = _orientation_wxyz(0.51)
        return actual


class SettledPoseDriftRunner(FakeSingleProbeRunner):
    def __init__(self, *, translation_xyz=None, orientation_deg=None):
        super().__init__()
        self._settled_translation = np.asarray(
            (
                np.zeros(3)
                if translation_xyz is None
                else translation_xyz
            ),
            dtype=float,
        )
        self._settled_orientation_deg = orientation_deg
        self._pose_read_count = 0
        self._settled_drift_applied = False
        self.rpc.get_right_ee_pose = self._get_right_ee_pose

    def _get_right_ee_pose(self):
        self._pose_read_count += 1
        if self._pose_read_count == 3 and not self._settled_drift_applied:
            self.pose[4:7] += self._settled_translation
            if self._settled_orientation_deg is not None:
                self.pose[:4] = _orientation_wxyz(
                    self._settled_orientation_deg
                )
            self._settled_drift_applied = True
        return SimpleNamespace(parameters=lambda: self.pose.copy())


single_runner = FakeSingleProbeRunner()
single_right = FakeSingleProbeRight()
single_record = _artifact_root / "single_probe_record.json"
single_report = execute_single_horizontal_probe(
    single_runner,
    single_right,
    "x",
    0.006,
    hold_window_s=0.0,
    journal_path=single_record,
    motion_token_sha256="a" * 64,
)
assert len(single_runner.moves) == 1
assert np.array_equal(single_runner.moves[0][0], [0.006, 0.0, 0.0])
assert single_runner.moves[0][1] == 0.0
assert single_right.observations == 2
assert single_report["status"] == "SINGLE_X_PROBE_COMMITTED"
assert np.allclose(
    single_report["motion"]["actual_settled_xyz_m"],
    [0.0058, -0.0001, 0.0],
)
assert single_report["motion"]["hold"]["verified"]
assert (
    single_report["context"]["depth_geometry_quality"][
        "accepted_for_future_descent_or_3d"
    ]
    is False
)
assert (
    single_report["context"]["depth_geometry_quality"][
        "horizontal_uv_probe_policy"
    ]
    == "record_only_not_motion_gate"
)
assert single_report["motion"]["initial_hold"]["torque_limit_nm"] == [
    1.0
] * 6
assert single_report["motion"]["hold"]["torque_limit_nm"] == [1.0] * 6
assert single_report["quality"]["usable_for_fit"]
assert single_report["execution"]["motion_attempted"]
assert single_report["execution"]["motion_command_completed"]
assert single_report["execution"]["immediate_motion_validated"]
assert single_report["execution"]["settled_motion_validated"]
assert single_report["execution"]["stage"] == "committed"
assert single_report["authorization"]["motion_token_sha256"] == "a" * 64
assert json.loads(single_record.read_text())["record_id"] == (
    single_report["record_id"]
)
assert (
    Path(single_report["observation"]["after"]["head_image"]).name
    == "001.png"
)
assert (
    Path(single_report["observation"]["after"]["right_image"]).name
    == "right.png"
)
loaded = load_probe_record(single_record)
assert loaded.record_id == single_report["record_id"]
assert verify_right_stationary(
    single_runner.rpc, duration_s=0.0
)["torque_limit_nm"] is None
duplicate_runner = FakeSingleProbeRunner()
try:
    execute_single_horizontal_probe(
        duplicate_runner,
        FakeSingleProbeRight(),
        "x",
        0.006,
        hold_window_s=0.0,
        probe_context={"context_id_sha256": "test-context"},
        journal_path=single_record,
    )
    raise AssertionError("existing one-shot probe journal was overwritten")
except FileExistsError:
    pass
assert len(duplicate_runner.moves) == 0

try:
    execute_single_horizontal_probe(
        FakeSingleProbeRunner(), FakeSingleProbeRight(), "z", 0.006
    )
    raise AssertionError("non-horizontal single probe was accepted")
except ValueError:
    pass

missing_torque_runner = FakeSingleProbeRunner()
del missing_torque_runner.torque_limit
try:
    execute_single_horizontal_probe(
        missing_torque_runner,
        FakeSingleProbeRight(),
        "x",
        0.006,
        hold_window_s=0.0,
    )
    raise AssertionError("probe without a torque limit was executed")
except RuntimeError as exc:
    assert "valid right-arm torque limit" in str(exc)
assert missing_torque_runner.observations == 0
assert len(missing_torque_runner.moves) == 0

negative_runner = FakeSingleProbeRunner()
negative_report = execute_single_horizontal_probe(
    negative_runner,
    FakeSingleProbeRight(),
    "y",
    -0.004,
    hold_window_s=0.0,
    probe_context={"context_id_sha256": "test-context"},
)
assert np.array_equal(
    negative_runner.moves[0][0], [0.0, -0.004, 0.0]
)
assert negative_report["motion"]["requested_xyz_m"] == [0.0, -0.004, 0.0]
assert negative_report["motion"]["actual_settled_xyz_m"][1] < 0.0

pre_motion_record = _artifact_root / "pre_motion_failure.json"
pre_motion_runner = MissingPreMotionEvidenceRunner()
try:
    execute_single_horizontal_probe(
        pre_motion_runner,
        FakeSingleProbeRight(),
        "x",
        0.006,
        hold_window_s=0.0,
        probe_context={"context_id_sha256": "test-context"},
        journal_path=pre_motion_record,
    )
    raise AssertionError("probe without pre-motion evidence was executed")
except RuntimeError as exc:
    assert "observation provenance" in str(exc)
assert len(pre_motion_runner.moves) == 0
pre_motion_payload = json.loads(pre_motion_record.read_text())
assert pre_motion_payload["execution"]["motion_attempted"] is False
assert pre_motion_payload["execution"]["failure_timing"] == (
    "before_motion_attempt"
)

placement_record = _artifact_root / "placement_failure.json"
placement_runner = FakeSingleProbeRunner()
scene_path = Path(placement_runner.args.scene_config)
scene_profile = json.loads(scene_path.read_text())
scene_profile["head_camera_placement_reference"]["rgb_sha256"] = "0" * 64
scene_path.write_text(json.dumps(scene_profile))
try:
    execute_single_horizontal_probe(
        placement_runner,
        FakeSingleProbeRight(),
        "x",
        0.006,
        hold_window_s=0.0,
        journal_path=placement_record,
    )
    raise AssertionError("invalid camera placement reached physical motion")
except RuntimeError as exc:
    assert "placement reference hash mismatch" in str(exc)
assert len(placement_runner.moves) == 0
placement_payload = json.loads(placement_record.read_text())
assert placement_payload["execution"]["failure_timing"] == (
    "before_motion_attempt"
)

high_torque_runner = FakeSingleProbeRunner()
high_torque_runner.torque_limit = np.full(6, 0.5)
high_torque_runner.rpc.get_right_joint_torque = lambda: np.ones(6)
post_motion_record = _artifact_root / "post_motion_failure.json"
try:
    execute_single_horizontal_probe(
        high_torque_runner,
        FakeSingleProbeRight(),
        "y",
        0.006,
        hold_window_s=0.0,
        probe_context={"context_id_sha256": "test-context"},
        journal_path=post_motion_record,
    )
    raise AssertionError("unstable high-torque probe was committed")
except RuntimeError as exc:
    assert "did not become stationary" in str(exc)
assert len(high_torque_runner.moves) == 1
assert high_torque_runner.holds == 1
post_motion_payload = json.loads(post_motion_record.read_text())
assert post_motion_payload["execution"]["motion_attempted"] is True
assert post_motion_payload["execution"]["motion_command_completed"] is True
assert post_motion_payload["execution"]["immediate_motion_validated"] is True
assert post_motion_payload["execution"]["settled_motion_validated"] is False
assert post_motion_payload["execution"]["failure_timing"] == (
    "during_or_after_motion_attempt"
)

zero_motion_runner = ZeroMotionRunner()
zero_motion_record = _artifact_root / "zero_motion_failure.json"
try:
    execute_single_horizontal_probe(
        zero_motion_runner,
        FakeSingleProbeRight(),
        "x",
        0.006,
        hold_window_s=0.0,
        probe_context={"context_id_sha256": "test-context"},
        journal_path=zero_motion_record,
    )
    raise AssertionError("zero-motion probe was committed")
except RuntimeError as exc:
    assert "below 0.5 mm" in str(exc)
assert len(zero_motion_runner.moves) == 1
zero_motion_payload = json.loads(zero_motion_record.read_text())
assert zero_motion_payload["status"] == "SINGLE_X_PROBE_FAILED"
assert zero_motion_payload["execution"]["failure_timing"] == (
    "during_or_after_motion_attempt"
)
assert zero_motion_payload["execution"]["motion_command_completed"] is True
assert zero_motion_payload["execution"]["immediate_motion_validated"] is False
assert zero_motion_payload["execution"]["settled_motion_validated"] is False
assert zero_motion_payload["motion"]["actual_immediate_xyz_m"] == [
    0.0,
    0.0,
    0.0,
]
assert "post_motion_state" in zero_motion_payload["motion"]

reverse_motion_runner = ReverseMotionRunner()
reverse_motion_record = _artifact_root / "reverse_motion_failure.json"
try:
    execute_single_horizontal_probe(
        reverse_motion_runner,
        FakeSingleProbeRight(),
        "y",
        -0.006,
        hold_window_s=0.0,
        probe_context={"context_id_sha256": "test-context"},
        journal_path=reverse_motion_record,
    )
    raise AssertionError("reverse-motion probe was committed")
except RuntimeError as exc:
    assert "opposed or diverged from signed request" in str(exc)
reverse_motion_payload = json.loads(reverse_motion_record.read_text())
assert reverse_motion_payload["execution"]["failure_timing"] == (
    "during_or_after_motion_attempt"
)
assert (
    reverse_motion_payload["motion"]["actual_immediate_quality"][
        "direction_cosine"
    ]
    == -1.0
)


def _assert_probe_quality_failure(
    runner,
    record_name,
    expected_message,
    *,
    immediate_validated,
):
    record = _artifact_root / record_name
    try:
        execute_single_horizontal_probe(
            runner,
            FakeSingleProbeRight(),
            "x",
            0.006,
            hold_window_s=0.0,
            probe_context={"context_id_sha256": "test-context"},
            journal_path=record,
        )
        raise AssertionError("unsafe probe was committed")
    except RuntimeError as exc:
        assert expected_message in str(exc)
    assert len(runner.moves) == 1
    assert np.array_equal(runner.moves[0][0], [0.006, 0.0, 0.0])
    payload = json.loads(record.read_text())
    assert payload["status"] == "SINGLE_X_PROBE_FAILED"
    assert "COMMITTED" not in payload["status"]
    assert payload["execution"]["motion_attempted"] is True
    assert payload["execution"]["motion_command_completed"] is True
    assert (
        payload["execution"]["immediate_motion_validated"]
        is immediate_validated
    )
    assert payload["execution"]["settled_motion_validated"] is False
    assert payload["execution"]["failure_timing"] == (
        "during_or_after_motion_attempt"
    )
    return payload


excess_immediate_payload = _assert_probe_quality_failure(
    ExcessImmediateHorizontalRunner(),
    "excess_immediate_horizontal_failure.json",
    "immediate horizontal motion exceeded 10 mm",
    immediate_validated=False,
)
assert (
    excess_immediate_payload["motion"]["actual_immediate_quality"][
        "maximum_horizontal_norm_m"
    ]
    == 0.010
)

vertical_immediate_payload = _assert_probe_quality_failure(
    ImmediateVerticalDriftRunner(),
    "immediate_vertical_drift_failure.json",
    "immediate vertical drift exceeded 0.5 mm",
    immediate_validated=False,
)
assert (
    vertical_immediate_payload["motion"]["actual_immediate_quality"][
        "maximum_abs_vertical_drift_m"
    ]
    == 0.0005
)

orientation_immediate_payload = _assert_probe_quality_failure(
    ImmediateOrientationDriftRunner(),
    "immediate_orientation_drift_failure.json",
    "immediate tool orientation changed",
    immediate_validated=False,
)
assert (
    orientation_immediate_payload["motion"][
        "immediate_orientation_quality"
    ]["maximum_change_deg"]
    == 0.5
)

excess_settled_payload = _assert_probe_quality_failure(
    SettledPoseDriftRunner(translation_xyz=[0.0043, 0.0, 0.0]),
    "excess_settled_horizontal_failure.json",
    "settled horizontal motion exceeded 10 mm",
    immediate_validated=True,
)
assert (
    excess_settled_payload["motion"]["actual_settled_quality"][
        "horizontal_norm_m"
    ]
    > 0.010
)

vertical_settled_payload = _assert_probe_quality_failure(
    SettledPoseDriftRunner(translation_xyz=[0.0, 0.0, 0.00051]),
    "settled_vertical_drift_failure.json",
    "settled vertical drift exceeded 0.5 mm",
    immediate_validated=True,
)
assert (
    abs(
        vertical_settled_payload["motion"]["actual_settled_quality"][
            "vertical_drift_m"
        ]
    )
    > 0.0005
)

orientation_settled_payload = _assert_probe_quality_failure(
    SettledPoseDriftRunner(orientation_deg=0.51),
    "settled_orientation_drift_failure.json",
    "settled tool orientation changed",
    immediate_validated=True,
)
assert (
    orientation_settled_payload["motion"][
        "settled_orientation_quality"
    ]["change_deg"]
    > 0.5
)

try:
    execute_single_horizontal_probe(
        MovingDuringSamRunner(),
        FakeSingleProbeRight(),
        "y",
        0.006,
        hold_window_s=0.0,
        probe_context={"context_id_sha256": "test-context"},
    )
    raise AssertionError("motion during SAM observation was committed")
except RuntimeError as exc:
    assert "while post-probe SAM was running" in str(exc)

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    claims = root / "claims"
    output = root / "run-one"
    claim = claim_motion_execution(
        "approval-001",
        claims,
        output,
        {"axis": "x", "distance_m": 0.006},
    )
    assert output.is_dir()
    claim.set_result({"status": "ok"})
    claim.finalize()
    payload = json.loads(claim.path.read_text())
    assert payload["status"] == "completed"
    assert payload["result"]["status"] == "ok"

    try:
        claim_motion_execution(
            "approval-001",
            claims,
            root / "run-two",
            {"axis": "x", "distance_m": 0.006},
        )
        raise AssertionError("consumed motion token was accepted twice")
    except RuntimeError as exc:
        assert "already consumed" in str(exc)

    failed = claim_motion_execution(
        "approval-002",
        claims,
        root / "run-failed",
        {"axis": "y", "distance_m": 0.006},
    )
    failed.finalize(RuntimeError("camera unavailable"))
    failed_payload = json.loads(failed.path.read_text())
    assert failed_payload["status"] == "failed"
    assert failed_payload["error"]["type"] == "RuntimeError"
    assert failed_payload["failure_timing"] == "before_motion_attempt"

    attempted = claim_motion_execution(
        "approval-003",
        claims,
        root / "run-attempted",
        {"axis": "y", "distance_m": -0.004},
    )
    attempted.mark_motion_attempt()
    attempted.finalize(RuntimeError("post-motion observation failed"))
    attempted_payload = json.loads(attempted.path.read_text())
    assert attempted_payload["motion_attempted"] is True
    assert attempted_payload["failure_timing"] == (
        "during_or_after_motion_attempt"
    )

    cleanup_claim = claim_motion_execution(
        "approval-004",
        claims,
        root / "run-cleanup-failed",
        {"axis": "x", "distance_m": 0.006},
    )
    cleanup_right_error = RuntimeError("right camera stop failed")
    cleanup_runner_error = RuntimeError("runner stop failed")
    cleanup_failure = _cleanup_staged_runtime(
        FakeCleanupTarget(cleanup_runner_error),
        FakeCleanupTarget(cleanup_right_error),
        cleanup_claim,
        None,
    )
    assert cleanup_failure is cleanup_right_error
    cleanup_payload = json.loads(cleanup_claim.path.read_text())
    assert cleanup_payload["status"] == "failed"
    assert cleanup_payload["error"]["message"] == str(cleanup_right_error)
    assert "runner.stop" in "\n".join(
        getattr(cleanup_right_error, "__notes__", ())
    )

    retry_claim = claim_motion_execution(
        "approval-005",
        claims,
        root / "run-finalize-retry",
        {"axis": "y", "distance_m": 0.006},
    )
    original_atomic_write = staged_pregrasp._atomic_write_json

    def fail_atomic_write(path, payload):
        raise OSError("injected claim write failure")

    try:
        staged_pregrasp._atomic_write_json = fail_atomic_write
        try:
            retry_claim.finalize()
            raise AssertionError("claim write failure was swallowed")
        except OSError as exc:
            assert "injected claim write failure" in str(exc)
    finally:
        staged_pregrasp._atomic_write_json = original_atomic_write
    assert retry_claim.finalized is False
    assert retry_claim.payload["status"] == "claimed"
    retry_claim.finalize()
    assert retry_claim.finalized is True
    assert json.loads(retry_claim.path.read_text())["status"] == "completed"

print("staged SAM pregrasp checks passed")

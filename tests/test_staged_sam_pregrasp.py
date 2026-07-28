#!/usr/bin/env python3

import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.local_feature_calibration import load_probe_record
from src.run_staged_sam_pregrasp import (
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
    image_paths = {
        "raw_image": output_dir / f"{prefix}_raw.png",
        "sam_input_png": output_dir / f"{prefix}_sam_input.png",
        "overlay_image": output_dir / f"{sequence:03d}.png",
        "lid_mask": output_dir / f"{prefix}_lid_mask.png",
        "gripper_mask": output_dir / f"{prefix}_gripper_mask.png",
    }
    for path in image_paths.values():
        assert cv2.imwrite(str(path), _artifact_image)
    request_path = output_dir / f"{prefix}_sam_request_q90.jpg"
    assert cv2.imwrite(
        str(request_path),
        _artifact_image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    roi_input_path = output_dir / f"{prefix}_sam_roi_input.png"
    assert cv2.imwrite(str(roi_input_path), _artifact_image)
    roi_request_path = output_dir / f"{prefix}_sam_roi_request_q90.jpg"
    assert cv2.imwrite(
        str(roi_request_path),
        _artifact_image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    depth_path = output_dir / f"{prefix}_depth.npz"
    np.savez_compressed(
        depth_path,
        depth_m=np.ones((240, 320), dtype=np.float32),
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
        "gripper": {"prompt": "blue clamp", "model": "fake-sam"},
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

print("staged SAM pregrasp checks passed")

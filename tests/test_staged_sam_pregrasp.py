#!/usr/bin/env python3

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_staged_sam_pregrasp import (
    bound_horizontal_step,
    claim_motion_execution,
    estimate_horizontal_displacement,
    execute_single_horizontal_probe,
    fit_horizontal_jacobian,
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


class FakeSingleProbeRunner:
    def __init__(self):
        self.moves = []
        self.holds = 0
        self.observations = 0
        self.rpc = SimpleNamespace()
        self.pose = np.array([1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.8])
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
        actual = np.array([0.0058, -0.0001, 0.0])
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
        return feature, error, "/tmp/head.png", timestamp


class FakeSingleProbeRight:
    def __init__(self):
        self.observations = 0

    def observe(self, require_lid):
        assert require_lid is False
        self.observations += 1
        geometry = SimpleNamespace(center_px=np.array([120.0, 90.0]))
        candidate = SimpleNamespace(score=0.8)
        return geometry, candidate, "/tmp/right.png", 124.0


class MovingDuringSamRunner(FakeSingleProbeRunner):
    def observe(self, clearance_m):
        result = super().observe(clearance_m)
        if self.observations == 2:
            self.pose[4] += 0.001
        return result


single_runner = FakeSingleProbeRunner()
single_right = FakeSingleProbeRight()
single_report = execute_single_horizontal_probe(
    single_runner, single_right, "x", 0.006, hold_window_s=0.0
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
assert single_report["quality"]["usable_for_fit"]
assert (
    single_report["observation"]["after"]["head_image"]
    == "/tmp/head.png"
)
assert (
    single_report["observation"]["after"]["right_image"]
    == "/tmp/right.png"
)

try:
    execute_single_horizontal_probe(
        FakeSingleProbeRunner(), FakeSingleProbeRight(), "z", 0.006
    )
    raise AssertionError("non-horizontal single probe was accepted")
except ValueError:
    pass

high_torque_runner = FakeSingleProbeRunner()
high_torque_runner.torque_limit = np.full(6, 0.5)
high_torque_runner.rpc.get_right_joint_torque = lambda: np.ones(6)
try:
    execute_single_horizontal_probe(
        high_torque_runner,
        FakeSingleProbeRight(),
        "y",
        0.006,
        hold_window_s=0.0,
    )
    raise AssertionError("unstable high-torque probe was committed")
except RuntimeError as exc:
    assert "did not become stationary" in str(exc)
assert len(high_torque_runner.moves) == 1
assert high_torque_runner.holds == 1

try:
    execute_single_horizontal_probe(
        MovingDuringSamRunner(),
        FakeSingleProbeRight(),
        "y",
        0.006,
        hold_window_s=0.0,
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

print("staged SAM pregrasp checks passed")

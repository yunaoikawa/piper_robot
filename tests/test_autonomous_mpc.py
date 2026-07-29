#!/usr/bin/env python3
"""Hardware-free checks for the autonomous SAM/RGB-D motion core."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.autonomous_mpc import (
    AtomicRunState,
    AutonomousStop,
    ChunkExecutor,
    ESDFGrid,
    MuJoCoIKValidator,
    Pose,
    ReplanPolicy,
    SceneSnapshot,
    check_pregrasp,
    decide_replan,
    plan_lift_translate_descend,
    validate_calibration,
)


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def snapshot(
    *,
    timestamp=10.0,
    target=(0.40, 0.05, 0.70),
    instance="lid-1",
    ee=(0.30, 0.00, 0.80),
    clearance=0.03,
    marker=(189.7, 296.8),
    distance=0.008,
):
    return SceneSnapshot(
        timestamp,
        target,
        instance,
        Pose(IDENTITY, ee),
        target is not None,
        marker is not None,
        marker,
        distance,
        clearance,
    )


start = Pose(IDENTITY, (0.30, 0.00, 0.75))
plan = plan_lift_translate_descend(
    start,
    (0.40, 0.05, 0.70),
    validator=lambda pose: 0.025,
    now_s=123.0,
)
assert [stage for stage in dict.fromkeys(w.stage for w in plan.waypoints)] == [
    "lift",
    "translate_xy",
    "approach",
    "descend",
]
assert np.allclose(plan.waypoints[-1].pose.xyz, (0.40, 0.05, 0.703))
assert plan.minimum_clearance_m == 0.025
assert all(1 <= len(chunk) <= 16 for chunk in plan.chunks())
assert all(
    later.t_s > earlier.t_s
    for earlier, later in zip(plan.waypoints, plan.waypoints[1:])
)

try:
    plan_lift_translate_descend(start, (0.4, 0.0, 0.7), validator=lambda pose: None)
except AutonomousStop:
    pass
else:
    raise AssertionError("unknown collision space was treated as free")

grid = ESDFGrid(
    np.full((4, 4, 4), 0.050),
    (0.0, 0.0, 0.0),
    0.1,
    body_radius_m=0.020,
)
assert abs(grid.clearance(Pose(IDENTITY, (0.15, 0.15, 0.15))) - 0.030) < 1e-6
assert grid.clearance(Pose(IDENTITY, (-0.1, 0.0, 0.0))) is None

model_validator = MuJoCoIKValidator(
    "robot/cone-e-description/robot-welded-base-and-lift.mjcf",
    [0.0268955231, 1.8055955172, -0.6696653962, 0.0435110591, -1.1529295444, 2.3609592915],
    left_q=[0.0, 1.58, -0.58, 0.0, -0.91, 1.40],
)
measured_model_pose = Pose.from_se3(model_validator.solver.forward_kinematics())
assert np.allclose(
    model_validator.validate(measured_model_pose),
    model_validator.previous_q,
)

reference = snapshot()
assert (
    decide_replan(
        now_s=10.5,
        reference=reference,
        current=snapshot(),
        commanded_pose=Pose(IDENTITY, (0.30, 0.00, 0.80)),
    ).action
    == "CONTINUE"
)
shifted = snapshot(target=(0.411, 0.05, 0.70))
decision = decide_replan(
    now_s=10.5,
    reference=reference,
    current=shifted,
    commanded_pose=shifted.ee_pose,
)
assert decision.action == "REPLAN" and "target_shift" in decision.reasons

stale = snapshot(timestamp=8.0)
assert (
    decide_replan(
        now_s=10.0,
        reference=reference,
        current=stale,
        commanded_pose=stale.ee_pose,
    ).action
    == "HOLD"
)

tracking_error = snapshot(ee=(0.306, 0.00, 0.80))
assert "trajectory_position_error" in decide_replan(
    now_s=10.5,
    reference=reference,
    current=tracking_error,
    commanded_pose=Pose(IDENTITY, (0.30, 0.00, 0.80)),
).reasons

assert check_pregrasp(reference, goal_px=(189.7, 296.8)).allowed
bad_gate = check_pregrasp(
    snapshot(marker=(210.0, 296.8), distance=0.012),
    goal_px=(189.7, 296.8),
)
assert not bad_gate.allowed
assert set(bad_gate.reasons) == {
    "right_goal_too_far",
    "gripper_lid_distance_too_large_or_unknown",
}

calibration = {
    "accepted": True,
    "accepted_at_s": 100.0,
    "record3d_udid": "head",
    "T_robot_camera": np.eye(4).tolist(),
}
assert np.allclose(
    validate_calibration(
        calibration,
        camera_udid="head",
        maximum_age_s=10.0,
        now_s=105.0,
    ),
    np.eye(4),
)
for broken in (
    {**calibration, "accepted": False},
    {**calibration, "record3d_udid": "wrong"},
    {**calibration, "T_robot_camera": [[1.0]]},
):
    try:
        validate_calibration(broken, camera_udid="head")
    except AutonomousStop:
        pass
    else:
        raise AssertionError("invalid calibration was accepted")


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class FakeRPC:
    def __init__(self, torques):
        self.torques = [np.asarray(value, dtype=float) for value in torques]
        self.commands = []
        self.holds = 0
        self.pose = mink.SE3(np.array([1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.8]))

    def get_right_joint_torque(self):
        return self.torques.pop(0) if self.torques else np.zeros(6)

    def set_right_ee_target(self, **kwargs):
        self.commands.append(kwargs)

    def get_right_ee_pose(self):
        self.holds += 1
        return self.pose


clock = FakeClock()
rpc = FakeRPC([np.zeros(6)] * 10)
executor = ChunkExecutor(
    rpc,
    torque_limit_nm=np.ones(6),
    clock=clock,
    sleep=clock.sleep,
)
executor.execute(plan.waypoints[:3])
assert len(rpc.commands) == 3
assert all(command["preview_time"] == 0.05 for command in rpc.commands)
assert abs(clock.value - 0.1) < 1e-9

high_torque = FakeRPC([np.ones(6) * 2.0, np.ones(6) * 2.0])
executor = ChunkExecutor(
    high_torque,
    torque_limit_nm=np.ones(6),
    clock=clock,
    sleep=clock.sleep,
)
try:
    executor.execute(plan.waypoints[:3])
except AutonomousStop:
    pass
else:
    raise AssertionError("sustained high torque did not stop execution")
assert high_torque.holds == 1
assert len(high_torque.commands) == 2  # first target, then measured-pose hold

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "run_state.json"
    state = AtomicRunState(path)
    state.event("perception", target="lid-1")
    state.update("PLANNED", plan_id="plan-1")
    resumed = AtomicRunState(path, resume=True)
    assert resumed.payload["status"] == "PLANNED"
    assert resumed.payload["events"][0]["target"] == "lid-1"
    json.loads(path.read_text())

print("autonomous MPC checks passed")

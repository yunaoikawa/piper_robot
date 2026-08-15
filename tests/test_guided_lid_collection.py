from pathlib import Path

import h5py
import numpy as np
import pytest

from rollout.guided_lid_collection import (
    BaselineTrajectory,
    CollectionPhase,
    GuidedLidCycle,
    build_grasp_prefix,
    build_reposition_commands,
    rebase_post_grasp,
)


def _baseline_hdf5(path: Path, *, task: str) -> Path:
    count = 90
    timestamps = np.arange(count, dtype=float) / 30.0
    xyz = np.zeros((count, 3), dtype=float)
    xyz[:, 0] = 0.20
    xyz[:, 1] = -0.01
    xyz[:, 2] = 0.80
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
    commanded = np.full((count, 16), np.nan)
    commanded[:, 7:14] = np.concatenate([quat, xyz], axis=1)
    commanded[:, 15] = 1.0
    if task == "lid_open":
        commanded[20:, 15] = 0.0
        xyz[30:, 2] += np.linspace(0.0, 0.04, count - 30)
        xyz[45:, 1] -= np.linspace(0.0, 0.06, count - 45)
    else:
        # A transient close/open must not be selected as the successful grasp.
        commanded[10:14, 15] = 0.0
        commanded[40:75, 15] = 0.0
        commanded[75:, 15] = 1.0
        xyz[48:, 2] += np.linspace(0.0, 0.035, count - 48)
        xyz[60:, 1] -= np.linspace(0.0, 0.05, count - 60)
    commanded[:, 7:14] = np.concatenate([quat, xyz], axis=1)
    with h5py.File(path, "w") as output:
        output["active_timestamps"] = timestamps
        output["right_ee_pos"] = xyz
        output["right_ee_quat"] = quat
        output["commanded_target_quat16"] = commanded
    return path


@pytest.mark.parametrize("task,expected_grasp", [("lid_open", 20), ("lid_close", 40)])
def test_loads_final_successful_close_and_complete_transport(tmp_path, task, expected_grasp):
    baseline = BaselineTrajectory.load(
        _baseline_hdf5(tmp_path / f"{task}.hdf5", task=task), task=task
    )
    assert baseline.source_grasp_index == expected_grasp
    assert baseline.source_review_index > expected_grasp
    assert len(baseline.post_review) > 2
    if task == "lid_open":
        assert baseline.post_review[-1].stage == "release"
        assert baseline.post_review[-1].gripper_open_ratio == pytest.approx(1.0)


def test_grasp_prefix_has_one_descent_with_fixed_xy_and_requested_speed(tmp_path):
    baseline = BaselineTrajectory.load(
        _baseline_hdf5(tmp_path / "open.hdf5", task="lid_open"),
        task="lid_open",
    )
    start = np.array([1, 0, 0, 0, 0.30, 0.04, 0.83], dtype=float)
    commands = build_grasp_prefix(
        start,
        baseline,
        [0.02, -0.01, -0.005],
        hover_clearance_m=0.03,
        transit_speed_m_s=0.05,
        descent_speed_m_s=0.10,
        close_duration_s=1.2,
        verification_lift_m=0.02,
    )
    stages = [command.stage for command in commands]
    descent = [command for command in commands if command.stage == "continuous_descent"]
    assert descent
    assert max(np.ptp(np.asarray([c.pose()[4:6] for c in descent]), axis=0)) < 1e-12
    assert descent[-1].t_s - descent[0].t_s <= 0.03 / 0.10 + 1 / 30
    assert stages.index("continuous_descent") < stages.index("close")
    assert stages.index("clearance") < stages.index("level_at_clearance")
    assert stages.index("level_at_clearance") < stages.index("hover_transit")
    assert stages.index("close") < stages.index("vertical_verification_lift")
    assert not any(
        command.stage == "continuous_descent" and command.gripper_open_ratio < 0.999
        for command in commands
    )


def test_post_grasp_rebase_preserves_relative_transport(tmp_path):
    baseline = BaselineTrajectory.load(
        _baseline_hdf5(tmp_path / "open.hdf5", task="lid_open"),
        task="lid_open",
    )
    measured = np.asarray(baseline.review_pose_wxyz_xyz).copy()
    measured[4:7] += [0.03, -0.02, 0.01]
    commands = rebase_post_grasp(baseline, measured)
    source_delta = (
        baseline.post_review[-1].pose()[4:7]
        - np.asarray(baseline.review_pose_wxyz_xyz)[4:7]
    )
    actual_delta = commands[-1].pose()[4:7] - measured[4:7]
    assert actual_delta == pytest.approx(source_delta)


def test_cycle_uses_physical_right_sweep_and_keeps_successful_depth():
    cycle = GuidedLidCycle()
    cycle.adjust("x", 20)
    cycle.adjust("z", -5)
    cycle.start_attempt()
    assert cycle.active_attempt_correction_m.tolist() == pytest.approx([0.02, 0, -0.005])
    cycle.enter_review(); cycle.succeed(); cycle.task_complete()
    assert cycle.task == "lid_close"
    cycle.start_attempt(); cycle.enter_review(); cycle.succeed()
    assert cycle.task_complete() == "reposition"
    first_delta = cycle.next_placement_robot_xyz_m - cycle.placement_robot_xyz_m
    assert first_delta.tolist() == pytest.approx([0.0, -0.01, 0.0])
    cycle.reposition_complete()
    assert cycle.task == "lid_open"
    assert cycle.placement_right_mm == 10
    assert cycle.attempt_correction_m[2] == pytest.approx(-0.005)
    assert cycle.attempt_correction_m[0] == pytest.approx(0.02)


def test_review_adjustment_applies_only_to_next_attempt():
    cycle = GuidedLidCycle()
    cycle.adjust("z", -5)
    cycle.start_attempt(); cycle.enter_review()
    cycle.adjust("z", -5)
    cycle.succeed()
    assert cycle.successful_correction_by_task_m["lid_open"][2] == pytest.approx(-0.005)
    assert cycle.attempt_correction_m[2] == pytest.approx(-0.005)


def test_failure_retains_next_correction_but_does_not_promote_it():
    cycle = GuidedLidCycle()
    cycle.start_attempt(); cycle.enter_review()
    cycle.adjust("y", -10)
    cycle.fail()
    assert cycle.attempt_correction_m.tolist() == pytest.approx([0, -0.01, 0])
    assert "lid_open" not in cycle.successful_correction_by_task_m
    assert cycle.phase == CollectionPhase.READY


def test_reposition_moves_only_in_plane_between_vertical_segments():
    pose = [1, 0, 0, 0, 0.2, 0.0, 0.8]
    current = [1, 0, 0, 0, 0.3, 0.04, 0.85]
    commands = build_reposition_commands(current, pose, [0, -0.01, 0])
    planar = [c for c in commands if c.stage == "reposition_planar"]
    assert planar
    assert planar[-1].pose()[5] == pytest.approx(-0.01)
    assert all(c.pose()[6] == pytest.approx(0.82) for c in planar)
    release = [c for c in commands if c.stage == "reposition_release"]
    assert release[-1].gripper_open_ratio == pytest.approx(1.0)


def test_sweep_is_bounded_and_reverses_without_drift():
    cycle = GuidedLidCycle()
    seen = []
    for _ in range(len(cycle.sweep_right_mm) * 2):
        seen.append(cycle.placement_right_mm)
        cycle.phase = CollectionPhase.REPOSITIONING
        cycle.reposition_complete()
    assert max(seen) == 30
    assert min(seen) == -30
    assert seen[: len(cycle.sweep_right_mm)] == list(cycle.sweep_right_mm)
    assert seen[: len(cycle.sweep_right_mm)] == seen[len(cycle.sweep_right_mm) :]

from pathlib import Path
import json
import tempfile

import h5py
import mink
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rollout.dish_transport_rehearsal import (
    CartesianAirTransportStreamer,
    TransportEpisode,
    TransportPlan,
    choose_route_medoid,
    load_transport_episode,
    sample_pose_path_at_speed,
    split_checkpoint_chunks,
)
from src.dish_transport_rehearsal_ui import CheckpointApprovalStore
from src.run_dish_transport_rehearsal import _validate_scene_layout_contract
from rollout.gripper_level import JawLevelReference


def _write_episode(path: Path, gripper):
    count = len(gripper)
    positions = np.c_[
        np.linspace(0.30, 0.12, count),
        np.linspace(0.05, 0.25, count),
        0.85 + 0.08 * np.sin(np.linspace(0.0, np.pi, count)),
    ]
    with h5py.File(path, "w") as recording:
        recording.create_dataset("right_ee_pos", data=positions)
        recording.create_dataset(
            "right_ee_quat", data=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
        )
        recording.create_dataset("right_gripper", data=np.asarray(gripper))
        recording.create_dataset(
            "right_joint_positions", data=np.zeros((count, 6), dtype=float)
        )


def test_episode_requires_exact_open_close_open():
    with tempfile.TemporaryDirectory() as directory:
        clean = Path(directory) / "clean.hdf5"
        _write_episode(clean, [1, 1, 0, 0, 0, 1, 1])
        episode = load_transport_episode(
            clean, source_lift_m=0.05, arrival_hover_m=0.03
        )
        assert episode.close_frame == 2
        assert episode.release_frame == 5
        assert episode.arrival_hover_frame == 5
        noisy = Path(directory) / "noisy.hdf5"
        _write_episode(noisy, [1, 0, 1, 0, 1])
        with pytest.raises(ValueError, match="exactly close then release"):
            load_transport_episode(
                noisy, source_lift_m=0.05, arrival_hover_m=0.03
            )


def test_round_trip_destination_is_turnaround_not_release():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "round_trip.hdf5"
        gripper = [1, 1, 0, 0, 0, 0, 0, 1]
        positions = np.asarray(
            [
                [0.30, 0.05, 0.85],
                [0.30, 0.05, 0.85],
                [0.30, 0.05, 0.85],
                [0.20, 0.10, 0.86],
                [0.10, 0.20, 0.87],  # named station / turnaround
                [0.20, 0.10, 0.86],
                [0.29, 0.05, 0.85],
                [0.30, 0.05, 0.85],
            ]
        )
        with h5py.File(path, "w") as recording:
            recording.create_dataset("right_ee_pos", data=positions)
            recording.create_dataset(
                "right_ee_quat", data=np.tile([1.0, 0.0, 0.0, 0.0], (8, 1))
            )
            recording.create_dataset("right_gripper", data=np.asarray(gripper))
            recording.create_dataset(
                "right_joint_positions", data=np.zeros((8, 6), dtype=float)
            )
        episode = load_transport_episode(
            path, source_lift_m=0.05, arrival_hover_m=0.03
        )
        assert episode.close_frame == 2
        assert episode.release_frame == 7
        assert episode.arrival_hover_frame == 4
        assert len(episode.transport_poses) == 3
        np.testing.assert_allclose(episode.transport_poses[-1, 4:], positions[4])


def test_scene_layout_contract_rejects_mirrored_microscope(tmp_path):
    report = {
        "objects": [
            {
                "semantic_name": "microscope",
                "geometry": {"center_xyz_m": [0.8, 1.2, 0.0]},
            }
        ],
        "robot_placement": {
            "base_xyz_level_m": {
                "left/base_link": [-0.5, 0.5, 0.0],
                "right/base_link": [0.5, 0.5, 0.0],
            }
        },
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="mirrored/stale"):
        _validate_scene_layout_contract(
            {
                "semantic_scene_report": str(path),
                "scene_layout_contract": {
                    "station_in_front_of_arm": {"microscope": "left"}
                },
            }
        )


def _episode(name: str, lateral: float) -> TransportEpisode:
    positions = np.asarray(
        [
            [0.0, 0.0, 0.9],
            [0.1, lateral, 1.0],
            [0.2, 0.0, 0.93],
        ]
    )
    return TransportEpisode(
        path=Path(name),
        close_frame=0,
        release_frame=2,
        source_lift_frame=0,
        arrival_hover_frame=2,
        positions_m=positions,
        quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        right_joint_positions_rad=None,
    )


def test_route_medoid_rejects_shape_outlier():
    episodes = (_episode("center", 0.01), _episode("near", 0.012), _episode("far", 0.20))
    assert choose_route_medoid(episodes, comparison_samples=31) == 1


def test_pose_sampling_hits_endpoint_and_speed_duration():
    poses = np.asarray(
        [
            [1, 0, 0, 0, 0.0, 0.0, 0.8],
            [1, 0, 0, 0, 0.1, 0.0, 0.8],
        ],
        dtype=float,
    )
    samples = sample_pose_path_at_speed(poses, speed_m_s=0.05, control_hz=30.0)
    assert np.allclose(samples[-1].pose_wxyz_xyz, poses[-1])
    assert samples[-1].t_s == pytest.approx(2.0)
    assert len(samples) == 60


def test_checkpoint_chunks_overlap_only_at_stops():
    poses = np.tile([1, 0, 0, 0, 0, 0, 1], (10, 1)).astype(float)
    poses[:, 4] = np.arange(10)
    plan = TransportPlan(
        name="x",
        source="a",
        destination="b",
        physical_arm="right",
        medoid_hdf5="demo.hdf5",
        medoid_sha256="abc",
        coordinate_retarget="recorded",
        poses_wxyz_xyz=poses,
        q_physical_rad=np.zeros((10, 6)),
        checkpoint_indices=(2, 5, 7),
        checkpoint_names=("source_lifted", "route_midpoint", "arrival_hover"),
        maximum_planned_tilt_deg=0.0,
        collision_audit={"accepted": True},
    )
    chunks = split_checkpoint_chunks(plan)
    assert [len(value) for value in chunks] == [3, 4, 3, 3]
    assert np.allclose(chunks[0][-1], chunks[1][0])
    assert np.allclose(chunks[1][-1], chunks[2][0])
    assert np.allclose(chunks[2][-1], chunks[3][0])


def test_five_checkpoint_chunks_cover_route_and_return():
    poses = np.tile([1, 0, 0, 0, 0, 0, 1], (21, 1)).astype(float)
    poses[:, 4] = np.arange(21)
    plan = TransportPlan(
        name="x",
        source="a",
        destination="b",
        physical_arm="right",
        medoid_hdf5="demo.hdf5",
        medoid_sha256="abc",
        coordinate_retarget="recorded",
        poses_wxyz_xyz=poses,
        q_physical_rad=np.zeros((21, 6)),
        checkpoint_indices=(4, 7, 10, 13, 16),
        checkpoint_names=("departure", "quarter", "midpoint", "three_quarters", "arrival"),
        maximum_planned_tilt_deg=0.0,
        collision_audit={"accepted": True},
    )
    chunks = split_checkpoint_chunks(plan)
    assert [len(value) for value in chunks] == [5, 4, 4, 4, 4, 5]
    assert all(np.allclose(first[-1], second[0]) for first, second in zip(chunks, chunks[1:]))


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += max(0.0, float(duration))


class _RPC:
    def __init__(self):
        self.pose = mink.SE3(np.asarray([1, 0, 0, 0, 0, 0, 0.8], dtype=float))
        self.left_commands = []

    def set_left_ee_target(self, pose, gripper_target, preview_time):
        self.pose = pose
        self.left_commands.append((pose, gripper_target, preview_time))
        return True

    def get_left_ee_pose(self):
        return self.pose

    def get_left_gripper_exact(self):
        return 1.0

    def get_left_joint_torque(self):
        return np.zeros(6)


def test_streamer_uses_physical_left_teleop_rpc_path():
    rpc = _RPC()
    clock = _Clock()
    streamer = CartesianAirTransportStreamer(
        rpc,
        "left",
        torque_limit_nm=np.ones(6),
        tracking_interval=1,
        final_settle_s=0.2,
        clock=clock,
        sleep=clock.sleep,
    )
    poses = np.asarray(
        [
            [1, 0, 0, 0, 0.0, 0.0, 0.8],
            [1, 0, 0, 0, 0.02, 0.0, 0.8],
        ],
        dtype=float,
    )
    report = streamer.execute(
        poses, speed_m_s=0.04, gripper_open_ratio=1.0, stage="test"
    )
    assert report["command_path"] == "set_left_ee_target"
    assert len(rpc.left_commands) == (
        report["sample_count"] + report["endpoint_settle_command_count"] + 1
    )  # samples, endpoint settle, final measured hold
    assert all(command[1] == 1.0 for command in rpc.left_commands)


def test_bounded_level_refinement_keeps_measured_xyz():
    rpc = _RPC()
    clock = _Clock()
    level = Rotation.from_euler("x", 90, degrees=True)
    tilted = Rotation.from_euler("y", 4, degrees=True) * level
    xyzw = tilted.as_quat()
    xyz = np.asarray([0.21, -0.12, 0.91])
    rpc.pose = mink.SE3(np.r_[xyzw[[3, 0, 1, 2]], xyz])
    streamer = CartesianAirTransportStreamer(
        rpc,
        "left",
        torque_limit_nm=np.ones(6),
        final_settle_s=0.0,
        clock=clock,
        sleep=clock.sleep,
    )
    report = streamer.refine_jaw_level(
        JawLevelReference(),
        gripper_open_ratio=1.0,
        maximum_attempts=2,
        correction_duration_s=0.2,
        settle_s=0.2,
    )
    assert report["accepted"] is True
    assert report["attempts_used"] == 1
    assert np.allclose(report["fixed_xyz_m"], xyz)
    assert np.allclose(report["final_pose_wxyz_xyz"][4:], xyz)


def test_ui_rejects_continue_when_level_gate_failed():
    store = CheckpointApprovalStore()
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    revision = store.publish(
        segment="1/3",
        checkpoint="1/3",
        physical_arm="right",
        metrics={},
        head_bgr=image,
        left_bgr=image,
        right_bgr=image,
        continue_allowed=False,
    )
    with pytest.raises(ValueError, match="水平姿勢ゲート"):
        store.decide(revision, "continue")
    assert store.decide(revision, "abort_hold")["decision"] == "abort_hold"

import json
from dataclasses import replace
import threading
import urllib.request

import h5py
import numpy as np
import pytest

from rollout.agent_collection import (
    AgentEpisodeRecorder, AgentRecordingSample, ControllerClaim,
    GripperCloseLatch, InterventionState, gripper_latch_config,
    intervention_slice_mask,
    summarize_sampling_timing,
)
from src.convert_to_lerobot import find_episode_pairs, load_episode
from rollout.agent_collection_ui import AgentCollectionUI


class Rotation:
    wxyz = np.array([1.0, 0.0, 0.0, 0.0])


class Pose:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=float)

    def translation(self):
        return self.xyz

    def rotation(self):
        return Rotation()


def sample(index, revision=0):
    pose = Pose([index / 1000, 0.0, 0.5])
    action = np.full(16, np.nan)
    action[7:14] = [1, 0, 0, 0, index / 1000, 0, .5]
    action[15] = .4
    return AgentRecordingSample(
        wall_timestamp=100 + index / 30,
        active_timestamp=index / 30,
        left_ee_pose=Pose([0, 0, .7]), right_ee_pose=pose,
        left_gripper_exact=1, right_gripper_exact=.4,
        left_gripper=1, right_gripper=0,
        left_joint_positions=np.zeros(6), right_joint_positions=np.ones(6),
        head_rgb=None, left_rgb=None, right_rgb=None,
        camera_timestamps=(1, 2, 3), camera_frame_ids=(10, 20, 30),
        policy_action_quat16=action.copy(), commanded_target_quat16=action.copy(),
        xyz_bias_left_right=np.array([0, 0, 0, 0, 0, -.03]),
        chunk_index=index, action_generation=2,
        action_observation_timestamp=99, intervention_revision=revision,
        safety_rejected_count=0,
    )


def test_bias_mailbox_and_slice():
    state = InterventionState()
    state.set_bias("right", np.array([.001, -.002, -.03]))
    assert state.snapshot()["bias"]["right"] == [.001, -.002, -.03]
    with pytest.raises(ValueError):
        state.set_bias("right", np.array([0, 0, -.061]))
    revisions = np.array([0, 0, 1, 1])
    assert intervention_slice_mask(revisions, "all").tolist() == [True] * 4
    assert intervention_slice_mask(revisions, "post-intervention").tolist() == [False, False, True, True]


def test_sampling_timing_rejects_synthetic_30hz_claim_for_10hz_data():
    timing = summarize_sampling_timing(np.arange(419) * 0.0965, target_hz=30)
    assert timing["effective_hz"] == pytest.approx(10.3627, rel=1e-3)
    assert timing["interval_deadline_misses"] == 418
    assert timing["eligible"] is False


def test_episode_active_clock_excludes_pause_and_inference_wait(tmp_path):
    recorder = AgentEpisodeRecorder(tmp_path / "agent", threading.Event())
    recorder.set_sampling_active(True, now=10.0)
    assert recorder.active_timestamp(now=10.1) == pytest.approx(.1)
    recorder.set_sampling_active(False, now=10.2)
    # Arbitrarily long pause / fresh-inference wait does not advance the clock.
    assert recorder.active_timestamp(now=50.0) == pytest.approx(.2)
    recorder.set_sampling_active(True, now=50.0)
    assert recorder.active_timestamp(now=50.1) == pytest.approx(.3)
    recorder.set_sampling_active(False, now=50.1)
    recorder.stop()


def test_gripper_latch_blocks_open_until_measured_transport_then_releases():
    latch = GripperCloseLatch(
        minimum_transport_distance_m=.03,
        minimum_release_command_time_s=.15,
    )
    grasp_xyz = np.array([.20, -.10, .70])
    steps = (
        (1.0, 1.0, 0.0, grasp_xyz),
        (.3, .7, 1.0, grasp_xyz),
        (.2, .61, 1.7, grasp_xyz),
        (.15, .605, 1.9, grasp_xyz),
        (.1, .60, 2.1, grasp_xyz),
        # A replan's temporary open beside the grasp is suppressed.
        (.9, .60, 2.2, grasp_xyz + [.029, 0, 0]),
        # Crossing the task-independent Cartesian transport gate enables, but
        # does not immediately execute, a one-frame open prediction.
        (1.0, .60, 2.3, grasp_xyz + [.031, 0, 0]),
        (1.0, .60, 2.46, grasp_xyz + [.031, 0, 0]),
        # Completion is measured, not inferred from sending the open command.
        (.1, .80, 2.50, grasp_xyz + [.031, 0, 0]),
    )
    output = [
        latch.apply(command, measured, measured_xyz=xyz, now=now)[0]
        for command, measured, now, xyz in steps
    ]
    assert output == pytest.approx([1.0, .3, .2, .15, .1, .1, .1, 1.0, 1.0])
    assert latch.release_enabled is True
    assert latch.released is True
    assert latch.latched is False
    assert latch.grasp_xyz_m == pytest.approx(grasp_xyz)
    assert latch.transport_displacement_m == pytest.approx(.031)


def test_gripper_latch_rejects_transient_open_even_after_transport():
    latch = GripperCloseLatch(
        minimum_transport_distance_m=.02,
        minimum_release_command_time_s=.15,
    )
    grasp_xyz = np.array([0.0, 0.0, .5])
    latch.apply(1.0, 1.0, measured_xyz=grasp_xyz, now=0)
    latch.apply(.2, .6, measured_xyz=grasp_xyz, now=1)
    latch.apply(.1, .6, measured_xyz=grasp_xyz, now=1.9)
    latch.apply(.1, .6, measured_xyz=grasp_xyz, now=2.1)
    assert latch.latched

    transported = grasp_xyz + [0, -.03, .02]
    assert latch.apply(1.0, .6, measured_xyz=transported, now=2.2)[0] == .1
    # A close command resets the persistence timer.
    assert latch.apply(.1, .6, measured_xyz=transported, now=2.3)[0] == .1
    assert latch.apply(1.0, .6, measured_xyz=transported, now=2.4)[0] == .1
    assert latch.apply(1.0, .6, measured_xyz=transported, now=2.56)[0] == 1.0
    assert latch.release_commanded
    assert not latch.released
    # Even if a later policy chunk asks to close, the terminal release remains
    # open until the measured jaws confirm it.
    assert latch.apply(.1, .8, measured_xyz=transported, now=2.6)[0] == 1.0
    assert latch.released


def test_gripper_latch_fails_closed_without_valid_measured_pose():
    latch = GripperCloseLatch(minimum_release_command_time_s=0)
    latch.apply(1.0, 1.0, now=0)
    latch.apply(.2, .6, now=1)
    latch.apply(.1, .6, now=1.9)
    latch.apply(.1, .6, now=2.1)
    assert latch.latched
    assert latch.grasp_xyz_m is None
    assert latch.apply(1.0, .6, measured_xyz=[np.nan, 0, 0], now=3)[0] == .1
    assert latch.release_enabled is False
    assert latch.released is False


def test_gripper_latch_recovers_when_pose_arrives_after_grasp_confirmation():
    latch = GripperCloseLatch(
        minimum_transport_distance_m=.02,
        minimum_release_command_time_s=0,
    )
    latch.apply(1.0, 1.0, now=0)
    latch.apply(.2, .6, now=1)
    latch.apply(.1, .6, now=1.9)
    latch.apply(.1, .6, now=2.1)
    assert latch.latched and latch.grasp_xyz_m is None

    late_grasp_pose = np.array([.1, .2, .7])
    assert latch.apply(.1, .6, measured_xyz=late_grasp_pose, now=2.2)[0] == .1
    assert latch.grasp_xyz_m == pytest.approx(late_grasp_pose)
    moved = late_grasp_pose + [0, 0, .021]
    assert latch.apply(1.0, .6, measured_xyz=moved, now=2.3)[0] == 1.0
    assert latch.release_enabled and latch.release_commanded


def test_gripper_latch_detects_sustained_opening_loss_during_transport():
    latch = GripperCloseLatch(
        minimum_transport_distance_m=.05,
        drop_monitor_transport_distance_m=.01,
        drop_opening_decrease=.20,
        drop_confirmation_time_s=.20,
    )
    grasp_xyz = np.array([.1, .2, .7])
    latch.apply(1.0, 1.0, measured_xyz=grasp_xyz, now=0)
    latch.apply(.1, .66, measured_xyz=grasp_xyz, now=1)
    latch.apply(.1, .66, measured_xyz=grasp_xyz, now=1.9)
    latch.apply(.1, .66, measured_xyz=grasp_xyz, now=2.1)
    assert latch.latched
    assert latch.grasp_measured_opening == pytest.approx(.66)

    moved = grasp_xyz + [0, .012, 0]
    # A 0.13 decrease, like the latest real run, remains holding evidence and
    # is not misclassified as an empty-gripper slip.
    latch.apply(.1, .53, measured_xyz=moved, now=2.2)
    latch.apply(.1, .53, measured_xyz=moved, now=2.5)
    assert not latch.dropped
    # A larger loss must persist; a single sample cannot stop the cycle.
    latch.apply(.1, .43, measured_xyz=moved, now=2.6)
    assert not latch.dropped
    latch.apply(.1, .55, measured_xyz=moved, now=2.7)
    assert not latch.dropped
    latch.apply(.1, .43, measured_xyz=moved, now=2.8)
    latch.apply(.1, .43, measured_xyz=moved, now=3.01)
    assert latch.dropped
    assert not latch.released
    assert not latch.release_commanded


def test_gripper_drop_monitor_is_disabled_before_transport_and_final_release():
    latch = GripperCloseLatch(
        minimum_transport_distance_m=.03,
        drop_monitor_transport_distance_m=.01,
        drop_opening_decrease=.1,
        drop_confirmation_time_s=0,
        minimum_release_command_time_s=0,
    )
    grasp_xyz = np.array([0, 0, .5])
    latch.apply(1.0, 1.0, measured_xyz=grasp_xyz, now=0)
    latch.apply(.1, .6, measured_xyz=grasp_xyz, now=1)
    latch.apply(.1, .6, measured_xyz=grasp_xyz, now=1.9)
    latch.apply(.1, .6, measured_xyz=grasp_xyz, now=2.1)
    latch.apply(.1, .2, measured_xyz=grasp_xyz, now=2.2)
    assert not latch.dropped

    released_xyz = grasp_xyz + [.031, 0, 0]
    latch.apply(1.0, .6, measured_xyz=released_xyz, now=2.3)
    assert latch.release_commanded
    latch.apply(1.0, .2, measured_xyz=released_xyz, now=2.4)
    assert not latch.dropped


def test_gripper_latch_config_applies_demo_gate_per_task():
    profile = {
        "intervention": {"gripper_transport_release": {
            "minimum_transport_distance_m": .055,
            "drop_opening_decrease": .2,
        }},
        "tasks": {
            "lid_open": {"gripper_transport_release": {
                "minimum_transport_distance_m": .0544,
            }},
            "lid_close": {"gripper_transport_release": {
                "minimum_transport_distance_m": .0476,
            }},
        },
    }
    assert gripper_latch_config(profile, "lid_open")[
        "minimum_transport_distance_m"
    ] == pytest.approx(.0544)
    assert gripper_latch_config(profile, "lid_close")[
        "minimum_transport_distance_m"
    ] == pytest.approx(.0476)


def test_gripper_latch_does_not_hold_an_empty_fully_closed_gripper():
    latch = GripperCloseLatch()
    assert latch.apply(.2, .2, now=0) == pytest.approx((.2, False))
    assert latch.apply(.9, .9, now=1) == pytest.approx((.9, False))
    assert latch.apply(.2, .7, now=2) == pytest.approx((.2, False))
    assert latch.apply(.1, .04, now=3.1) == pytest.approx((.1, False))
    assert latch.apply(1.0, .01, now=3.2) == pytest.approx((1.0, False))
    assert latch.latched is False
    latch.reset()
    assert latch.apply(.9, .9, now=4) == pytest.approx((.9, False))


def test_controller_claim_is_exclusive(tmp_path):
    first, second = ControllerClaim(tmp_path / "controller.lock"), ControllerClaim(tmp_path / "controller.lock")
    first.acquire()
    with pytest.raises(RuntimeError):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


def test_success_promotion_and_converter_slice(tmp_path):
    stop = threading.Event()
    recorder = AgentEpisodeRecorder(tmp_path / "agent", stop)
    recorder.configure_episode({
        "task": "lid_open", "target_selection": {"u": .5, "v": .5},
        "initial_bias_m": {"left": [0, 0, 0], "right": [0, 0, -.03]},
    })
    recorder.start_episode()
    for index in range(5):
        recorder.record_sample(sample(index, revision=0 if index < 2 else 1))
    recorder.end_episode()
    destination = recorder.finalize("success")
    recorder.stop()

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["training_eligible"] is True
    assert manifest["sample_count"] == 5
    assert "success/lid_open" in str(destination)
    pairs = find_episode_pairs(str(tmp_path / "agent"), [], require_success_manifest=True)
    assert len(pairs) == 1
    episode = load_episode(pairs[0][0], intervention_slice="post-intervention")
    assert len(episode["state"]) == 3
    assert episode["video_frame_start"] == 2
    # Teacher is the actual next measured pose, never the policy proposal.
    assert episode["action"][0, 10] == pytest.approx(.003)


def test_converter_can_match_existing_right_only_act_checkpoint(tmp_path):
    stop = threading.Event()
    recorder = AgentEpisodeRecorder(tmp_path / "agent", stop)
    recorder.configure_episode({
        "task": "lid_open", "target_selection": {"u": .5, "v": .5},
        "initial_bias_m": {"left": [0, 0, 0], "right": [0, 0, -.03]},
    })
    recorder.start_episode()
    recorder.record_sample(sample(0))
    recorder.record_sample(sample(1))
    recorder.end_episode()
    destination = recorder.finalize("success")
    recorder.stop()
    episode = load_episode(
        str(next(destination.glob("*.hdf5"))), active_arm="right"
    )
    assert episode["state"].shape == (2, 10)
    assert episode["action"].shape == (2, 10)
    assert episode["action"][0, 0] == pytest.approx(.001)


def test_missing_required_right_camera_quarantines_nominal_success(tmp_path):
    recorder = AgentEpisodeRecorder(tmp_path / "agent", threading.Event())
    recorder.configure_episode({
        "task": "lid_open", "target_selection": {"source": "test"},
        "initial_bias_m": {"left": [0, 0, 0], "right": [0, 0, 0]},
        "required_cameras": ["head", "right"],
    })
    recorder.start_episode()
    for index in range(3):
        recorder.record_sample(replace(
            sample(index), camera_timestamps=(1.0, 2.0, np.nan)
        ))
    recorder.end_episode()
    destination = recorder.finalize("success")
    recorder.stop()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["outcome"] == "success"
    assert manifest["training_eligible"] is False
    assert manifest["camera_completeness"]["right"]["complete"] is False


def test_failure_is_quarantined(tmp_path):
    recorder = AgentEpisodeRecorder(tmp_path / "agent", threading.Event())
    recorder.configure_episode({
        "task": "lid_close", "target_selection": {"u": .2, "v": .3},
        "initial_bias_m": {"left": [0, 0, 0], "right": [0, 0, -.03]},
    })
    recorder.start_episode()
    recorder.record_sample(sample(0))
    recorder.record_sample(sample(1))
    recorder.end_episode()
    destination = recorder.finalize("failure", reason="grasp_miss")
    recorder.stop()
    assert "failures/lid_close" in str(destination)
    assert find_episode_pairs(str(tmp_path / "agent"), [], require_success_manifest=True) == []


def test_phone_ui_command_round_trip():
    state = InterventionState()

    def callback(command, payload):
        return {"command": command, "payload": payload}

    ui = AgentCollectionUI("127.0.0.1", 0, "secret", state, callback,
                           lambda _camera: (None, None))
    ui.start()
    request = urllib.request.Request(
        f"http://127.0.0.1:{ui.port}/api/command?token=secret",
        data=json.dumps({"command": "pause", "payload": {}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    response = json.loads(urllib.request.urlopen(request, timeout=1).read())
    ui.stop()
    assert response["command"] == "pause"

import json
import threading
import urllib.request

import h5py
import numpy as np
import pytest

from rollout.agent_collection import (
    AgentEpisodeRecorder, AgentRecordingSample, ControllerClaim,
    InterventionState, intervention_slice_mask,
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

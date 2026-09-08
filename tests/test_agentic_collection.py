import json
from pathlib import Path
import threading

import h5py
import numpy as np
import pytest

from rollout.agentic_collection import (
    AgenticEpisode,
    Checkpoint,
    CheckpointVerifier,
    EpisodeClass,
    PetriTaskScheduler,
    ReferenceEnvelope,
    SemanticObservation,
    SkillEvidence,
    SkillRegistry,
    TaskSpec,
    VerifierConfig,
    Verdict,
    normalized_visual_difference,
)
from rollout.agentic_policy_supervisor import AgenticPolicySupervisor
from rollout.recorder import DataRecorder
from src.convert_to_lerobot import agentic_classification


def _task(name="petri2microscope", source="bench", destination="microscope"):
    return TaskSpec(name, f"move to {destination}", source, destination, "right")


def _config():
    return VerifierConfig(
        object_diameter_m=0.1,
        maximum_observation_age_s=1.0,
        maximum_camera_skew_s=0.1,
        reference_position_tolerance_diameters=0.5,
        minimum_lift_diameters=0.1,
        minimum_visual_change_fraction=0.01,
        stable_observations=2,
    )


def _reference():
    return ReferenceEnvelope(
        (0.1, 0.2, 0.3),
        (0.5, 0.6, 0.35),
        0.2,
        0.9,
        ("demo.hdf5",),
    )


def _obs(
    xyz=(0.1, 0.2, 0.3),
    grip=1.0,
    *,
    station="bench",
    attached=None,
    supported=None,
    timestamp=10.0,
):
    return SemanticObservation(
        timestamp=timestamp,
        ee_position_xyz=xyz,
        gripper_open_fraction=grip,
        camera_timestamps={"head": timestamp, "left": timestamp, "right": timestamp},
        visual_change_fraction={"head": 0.05},
        target_visible=True,
        target_station=station,
        target_attached=attached,
        target_supported=supported,
    )


def test_verified_episode_becomes_clean_success():
    task = _task()
    initial = _obs()
    episode = AgenticEpisode(task, CheckpointVerifier(_config()), _reference(), initial)
    assert episode.check(Checkpoint.INITIAL, initial, now=10).verdict is Verdict.ACCEPT
    assert episode.check(Checkpoint.PRE_GRASP, initial, now=10).verdict is Verdict.ACCEPT
    lifted = _obs((0.1, 0.2, 0.32), 0.2, attached=True)
    assert episode.check(Checkpoint.POST_GRASP, lifted, now=10).verdict is Verdict.ACCEPT
    at_goal = _obs((0.5, 0.6, 0.35), 0.2, station="bench", attached=True)
    assert episode.check(Checkpoint.PRE_PLACE, at_goal, now=10).verdict is Verdict.ACCEPT
    released = _obs(
        (0.5, 0.6, 0.35), 0.9, station="microscope",
        attached=False, supported=True,
    )
    assert episode.check(Checkpoint.POST_RELEASE, released, now=10).verdict is Verdict.ACCEPT
    assert episode.classify() is EpisodeClass.CLEAN_SUCCESS


def test_missing_support_is_uncertain_and_never_training_success():
    verifier = CheckpointVerifier(_config())
    decision = verifier.evaluate(
        Checkpoint.POST_RELEASE,
        _obs((0.5, 0.6, 0.35), 0.9, station="microscope", attached=False),
        task=_task(),
        reference=_reference(),
        initial=_obs(),
        now=10,
    )
    assert decision.verdict is Verdict.UNCERTAIN
    assert "support_state_unknown" in decision.reasons


def test_success_like_finger_opening_cannot_prove_attachment():
    decision = CheckpointVerifier(_config()).evaluate(
        Checkpoint.POST_GRASP,
        _obs((0.1, 0.2, 0.32), 0.2, attached=None),
        task=_task(),
        reference=_reference(),
        initial=_obs(),
        now=10,
    )
    assert decision.verdict is Verdict.UNCERTAIN
    assert "attachment_only_inferred_from_gripper" in decision.reasons


def test_stale_or_desynchronized_cameras_reject():
    value = _obs(timestamp=5.0)
    value = SemanticObservation(
        **{
            **value.__dict__,
            "camera_timestamps": {"head": 5.0, "left": 5.4, "right": 5.0},
        }
    )
    decision = CheckpointVerifier(_config()).evaluate(
        Checkpoint.INITIAL,
        value,
        task=_task(),
        reference=_reference(),
        initial=value,
        now=10.0,
    )
    assert decision.verdict is Verdict.REJECT
    assert "stale_observation" in decision.reasons
    assert "camera_timestamp_skew" in decision.reasons


def test_petritask_scheduler_uses_bench_as_reversible_hub():
    tasks = [
        _task("to_microscope", "bench", "microscope"),
        _task("back_from_microscope", "microscope", "bench"),
        _task("to_incubator", "bench", "incubator"),
        _task("back_from_incubator", "incubator", "bench"),
    ]
    scheduler = PetriTaskScheduler(tasks, initial_station="bench")
    names = []
    for _ in range(4):
        task = scheduler.next_task()
        names.append(task.name)
        scheduler.record(task, success=True)
    assert names == [
        "to_microscope", "back_from_microscope",
        "to_incubator", "back_from_incubator",
    ]


def test_reference_envelope_uses_only_clean_open_close_open(tmp_path):
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as recording:
        recording.create_dataset(
            "right_ee_pos",
            data=np.asarray([[0, 0, 0], [1, 2, 3], [2, 3, 4], [5, 6, 7]]),
        )
        recording.create_dataset("right_gripper", data=[1, 0, 0, 1])
        recording.create_dataset("right_gripper_exact", data=[0.9, 0.2, 0.25, 0.95])
    result = ReferenceEnvelope.from_hdf5([path], arm="right")
    assert result.close_position_xyz == (1.0, 2.0, 3.0)
    assert result.release_position_xyz == (5.0, 6.0, 7.0)
    assert result.closed_gripper_median == pytest.approx(0.225)


def test_visual_difference_is_resolution_independent():
    first = np.zeros((480, 640, 3), np.uint8)
    second = first.copy()
    second[100:200, 100:200] = 255
    result = normalized_visual_difference(first, second)
    assert 0 < result["changed_pixel_fraction"] < 1
    assert result["brightness_after"] > result["brightness_before"]


def test_skill_promotion_requires_validation_and_multiple_start_bins(tmp_path):
    registry = SkillRegistry(tmp_path / "skills.json")
    trace = [{"primitive": "retry_grasp", "object_frame_delta": [0, 0, 0.01]}]
    for index in range(5):
        registry.add_candidate(
            "retry_grasp_v1",
            primitive_trace=trace,
            preconditions={"target_visible": True},
            postconditions={"target_attached": True},
            evidence=SkillEvidence(
                f"ep{index}", "petri", "left" if index < 3 else "right", "ring", True
            ),
        )
    with pytest.raises(ValueError, match="simulation and shadow"):
        registry.promote("retry_grasp_v1")
    registry.mark_validation("retry_grasp_v1", simulation=True, shadow=True)
    registry.promote("retry_grasp_v1")
    assert json.loads((tmp_path / "skills.json").read_text())["skills"]["retry_grasp_v1"]["promoted"]


def _write_supervisor_fixture(tmp_path):
    demo = tmp_path / "reference.hdf5"
    with h5py.File(demo, "w") as recording:
        recording.create_dataset(
            "right_ee_pos",
            data=np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.5, 0.6, 0.35]]),
        )
        recording.create_dataset("right_gripper", data=[1, 0, 1])
        recording.create_dataset("right_gripper_exact", data=[1, 0.2, 1])
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "piper_robot.agentic_collection_profile/v1",
                "initial_station": "bench",
                "tasks": [
                    {
                        "name": "petri2microscope",
                        "instruction": "move petri",
                        "source": "bench",
                        "destination": "microscope",
                        "arm": "right",
                        "reference_glob": str(demo),
                    },
                    {
                        "name": "petri2bench",
                        "instruction": "return petri",
                        "source": "microscope",
                        "destination": "bench",
                        "arm": "right",
                        "reference_glob": str(demo),
                    },
                ],
                "verifier": {
                    "object_diameter_m": 0.1,
                    "maximum_observation_age_s": 1,
                    "maximum_camera_skew_s": 0.1,
                    "reference_position_tolerance_diameters": 0.5,
                    "minimum_lift_diameters": 0.1,
                    "minimum_visual_change_fraction": 0.001,
                    "stable_observations": 2,
                },
            }
        )
    )
    return profile


def _raw(clock, xyz, grip, *, semantic=None, changed=False):
    qpos = np.zeros(20)
    qpos[10:13] = xyz
    qpos[19] = grip
    image = np.full((40, 60, 3), 200 if not changed else 100, np.uint8)
    return {
        "timestamp": clock,
        "qpos": qpos,
        "images": {"cam_high": image, "cam_left_wrist": image, "cam_right_wrist": image},
        "camera_timestamps": {"cam_high": clock, "cam_left_wrist": clock, "cam_right_wrist": clock},
        "semantic": semantic or {},
    }


def test_policy_supervisor_completes_clean_semantic_cycle(tmp_path):
    clock = [10.0]
    supervisor = AgenticPolicySupervisor(
        _write_supervisor_fixture(tmp_path),
        mode="auto",
        condition="ring",
        armed=True,
        output_dir=tmp_path / "runs",
        now=lambda: clock[0],
    )
    initial = _raw(
        clock[0], [0.1, 0.2, 0.3], 1.0,
        semantic={"target_station": "bench", "target_visible": True},
    )
    supervisor.begin_episode(initial)
    assert supervisor.before_action({"right_gripper": 0.0})
    supervisor.after_action({"right_gripper": 0.0})
    clock[0] += 0.1
    supervisor.observe(
        _raw(clock[0], [0.1, 0.2, 0.32], 0.2, semantic={"target_attached": True}, changed=True)
    )
    clock[0] += 0.1
    supervisor.observe(
        _raw(clock[0], [0.5, 0.6, 0.35], 0.2, semantic={"target_attached": True}, changed=True)
    )
    assert supervisor.before_action({"right_gripper": 1.0})
    supervisor.after_action({"right_gripper": 1.0})
    for _ in range(2):
        clock[0] += 0.1
        supervisor.observe(
            _raw(
                clock[0], [0.5, 0.6, 0.35], 1.0,
                semantic={
                    "target_attached": False,
                    "target_supported": True,
                    "target_station": "microscope",
                },
                changed=True,
            )
        )
    assert supervisor.terminal_request() == (EpisodeClass.CLEAN_SUCCESS, "verified_success")


def test_shadow_supervisor_never_authorizes_commands(tmp_path):
    supervisor = AgenticPolicySupervisor(
        _write_supervisor_fixture(tmp_path),
        mode="shadow",
        condition="ring",
        output_dir=tmp_path / "runs",
        now=lambda: 10.0,
    )
    supervisor.begin_episode(
        _raw(10.0, [0.1, 0.2, 0.3], 1.0, semantic={"target_station": "bench"})
    )
    assert not supervisor.commands_enabled
    assert not supervisor.before_action({"right_gripper": 0.0})


def test_operator_override_can_only_continue_uncertain_gate(tmp_path):
    supervisor = AgenticPolicySupervisor(
        _write_supervisor_fixture(tmp_path),
        mode="auto",
        condition="ring",
        armed=True,
        output_dir=tmp_path / "runs",
        now=lambda: 10.0,
    )
    raw = _raw(10.0, [0.1, 0.2, 0.3], 1.0, semantic={"target_station": "bench"})
    # Removing synchronized timestamps makes the initial gate reject, not
    # merely uncertain, so the phone override must fail closed.
    raw["camera_timestamps"] = {}
    supervisor.begin_episode(raw)
    with pytest.raises(ValueError, match="only an uncertain"):
        supervisor.override_uncertain_checkpoint("looks okay")


def test_uncertain_gate_holds_then_replans_after_operator_review(tmp_path):
    supervisor = AgenticPolicySupervisor(
        _write_supervisor_fixture(tmp_path),
        mode="auto",
        condition="ring",
        armed=True,
        output_dir=tmp_path / "runs",
        now=lambda: 10.0,
    )
    supervisor.begin_episode(_raw(10.0, [0.1, 0.2, 0.3], 1.0))
    assert supervisor.terminal_request() is None
    assert supervisor.held_uncertain
    assert not supervisor.before_action({"right_gripper": 0.0})
    supervisor.override_uncertain_checkpoint("operator sees the dish")
    assert not supervisor.held_uncertain
    assert supervisor.consume_resume_replan()
    assert not supervisor.consume_resume_replan()


def test_semantic_provider_is_called_only_for_uncertainty(tmp_path):
    calls = []

    def provider(checkpoint, raw, semantic):
        calls.append(checkpoint)
        return {
            "provider": "test_reasoner",
            "target_visible": True,
            "target_station": "bench",
        }

    supervisor = AgenticPolicySupervisor(
        _write_supervisor_fixture(tmp_path),
        mode="shadow",
        condition="ring",
        output_dir=tmp_path / "runs",
        semantic_provider=provider,
        now=lambda: 10.0,
    )
    supervisor.begin_episode(_raw(10.0, [0.1, 0.2, 0.3], 1.0))
    assert calls == [Checkpoint.INITIAL]
    assert supervisor.episode.decisions[-1].verdict is Verdict.ACCEPT


def test_recorder_writes_agentic_sidecar_and_camera_timestamps(tmp_path):
    stopped = threading.Event()
    stopped.set()
    recorder = DataRecorder(tmp_path, stopped)
    recorder._episode_name = "episode_0000_20260828_120000"
    recorder.episode_data = {
        "timestamps": [1.0],
        "left_ee_pos": [[0, 0, 0]],
        "left_ee_quat": [[1, 0, 0, 0]],
        "left_gripper_exact": [0.1],
        "left_gripper": [0.0],
        "right_ee_pos": [[0, 0, 0]],
        "right_ee_quat": [[1, 0, 0, 0]],
        "right_gripper_exact": [0.2],
        "right_gripper": [0.0],
        "rgb_frame_timestamps": [0.99],
        "left_wrist_rgb_timestamps": [0.98],
        "right_wrist_rgb_timestamps": [0.97],
        "left_joint_positions": [[0] * 7],
        "right_joint_positions": [[0] * 7],
    }
    recorder.set_episode_context({"task": "petri2microscope"})
    recorder.set_episode_outcome({"classification": "clean_success"})
    recorder._save_hdf5()
    recorder._save_agentic_sidecar()

    h5_path = tmp_path / f"{recorder._episode_name}.hdf5"
    with h5py.File(h5_path, "r") as recording:
        assert recording.attrs["agentic_classification"] == "clean_success"
        assert recording["right_wrist_rgb_timestamps"][0] == pytest.approx(0.97)
    sidecar = json.loads(h5_path.with_suffix(".agentic.json").read_text())
    assert sidecar["context"]["task"] == "petri2microscope"
    assert sidecar["outcome"]["classification"] == "clean_success"
    recorder.stop()


def test_agentic_classification_prefers_sidecar(tmp_path):
    h5 = tmp_path / "episode.hdf5"
    with h5py.File(h5, "w") as recording:
        recording.attrs["agentic_classification"] = "failure"
    h5.with_suffix(".agentic.json").write_text(
        json.dumps({"outcome": {"classification": "clean_success"}})
    )
    assert agentic_classification(str(h5)) == "clean_success"

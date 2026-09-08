import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rollout.contact_placement_pipeline import (
    Action,
    ArmIdentity,
    ContactPlacementConfig,
    ContactPlacementPolicy,
    FrameEvidence,
    GoalEstimate,
    MobileEvidencePublisher,
    PipelineState,
    RuntimeObservation,
    Stage,
    build_level_transfer_plan,
    deproject_normalized_goal,
    validate_fresh_frames,
)
from rollout.gripper_level import JawLevelReference


def _config(**overrides):
    values = {
        "physical_arm": "right",
        "required_cameras": ("head", "manipulator", "observer"),
        "maximum_frame_age_s": 1.0,
        "maximum_camera_skew_s": 0.1,
    }
    values.update(overrides)
    return ContactPlacementConfig(**values)


def _goal(position=(0.4, 0.1, 0.2), scale=0.1):
    return GoalEstimate(
        semantic_name="destination_support_center",
        position_robot_m=position,
        support_normal_robot=(0.0, 0.0, 1.0),
        characteristic_scale_m=scale,
        source="semantic_scene_object",
        scene_revision="scene-7",
    )


def _frames(now, revision):
    return tuple(
        FrameEvidence(name, f"{name}-{revision}", now - index * 0.01)
        for index, name in enumerate(("head", "manipulator", "observer"))
    )


def _observation(now, revision, **overrides):
    values = {
        "physical_arm": "right",
        "scene_revision": "scene-7",
        "frames": _frames(now, revision),
    }
    values.update(overrides)
    return RuntimeObservation(**values)


def test_physical_arm_mapping_is_explicit_and_not_image_dependent():
    right = ArmIdentity.for_physical_arm("right")
    left = ArmIdentity.for_physical_arm("left")
    assert right.production_branch == "left_arm"
    assert right.semantic_branch == "right"
    assert right.observer_arm == "left"
    assert left.production_branch == "right_arm"
    assert left.semantic_branch == "left"


def test_normalized_rgbd_goal_is_resolution_independent():
    transform = np.eye(4)
    first = deproject_normalized_goal(
        normalized_uv=(0.25, 0.5),
        depth_m=np.full((100, 200), 0.8),
        intrinsics_fx_fy_cx_cy=(160.0, 160.0, 100.0, 50.0),
        camera_to_robot_4x4=transform,
        semantic_name="support",
        support_normal_robot=(0, 0, 1),
        characteristic_scale_m=0.08,
        scene_revision="r1",
    )
    second = deproject_normalized_goal(
        normalized_uv=(0.25, 0.5),
        depth_m=np.full((200, 400), 0.8),
        intrinsics_fx_fy_cx_cy=(320.0, 320.0, 200.0, 100.0),
        camera_to_robot_4x4=transform,
        semantic_name="support",
        support_normal_robot=(0, 0, 1),
        characteristic_scale_m=0.08,
        scene_revision="r1",
    )
    np.testing.assert_allclose(first.position_robot_m, second.position_robot_m)


def test_transfer_plan_retargets_arbitrary_measured_start_and_goal():
    config = _config()
    reference = JawLevelReference()
    first = build_level_transfer_plan(
        start_pose_wxyz_xyz=(1, 0, 0, 0, 0.25, -0.1, 0.35),
        goal=_goal((0.45, 0.2, 0.18), scale=0.08),
        config=config,
        level_reference=reference,
    )
    second = build_level_transfer_plan(
        start_pose_wxyz_xyz=(1, 0, 0, 0, -0.2, 0.3, 0.42),
        goal=_goal((-0.3, -0.15, 0.25), scale=0.16),
        config=config,
        level_reference=reference,
    )
    assert first.waypoints_wxyz_xyz != second.waypoints_wxyz_xyz
    np.testing.assert_allclose(first.waypoints_wxyz_xyz[-1][4:6], (0.45, 0.2))
    np.testing.assert_allclose(second.waypoints_wxyz_xyz[-1][4:6], (-0.3, -0.15))
    assert second.route_clearance_m > first.route_clearance_m
    assert first.production_branch == "left_arm"
    assert first.motion_ready is False


def test_fresh_frame_gate_rejects_stale_skewed_and_reused_frames():
    frames = _frames(10.0, "a")
    accepted = validate_fresh_frames(
        frames,
        required_cameras=("head", "manipulator", "observer"),
        now_s=10.0,
        maximum_age_s=1.0,
        maximum_skew_s=0.1,
    )
    assert accepted.accepted
    reused = validate_fresh_frames(
        frames,
        required_cameras=("head", "manipulator", "observer"),
        now_s=10.0,
        maximum_age_s=1.0,
        maximum_skew_s=0.1,
        prior_frame_ids={frame.camera: frame.frame_id for frame in frames},
    )
    assert not reused.accepted
    assert any(reason.startswith("reused_frame") for reason in reused.reasons)
    stale = validate_fresh_frames(
        frames,
        required_cameras=("head", "manipulator", "observer"),
        now_s=12.0,
        maximum_age_s=1.0,
        maximum_skew_s=0.1,
    )
    assert not stale.accepted


def test_stalled_descent_without_contact_evidence_rebranches():
    now = 20.0
    policy = ContactPlacementPolicy(_config(), _goal())
    state = PipelineState(stage=Stage.DESCEND)
    transition = policy.advance(
        state,
        _observation(
            now,
            "1",
            requested_descent_m=0.004,
            measured_descent_m=0.0002,
            maximum_torque_change_nm=0.01,
            support_clearance_m=0.03,
        ),
        now_s=now,
    )
    assert transition.action == Action.REBRANCH
    assert transition.state.stage == Stage.ALIGN
    assert "stalled_without_contact_evidence" in transition.reasons


def test_pressure_or_repeated_support_evidence_proves_contact():
    now = 30.0
    policy = ContactPlacementPolicy(_config(required_contact_candidates=2), _goal())
    pressure = policy.advance(
        PipelineState(stage=Stage.DESCEND),
        _observation(now, "p", pressure_latched=True),
        now_s=now,
    )
    assert pressure.action == Action.HOLD_CONTACT
    assert pressure.state.stage == Stage.CONTACT

    first = policy.advance(
        PipelineState(stage=Stage.DESCEND),
        _observation(
            now,
            "1",
            requested_descent_m=0.004,
            measured_descent_m=0.0002,
            maximum_torque_change_nm=0.02,
            support_clearance_m=0.002,
        ),
        now_s=now,
    )
    assert first.action == Action.PROBE_DESCENT
    assert first.state.consecutive_contact_candidates == 1
    second = policy.advance(
        first.state,
        _observation(
            now + 0.2,
            "2",
            requested_descent_m=first.distance_m,
            measured_descent_m=0.0,
            maximum_torque_change_nm=0.02,
            support_clearance_m=0.001,
        ),
        now_s=now + 0.2,
    )
    assert second.action == Action.HOLD_CONTACT
    assert second.state.stage == Stage.CONTACT


def test_scene_or_arm_change_fails_closed():
    policy = ContactPlacementPolicy(_config(), _goal())
    wrong = policy.advance(
        PipelineState(),
        RuntimeObservation(
            physical_arm="left",
            scene_revision="scene-7",
            frames=_frames(5.0, "x"),
        ),
        now_s=5.0,
    )
    assert wrong.action == Action.HOLD
    assert wrong.state.stage == Stage.BLOCKED


def test_transient_stale_camera_holds_same_stage_for_fresh_resume():
    policy = ContactPlacementPolicy(_config(), _goal())
    held = policy.advance(
        PipelineState(stage=Stage.ALIGN),
        _observation(5.0, "old", normalized_goal_error=0.1),
        now_s=7.0,
    )
    assert held.action == Action.HOLD
    assert held.state.stage == Stage.ALIGN
    assert any(reason.startswith("stale_frame") for reason in held.reasons)


def test_mobile_publisher_persists_revision_and_rejects_stale_reuse(tmp_path):
    now = 40.0
    frames = []
    for index, camera in enumerate(("head", "manipulator", "observer")):
        image = tmp_path / f"{camera}.jpg"
        image.write_bytes(f"image-{camera}".encode())
        frames.append(
            FrameEvidence(
                camera=camera,
                frame_id=f"{camera}-1",
                captured_at_s=now - index * 0.01,
                image_path=str(image),
            )
        )
    output = tmp_path / "phone"
    manifest = MobileEvidencePublisher(output).publish(
        semantic_name="destination",
        physical_arm="right",
        stage=Stage.DESCEND,
        action=Action.PROBE_DESCENT,
        frames=frames,
        required_cameras=("head", "manipulator", "observer"),
        maximum_age_s=1.0,
        maximum_skew_s=0.1,
        now_s=now,
    )
    assert manifest["revision"] == 1
    assert (output / "index.html").is_file()
    assert json.loads((output / "current.json").read_text())["physical_arm"] == "right"
    with pytest.raises(RuntimeError, match="reused_frame"):
        MobileEvidencePublisher(output).publish(
            semantic_name="destination",
            physical_arm="right",
            stage=Stage.DESCEND,
            action=Action.PROBE_DESCENT,
            frames=frames,
            required_cameras=("head", "manipulator", "observer"),
            maximum_age_s=1.0,
            maximum_skew_s=0.1,
            now_s=now,
        )


def test_example_profile_has_no_task_pixel_or_lab_name():
    profile_path = Path("src/configs/contact_placement_profile.example.json")
    text = profile_path.read_text()
    profile = json.loads(text)
    assert profile["schema"] == "piper_robot.contact_placement_profile/v1"
    assert "pixel" not in text.lower()
    assert "pasteur" not in text.lower()


def test_cli_preview_plan_and_state_advance_are_codex_free(tmp_path):
    start = tmp_path / "start.json"
    goal = tmp_path / "goal.json"
    plan = tmp_path / "plan.json"
    state = tmp_path / "state.json"
    observation = tmp_path / "observation.json"
    transition = tmp_path / "transition.json"
    start.write_text(
        json.dumps(
            {
                "measured_q_physical_rad": [0.0] * 6,
                "measured_pose_wxyz_xyz": [1, 0, 0, 0, 0.2, 0.0, 0.4],
            }
        )
    )
    goal.write_text(json.dumps(_goal().to_dict()))
    profile = Path("src/configs/contact_placement_profile.example.json").resolve()
    result = subprocess.run(
        [
            sys.executable,
            "src/run_contact_placement_pipeline.py",
            "plan",
            "--profile",
            str(profile),
            "--start",
            str(start),
            "--goal",
            str(goal),
            "--output",
            str(plan),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan_document = json.loads(plan.read_text())
    assert plan_document["motion_ready"] is False
    assert plan_document["execution_contract"]["allowed"] is False

    observation.write_text(
        json.dumps(
            {
                "physical_arm": "right",
                "scene_revision": "scene-7",
                "frames": [frame.to_dict() for frame in _frames(100.0, "cli")],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "src/run_contact_placement_pipeline.py",
            "advance",
            "--profile",
            str(profile),
            "--goal",
            str(goal),
            "--state",
            str(state),
            "--observation",
            str(observation),
            "--output",
            str(transition),
            "--next-state",
            str(state),
            "--now-s",
            "100",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(transition.read_text())["action"] == "execute_approach"
    assert json.loads(state.read_text())["stage"] == "approach"

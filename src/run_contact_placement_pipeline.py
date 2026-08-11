#!/usr/bin/env python3
"""Plan and advance the generalized, evidence-gated placement pipeline.

This entrypoint is intentionally deterministic.  It does not use a language
model and does not move hardware by itself.  A site runner may execute only a
plan whose ``motion_ready`` field is true, then feed each fresh observation to
``advance`` before issuing the returned named primitive.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rollout.contact_placement_pipeline import (  # noqa: E402
    Action,
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
)
from rollout.dish_transport_rehearsal import (  # noqa: E402
    ProductionArmKinematics,
    audit_joint_path,
)
from rollout.gripper_level import JawLevelReference  # noqa: E402


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(profile_path: Path) -> tuple[dict, ContactPlacementConfig]:
    profile = _read_json(profile_path)
    if profile.get("schema") != "piper_robot.contact_placement_profile/v1":
        raise ValueError("unsupported contact-placement profile schema")
    return profile, ContactPlacementConfig.from_dict(profile["policy"])


def _load_level_reference(profile: dict) -> JawLevelReference:
    return JawLevelReference(**profile.get("level_reference", {}))


def _load_goal(goal_path: Path) -> GoalEstimate:
    value = _read_json(goal_path)
    if "position_robot_m" in value:
        return GoalEstimate(
            semantic_name=value["semantic_name"],
            position_robot_m=tuple(value["position_robot_m"]),
            support_normal_robot=tuple(value["support_normal_robot"]),
            characteristic_scale_m=float(value["characteristic_scale_m"]),
            source=str(value["source"]),
            scene_revision=str(value["scene_revision"]),
        )
    base = goal_path.parent
    depth = np.load(_resolve(base, value["depth_npy"]))
    return deproject_normalized_goal(
        normalized_uv=value["normalized_uv"],
        depth_m=depth,
        intrinsics_fx_fy_cx_cy=value["intrinsics_fx_fy_cx_cy"],
        camera_to_robot_4x4=value["camera_to_robot_4x4"],
        semantic_name=value["semantic_name"],
        support_normal_robot=value["support_normal_robot"],
        characteristic_scale_m=float(value["characteristic_scale_m"]),
        scene_revision=str(value["scene_revision"]),
        sampling_radius_fraction=float(value.get("sampling_radius_fraction", 0.01)),
    )


def command_plan(args) -> int:
    profile, config = _load_config(args.profile)
    start = _read_json(args.start)
    goal = _load_goal(args.goal)
    plan = build_level_transfer_plan(
        start_pose_wxyz_xyz=start["measured_pose_wxyz_xyz"],
        goal=goal,
        config=config,
        level_reference=_load_level_reference(profile),
    )
    planning = profile.get("planning", {})
    production_model = planning.get("production_model")
    scene_model = planning.get("semantic_scene_model")
    if bool(production_model) != bool(scene_model):
        raise ValueError(
            "production_model and semantic_scene_model must be supplied together"
        )
    if production_model:
        base = args.profile.parent
        kinematics = ProductionArmKinematics(
            _resolve(base, production_model), config.physical_arm
        )
        q_path, ik_reports = kinematics.solve_path(
            np.asarray(plan.poses_wxyz_xyz),
            seed_q=np.asarray(start["measured_q_physical_rad"], dtype=float),
            level_reference=_load_level_reference(profile),
            maximum_joint_delta_rad=planning.get("maximum_joint_delta_rad"),
        )
        envelope = profile["carried_object_collision_envelope"]
        audit = audit_joint_path(
            _resolve(base, scene_model),
            config.physical_arm,
            q_path,
            dish_radius_m=float(envelope["radius_m"]),
            dish_thickness_m=float(envelope["thickness_m"]),
            dish_center_offset_ee_m=envelope["center_offset_ee_m"],
            ignored_environment_bodies=tuple(
                planning.get("ignored_environment_bodies", ())
            ),
            maximum_baseline_penetration_increase_m=float(
                planning.get("maximum_baseline_penetration_increase_m", 0.003)
            ),
        )
        audit = {**audit, "ik_reports": ik_reports}
        plan = replace(
            plan,
            q_physical_rad=tuple(tuple(float(v) for v in q) for q in q_path),
            collision_audit=audit,
        )
    document = plan.to_dict()
    document["execution_contract"] = {
        "allowed": plan.motion_ready,
        "reason": (
            "exact CAD IK and semantic-scene collision audit accepted"
            if plan.motion_ready
            else "preview only; exact CAD IK and collision audit are both required"
        ),
        "first_command_must_start_at_measured_q": True,
        "pressure_guard_required_during_descent": True,
        "fresh_post_motion_frames_required": list(config.required_cameras),
    }
    _write_json(args.output, document)
    print(json.dumps({"output": str(args.output), "motion_ready": plan.motion_ready}))
    return 0


def command_advance(args) -> int:
    _, config = _load_config(args.profile)
    goal = _load_goal(args.goal)
    state = PipelineState.from_dict(_read_json(args.state)) if args.state.exists() else PipelineState()
    observation = RuntimeObservation.from_dict(_read_json(args.observation))
    transition = ContactPlacementPolicy(config, goal).advance(
        state, observation, now_s=args.now_s
    )
    _write_json(args.output, transition.to_dict())
    if args.next_state is not None:
        _write_json(args.next_state, transition.state.to_dict())
    print(json.dumps(transition.to_dict(), ensure_ascii=False))
    return 0 if transition.allowed else 2


def command_publish(args) -> int:
    _, config = _load_config(args.profile)
    payload = _read_json(args.evidence)
    frames = tuple(FrameEvidence(**item) for item in payload["frames"])
    publisher = MobileEvidencePublisher(args.directory)
    manifest = publisher.publish(
        semantic_name=payload["semantic_name"],
        physical_arm=payload["physical_arm"],
        stage=Stage(payload["stage"]),
        action=Action(payload["action"]),
        frames=frames,
        required_cameras=config.required_cameras,
        maximum_age_s=config.maximum_frame_age_s,
        maximum_skew_s=config.maximum_camera_skew_s,
        now_s=args.now_s,
        metrics=payload.get("metrics"),
    )
    print(json.dumps({"revision": manifest["revision"], "directory": str(args.directory)}))
    return 0


def command_serve(args) -> int:
    args.directory.mkdir(parents=True, exist_ok=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(args.directory.resolve()))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"mobile evidence: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--profile", required=True, type=Path)
    plan.add_argument("--start", required=True, type=Path)
    plan.add_argument("--goal", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.set_defaults(function=command_plan)

    advance = subparsers.add_parser("advance")
    advance.add_argument("--profile", required=True, type=Path)
    advance.add_argument("--goal", required=True, type=Path)
    advance.add_argument("--state", required=True, type=Path)
    advance.add_argument("--observation", required=True, type=Path)
    advance.add_argument("--output", required=True, type=Path)
    advance.add_argument("--next-state", type=Path)
    advance.add_argument("--now-s", type=float, default=time.time())
    advance.set_defaults(function=command_advance)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--profile", required=True, type=Path)
    publish.add_argument("--evidence", required=True, type=Path)
    publish.add_argument("--directory", required=True, type=Path)
    publish.add_argument("--now-s", type=float, default=time.time())
    publish.set_defaults(function=command_publish)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--directory", required=True, type=Path)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8774)
    serve.set_defaults(function=command_serve)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

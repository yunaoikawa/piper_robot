#!/usr/bin/env python3
"""Plan and execute a demo-free lid grasp through the teleop command path.

The default mode is simulation-only.  ``--execute`` is the sole hardware
motion switch.  No replay trajectory, successful grasp pose/image, or
demonstrated gripper width is accepted as an input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.rpc import RPCClient
from rollout.teleop_trajectory_stream import (
    ProductionRightFK,
    TeleopTrajectoryStreamer,
    sample_joint_knots,
)
from rollout.gripper_level import (
    JawLevelReference,
    RightJawLevelCheckpoint,
    assess_jaw_level,
    leveled_pose,
)
from rollout.orientation_monitor_policy import (
    CachedOrientationMonitor,
    OrientationMonitoringPolicyStore,
)
from rollout.torque_safety import torque_stop_enabled_from_config
from rollout.right_active_visual_search import select_unique_scene_target
from src.optimize_lid_grasp_trajectory import search


RUN_SCHEMA = "piper_robot.autonomous_shape_lid_grasp/v1"
TRAJECTORY_SCHEMA = "piper_robot.simulated_lid_grasp_trajectory/v2"


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).resolve().read_text())


def _load_level_reference(path: str | Path) -> JawLevelReference:
    value = _load_json(path)
    names = {
        "support_up_robot",
        "tip_baseline_ee",
        "approach_axis_ee",
        "open_tip_span_m",
        "maximum_checkpoint_tilt_deg",
        "maximum_planned_tilt_deg",
        "maximum_tip_height_difference_m",
    }
    kwargs = {name: value[name] for name in names if name in value}
    kwargs["source"] = str(value.get("schema", Path(path).resolve()))
    return JawLevelReference(**kwargs)


def validate_demo_free_trajectory(payload: dict) -> None:
    if payload.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError(
            f"trajectory must use {TRAJECTORY_SCHEMA}, got "
            f"{payload.get('schema')!r}"
        )
    if payload.get("commands_sent") is not False:
        raise ValueError("planner artifact must be simulation-only")
    policy = payload.get("closure_policy")
    if not isinstance(policy, dict):
        raise ValueError("trajectory lacks a closure_policy")
    if policy.get("demonstration_used") is not False:
        raise ValueError("trajectory closure must be demo-free")
    if policy.get("mode") != "close_until_obstructed":
        raise ValueError("unsupported physical closure policy")
    if "closure_demonstration" in payload:
        raise ValueError("demonstration-derived closure is forbidden")
    knots = payload.get("knots")
    if not isinstance(knots, list) or len(knots) < 2:
        raise ValueError("trajectory needs at least two knots")
    stages = [str(knot.get("stage")) for knot in knots]
    required = (
        "home",
        "depart_up",
        "hover_xy",
        "descend",
        "insert",
        "preclose_observe",
        "close",
    )
    missing = [stage for stage in required if stage not in stages]
    if missing:
        raise ValueError(f"trajectory is missing required stages: {missing}")
    if "verification_lift" not in stages:
        raise ValueError("trajectory lacks a straight verification lift")
    close_index = stages.index("close")
    lift_indices = [
        index for index, stage in enumerate(stages)
        if stage == "verification_lift"
    ]
    if min(lift_indices) <= close_index:
        raise ValueError("verification lift must follow closure")
    lift_q = [
        np.asarray(knots[index]["right_q_physical_rad"], dtype=float)
        for index in lift_indices
    ]
    if any(q.shape != (6,) or not np.all(np.isfinite(q)) for q in lift_q):
        raise ValueError("verification lift contains invalid joint targets")


def audit_physical_right_level(
    trajectory: dict,
    *,
    production_model: str | Path,
    reference: JawLevelReference,
) -> dict:
    """Reject plans that only look level in the semantic NYU model."""

    fk = ProductionRightFK(production_model)
    records_by_stage = {}
    required = {
        "descend_fast",
        "descend",
        "insert",
        "preclose_observe",
        "close",
        "verification_lift",
    }
    for sample in sample_joint_knots(trajectory["knots"]):
        stage = str(sample.stage)
        if stage not in required:
            continue
        assessment = assess_jaw_level(
            leveled_pose(
                fk.pose(sample.right_q_physical_rad).parameters(), reference
            ),
            reference,
            planned=True,
        )
        record = records_by_stage.setdefault(
            stage,
            {
                "stage": stage,
                "accepted": True,
                "maximum_combined_tilt_deg": 0.0,
                "maximum_tip_height_difference_m": 0.0,
                "sample_count": 0,
            },
        )
        record["sample_count"] += 1
        record["accepted"] = bool(record["accepted"] and assessment.accepted)
        record["maximum_combined_tilt_deg"] = max(
            record["maximum_combined_tilt_deg"],
            assessment.combined_tilt_deg,
        )
        record["maximum_tip_height_difference_m"] = max(
            record["maximum_tip_height_difference_m"],
            assessment.tip_height_difference_m,
        )
    records = list(records_by_stage.values())
    rejected = [record for record in records if not record["accepted"]]
    if rejected:
        first = rejected[0]
        raise ValueError(
            "physical-right jaw-level plan audit failed at "
            f"{first['stage']}: tilt="
            f"{first['maximum_combined_tilt_deg']:.2f}deg, "
            "tip_delta="
            f"{first['maximum_tip_height_difference_m'] * 1000:.2f}mm"
        )
    return {"accepted": True, "stages": records}


def _plan(args) -> tuple[dict, Path]:
    report = search(
        SimpleNamespace(
            model=args.model,
            object_scene=args.object_scene,
            output_dir=args.output_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    )
    if report.get("accepted") is not True:
        raise RuntimeError("MuJoCo did not find a physically verified grasp")
    trajectory_path = (
        Path(args.output_dir).resolve()
        / "best_lid_grasp_trajectory.json"
    )
    return report, trajectory_path


def _close_rpc(rpc) -> None:
    if rpc is None:
        return
    socket = getattr(rpc, "socket", None)
    context = getattr(rpc, "context", None)
    if socket is not None:
        socket.close(linger=0)
    if context is not None:
        context.term()


def execute_trajectory(
    trajectory: dict,
    *,
    production_model: str | Path,
    torque_config: dict,
    rpc,
    right_visual_goal_monitor=None,
    lid_motion_guard=None,
    level_reference: JawLevelReference | None = None,
    orientation_policy_store: OrientationMonitoringPolicyStore | None = None,
    cached_orientation_monitor: CachedOrientationMonitor | None = None,
) -> dict:
    validate_demo_free_trajectory(trajectory)
    if trajectory.get("simulation_validation", {}).get("success") is not True:
        raise ValueError("hardware execution requires successful MuJoCo validation")
    level_reference = level_reference or JawLevelReference()
    plan_level_audit = audit_physical_right_level(
        trajectory,
        production_model=production_model,
        reference=level_reference,
    )
    samples = sample_joint_knots(trajectory["knots"])
    fk = ProductionRightFK(production_model)
    initial_open_ratio = float(
        np.asarray(rpc.get_right_gripper_exact(), dtype=float).reshape(-1)[0]
    )
    streamer = TeleopTrajectoryStreamer(
        rpc,
        fk,
        torque_limit_nm=torque_config["thresholds"]["right"],
        consecutive_torque_samples=int(
            torque_config.get("consecutive_samples", 5)
        ),
        enforce_torque_stop=torque_stop_enabled_from_config(torque_config),
    )
    visual_gate = {}
    level_checkpoint = RightJawLevelCheckpoint(rpc, level_reference)
    policy = (
        orientation_policy_store.load()
        if orientation_policy_store is not None
        else None
    )
    low_stages = {
        "descend_fast",
        "descend",
        "insert",
        "preclose_observe",
        "close",
    }
    descent_checkpoint_done = False

    def stage_gate(stage: str) -> None:
        nonlocal descent_checkpoint_done
        if stage in {"descend_fast", "descend"} and not descent_checkpoint_done:
            level_checkpoint.require("before_descend")
            descent_checkpoint_done = True
        if stage == "close":
            level_checkpoint.require("before_close")
            if right_visual_goal_monitor is not None:
                visual_gate.update(
                    right_visual_goal_monitor.require_close_allowed(
                        maximum_age_s=3.0
                    )
                )

    def sample_gate(stage: str, _t_s: float) -> None:
        if stage not in low_stages:
            return
        if lid_motion_guard is not None:
            lid_motion_guard.require_motion_safe()
        if (
            policy is not None
            and policy.mode == "continuous_cached"
        ):
            if cached_orientation_monitor is None:
                raise RuntimeError(
                    "continuous_cached policy requires controller cache"
                )
            cached_orientation_monitor.require_level(maximum_age_s=0.2)

    def pose_transformer(stage: str, pose):
        if stage not in low_stages:
            return pose
        # Pure geometry on the already-planned pose: no camera or robot RPC.
        # This prevents joint-space interpolation from reintroducing roll or
        # pitch between level waypoints.
        return type(pose)(
            leveled_pose(pose.parameters(), level_reference)
        )

    try:
        execution = streamer.execute(
            samples,
            stage_gate=stage_gate,
            sample_gate=sample_gate,
            pose_transformer=pose_transformer,
        )
    except RuntimeError as error:
        text = str(error)
        low_pose_failure = any(
            marker in text
            for marker in (
                "before_close",
                "motion watchdog",
                "both lid motion cameras",
                "cached jaw orientation",
            )
        )
        failure_reason = None
        if "motion watchdog" in text or "lid motion cameras" in text:
            failure_reason = "lid_lateral_motion"
        elif "jaw" in text and ("level" in text or "orientation" in text):
            failure_reason = "jaw_tilt"
        if orientation_policy_store is not None and failure_reason is not None:
            orientation_policy_store.record_failure(failure_reason)
        if low_pose_failure:
            recovery = streamer.recover_vertical_then_open(
                clearance_m=0.020,
                support_up_robot=level_reference.support_up_robot,
            )
            setattr(error, "vertical_recovery", recovery)
        raise
    final_open_ratio = execution["final_right_gripper_open_ratio"]
    obstruction_threshold = max(0.01, 0.02 * initial_open_ratio)
    execution.update(
        {
            "closure_policy": trajectory["closure_policy"],
            "initial_right_gripper_open_ratio": initial_open_ratio,
            "obstruction_threshold_open_ratio": obstruction_threshold,
            "gripper_obstruction_detected": bool(
                final_open_ratio is not None
                and final_open_ratio > obstruction_threshold
            ),
            "success_requires_visual_target_follow_confirmation": True,
            "right_visual_goal_gate": visual_gate or None,
            "physical_right_level_plan_audit": plan_level_audit,
            "physical_right_level_checkpoints": level_checkpoint.records,
            "orientation_monitoring_mode": (
                "checkpoint" if policy is None else policy.mode
            ),
        }
    )
    return execution


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--object-scene")
    parser.add_argument("--trajectory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--production-model",
        default="robot/cone-e-description/robot-welded-base-and-lift.mjcf",
    )
    parser.add_argument(
        "--torque-config",
        default="src/configs/pasteur_lid_torque.json",
    )
    parser.add_argument(
        "--level-config",
        default="src/configs/pasteur_fast_lid_grasp_level.json",
    )
    parser.add_argument(
        "--orientation-policy-state",
        default=(
            "data/runs/pasteur/fast_lid_grasp_orientation_policy.json"
        ),
    )
    parser.add_argument(
        "--right-visual-goal-selection",
        default="src/configs/pasteur_grasp_window_selection.json",
    )
    parser.add_argument(
        "--task-config",
        default="src/configs/pasteur_lid_sam_task.json",
    )
    parser.add_argument(
        "--semantic-scene",
        help=(
            "operator-confirmed SAM-first semantic scene used to lock the "
            "target instance before wrist visual search"
        ),
    )
    parser.add_argument(
        "--sam-endpoint",
        default="tcp://127.0.0.1:15563",
    )
    parser.add_argument(
        "--disable-right-visual-goal",
        action="store_true",
        help="diagnostic only: execute without the right-wrist close gate",
    )
    parser.add_argument("--rpc-host", default="localhost")
    parser.add_argument("--rpc-port", type=int, default=8081)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send the accepted plan to the physical right arm",
    )
    args = parser.parse_args(argv)
    if args.trajectory:
        if args.model or args.object_scene:
            parser.error(
                "--trajectory cannot be combined with --model/--object-scene"
            )
        trajectory_path = Path(args.trajectory).resolve()
        planning_report = None
    else:
        if not args.model or not args.object_scene:
            parser.error(
                "provide --trajectory or both --model and --object-scene"
            )
        planning_report, trajectory_path = _plan(args)
    trajectory = _load_json(trajectory_path)
    validate_demo_free_trajectory(trajectory)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": RUN_SCHEMA,
        "created_at_s": time.time(),
        "trajectory": str(trajectory_path),
        "trajectory_schema": trajectory["schema"],
        "demonstration_used": False,
        "planning_report": planning_report,
        "hardware_execution_requested": bool(args.execute),
        "hardware_execution_started": False,
        "commands_sent": False,
        "execution": None,
        "failure": None,
    }
    rpc = None
    right_visual_goal_monitor = None
    execution_error = None
    if args.execute:
        try:
            if not args.disable_right_visual_goal:
                from rollout.right_visual_goal_monitor import (
                    RightVisualGoalMonitor,
                )

                if not args.semantic_scene:
                    raise ValueError(
                        "--execute requires --semantic-scene so the wrist "
                        "monitor cannot relabel a nearby dish as the lid"
                    )
                task = _load_json(args.task_config)
                identity = select_unique_scene_target(
                    _load_json(args.semantic_scene),
                    semantic_role=task["target"]["semantic_role"],
                )
                result["locked_target_identity"] = {
                    "semantic_role": identity.semantic_role,
                    "instance_id": identity.instance_id,
                    "semantic_name": identity.semantic_name,
                }
                right_visual_goal_monitor = RightVisualGoalMonitor.from_files(
                    sam_endpoint=args.sam_endpoint,
                    output_dir=output_dir / "right_visual_goal",
                    selection_path=args.right_visual_goal_selection,
                    task_path=args.task_config,
                    locked_instance_id=identity.instance_id,
                )
                right_visual_goal_monitor.start()
            rpc = RPCClient(
                args.rpc_host,
                args.rpc_port,
                timeout_ms=10000,
            )
            result["hardware_execution_started"] = True
            # execute_trajectory prepares the physical controller before its
            # first sampled waypoint (measured-pose latch, gains, MIT mode).
            # From this point onward a failure journal must conservatively
            # state that hardware commands were sent, including a visual-gate
            # rejection after the open approach.
            result["commands_sent"] = True
            result["execution"] = execute_trajectory(
                trajectory,
                production_model=args.production_model,
                torque_config=_load_json(args.torque_config),
                rpc=rpc,
                right_visual_goal_monitor=right_visual_goal_monitor,
                level_reference=_load_level_reference(args.level_config),
                orientation_policy_store=(
                    OrientationMonitoringPolicyStore(
                        args.orientation_policy_state
                    )
                ),
            )
        except BaseException as error:
            execution_error = error
            result["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            recovery = getattr(error, "vertical_recovery", None)
            if recovery is not None:
                result["failure"]["vertical_recovery"] = recovery
        finally:
            try:
                if right_visual_goal_monitor is not None:
                    right_visual_goal_monitor.stop()
            except BaseException as cleanup_error:
                if result["failure"] is None:
                    execution_error = cleanup_error
                    result["failure"] = {
                        "type": type(cleanup_error).__name__,
                        "message": str(cleanup_error),
                        "traceback": traceback.format_exc(),
                    }
                else:
                    result["failure"]["monitor_cleanup_error"] = repr(
                        cleanup_error
                    )
            finally:
                _close_rpc(rpc)
    report_path = output_dir / "autonomous_shape_lid_grasp_run.json"
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if execution_error is not None:
        raise execution_error


if __name__ == "__main__":
    main()

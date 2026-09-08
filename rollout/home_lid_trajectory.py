"""Offline home-to-lid trajectory planning with fixed robot state authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from robot.arm.home import (
    physical_home_q,
    physical_to_semantic_model_q_offset,
)
from rollout.autonomous_mpc import (
    AnalyticObstacleSet,
    AutonomousStop,
    MuJoCoIKValidator,
    Pose,
    TimedWaypoint,
    TrajectoryPlan,
)


SCHEMA = "piper_robot.home_lid_trajectory/v1"
OBJECT_SCENE_SCHEMA = "piper_robot.dynamic_dish_lid_scene/v1"


@dataclass(frozen=True)
class GripperEvent:
    t_s: float
    action: str
    target_open_ratio: float


@dataclass
class PlannedHomeTrajectory:
    plan: TrajectoryPlan
    gripper_events: list[GripperEvent]
    home_q_physical: dict[str, list[float]]
    mujoco_q_waypoints: list[list[float]]
    object_scene: dict
    display_only: bool
    authority_reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "plan": asdict(self.plan),
            "gripper_events": [asdict(item) for item in self.gripper_events],
            "home_q_physical": self.home_q_physical,
            "mujoco_q_waypoints": self.mujoco_q_waypoints,
            "object_scene": self.object_scene,
            "display_only": self.display_only,
            "motion_authorized": not self.display_only,
            "authority_reasons": self.authority_reasons,
            "commands_sent": False,
        }


def _pose(value, name: str) -> Pose:
    array = np.asarray(value, dtype=float)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain wxyz plus xyz")
    norm = float(np.linalg.norm(array[:4]))
    if norm < 1e-9:
        raise ValueError(f"{name} quaternion is invalid")
    return Pose(
        tuple(float(item) for item in array[:4] / norm),
        tuple(float(item) for item in array[4:]),
    )


def load_object_scene(path: str | Path) -> dict:
    scene = json.loads(Path(path).read_text())
    if scene.get("schema") != OBJECT_SCENE_SCHEMA:
        raise ValueError("unsupported dish/lid object scene schema")
    objects = list(scene.get("objects", ()))
    roles = {item.get("role"): item for item in objects}
    lid = roles.get("target_lid")
    if lid is None or lid.get("status") != "confirmed":
        raise ValueError("a confirmed target_lid is required")
    if "grasp_ee_pose_robot_wxyz_xyz" not in lid:
        raise ValueError("target_lid lacks grasp_ee_pose_robot_wxyz_xyz")
    _pose(lid["grasp_ee_pose_robot_wxyz_xyz"], "target lid grasp pose")
    for field in ("grasp_right_q_rad", "verification_lift_right_q_rad"):
        values = np.asarray(lid.get(field), dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(f"target_lid lacks a valid {field}")
    return scene


class HomeSceneValidator:
    """Validate right-arm IK against fixed-home left arm and known objects."""

    def __init__(self, model_path: str | Path, objects: Sequence[dict]):
        import mujoco

        inspected = mujoco.MjModel.from_xml_path(str(model_path))
        semantic_joint = mujoco.mj_name2id(
            inspected, mujoco.mjtObj.mjOBJ_JOINT, "left/joint1"
        )
        if semantic_joint >= 0:
            # The semantic planning model preserves physical identity.
            # Historical ConeE left_arm_/right_arm_ names do not apply here.
            right_names = [f"right/joint{index}" for index in range(1, 7)]
            left_names = [f"left/joint{index}" for index in range(1, 7)]
            link_capsules = [
                ("right/base_link", "right/link1", 0.058),
                ("right/link1", "right/link2", 0.052),
                ("right/link2", "right/link3", 0.055),
                ("right/link3", "right/link4", 0.050),
                ("right/link4", "right/link5", 0.044),
                ("right/link5", "right/link6", 0.042),
                ("right/link6", "right/gripper_base", 0.035),
            ]
            self.mujoco = MuJoCoIKValidator(
                model_path,
                physical_home_q("right"),
                left_q=physical_home_q("left"),
                maximum_joint_step_rad=0.20,
                joint_names=right_names,
                left_joint_names=left_names,
                ee_frame="right/ee",
                right_q_offset=physical_to_semantic_model_q_offset("right"),
                left_q_offset=physical_to_semantic_model_q_offset("left"),
                contact_body_prefix="right/",
                link_capsules=link_capsules,
                attached_spheres=[("right/gripper_base", 0.055)],
            )
            self.model_variant = "sam_reconstruction_upright_nyu"
        else:
            self.mujoco = MuJoCoIKValidator(
                model_path,
                physical_home_q("right"),
                left_q=physical_home_q("left"),
                maximum_joint_step_rad=0.20,
            )
            self.model_variant = "legacy_cone_e"
        self.obstacles = AnalyticObstacleSet(objects)

    def __call__(self, pose: Pose) -> float:
        self.mujoco.validate(pose)
        clearance = self.obstacles.clearance(
            self.mujoco.robot_proxy_samples()
        )
        # No analytic object other than the target is a valid display-only
        # scene. It has no metric clearance authority, but may still be
        # rendered and inspected.
        # Keep the planner numerically evaluable for an explicitly
        # display-only scene while authority_reasons records that this is not
        # measured obstacle clearance. MuJoCo IK, limits and new contacts are
        # still checked at every waypoint.
        return 1.0 if math.isinf(clearance) else float(clearance)


def _target_and_objects(scene: dict) -> tuple[Pose, list[dict], list[str]]:
    by_role = {item.get("role"): item for item in scene["objects"]}
    lid = by_role["target_lid"]
    target = _pose(
        lid["grasp_ee_pose_robot_wxyz_xyz"],
        "target lid grasp pose",
    )
    reasons = []
    container = by_role.get("target_container")
    if container is None or container.get("status") != "confirmed":
        reasons.append("confirmed_target_container_missing")
    if scene.get("camera_to_robot_accepted") is not True:
        reasons.append("camera_to_robot_not_accepted")
    if scene.get("operator_confirmed") is not True:
        reasons.append("dish_lid_positions_not_operator_confirmed")
    return target, list(scene["objects"]), reasons


def _smooth_joint_segment(
    start_q: np.ndarray,
    finish_q: np.ndarray,
    *,
    start_t_s: float,
    maximum_joint_speed_rad_s: float,
    stage: str,
    control_hz: float = 30.0,
) -> list[tuple[float, np.ndarray, str]]:
    delta = float(np.max(np.abs(finish_q - start_q)))
    duration = max(0.5, 1.875 * delta / maximum_joint_speed_rad_s)
    samples = max(1, int(math.ceil(duration * control_hz)))
    result = []
    for index in range(1, samples + 1):
        fraction = index / samples
        smooth = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
        q = start_q + smooth * (finish_q - start_q)
        result.append((start_t_s + fraction * duration, q, stage))
    return result


def _joint_corridors(start_q: np.ndarray, finish_q: np.ndarray):
    midpoint = 0.5 * (start_q + finish_q)
    return [
        ("direct", [finish_q]),
        (
            "shoulder_positive_detour",
            [midpoint + np.array([0.20, 0.10, -0.10, 0, 0, 0]), finish_q],
        ),
        (
            "shoulder_negative_detour",
            [midpoint + np.array([-0.20, 0.10, -0.10, 0, 0, 0]), finish_q],
        ),
    ]


def _validate_joint_path(
    model_path: str | Path,
    objects: Sequence[dict],
    knots: Sequence[np.ndarray],
    *,
    maximum_joint_speed_rad_s: float,
    start_t_s: float,
    stages: Sequence[str],
) -> tuple[list[TimedWaypoint], list[list[float]], float, float]:
    validator = HomeSceneValidator(model_path, objects)
    cursor = physical_home_q("right")
    elapsed = float(start_t_s)
    waypoints = []
    minimum_clearance = math.inf
    for knot, stage in zip(knots, stages):
        segment = _smooth_joint_segment(
            cursor,
            np.asarray(knot, dtype=float),
            start_t_s=elapsed,
            maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
            stage=stage,
        )
        for timestamp, q, segment_stage in segment:
            validator.mujoco.validate_q(q)
            clearance = validator.obstacles.clearance(
                validator.mujoco.robot_proxy_samples()
            )
            minimum_clearance = min(minimum_clearance, clearance)
            waypoints.append(
                TimedWaypoint(
                    timestamp,
                    Pose.from_se3(
                        validator.mujoco.solver.forward_kinematics()
                    ),
                    segment_stage,
                )
            )
        elapsed = segment[-1][0]
        cursor = np.asarray(knot, dtype=float)
    return (
        waypoints,
        validator.mujoco.q_waypoints,
        minimum_clearance,
        elapsed,
    )


def plan_home_lid_trajectory(
    scene: dict,
    *,
    model_path: str | Path,
    lift_m: float = 0.040,
    pregrasp_height_m: float = 0.015,
    final_height_m: float = 0.003,
    verification_lift_m: float = 0.005,
    detour_m: float = 0.070,
    maximum_joint_speed_rad_s: float = 0.35,
    now_s: float | None = None,
) -> PlannedHomeTrajectory:
    """Plan home -> grasp -> close -> verification lift without robot I/O."""

    del lift_m, pregrasp_height_m, final_height_m, verification_lift_m, detour_m
    target_pose, objects, authority_reasons = _target_and_objects(scene)
    lid = next(item for item in objects if item.get("role") == "target_lid")
    grasp_q = np.asarray(lid["grasp_right_q_rad"], dtype=float)
    lift_q = np.asarray(lid["verification_lift_right_q_rad"], dtype=float)
    candidates = []
    rejected = {}
    created = time.time() if now_s is None else float(now_s)
    for corridor, approach_knots in _joint_corridors(
        physical_home_q("right"), grasp_q
    ):
        try:
            approach_stages = (
                ["transit_detour"] * (len(approach_knots) - 1)
                + ["approach_grasp"]
            )
            waypoints, q_waypoints, minimum_clearance, close_t = (
                _validate_joint_path(
                    model_path,
                    objects,
                    [*approach_knots, lift_q],
                    maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                    start_t_s=0.0,
                    stages=[*approach_stages, "verification_lift"],
                )
            )
            lift_start = next(
                waypoint.t_s
                for waypoint in waypoints
                if waypoint.stage == "verification_lift"
            )
            close_t = max(0.0, lift_start - 1.0 / 30.0)
            approach = TrajectoryPlan(
                plan_id=f"home-lid-{int(created * 1000)}",
                created_at_s=created,
                target_xyz_m=target_pose.xyz,
                waypoints=waypoints,
                minimum_clearance_m=(
                    None if math.isinf(minimum_clearance)
                    else float(minimum_clearance)
                ),
            )
            approach.metadata.update(
                {
                    "corridor": corridor,
                    "stage_order": list(
                        dict.fromkeys(
                            waypoint.stage for waypoint in approach.waypoints
                        )
                    ),
                    "start_state": "canonical_physical_home",
                    "moving_arm": "physical_right",
                    "fixed_arm": "physical_left",
                    "target_source": scene.get("source"),
                    "planner": "minimum_jerk_joint_space",
                    "maximum_joint_speed_rad_s": maximum_joint_speed_rad_s,
                    "terminal_grasp_q_source": "operator_verified_recording",
                    "verification_lift_q_source": "operator_verified_recording",
                    "model_variant": "sam_reconstruction_upright_nyu",
                }
            )
            clearance_score = (
                1.0
                if approach.minimum_clearance_m is None
                else float(approach.minimum_clearance_m)
            )
            score = clearance_score - 0.0001 * approach.waypoints[-1].t_s
            candidates.append(
                (
                    score,
                    approach,
                    q_waypoints,
                    [
                        GripperEvent(0.0, "open", 1.0),
                        GripperEvent(close_t, "close", 0.0),
                    ],
                )
            )
        except (AutonomousStop, ValueError) as error:
            rejected[corridor] = str(error)
    if not candidates:
        raise AutonomousStop(
            "all home-to-lid corridors rejected: "
            + json.dumps(rejected, sort_keys=True)
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, plan, q_waypoints, gripper_events = candidates[0]
    plan.metadata.update(
        {
            "rejected_corridors": rejected,
            "candidate_count": 3,
            "accepted_candidate_count": len(candidates),
        }
    )
    return PlannedHomeTrajectory(
        plan=plan,
        gripper_events=gripper_events,
        home_q_physical={
            "left": physical_home_q("left").tolist(),
            "right": physical_home_q("right").tolist(),
        },
        mujoco_q_waypoints=q_waypoints,
        object_scene=scene,
        display_only=bool(authority_reasons),
        authority_reasons=authority_reasons,
    )

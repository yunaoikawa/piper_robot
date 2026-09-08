"""Reconstruct a time-indexed MuJoCo replay from measured stopped states."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from robot.arm.home import (
    physical_home_q,
    physical_to_semantic_model_q_offset,
    semantic_model_home_q,
)


SCHEMA = "piper_robot.measured_keyframe_trajectory_replay/v1"


def _minimum_jerk(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def _load_measured_keyframe(item: dict) -> dict:
    capture = Path(item["capture"]).resolve()
    manifest = json.loads((capture / "manifest.json").read_text())
    state = manifest.get("robot_state", {})
    if state.get("commands_sent") is not False:
        raise ValueError(f"unsafe capture provenance: {capture}")
    if state.get("stability", {}).get("stationary") is not True:
        raise ValueError(f"trajectory keyframe was not stationary: {capture}")
    after = state["after"]
    q = np.asarray(after["right_joint_positions_rad"], dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError(f"invalid measured right q in {capture}")
    return {
        "name": str(item["name"]),
        "stage": str(item.get("stage", item["name"])),
        "source_kind": "stopped_rgbd_capture_with_read_only_robot_state",
        "capture": str(capture),
        "session_id": manifest["session_id"],
        "capture_time_utc": manifest["created_at_utc"],
        "right_q_physical_rad": q.tolist(),
        "right_gripper_open_ratio": float(
            after["right_gripper_open_ratio"]
        ),
        "measured": True,
    }


def _robot_environment_contacts(
    model,
    data,
    *,
    moving_body_prefix: str,
) -> list[dict]:
    records = []
    for contact in data.contact:
        first_body = model.body(model.geom_bodyid[contact.geom1]).name
        second_body = model.body(model.geom_bodyid[contact.geom2]).name
        first_robot = first_body.startswith(("left/", "right/"))
        second_robot = second_body.startswith(("left/", "right/"))
        if first_robot == second_robot:
            continue
        records.append(
            {
                "robot_body": first_body if first_robot else second_body,
                "environment_body": second_body if first_robot else first_body,
                "penetration_depth_m": max(0.0, -float(contact.dist)),
                "moving_arm_contact": (
                    (first_body if first_robot else second_body).startswith(
                        moving_body_prefix
                    )
                ),
            }
        )
    return records


def _model_mapping(
    model,
    keyframe: str,
    physical_right_model_branch: str,
):
    import mujoco

    if mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{physical_right_model_branch}/joint1",
    ) < 0:
        raise ValueError("recorded replay requires semantic Piper joint CAD")
    physical_left_model_branch = (
        "right" if physical_right_model_branch == "left" else "left"
    )
    right_ids = [
        int(
            model.joint(
                f"{physical_right_model_branch}/joint{index}"
            ).qposadr[0]
        )
        for index in range(1, 7)
    ]
    left_ids = [
        int(
            model.joint(
                f"{physical_left_model_branch}/joint{index}"
            ).qposadr[0]
        )
        for index in range(1, 7)
    ]
    key_id = int(model.key(keyframe).id)
    return (
        right_ids,
        left_ids,
        key_id,
        physical_right_model_branch,
        physical_left_model_branch,
    )


def _segment_duration(
    start_q: np.ndarray,
    finish_q: np.ndarray,
    maximum_joint_speed_rad_s: float,
    minimum_duration_s: float,
) -> float:
    # A quintic minimum-jerk time law has peak scalar velocity 1.875 / T.
    peak_delta = float(np.max(np.abs(finish_q - start_q)))
    return max(
        minimum_duration_s,
        1.875 * peak_delta / maximum_joint_speed_rad_s,
    )


class _JointTree:
    def __init__(self, value: np.ndarray):
        self.values = [value.copy()]
        self.parents = [-1]

    def nearest(self, target: np.ndarray) -> int:
        return int(
            np.argmin(
                [np.linalg.norm(value - target) for value in self.values]
            )
        )

    def add(self, value: np.ndarray, parent: int) -> int:
        self.values.append(value.copy())
        self.parents.append(parent)
        return len(self.values) - 1

    def path(self, index: int) -> list[np.ndarray]:
        result = []
        while index >= 0:
            result.append(self.values[index])
            index = self.parents[index]
        return result[::-1]


def _plan_collision_free_joint_path(
    model,
    data,
    moving_ids: list[int],
    static_ids: list[int],
    static_q: np.ndarray,
    moving_branch: str,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    seed: int,
    edge_step_rad: float,
    extension_step_rad: float,
    maximum_iterations: int,
) -> tuple[list[np.ndarray], dict]:
    """Deterministic bidirectional RRT-Connect followed by shortcutting."""

    import mujoco

    cache = {}

    def valid(q: np.ndarray) -> bool:
        key = tuple(np.round(q, 5))
        if key in cache:
            return cache[key]
        data.qpos[moving_ids] = q
        data.qpos[static_ids] = static_q
        mujoco.mj_forward(model, data)
        result = not any(
            item["moving_arm_contact"]
            for item in _robot_environment_contacts(
                model,
                data,
                moving_body_prefix=f"{moving_branch}/",
            )
        )
        cache[key] = result
        return result

    def edge_valid(first: np.ndarray, second: np.ndarray) -> bool:
        count = max(
            1,
            int(
                math.ceil(
                    float(np.max(np.abs(second - first)))
                    / edge_step_rad
                )
            ),
        )
        return all(
            valid(first + (index / count) * (second - first))
            for index in range(1, count + 1)
        )

    if not valid(start) or not valid(goal):
        raise ValueError("measured trajectory endpoint is in collision")
    if edge_valid(start, goal):
        return [start.copy(), goal.copy()], {
            "planner": "direct",
            "iterations": 0,
            "collision_checks": len(cache),
        }
    ranges = np.asarray(
        [
            model.jnt_range[
                int(np.flatnonzero(model.jnt_qposadr == index)[0])
            ]
            for index in moving_ids
        ],
        dtype=float,
    )
    lower, upper = ranges[:, 0], ranges[:, 1]
    generator = np.random.default_rng(seed)
    first_tree = _JointTree(start)
    second_tree = _JointTree(goal)
    swapped = False
    result_path = None

    def extend(tree: _JointTree, target: np.ndarray):
        near_index = tree.nearest(target)
        near = tree.values[near_index]
        delta = target - near
        distance = float(np.linalg.norm(delta))
        candidate = (
            target
            if distance <= extension_step_rad
            else near + extension_step_rad * delta / distance
        )
        if not edge_valid(near, candidate):
            return None, False
        index = tree.add(candidate, near_index)
        return index, distance <= extension_step_rad

    iteration = -1
    for iteration in range(maximum_iterations):
        if generator.random() < 0.80:
            fraction = generator.random()
            center = start + fraction * (goal - start)
            spread = np.maximum(0.35, 0.40 * np.abs(goal - start))
            target = center + generator.normal(0.0, spread)
        else:
            target = generator.uniform(lower, upper)
        target = np.clip(target, lower, upper)
        first_index, _ = extend(first_tree, target)
        if first_index is not None:
            connection = first_tree.values[first_index]
            while True:
                second_index, reached = extend(second_tree, connection)
                if second_index is None:
                    break
                if reached:
                    first_path = first_tree.path(first_index)
                    second_path = second_tree.path(second_index)
                    result_path = (
                        first_path + second_path[::-1][1:]
                        if not swapped
                        else second_path + first_path[::-1][1:]
                    )
                    break
            if result_path is not None:
                break
        first_tree, second_tree = second_tree, first_tree
        swapped = not swapped
    if result_path is None:
        raise ValueError(
            "no collision-free joint path found between measured endpoints"
        )
    original_nodes = len(result_path)
    shortcut_attempts = int(min(3000, 100 * original_nodes))
    for _ in range(shortcut_attempts):
        if len(result_path) <= 2:
            break
        first_index, second_index = sorted(
            generator.choice(len(result_path), 2, replace=False)
        )
        if (
            second_index > first_index + 1
            and edge_valid(
                result_path[first_index],
                result_path[second_index],
            )
        ):
            result_path = (
                result_path[:first_index + 1]
                + result_path[second_index:]
            )
    return result_path, {
        "planner": "bidirectional_rrt_connect",
        "iterations": iteration + 1,
        "collision_checks": len(cache),
        "nodes_before_shortcut": original_nodes,
        "nodes_after_shortcut": len(result_path),
        "seed": seed,
        "edge_step_rad": edge_step_rad,
        "extension_step_rad": extension_step_rad,
    }


def build_recorded_replay(
    *,
    model_path: str | Path,
    object_scene_path: str | Path,
    measured_keyframes: Iterable[dict],
    output_path: str | Path,
    keyframe: str = "home",
    control_hz: float = 30.0,
    maximum_joint_speed_rad_s: float = 0.35,
    minimum_segment_duration_s: float = 0.5,
    physical_right_model_branch: str = "right",
    planning_seed: int = 22,
    planning_seed_candidates: Iterable[int] | None = None,
    planning_edge_step_rad: float = 0.035,
    planning_extension_step_rad: float = 0.12,
    planning_maximum_iterations: int = 20000,
    allow_colliding_display_only: bool = False,
) -> dict:
    """Build home -> exact measured states with an audited reconstructed path.

    By default every segment must be collision-free. The explicit
    ``allow_colliding_display_only`` mode exists only to visualize a rejected
    scene: it connects exact stopped states directly, audits every sample, and
    keeps motion authority false.
    """

    import mujoco

    model_path = Path(model_path).resolve()
    object_scene_path = Path(object_scene_path).resolve()
    measured = [_load_measured_keyframe(item) for item in measured_keyframes]
    if not measured:
        raise ValueError("at least one measured keyframe is required")
    home_q = physical_home_q("right")
    keyframes = [
        {
            "name": "home",
            "stage": "home",
            "source_kind": "repository_physical_home_q_plus_user_home_assertion",
            "capture": None,
            "session_id": None,
            "capture_time_utc": None,
            "right_q_physical_rad": home_q.tolist(),
            "right_gripper_open_ratio": 1.0,
            "measured": False,
        },
        *measured,
    ]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if physical_right_model_branch not in {"left", "right"}:
        raise ValueError("physical right model branch must be left or right")
    (
        right_ids,
        left_ids,
        key_id,
        moving_branch,
        static_branch,
    ) = _model_mapping(model, keyframe, physical_right_model_branch)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    # This semantic model preserves physical identity: physical-right maps to
    # right/ and physical-left maps to left/. Production ConeE branch names
    # are intentionally not reused here. Joint coordinates use the physical
    # Piper values directly; old MJCF zero offsets are not applied.
    model_home_right = semantic_model_home_q("right")
    model_home_left = semantic_model_home_q("left")
    data.qpos[right_ids] = model_home_right
    data.qpos[left_ids] = model_home_left
    mujoco.mj_forward(model, data)
    right_offset = physical_to_semantic_model_q_offset("right")
    samples = []
    cursor_s = 0.0
    maximum_contact_count = 0
    maximum_moving_contact_count = 0
    maximum_penetration_m = 0.0
    contact_pairs = set()
    segment_records = []

    def append_sample(
        timestamp_s: float,
        q_physical: np.ndarray,
        gripper: float,
        stage: str,
        measured_endpoint: bool,
        source_session_id: str | None,
    ) -> None:
        nonlocal maximum_contact_count
        nonlocal maximum_moving_contact_count, maximum_penetration_m
        q_model = q_physical + right_offset
        data.qpos[right_ids] = q_model
        data.qpos[left_ids] = model_home_left
        mujoco.mj_forward(model, data)
        contacts = _robot_environment_contacts(
            model,
            data,
            moving_body_prefix=f"{moving_branch}/",
        )
        moving_contacts = [
            item for item in contacts if item["moving_arm_contact"]
        ]
        maximum_contact_count = max(maximum_contact_count, len(contacts))
        maximum_moving_contact_count = max(
            maximum_moving_contact_count,
            len(moving_contacts),
        )
        for contact in moving_contacts:
            maximum_penetration_m = max(
                maximum_penetration_m,
                contact["penetration_depth_m"],
            )
            contact_pairs.add(
                (
                    contact["robot_body"],
                    contact["environment_body"],
                )
            )
        previous = (
            np.asarray(samples[-1]["right_q_physical_rad"], dtype=float)
            if samples
            else q_physical
        )
        samples.append(
            {
                "t_s": float(timestamp_s),
                "stage": stage,
                "right_q_physical_rad": q_physical.tolist(),
                "right_q_model_rad": q_model.tolist(),
                "right_q_increment_rad": (q_physical - previous).tolist(),
                "right_gripper_open_ratio": float(gripper),
                "measured_endpoint": bool(measured_endpoint),
                "source_session_id": source_session_id,
                "robot_environment_contact_count": len(contacts),
                "moving_arm_environment_contact_count": len(moving_contacts),
            }
        )

    append_sample(
        0.0,
        home_q,
        1.0,
        "home",
        True,
        None,
    )
    planner_records = []
    display_only_collision_fallback_used = False
    for pair_index, (start, finish) in enumerate(
        zip(keyframes, keyframes[1:])
    ):
        start_q = np.asarray(start["right_q_physical_rad"], dtype=float)
        finish_q = np.asarray(finish["right_q_physical_rad"], dtype=float)
        seeds = (
            [int(value) + pair_index for value in planning_seed_candidates]
            if planning_seed_candidates is not None
            else [planning_seed + pair_index]
        )
        candidate_plans = []
        for seed in seeds:
            try:
                candidate_path, candidate_record = (
                    _plan_collision_free_joint_path(
                        model,
                        data,
                        right_ids,
                        left_ids,
                        model_home_left,
                        moving_branch,
                        start_q,
                        finish_q,
                        seed=seed,
                        edge_step_rad=planning_edge_step_rad,
                        extension_step_rad=planning_extension_step_rad,
                        maximum_iterations=planning_maximum_iterations,
                    )
                )
            except ValueError as error:
                candidate_plans.append(
                    {
                        "seed": seed,
                        "accepted": False,
                        "error": str(error),
                    }
                )
                continue
            length = float(
                sum(
                    np.linalg.norm(second - first)
                    for first, second in zip(
                        candidate_path,
                        candidate_path[1:],
                    )
                )
            )
            candidate_record = {
                **candidate_record,
                "seed": seed,
                "accepted": True,
                "joint_path_length_rad": length,
            }
            candidate_plans.append(candidate_record)
            if candidate_record["planner"] == "direct":
                break
        accepted_candidates = [
            item for item in candidate_plans if item["accepted"]
        ]
        if not accepted_candidates:
            if not allow_colliding_display_only:
                raise ValueError(
                    f"all planner starts failed for {start['name']} -> "
                    f"{finish['name']}"
                )
            display_only_collision_fallback_used = True
            selected_seed = None
            path = [start_q.copy(), finish_q.copy()]
            planner_record = {
                "planner": "display_only_direct_interpolation",
                "iterations": 0,
                "collision_checks": 0,
                "rejected_for_motion": True,
                "failure_reason": (
                    "no collision-free plan; direct interpolation retained "
                    "only for diagnostic rendering"
                ),
            }
        else:
            selected = min(
                accepted_candidates,
                key=lambda item: item["joint_path_length_rad"],
            )
            selected_seed = int(selected["seed"])
            path, planner_record = _plan_collision_free_joint_path(
                model,
                data,
                right_ids,
                left_ids,
                model_home_left,
                moving_branch,
                start_q,
                finish_q,
                seed=selected_seed,
                edge_step_rad=planning_edge_step_rad,
                extension_step_rad=planning_extension_step_rad,
                maximum_iterations=planning_maximum_iterations,
            )
        planner_record.update(
            {
                "seed": selected_seed,
                "from": start["name"],
                "to": finish["name"],
                "joint_path_length_rad": float(
                    sum(
                        np.linalg.norm(second - first)
                        for first, second in zip(path, path[1:])
                    )
                ),
                "knots_physical_q_rad": [
                    value.tolist() for value in path
                ],
                "selection_policy": (
                    "minimum_joint_path_length_across_deterministic_seeds"
                ),
                "candidate_plans": candidate_plans,
            }
        )
        planner_records.append(planner_record)
        edge_lengths = np.asarray(
            [
                np.linalg.norm(second - first)
                for first, second in zip(path, path[1:])
            ],
            dtype=float,
        )
        total_length = float(np.sum(edge_lengths))
        completed_length = 0.0
        for edge_index, (edge_start, edge_finish) in enumerate(
            zip(path, path[1:])
        ):
            duration = _segment_duration(
                edge_start,
                edge_finish,
                maximum_joint_speed_rad_s,
                minimum_segment_duration_s,
            )
            count = max(1, int(math.ceil(duration * control_hz)))
            segment_start = cursor_s
            for index in range(1, count + 1):
                fraction = index / count
                blend = _minimum_jerk(fraction)
                q = edge_start + blend * (edge_finish - edge_start)
                path_fraction = (
                    completed_length + blend * edge_lengths[edge_index]
                ) / max(total_length, 1e-12)
                gripper = (
                    float(start["right_gripper_open_ratio"])
                    + path_fraction
                    * (
                        float(finish["right_gripper_open_ratio"])
                        - float(start["right_gripper_open_ratio"])
                    )
                )
                final_endpoint = (
                    edge_index == len(path) - 2 and index == count
                )
                append_sample(
                    segment_start + fraction * duration,
                    q,
                    gripper,
                    (
                        finish["stage"]
                        if final_endpoint
                        else (
                            f"display_only_colliding_transit_to_"
                            f"{finish['name']}"
                            if planner_record["planner"]
                            == "display_only_direct_interpolation"
                            else f"collision_free_transit_to_{finish['name']}"
                        )
                    ),
                    final_endpoint,
                    finish["session_id"] if final_endpoint else None,
                )
            cursor_s += duration
            completed_length += edge_lengths[edge_index]
            segment_records.append(
                {
                    "from": (
                        start["name"]
                        if edge_index == 0
                        else f"{start['name']}_detour_{edge_index}"
                    ),
                    "to": (
                        finish["name"]
                        if edge_index == len(path) - 2
                        else f"{start['name']}_detour_{edge_index + 1}"
                    ),
                    "endpoint_pair": [start["name"], finish["name"]],
                    "duration_s": duration,
                    "sample_count": count,
                    "maximum_joint_delta_rad": float(
                        np.max(np.abs(edge_finish - edge_start))
                    ),
                    "joint_path_length_rad": float(
                        np.linalg.norm(edge_finish - edge_start)
                    ),
                    "spatial_policy": (
                        "display_only_direct_interpolation"
                        if planner_record["planner"]
                        == "display_only_direct_interpolation"
                        else "collision_free_shortcut_joint_segment"
                    ),
                    "time_policy": (
                        "quintic_minimum_jerk_zero_endpoint_velocity"
                    ),
                }
            )
    exact_endpoint_errors = []
    endpoint_samples = [item for item in samples if item["measured_endpoint"]]
    for key, sample in zip(keyframes, endpoint_samples):
        exact_endpoint_errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(key["right_q_physical_rad"])
                        - np.asarray(sample["right_q_physical_rad"])
                    )
                )
            )
        )
    object_scene = json.loads(object_scene_path.read_text())
    moving_arm_path_clear = maximum_moving_contact_count == 0
    global_scene_home_clear = (
        samples[0]["robot_environment_contact_count"] == 0
    )
    report = {
        "schema": SCHEMA,
        "commands_sent": False,
        "observation_only": True,
        "model_path": str(model_path),
        "object_scene_path": str(object_scene_path),
        "object_scene": object_scene,
        "home_keyframe": keyframe,
        "home_left_q_model_rad": model_home_left.tolist(),
        "home_right_q_model_rad": model_home_right.tolist(),
        "home_right_q_physical_rad": home_q.tolist(),
        "right_physical_to_model_q_offset_rad": right_offset.tolist(),
        "physical_to_model_branch": {
            "right": moving_branch,
            "left": static_branch,
        },
        "keyframes": keyframes,
        "segments": segment_records,
        "planners": planner_records,
        "samples": samples,
        "duration_s": cursor_s,
        "control_hz": control_hz,
        "maximum_joint_speed_rad_s": maximum_joint_speed_rad_s,
        "trajectory_provenance": {
            "continuous_joint_log_available": False,
            "measured_states_exact": True,
            "between_state_motion_reconstructed": True,
            "display_only_collision_fallback_used": (
                display_only_collision_fallback_used
            ),
            "reconstruction_policy": (
                (
                    "When collision-free planning fails, directly interpolate "
                    "exact measured endpoints only for rejected diagnostic "
                    "rendering; audit all contacts and never authorize motion."
                    if display_only_collision_fallback_used
                    else
                    "Direct edge when collision-free; otherwise the shortest "
                    "collision-free shortcut path across configured "
                    "deterministic bidirectional RRT-Connect seeds, with a "
                    "quintic minimum-jerk time law per edge."
                )
                + " This reconstructs between exact measured endpoints and "
                "is not a claim that original teleop samples were recorded."
            ),
        },
        "validation": {
            "exact_keyframe_max_errors_rad": exact_endpoint_errors,
            "all_keyframes_exact": max(exact_endpoint_errors) <= 1e-12,
            "maximum_robot_environment_contact_count": (
                maximum_contact_count
            ),
            "maximum_moving_arm_environment_contact_count": (
                maximum_moving_contact_count
            ),
            "maximum_robot_environment_penetration_m": (
                maximum_penetration_m
            ),
            "contact_pairs": [
                {"robot_body": first, "environment_body": second}
                for first, second in sorted(contact_pairs)
            ],
            "moving_arm_path_clear": moving_arm_path_clear,
            "global_scene_home_clear": global_scene_home_clear,
            "simulation_path_clear": bool(
                moving_arm_path_clear and global_scene_home_clear
            ),
        },
        "motion_authorized": False,
        "motion_authority_reasons": [
            "offline_replay_never_authorizes_hardware",
            "continuous_original_teleop_samples_not_recorded",
            *(
                []
                if global_scene_home_clear
                else ["static_arm_or_stale_scene_home_contacts_remain"]
            ),
            *(
                []
                if moving_arm_path_clear
                else ["moving_arm_environment_contacts_remain"]
            ),
        ],
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report

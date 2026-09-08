"""Plan and audit horizontal, dish-sized air-transport rehearsals.

The reusable signal in the Pasteur demonstrations is the motion between a
verified close and release.  This module deliberately does not replay the
grasp/place portions: an air rehearsal starts above the demonstrated source,
keeps the jaw plane level, follows a robust successful route, and stops above
the demonstrated destination.

All public ``side`` arguments name the *physical* arm.  The production and
semantic MuJoCo models use different branch names; those mappings live here so
callers cannot silently exchange the two arms again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import tempfile
import time
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import h5py
import mink
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from robot.arm.home import (
    physical_home_q,
    physical_to_semantic_model_q_offset,
)
from robot.arm.ik_solver import SingleArmIK
from rollout.gripper_level import (
    JawLevelReference,
    assess_jaw_level,
    leveled_pose,
)


SCHEMA = "piper_robot.dish_transport_air_rehearsal/v2"
PRODUCTION_BRANCH = {"right": "left_arm", "left": "right_arm"}
PRODUCTION_EE = {"right": "left_arm_ee", "left": "right_arm_ee"}
SEMANTIC_BRANCH = {"right": "right", "left": "left"}


def _side(side: str) -> str:
    side = str(side).lower()
    if side not in {"left", "right"}:
        raise ValueError("physical arm side must be 'left' or 'right'")
    return side


def _pose(quaternion_wxyz, position_xyz) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    position = np.asarray(position_xyz, dtype=float).reshape(3)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("pose quaternion is zero")
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    result = np.r_[quaternion, position]
    if not np.all(np.isfinite(result)):
        raise ValueError("pose contains non-finite values")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quaternion_distance_rad(first, second) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(2.0 * math.acos(np.clip(abs(first @ second), -1.0, 1.0)))


def _arc_resample(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("arc resampling needs at least two vectors")
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    values = values[keep]
    cumulative = cumulative[keep]
    if cumulative[-1] <= 1e-9:
        raise ValueError("trajectory has no Cartesian motion")
    query = np.linspace(0.0, cumulative[-1], int(count))
    return np.column_stack(
        [np.interp(query, cumulative, values[:, axis]) for axis in range(values.shape[1])]
    )


def _resample_pose_path(poses: np.ndarray, count: int) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    positions = poses[:, 4:]
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    poses = poses[keep]
    cumulative = cumulative[keep]
    if cumulative[-1] <= 1e-9:
        raise ValueError("pose path has no Cartesian motion")
    query = np.linspace(0.0, cumulative[-1], int(count))
    positions = np.column_stack(
        [np.interp(query, cumulative, poses[:, 4 + axis]) for axis in range(3)]
    )
    xyzw = poses[:, [1, 2, 3, 0]]
    rotations = Slerp(cumulative, Rotation.from_quat(xyzw))(query).as_quat()
    quaternions = rotations[:, [3, 0, 1, 2]]
    return np.c_[quaternions, positions]


@dataclass(frozen=True)
class TransportEpisode:
    path: Path
    close_frame: int
    release_frame: int
    source_lift_frame: int
    arrival_hover_frame: int
    positions_m: np.ndarray
    quaternions_wxyz: np.ndarray
    right_joint_positions_rad: np.ndarray | None

    @property
    def transport_poses(self) -> np.ndarray:
        sl = slice(self.source_lift_frame, self.arrival_hover_frame + 1)
        return np.asarray(
            [
                _pose(q, p)
                for q, p in zip(self.quaternions_wxyz[sl], self.positions_m[sl])
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class PlannedCheckpoint:
    name: str
    pose_index: int
    pose_wxyz_xyz: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TransportPlan:
    name: str
    source: str
    destination: str
    physical_arm: str
    medoid_hdf5: str
    medoid_sha256: str
    coordinate_retarget: str
    poses_wxyz_xyz: np.ndarray
    q_physical_rad: np.ndarray
    checkpoint_indices: tuple[int, ...]
    checkpoint_names: tuple[str, ...]
    maximum_planned_tilt_deg: float
    collision_audit: dict

    def checkpoint(self, number: int) -> PlannedCheckpoint:
        index = self.checkpoint_indices[number]
        return PlannedCheckpoint(
            self.checkpoint_names[number],
            int(index),
            tuple(float(value) for value in self.poses_wxyz_xyz[index]),
        )

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "source": self.source,
            "destination": self.destination,
            "physical_arm": self.physical_arm,
            "medoid_hdf5": self.medoid_hdf5,
            "medoid_sha256": self.medoid_sha256,
            "coordinate_retarget": self.coordinate_retarget,
            "poses_wxyz_xyz": self.poses_wxyz_xyz.tolist(),
            "q_physical_rad": self.q_physical_rad.tolist(),
            "checkpoints": [
                self.checkpoint(i).to_dict() for i in range(len(self.checkpoint_indices))
            ],
            "maximum_planned_tilt_deg": self.maximum_planned_tilt_deg,
            "collision_audit": self.collision_audit,
        }


@dataclass(frozen=True)
class StoppedObserverPlan:
    """Left-camera poses executed only while the right carrier is stopped."""

    physical_arm: str
    carrier_arm: str
    checkpoint_poses_wxyz_xyz: np.ndarray
    checkpoint_q_physical_rad: np.ndarray
    transition_pose_paths: tuple[np.ndarray, ...]
    transition_q_paths: tuple[np.ndarray, ...]
    return_pose_path: np.ndarray
    return_q_path: np.ndarray
    audit: dict

    def to_dict(self) -> dict:
        return {
            "physical_arm": self.physical_arm,
            "carrier_arm": self.carrier_arm,
            "checkpoint_poses_wxyz_xyz": self.checkpoint_poses_wxyz_xyz.tolist(),
            "checkpoint_q_physical_rad": self.checkpoint_q_physical_rad.tolist(),
            "transition_pose_paths": [value.tolist() for value in self.transition_pose_paths],
            "transition_q_paths": [value.tolist() for value in self.transition_q_paths],
            "return_pose_path": self.return_pose_path.tolist(),
            "return_q_path": self.return_q_path.tolist(),
            "audit": self.audit,
        }


def load_transport_episode(
    path: str | Path,
    *,
    source_lift_m: float,
    arrival_hover_m: float,
    open_threshold: float = 0.5,
) -> TransportEpisode:
    """Load one clean success and remove its low grasp/place portions."""

    path = Path(path).resolve()
    with h5py.File(path, "r") as recording:
        positions = np.asarray(recording["right_ee_pos"], dtype=float)
        quaternions = np.asarray(recording["right_ee_quat"], dtype=float)
        gripper = np.asarray(recording["right_gripper"], dtype=float)
        joints = (
            np.asarray(recording["right_joint_positions"], dtype=float)
            if "right_joint_positions" in recording
            else None
        )
    if not (
        positions.ndim == 2
        and positions.shape[1] == 3
        and quaternions.shape == (len(positions), 4)
        and gripper.shape == (len(positions),)
    ):
        raise ValueError(f"{path}: unsupported transport recording schema")
    open_state = gripper > float(open_threshold)
    transitions = np.flatnonzero(open_state[1:] != open_state[:-1]) + 1
    if len(transitions) != 2:
        raise ValueError(
            f"{path}: expected exactly close then release; got {len(transitions)} transitions"
        )
    close_frame, release_frame = (int(value) for value in transitions)
    if open_state[close_frame] or not open_state[release_frame]:
        raise ValueError(f"{path}: gripper transitions are not open-close-open")
    carry = positions[close_frame : release_frame + 1]
    # A transport recording may include the return trip before the operator
    # opens the gripper.  In that common teleop pattern, release is back near
    # the source and is not the named destination.  The destination is the
    # turnaround: the carried pose furthest from the grasp pose.  This also
    # reduces to the release frame for a one-way recording whose displacement
    # grows monotonically, so the rule is station-name independent.
    destination_offset = int(
        np.argmax(np.linalg.norm(carry - carry[0], axis=1))
    )
    arrival_frame = close_frame + destination_offset
    if arrival_frame <= close_frame:
        raise ValueError(f"{path}: carried motion has no distinct destination")
    # Route-medoid selection uses only the outbound carried motion.  Including
    # a differently timed return leg makes a medoid look like a small loop and
    # can place the destination back at the source.
    source_frame = close_frame
    return TransportEpisode(
        path=path,
        close_frame=close_frame,
        release_frame=release_frame,
        source_lift_frame=source_frame,
        arrival_hover_frame=arrival_frame,
        positions_m=positions,
        quaternions_wxyz=quaternions,
        right_joint_positions_rad=joints,
    )


def choose_route_medoid(
    episodes: Sequence[TransportEpisode], *, comparison_samples: int = 121
) -> int:
    """Select the successful route with the smallest normalized shape error."""

    if len(episodes) < 3:
        raise ValueError("at least three clean successful demonstrations are required")
    normalized = []
    for episode in episodes:
        path = _arc_resample(episode.transport_poses[:, 4:], comparison_samples)
        normalized.append(path - path[0])
    scores = np.zeros(len(episodes), dtype=float)
    for first in range(len(episodes)):
        for second in range(len(episodes)):
            scores[first] += float(
                np.mean(np.linalg.norm(normalized[first] - normalized[second], axis=1))
            )
    return int(np.argmin(scores))


class ProductionArmKinematics:
    """Production-CAD FK/IK for one physical arm, with explicit branch map."""

    def __init__(self, production_model: str | Path, physical_arm: str):
        self.physical_arm = _side(physical_arm)
        branch = PRODUCTION_BRANCH[self.physical_arm]
        self.joint_names = tuple(f"{branch}_joint{index}" for index in range(1, 7))
        self.ee_frame = PRODUCTION_EE[self.physical_arm]
        self.solver = SingleArmIK(
            str(Path(production_model).resolve()),
            joint_names=list(self.joint_names),
            ee_frame=self.ee_frame,
        )
        self.lower = np.asarray(
            [self.solver.model.joint(name).range[0] for name in self.joint_names],
            dtype=float,
        )
        self.upper = np.asarray(
            [self.solver.model.joint(name).range[1] for name in self.joint_names],
            dtype=float,
        )

    def pose(self, q_physical_rad) -> np.ndarray:
        q = np.asarray(q_physical_rad, dtype=float).reshape(6)
        self.solver.init(q)
        return np.asarray(self.solver.forward_kinematics().parameters(), dtype=float)

    def solve_pose(
        self,
        target_pose_wxyz_xyz,
        seed_q_physical_rad,
        *,
        maximum_position_error_m: float = 0.008,
        maximum_rotation_error_rad: float = math.radians(8.0),
        level_reference: JawLevelReference | None = None,
        allow_multistart: bool = True,
        maximum_joint_delta_rad: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        target = np.asarray(target_pose_wxyz_xyz, dtype=float).reshape(7)
        seed = np.asarray(seed_q_physical_rad, dtype=float).reshape(6)
        target_rotation = Rotation.from_quat(target[[1, 2, 3, 0]])

        def residual(q):
            self.solver.update_configuration(q)
            actual = np.asarray(self.solver.forward_kinematics().parameters(), dtype=float)
            actual_rotation = Rotation.from_quat(actual[[1, 2, 3, 0]])
            rotation_error = (target_rotation.inv() * actual_rotation).as_rotvec()
            return np.r_[
                (actual[4:] - target[4:]) / 0.002,
                rotation_error / math.radians(0.75),
            ]

        starts = [seed]
        # A level wrist can sit on either side of a wrist-roll local minimum.
        # Retry only after the continuous seed fails; execution still receives
        # Cartesian targets and the selected audit path is checked for jumps.
        if allow_multistart:
            for joint4_delta, joint6_delta in (
                (0.0, -0.8),
                (0.0, 0.8),
                (-0.35, -0.7),
                (0.35, 0.7),
            ):
                candidate = seed.copy()
                candidate[3] += joint4_delta
                candidate[5] += joint6_delta
                starts.append(np.clip(candidate, self.lower, self.upper))
            starts.append(
                np.clip(physical_home_q(self.physical_arm), self.lower, self.upper)
            )
        candidates = []
        for start_index, start in enumerate(starts):
            lower = self.lower
            upper = self.upper
            if maximum_joint_delta_rad is not None:
                delta = float(maximum_joint_delta_rad)
                if not np.isfinite(delta) or delta <= 0:
                    raise ValueError("maximum joint delta must be positive")
                lower = np.maximum(lower, seed - delta)
                upper = np.minimum(upper, seed + delta)
                start = np.clip(start, lower, upper)
            result = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=240,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
            )
            q = np.asarray(result.x, dtype=float)
            actual = self.pose(q)
            position_error = float(np.linalg.norm(actual[4:] - target[4:]))
            rotation_error = _quaternion_distance_rad(actual[:4], target[:4])
            level_assessment = (
                # This is an IK/collision proxy, not the commanded pose.  The
                # command is exactly level; allow the existing 3-degree
                # checkpoint tolerance for the proxy and re-check measured EE
                # at each human stop.
                assess_jaw_level(actual, level_reference, planned=False)
                if level_reference is not None
                else None
            )
            accepted = bool(
                position_error <= maximum_position_error_m
                and rotation_error <= maximum_rotation_error_rad
            )
            score = (
                position_error / maximum_position_error_m
                + rotation_error / maximum_rotation_error_rad
                + 0.02 * float(np.linalg.norm(q - seed))
            )
            candidates.append(
                (
                    accepted,
                    score,
                    q,
                    result,
                    position_error,
                    rotation_error,
                    start_index,
                    level_assessment,
                )
            )
            if accepted and start_index == 0:
                break
        (
            accepted,
            _,
            q,
            result,
            position_error,
            rotation_error,
            start_index,
            level_assessment,
        ) = min(
            candidates, key=lambda item: (not item[0], item[1])
        )
        report = {
            "accepted": accepted,
            "position_error_m": position_error,
            "rotation_error_deg": math.degrees(rotation_error),
            "optimizer_success": bool(result.success),
            "optimizer_cost": float(result.cost),
            "multistart_index": int(start_index),
            "maximum_joint_delta_rad": maximum_joint_delta_rad,
            "actual_jaw_level": (
                None if level_assessment is None else level_assessment.to_dict()
            ),
        }
        if not accepted:
            raise ValueError(f"{self.physical_arm} IK rejected: {report}")
        return q, report

    def solve_path(
        self,
        poses_wxyz_xyz,
        *,
        seed_q=None,
        level_reference: JawLevelReference | None = None,
        allow_multistart: bool = False,
        maximum_position_error_m: float = 0.008,
        maximum_rotation_error_rad: float = math.radians(8.0),
        maximum_joint_delta_rad: float | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        seed = physical_home_q(self.physical_arm) if seed_q is None else np.asarray(seed_q)
        q_path = []
        reports = []
        for index, pose in enumerate(np.asarray(poses_wxyz_xyz, dtype=float)):
            try:
                seed, report = self.solve_pose(
                    pose,
                    seed,
                    level_reference=level_reference,
                    allow_multistart=allow_multistart,
                    maximum_position_error_m=maximum_position_error_m,
                    maximum_rotation_error_rad=maximum_rotation_error_rad,
                    maximum_joint_delta_rad=maximum_joint_delta_rad,
                )
            except ValueError as error:
                raise ValueError(f"IK path sample {index} rejected: {error}") from error
            q_path.append(seed.copy())
            reports.append(report)
        return np.asarray(q_path), reports


def _level_path(
    poses: np.ndarray,
    reference: JawLevelReference,
    *,
    fixed_orientation: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    result = np.asarray(poses, dtype=float).copy()
    if fixed_orientation is None:
        for index in range(len(result)):
            result[index] = leveled_pose(result[index], reference)
    else:
        result[:, :4] = np.asarray(fixed_orientation, dtype=float).reshape(4)
    assessments = [assess_jaw_level(pose, reference, planned=True) for pose in result]
    maximum = max(item.combined_tilt_deg for item in assessments)
    rejected = [item for item in assessments if not item.accepted]
    if rejected:
        raise ValueError(f"horizontal gripper gate rejected {len(rejected)} poses")
    return result, float(maximum)


def _append_linear_pose_path(first, second, maximum_step_m: float) -> np.ndarray:
    first = np.asarray(first, dtype=float).reshape(7)
    second = np.asarray(second, dtype=float).reshape(7)
    distance = float(np.linalg.norm(second[4:] - first[4:]))
    rotation_distance = _quaternion_distance_rad(first[:4], second[:4])
    count = max(
        2,
        int(math.ceil(distance / maximum_step_m)) + 1,
        int(math.ceil(rotation_distance / math.radians(5.0))) + 1,
    )
    alpha = np.linspace(0.0, 1.0, count)
    position = first[4:] + alpha[:, None] * (second[4:] - first[4:])
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_quat(np.asarray([first, second])[:, [1, 2, 3, 0]]),
    )(alpha).as_quat()
    return np.c_[rotations[:, [3, 0, 1, 2]], position]


def _append_joint_branch_path(first, second, maximum_step_rad: float) -> np.ndarray:
    """Interpolate one already-selected IK branch without re-solving it."""

    first = np.asarray(first, dtype=float).reshape(6)
    second = np.asarray(second, dtype=float).reshape(6)
    count = max(
        2,
        int(math.ceil(np.max(np.abs(second - first)) / maximum_step_rad)) + 1,
    )
    alpha = np.linspace(0.0, 1.0, count)
    return first + alpha[:, None] * (second - first)


def _choose_horizontal_orientation(
    kinematics: ProductionArmKinematics,
    anchor_positions_m: np.ndarray,
    home_pose_wxyz_xyz: np.ndarray,
    level_reference: JawLevelReference,
) -> tuple[np.ndarray, dict]:
    """Find a level yaw that is reachable at all three station anchors."""

    home_level = leveled_pose(home_pose_wxyz_xyz, level_reference)
    home_rotation = Rotation.from_quat(home_level[[1, 2, 3, 0]])
    yaw_candidates_deg = (0, -15, 15, -30, 30, -45, 45, -60, 60, -75, 75, -90, 90)
    candidates = []
    for yaw_deg in yaw_candidates_deg:
        rotation = Rotation.from_euler("z", yaw_deg, degrees=True) * home_rotation
        xyzw = rotation.as_quat()
        quaternion = xyzw[[3, 0, 1, 2]]
        seed = physical_home_q(kinematics.physical_arm)
        reports = []
        q_values = []
        try:
            for position in np.asarray(anchor_positions_m, dtype=float):
                pose = np.r_[quaternion, position]
                seed, report = kinematics.solve_pose(
                    pose, seed, level_reference=level_reference
                )
                reports.append(report)
                q_values.append(seed.copy())
        except ValueError:
            continue
        proxy_tilt = max(
            report["actual_jaw_level"]["combined_tilt_deg"] for report in reports
        )
        proxy_position = max(report["position_error_m"] for report in reports)
        accepted = bool(
            all(report["actual_jaw_level"]["accepted"] for report in reports)
            and proxy_position <= 0.008
        )
        joint_travel = float(
            np.sum(
                np.linalg.norm(
                    np.diff(
                        np.vstack((physical_home_q(kinematics.physical_arm), q_values)),
                        axis=0,
                    ),
                    axis=1,
                )
            )
        )
        score = proxy_tilt + 100.0 * proxy_position + 0.03 * joint_travel
        candidates.append(
            {
                "accepted": accepted,
                "score": score,
                "yaw_from_level_home_deg": float(yaw_deg),
                "quaternion_wxyz": quaternion,
                "maximum_proxy_tilt_deg": proxy_tilt,
                "maximum_proxy_position_error_m": proxy_position,
                "joint_travel_rad": joint_travel,
                "anchor_q_physical_rad": np.asarray(q_values).tolist(),
            }
        )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    if not accepted:
        raise ValueError(
            f"{kinematics.physical_arm} has no horizontal yaw reachable at all anchors"
        )
    selected = min(accepted, key=lambda candidate: candidate["score"])
    report = {
        key: value
        for key, value in selected.items()
        if key not in {"quaternion_wxyz", "score", "anchor_q_physical_rad"}
    }
    report["candidate_count"] = len(candidates)
    report["branch_seed_source"] = "selected_horizontal_anchor_solution"
    return (
        np.asarray(selected["quaternion_wxyz"], dtype=float),
        np.asarray(selected["anchor_q_physical_rad"], dtype=float),
        report,
    )


def _insert_virtual_dish(scene_path: Path, radius_m: float, thickness_m: float) -> Path:
    tree = ET.parse(scene_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"{scene_path}: missing worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": "virtual-carried-dish", "mocap": "true", "pos": "0 0 2"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "virtual-carried-dish-collision",
            "type": "cylinder",
            "size": f"{radius_m:.9g} {0.5 * thickness_m:.9g}",
            "rgba": "0.2 0.7 1 0.35",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    handle = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".mjcf", prefix="dish-audit-", dir=scene_path.parent, delete=False
    )
    handle.close()
    generated = Path(handle.name)
    tree.write(generated, encoding="utf-8", xml_declaration=True)
    return generated


def audit_joint_path(
    planning_model: str | Path,
    physical_arm: str,
    q_path,
    *,
    dish_radius_m: float,
    dish_thickness_m: float,
    dish_center_offset_ee_m: Sequence[float],
    ignored_environment_bodies: Sequence[str] = (),
    maximum_baseline_penetration_increase_m: float = 0.003,
    carrying_sample_range: tuple[int, int] | None = None,
    minimum_dish_clearance_m: float = 0.0,
) -> dict:
    """Audit arm and a horizontal virtual dish against the calibrated scene."""

    import mujoco

    side = _side(physical_arm)
    path = np.asarray(q_path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2:
        raise ValueError("joint path must be finite Nx6")
    scene = Path(planning_model).resolve()
    generated = _insert_virtual_dish(scene, dish_radius_m, dish_thickness_m)
    try:
        model = mujoco.MjModel.from_xml_path(str(generated))
    finally:
        generated.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    active_prefix = SEMANTIC_BRANCH[side]
    inactive_side = "left" if side == "right" else "right"
    active_ids = np.asarray(
        [model.joint(f"{active_prefix}/joint{index}").qposadr[0] for index in range(1, 7)]
    )
    inactive_prefix = SEMANTIC_BRANCH[inactive_side]
    inactive_ids = np.asarray(
        [model.joint(f"{inactive_prefix}/joint{index}").qposadr[0] for index in range(1, 7)]
    )
    data.qpos[inactive_ids] = physical_home_q(inactive_side) + physical_to_semantic_model_q_offset(inactive_side)
    offset = physical_to_semantic_model_q_offset(side)
    data.qpos[active_ids] = physical_home_q(side) + offset
    dish_body = model.body("virtual-carried-dish")
    mocap_id = int(dish_body.mocapid[0])
    dish_geom_id = int(model.geom("virtual-carried-dish-collision").id)
    ee_site = model.site(f"{active_prefix}/ee")
    tool_offset = np.asarray(dish_center_offset_ee_m, dtype=float).reshape(3)
    ignored_bodies = {str(value) for value in ignored_environment_bodies}
    # The calibrated semantic scene can contain coarse voxel contacts in the
    # canonical home pose (for example where an arm passes through a platform
    # cut-out).  They are not newly created by this route.  Preserve their
    # signed depth and reject them only if the candidate path makes the
    # penetration materially worse.
    data.mocap_pos[mocap_id] = np.asarray([0.0, 0.0, 2.0])
    mujoco.mj_forward(model, data)
    baseline_contacts: dict[tuple[str, str], float] = {}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = model.geom(int(contact.geom1))
        geom2 = model.geom(int(contact.geom2))
        body1 = model.body(int(geom1.bodyid[0])).name
        body2 = model.body(int(geom2.bodyid[0])).name
        if body1.startswith(active_prefix + "/") or body2.startswith(active_prefix + "/"):
            pair = tuple(sorted((geom1.name or body1, geom2.name or body2)))
            baseline_contacts[pair] = min(
                baseline_contacts.get(pair, float("inf")), float(contact.dist)
            )
    # The first planned sample is the horizontalized home pose.  Coarse scene
    # voxels can report a new contact after this in-place wrist normalization;
    # preserve it as a start-state mismatch but still reject any worsening or
    # any new pair later on the route.
    data.qpos[active_ids] = path[0] + offset
    mujoco.mj_forward(model, data)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = model.geom(int(contact.geom1))
        geom2 = model.geom(int(contact.geom2))
        body1 = model.body(int(geom1.bodyid[0])).name
        body2 = model.body(int(geom2.bodyid[0])).name
        if body1.startswith(active_prefix + "/") or body2.startswith(active_prefix + "/"):
            pair = tuple(sorted((geom1.name or body1, geom2.name or body2)))
            baseline_contacts[pair] = min(
                baseline_contacts.get(pair, float("inf")), float(contact.dist)
            )
    disallowed: set[tuple[str, str]] = set()
    disallowed_minimum_distance: dict[tuple[str, str], float] = {}
    first_disallowed_sample = None
    minimum_dish_environment_distance_m = float("inf")
    minimum_dish_environment_pair = None
    minimum_dish_environment_sample = None
    carry_start, carry_stop = (
        (0, len(path) - 1)
        if carrying_sample_range is None
        else tuple(int(value) for value in carrying_sample_range)
    )
    if not 0 <= carry_start <= carry_stop < len(path):
        raise ValueError("carrying sample range is outside the joint path")
    for sample_index, q_physical in enumerate(path):
        data.qpos[active_ids] = q_physical + offset
        mujoco.mj_forward(model, data)
        carrying = carry_start <= sample_index <= carry_stop
        if carrying:
            rotation = np.asarray(data.site_xmat[ee_site.id], dtype=float).reshape(3, 3)
            data.mocap_pos[mocap_id] = data.site_xpos[ee_site.id] + rotation @ tool_offset
            data.mocap_quat[mocap_id] = np.asarray([1.0, 0.0, 0.0, 0.0])
        else:
            data.mocap_pos[mocap_id] = np.asarray([0.0, 0.0, 2.0])
        mujoco.mj_forward(model, data)
        if carrying:
            for geom_id in range(model.ngeom):
                if geom_id == dish_geom_id:
                    continue
                geom = model.geom(geom_id)
                body_name = model.body(int(geom.bodyid[0])).name
                if (
                    body_name in ignored_bodies
                    or body_name.startswith(active_prefix + "/")
                    or (
                        int(model.geom_contype[geom_id]) == 0
                        and int(model.geom_conaffinity[geom_id]) == 0
                    )
                ):
                    continue
                distance = float(
                    mujoco.mj_geomDistance(
                        model, data, dish_geom_id, geom_id, 1.0, None
                    )
                )
                if distance < minimum_dish_environment_distance_m:
                    minimum_dish_environment_distance_m = distance
                    minimum_dish_environment_pair = [
                        "virtual-carried-dish-collision",
                        geom.name or body_name,
                    ]
                    minimum_dish_environment_sample = sample_index
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = model.geom(int(contact.geom1))
            geom2 = model.geom(int(contact.geom2))
            body1 = model.body(int(geom1.bodyid[0])).name
            body2 = model.body(int(geom2.bodyid[0])).name
            names = (geom1.name or body1, geom2.name or body2)
            if body1 in ignored_bodies or body2 in ignored_bodies:
                continue
            active1 = body1.startswith(active_prefix + "/")
            active2 = body2.startswith(active_prefix + "/")
            dish1 = body1 == "virtual-carried-dish"
            dish2 = body2 == "virtual-carried-dish"
            involves_active_or_dish = active1 or active2 or dish1 or dish2
            if not involves_active_or_dish:
                continue
            intentional_attachment = (dish1 and active2) or (dish2 and active1)
            if intentional_attachment:
                continue
            pair = tuple(sorted(names))
            if pair in baseline_contacts and float(contact.dist) >= (
                baseline_contacts[pair] - float(maximum_baseline_penetration_increase_m)
            ):
                continue
            disallowed.add(pair)
            disallowed_minimum_distance[pair] = min(
                disallowed_minimum_distance.get(pair, float("inf")),
                float(contact.dist),
            )
            if first_disallowed_sample is None:
                first_disallowed_sample = sample_index
            if carrying and (dish1 or dish2):
                minimum_dish_environment_distance_m = min(
                    minimum_dish_environment_distance_m, float(contact.dist)
                )
    clearance = (
        None
        if math.isinf(minimum_dish_environment_distance_m)
        else float(minimum_dish_environment_distance_m)
    )
    clearance_accepted = bool(
        clearance is not None and clearance >= float(minimum_dish_clearance_m)
    )
    return {
        "accepted": bool(not disallowed and clearance_accepted),
        "physical_arm": side,
        "semantic_branch": active_prefix,
        "sample_count": int(len(path)),
        "first_disallowed_sample": first_disallowed_sample,
        "non_attachment_contacts": [list(pair) for pair in sorted(disallowed)],
        "non_attachment_contact_minimum_distance_m": [
            {"geoms": list(pair), "distance_m": distance}
            for pair, distance in sorted(disallowed_minimum_distance.items())
        ],
        "accepted_home_baseline_contacts": [
            {"geoms": list(pair), "distance_m": distance}
            for pair, distance in sorted(baseline_contacts.items())
        ],
        "ignored_environment_bodies": sorted(ignored_bodies),
        "minimum_dish_environment_distance_m": clearance,
        "minimum_dish_environment_pair": minimum_dish_environment_pair,
        "minimum_dish_environment_sample": minimum_dish_environment_sample,
        "minimum_required_dish_clearance_m": float(minimum_dish_clearance_m),
        "dish_clearance_accepted": clearance_accepted,
        "carrying_sample_range": [carry_start, carry_stop],
        "virtual_dish": {
            "radius_m": float(dish_radius_m),
            "thickness_m": float(dish_thickness_m),
            "center_offset_ee_m": tool_offset.tolist(),
            "orientation": "horizontal_in_robot_world",
        },
    }


def build_transport_plan(
    *,
    name: str,
    source: str,
    destination: str,
    physical_arm: str,
    demonstration_paths: Iterable[str | Path],
    production_model: str | Path,
    planning_model: str | Path,
    level_reference: JawLevelReference,
    source_lift_m: float = 0.05,
    arrival_hover_m: float = 0.03,
    route_samples: int = 61,
    maximum_cartesian_step_m: float = 0.012,
    dish_radius_m: float = 0.045,
    dish_thickness_m: float = 0.014,
    dish_center_offset_ee_m: Sequence[float] = (-0.075, 0.0, 0.0),
    ignored_environment_bodies: Sequence[str] = (),
    require_collision_free: bool = True,
    minimum_dish_clearance_m: float = 0.010,
    low_route_search_step_m: float = 0.005,
    maximum_low_route_lift_m: float = 0.120,
    maximum_ik_joint_step_rad: float = 0.12,
) -> TransportPlan:
    """Compile successes into the lowest level, collision-audited air route.

    The demonstration supplies station positions and route shape in XY.  Its
    hand-authored high waypoint is deliberately not reused: the vertical
    profile starts at the two station lifts and is raised in small increments
    until the carried-dish clearance, arm collision, and branch-locked IK gates
    all pass.
    """

    side = _side(physical_arm)
    episodes = []
    rejected = []
    for path in demonstration_paths:
        try:
            episodes.append(
                load_transport_episode(
                    path,
                    source_lift_m=source_lift_m,
                    arrival_hover_m=arrival_hover_m,
                )
            )
        except ValueError as error:
            rejected.append(str(error))
    if len(episodes) < 3:
        raise ValueError(
            f"{name}: only {len(episodes)} clean successes; rejected={rejected}"
        )
    medoid = episodes[choose_route_medoid(episodes)]
    demonstrated_route = _resample_pose_path(medoid.transport_poses, route_samples)
    demonstrated_route[0, 4:] = medoid.positions_m[medoid.close_frame]
    demonstrated_route[0, 6] += float(source_lift_m)
    demonstrated_route[-1, 4:] = medoid.positions_m[medoid.arrival_hover_frame]
    demonstrated_route[-1, 6] += float(arrival_hover_m)

    kinematics = ProductionArmKinematics(production_model, side)
    home_q = physical_home_q(side)
    home_pose = kinematics.pose(home_q)
    coordinate_retarget = "recorded_physical_right_robot_frame"
    if side == "left":
        right_home = ProductionArmKinematics(production_model, "right").pose(
            physical_home_q("right")
        )
        demonstrated_route[:, 4:] += home_pose[4:] - right_home[4:]
        coordinate_retarget = (
            "operator_confirmed_air_rehearsal_home_relative_translation_from_right_demo"
        )
    # Retain the demonstrated middle XY bend, but replace its height with the
    # straight station-to-station envelope.  Search upward from there; the
    # first accepted candidate is the lowest route at the configured grid.
    middle_start = max(1, len(demonstrated_route) // 5)
    middle_stop = min(len(demonstrated_route) - 1, 4 * len(demonstrated_route) // 5)
    # Preserve only the demonstrated obstacle-avoiding XY bend.  Its height is
    # discarded below, so a cautious human lift cannot force every replay high.
    middle_index = middle_start + int(
        np.argmax(demonstrated_route[middle_start:middle_stop, 6])
    )
    base_anchors = demonstrated_route[[0, middle_index, len(demonstrated_route) - 1]].copy()
    base_anchors[:, 6] = np.linspace(base_anchors[0, 6], base_anchors[-1, 6], 3)
    candidate_reports = []
    selected = None
    lift_values = np.arange(
        0.0,
        float(maximum_low_route_lift_m) + 0.5 * float(low_route_search_step_m),
        float(low_route_search_step_m),
    )
    for added_lift_m in lift_values:
        anchors = base_anchors.copy()
        anchors[:, 6] += float(added_lift_m)
        try:
            (
                fixed_orientation,
                horizontal_anchor_q,
                yaw_selection,
            ) = _choose_horizontal_orientation(
                kinematics, anchors[:, 4:], home_pose, level_reference
            )
            anchors[:, :4] = fixed_orientation
            first_half = _append_linear_pose_path(
                anchors[0], anchors[1], maximum_cartesian_step_m
            )
            second_half = _append_linear_pose_path(
                anchors[1], anchors[2], maximum_cartesian_step_m
            )
            route = np.vstack((first_half, second_half[1:]))
            route, maximum_tilt = _level_path(
                route, level_reference, fixed_orientation=fixed_orientation
            )
            # Horizontal level is a carried-object invariant, not a homing
            # invariant. First solve the branch-selected carried route.
            route_q, route_reports = kinematics.solve_path(
                route,
                seed_q=horizontal_anchor_q[0],
                level_reference=level_reference,
                allow_multistart=False,
                maximum_joint_delta_rad=maximum_ik_joint_step_rad,
            )
            # Connect home to that exact branch in joint space, then export
            # its FK poses to the same Cartesian teleop streamer used by the
            # operator. Re-solving a Cartesian home approach discarded the
            # selected wrist branch and produced a false 4--7 degree tilt.
            branch_step_rad = min(0.04, float(maximum_ik_joint_step_rad))
            approach_q = _append_joint_branch_path(
                home_q, route_q[0], branch_step_rad
            )
            # Do not cut diagonally from the destination to home: at the
            # microscope this crosses the front support. Retrace the audited
            # low transport branch to the source, then reverse the collision-
            # checked home approach.
            retreat_q = np.vstack(
                (route_q[::-1], approach_q[-2::-1])
            )
            approach = np.asarray([kinematics.pose(q) for q in approach_q])
            retreat = np.asarray([kinematics.pose(q) for q in retreat_q])
            poses = np.vstack((approach, route[1:], retreat[1:]))
            source_index = len(approach) - 1
            arrival_index = source_index + len(route) - 1
            q_path = np.vstack((approach_q, route_q[1:], retreat_q[1:]))
            audit = audit_joint_path(
                planning_model,
                side,
                q_path,
                dish_radius_m=dish_radius_m,
                dish_thickness_m=dish_thickness_m,
                dish_center_offset_ee_m=dish_center_offset_ee_m,
                ignored_environment_bodies=ignored_environment_bodies,
                carrying_sample_range=(source_index, arrival_index),
                minimum_dish_clearance_m=minimum_dish_clearance_m,
            )
            audit["maximum_ik_position_error_m"] = max(
                report["position_error_m"] for report in route_reports
            )
            audit["maximum_ik_rotation_error_deg"] = max(
                report["rotation_error_deg"] for report in route_reports
            )
            audit["horizontal_yaw_selection"] = yaw_selection
            carrying_ik_reports = route_reports
            audit["maximum_ik_proxy_jaw_tilt_deg"] = max(
                report["actual_jaw_level"]["combined_tilt_deg"]
                for report in carrying_ik_reports
            )
            audit["ik_proxy_jaw_level_accepted"] = all(
                report["actual_jaw_level"]["accepted"]
                for report in carrying_ik_reports
            )
            audit["ik_proxy_jaw_level_scope"] = (
                "source_lifted_through_arrival_hover"
            )
            audit["maximum_ik_joint_step_rad"] = float(
                np.max(np.abs(np.diff(q_path, axis=0)))
            )
            audit["maximum_allowed_ik_joint_step_rad"] = float(
                maximum_ik_joint_step_rad
            )
            audit["ik_branch_continuity_accepted"] = bool(
                audit["maximum_ik_joint_step_rad"]
                <= float(maximum_ik_joint_step_rad)
            )
            accepted = bool(
                audit["accepted"]
                and audit["ik_proxy_jaw_level_accepted"]
                and audit["ik_branch_continuity_accepted"]
            )
            candidate_reports.append(
                {
                    "added_lift_m": float(added_lift_m),
                    "accepted": accepted,
                    "minimum_dish_environment_distance_m": audit[
                        "minimum_dish_environment_distance_m"
                    ],
                    "minimum_dish_environment_pair": audit[
                        "minimum_dish_environment_pair"
                    ],
                    "minimum_dish_environment_sample": audit[
                        "minimum_dish_environment_sample"
                    ],
                    "maximum_ik_joint_step_rad": audit[
                        "maximum_ik_joint_step_rad"
                    ],
                    "horizontal_yaw_selection": audit[
                        "horizontal_yaw_selection"
                    ],
                    "first_disallowed_sample": audit[
                        "first_disallowed_sample"
                    ],
                    "non_attachment_contacts": audit[
                        "non_attachment_contacts"
                    ],
                }
            )
            if accepted:
                selected = (
                    poses,
                    q_path,
                    route,
                    source_index,
                    arrival_index,
                    maximum_tilt,
                    audit,
                    float(added_lift_m),
                )
                break
        except ValueError as error:
            candidate_reports.append(
                {
                    "added_lift_m": float(added_lift_m),
                    "accepted": False,
                    "error": str(error),
                }
            )
    if selected is None:
        raise ValueError(
            f"{name}: no low horizontal route passed clearance/IK/collision gates; "
            f"candidates={candidate_reports}"
        )
    (
        poses,
        q_path,
        route,
        source_index,
        arrival_index,
        maximum_tilt,
        audit,
        selected_lift_m,
    ) = selected
    checkpoint_names = (
        "departure",
        "quarter",
        "midpoint",
        "three_quarters",
        "arrival",
    )
    checkpoint_indices = tuple(
        source_index + int(round(fraction * (arrival_index - source_index)))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    audit["low_route_search"] = {
        "selected_added_lift_m": selected_lift_m,
        "step_m": float(low_route_search_step_m),
        "maximum_lift_m": float(maximum_low_route_lift_m),
        "candidates": candidate_reports,
    }
    audit["rejected_demonstrations"] = rejected
    if require_collision_free and not audit["accepted"]:
        raise ValueError(f"{name}: collision audit rejected route: {audit}")
    return TransportPlan(
        name=str(name),
        source=str(source),
        destination=str(destination),
        physical_arm=side,
        medoid_hdf5=str(medoid.path),
        medoid_sha256=_file_sha256(medoid.path),
        coordinate_retarget=coordinate_retarget,
        poses_wxyz_xyz=poses,
        q_physical_rad=q_path,
        checkpoint_indices=checkpoint_indices,
        checkpoint_names=checkpoint_names,
        maximum_planned_tilt_deg=maximum_tilt,
        collision_audit=audit,
    )


def split_checkpoint_chunks(plan: TransportPlan) -> tuple[np.ndarray, ...]:
    """Split at every checkpoint and append the audited home-return chunk."""

    indices = tuple(int(value) for value in plan.checkpoint_indices)
    if len(indices) < 2 or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("checkpoint indices must be strictly increasing")
    poses = plan.poses_wxyz_xyz
    chunks = [poses[: indices[0] + 1]]
    chunks.extend(poses[first : second + 1] for first, second in zip(indices, indices[1:]))
    chunks.append(poses[indices[-1] :])
    return tuple(chunks)


def _audit_coordinated_samples(
    planning_model: str | Path,
    samples: Sequence[tuple[np.ndarray, np.ndarray, bool, str]],
    *,
    dish_radius_m: float,
    dish_thickness_m: float,
    dish_center_offset_ee_m: Sequence[float],
    ignored_environment_bodies: Sequence[str],
    minimum_dish_clearance_m: float,
    calibration_baseline_samples: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    maximum_baseline_penetration_increase_m: float = 0.003,
) -> dict:
    """Audit a sequential two-arm schedule with a right-carried virtual dish."""

    import mujoco

    scene = Path(planning_model).resolve()
    generated = _insert_virtual_dish(scene, dish_radius_m, dish_thickness_m)
    try:
        model = mujoco.MjModel.from_xml_path(str(generated))
    finally:
        generated.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    qids = {
        side: np.asarray(
            [
                model.joint(f"{SEMANTIC_BRANCH[side]}/joint{index}").qposadr[0]
                for index in range(1, 7)
            ]
        )
        for side in ("left", "right")
    }
    offsets = {
        side: physical_to_semantic_model_q_offset(side)
        for side in ("left", "right")
    }
    ignored = {str(value) for value in ignored_environment_bodies}
    dish_body = model.body("virtual-carried-dish")
    dish_geom_id = int(model.geom("virtual-carried-dish-collision").id)
    mocap_id = int(dish_body.mocapid[0])
    right_ee_site = model.site(f"{SEMANTIC_BRANCH['right']}/ee")
    tool_offset = np.asarray(dish_center_offset_ee_m, dtype=float).reshape(3)

    def set_q(left_q, right_q):
        data.qpos[qids["left"]] = np.asarray(left_q, dtype=float) + offsets["left"]
        data.qpos[qids["right"]] = np.asarray(right_q, dtype=float) + offsets["right"]
        data.mocap_pos[mocap_id] = np.asarray([0.0, 0.0, 2.0])
        mujoco.mj_forward(model, data)

    def contact_pairs():
        result = {}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first = model.geom(int(contact.geom1))
            second = model.geom(int(contact.geom2))
            body1 = model.body(int(first.bodyid[0])).name
            body2 = model.body(int(second.bodyid[0])).name
            if body1 in ignored or body2 in ignored:
                continue
            names = tuple(sorted((first.name or body1, second.name or body2)))
            result[names] = min(result.get(names, float("inf")), float(contact.dist))
        return result

    set_q(physical_home_q("left"), physical_home_q("right"))
    baseline = contact_pairs()
    calibration_baseline_count = 0
    for left_q, right_q in calibration_baseline_samples:
        set_q(left_q, right_q)
        for pair, distance in contact_pairs().items():
            baseline[pair] = min(baseline.get(pair, float("inf")), distance)
        calibration_baseline_count += 1
    disallowed = {}
    minimum_clearance = float("inf")
    first_disallowed_stage = None
    for left_q, right_q, carrying, stage in samples:
        set_q(left_q, right_q)
        if carrying:
            rotation = np.asarray(
                data.site_xmat[right_ee_site.id], dtype=float
            ).reshape(3, 3)
            data.mocap_pos[mocap_id] = (
                data.site_xpos[right_ee_site.id] + rotation @ tool_offset
            )
            data.mocap_quat[mocap_id] = np.asarray([1.0, 0.0, 0.0, 0.0])
            mujoco.mj_forward(model, data)
            for geom_id in range(model.ngeom):
                if geom_id == dish_geom_id:
                    continue
                geom = model.geom(geom_id)
                body_name = model.body(int(geom.bodyid[0])).name
                if (
                    body_name in ignored
                    or body_name.startswith("right/")
                    or (
                        int(model.geom_contype[geom_id]) == 0
                        and int(model.geom_conaffinity[geom_id]) == 0
                    )
                ):
                    continue
                minimum_clearance = min(
                    minimum_clearance,
                    float(
                        mujoco.mj_geomDistance(
                            model, data, dish_geom_id, geom_id, 1.0, None
                        )
                    ),
                )
        for pair, distance in contact_pairs().items():
            if "virtual-carried-dish-collision" in pair:
                other = pair[0] if pair[1] == "virtual-carried-dish-collision" else pair[1]
                if other.startswith("right/"):
                    continue
            if pair in baseline and distance >= (
                baseline[pair] - float(maximum_baseline_penetration_increase_m)
            ):
                continue
            disallowed[pair] = min(disallowed.get(pair, float("inf")), distance)
            if first_disallowed_stage is None:
                first_disallowed_stage = stage
    clearance = None if math.isinf(minimum_clearance) else float(minimum_clearance)
    clearance_accepted = bool(
        clearance is not None and clearance >= float(minimum_dish_clearance_m)
    )
    return {
        "accepted": bool(not disallowed and clearance_accepted),
        "sample_count": len(samples),
        "first_disallowed_stage": first_disallowed_stage,
        "non_attachment_contacts": [list(pair) for pair in sorted(disallowed)],
        "non_attachment_contact_minimum_distance_m": [
            {"geoms": list(pair), "distance_m": distance}
            for pair, distance in sorted(disallowed.items())
        ],
        "minimum_dish_environment_distance_m": clearance,
        "minimum_required_dish_clearance_m": float(minimum_dish_clearance_m),
        "dish_clearance_accepted": clearance_accepted,
        "motion_policy": "one_arm_moves_while_the_other_holds",
        "accepted_calibration_baseline_sample_count": calibration_baseline_count,
        "accepted_calibration_baseline_contacts": [
            {"geoms": list(pair), "distance_m": distance}
            for pair, distance in sorted(baseline.items())
        ],
    }


def build_stopped_observer_plan(
    carrier_plan: TransportPlan,
    *,
    production_model: str | Path,
    planning_model: str | Path,
    reference_observer_pose_wxyz_xyz: Sequence[float],
    reference_carrier_pose_wxyz_xyz: Sequence[float],
    safe_waypoint_pose_wxyz_xyz: Sequence[float] | None = None,
    maximum_cartesian_step_m: float,
    maximum_ik_joint_step_rad: float,
    dish_radius_m: float,
    dish_thickness_m: float,
    dish_center_offset_ee_m: Sequence[float],
    ignored_environment_bodies: Sequence[str] = (),
    minimum_dish_clearance_m: float = 0.010,
) -> StoppedObserverPlan:
    """Plan relative-pose left observations without simultaneous arm motion."""

    if carrier_plan.physical_arm != "right":
        raise ValueError("stopped observer currently requires physical right carrier")
    observer_side = "left"
    kinematics = ProductionArmKinematics(production_model, observer_side)
    home_q = physical_home_q(observer_side)
    home_pose = kinematics.pose(home_q)
    reference_observer = np.asarray(reference_observer_pose_wxyz_xyz, dtype=float).reshape(7)
    reference_carrier = np.asarray(reference_carrier_pose_wxyz_xyz, dtype=float).reshape(7)
    checkpoint_poses = []
    for checkpoint_index in carrier_plan.checkpoint_indices:
        carrier_pose = carrier_plan.poses_wxyz_xyz[checkpoint_index]
        target = reference_observer.copy()
        # Follow the carrier on the bench plane while retaining the physically
        # verified observer height.  Copying the carrier's lower transport Z
        # drove the observer down into the microscope in both the semantic
        # audit and the earlier lab trial.
        target[4:6] += carrier_pose[4:6] - reference_carrier[4:6]
        checkpoint_poses.append(target)
    checkpoint_poses = np.asarray(checkpoint_poses)

    transition_pose_paths = []
    transition_q_paths = []
    checkpoint_q = []
    current_pose = home_pose
    current_q = home_q
    for checkpoint_number, target in enumerate(checkpoint_poses):
        if checkpoint_number == 0 and safe_waypoint_pose_wxyz_xyz is not None:
            waypoint = np.asarray(safe_waypoint_pose_wxyz_xyz, dtype=float).reshape(7)
            first = _append_linear_pose_path(
                current_pose, waypoint, maximum_cartesian_step_m
            )
            second = _append_linear_pose_path(
                waypoint, target, maximum_cartesian_step_m
            )
            pose_path = np.vstack((first, second[1:]))
        else:
            pose_path = _append_linear_pose_path(
                current_pose, target, maximum_cartesian_step_m
            )
        q_path, _ = kinematics.solve_path(
            pose_path,
            seed_q=current_q,
            allow_multistart=False,
            maximum_position_error_m=0.012,
        )
        if checkpoint_number == 0:
            # The Cartesian chord from home to the side-view pose crosses the
            # coarse microscope volume.  Interpolate the verified continuous
            # joint branch, then send its FK poses through the same Cartesian
            # teleop RPC used everywhere else.
            joint_count = max(
                2,
                int(
                    math.ceil(
                        float(np.max(np.abs(q_path[-1] - current_q))) / 0.04
                    )
                )
                + 1,
            )
            q_path = np.linspace(current_q, q_path[-1], joint_count)
            pose_path = np.asarray([kinematics.pose(q) for q in q_path])
        transition_pose_paths.append(pose_path)
        transition_q_paths.append(q_path)
        checkpoint_q.append(q_path[-1].copy())
        current_pose = target
        current_q = q_path[-1]
    return_pose_path = _append_linear_pose_path(
        current_pose, home_pose, maximum_cartesian_step_m
    )
    return_q_path, _ = kinematics.solve_path(
        return_pose_path,
        seed_q=current_q,
        allow_multistart=False,
        maximum_position_error_m=0.012,
    )
    return_count = max(
        2,
        int(math.ceil(float(np.max(np.abs(home_q - current_q))) / 0.04)) + 1,
    )
    return_q_path = np.linspace(current_q, home_q, return_count)
    return_pose_path = np.asarray([kinematics.pose(q) for q in return_q_path])

    maximum_step = max(
        float(np.max(np.abs(np.diff(path, axis=0))))
        for path in (*transition_q_paths, return_q_path)
        if len(path) > 1
    )
    samples = []
    right_q = carrier_plan.q_physical_rad
    first_checkpoint = carrier_plan.checkpoint_indices[0]
    for q in right_q[: first_checkpoint + 1]:
        samples.append((home_q, q, False, "right_approach"))
    for q in transition_q_paths[0]:
        samples.append((q, right_q[first_checkpoint], True, "left_observer_0"))
    for number in range(1, len(carrier_plan.checkpoint_indices)):
        first = carrier_plan.checkpoint_indices[number - 1]
        second = carrier_plan.checkpoint_indices[number]
        held_left = checkpoint_q[number - 1]
        for q in right_q[first : second + 1]:
            samples.append((held_left, q, True, f"right_carry_{number}"))
        for q in transition_q_paths[number]:
            samples.append((q, right_q[second], True, f"left_observer_{number}"))
    arrival = carrier_plan.checkpoint_indices[-1]
    for q in return_q_path:
        samples.append((q, right_q[arrival], True, "left_return_home"))
    for q in right_q[arrival:]:
        samples.append((home_q, q, False, "right_return_home"))
    audit = _audit_coordinated_samples(
        planning_model,
        samples,
        dish_radius_m=dish_radius_m,
        dish_thickness_m=dish_thickness_m,
        dish_center_offset_ee_m=dish_center_offset_ee_m,
        ignored_environment_bodies=ignored_environment_bodies,
        minimum_dish_clearance_m=minimum_dish_clearance_m,
        # The first target differs from the operator-verified reference by only
        # the checkpoint XY retarget.  Treat its scene-only microscope overlap
        # as the calibrated model mismatch baseline, then reject any new pair
        # or penetration worsening during subsequent following motion.
        calibration_baseline_samples=(
            (transition_q_paths[0][-1], right_q[first_checkpoint]),
        ),
    )
    audit["maximum_ik_joint_step_rad"] = maximum_step
    audit["maximum_allowed_ik_joint_step_rad"] = float(maximum_ik_joint_step_rad)
    audit["ik_branch_continuity_accepted"] = bool(
        maximum_step <= float(maximum_ik_joint_step_rad)
    )
    audit["accepted"] = bool(
        audit["accepted"] and audit["ik_branch_continuity_accepted"]
    )
    return StoppedObserverPlan(
        physical_arm=observer_side,
        carrier_arm="right",
        checkpoint_poses_wxyz_xyz=checkpoint_poses,
        checkpoint_q_physical_rad=np.asarray(checkpoint_q),
        transition_pose_paths=tuple(transition_pose_paths),
        transition_q_paths=tuple(transition_q_paths),
        return_pose_path=return_pose_path,
        return_q_path=return_q_path,
        audit=audit,
    )


@dataclass(frozen=True)
class TimedCartesianPose:
    t_s: float
    pose_wxyz_xyz: np.ndarray


def sample_pose_path_at_speed(
    poses_wxyz_xyz,
    *,
    speed_m_s: float,
    control_hz: float = 30.0,
) -> list[TimedCartesianPose]:
    """Sample a pose polyline continuously at the teleoperation rate."""

    poses = np.asarray(poses_wxyz_xyz, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2:
        raise ValueError("pose path must be finite Nx7")
    if not np.all(np.isfinite(poses)) or min(speed_m_s, control_hz) <= 0.0:
        raise ValueError("pose path and sampling parameters must be finite and positive")
    linear_distance = np.linalg.norm(np.diff(poses[:, 4:], axis=0), axis=1)
    angular_distance = np.asarray(
        [
            _quaternion_distance_rad(first[:4], second[:4])
            for first, second in zip(poses, poses[1:])
        ]
    )
    # Treat one radian of wrist rotation as 12 cm of path length.  At the
    # default 40 mm/s this caps in-place orientation changes near 19 deg/s.
    distance = np.maximum(linear_distance, 0.12 * angular_distance)
    cumulative = np.r_[0.0, np.cumsum(distance)]
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    poses = poses[keep]
    cumulative = cumulative[keep]
    duration = float(cumulative[-1] / speed_m_s)
    if duration <= 0.0:
        raise ValueError("pose path has no Cartesian motion")
    count = max(1, int(math.ceil(duration * control_hz)))
    times = np.linspace(duration / count, duration, count)
    distances = np.minimum(times * speed_m_s, cumulative[-1])
    positions = np.column_stack(
        [np.interp(distances, cumulative, poses[:, 4 + axis]) for axis in range(3)]
    )
    rotations = Slerp(
        cumulative,
        Rotation.from_quat(poses[:, [1, 2, 3, 0]]),
    )(distances).as_quat()
    sampled = np.c_[rotations[:, [3, 0, 1, 2]], positions]
    sampled[-1] = poses[-1]
    return [
        TimedCartesianPose(float(timestamp), pose.copy())
        for timestamp, pose in zip(times, sampled)
    ]


class CartesianAirTransportStreamer:
    """Send one physical arm through the same 30 Hz Cartesian path as teleop."""

    def __init__(
        self,
        rpc,
        physical_arm: str,
        *,
        torque_limit_nm: Sequence[float],
        torque_samples: int = 5,
        control_hz: float = 30.0,
        preview_time_s: float = 0.05,
        tracking_interval: int = 15,
        maximum_tracking_position_error_m: float = 0.10,
        maximum_tracking_rotation_error_rad: float = 1.5,
        final_settle_s: float = 0.8,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self.rpc = rpc
        self.side = _side(physical_arm)
        self.torque_limit = np.asarray(torque_limit_nm, dtype=float).reshape(6)
        if np.any(self.torque_limit <= 0.0):
            raise ValueError("torque limits must be positive")
        self.torque_samples = int(torque_samples)
        self.control_hz = float(control_hz)
        self.preview_time_s = float(preview_time_s)
        self.tracking_interval = int(tracking_interval)
        self.maximum_tracking_position_error_m = float(
            maximum_tracking_position_error_m
        )
        self.maximum_tracking_rotation_error_rad = float(
            maximum_tracking_rotation_error_rad
        )
        self.final_settle_s = float(final_settle_s)
        if self.final_settle_s < 0.0 or self.final_settle_s > 3.0:
            raise ValueError("final_settle_s must be within [0, 3] seconds")
        self.clock = clock
        self.sleep = sleep
        self.torque_warning_count = 0
        self.last_torque_warning = None
        self._torque_strikes = 0

    def _call(self, stem: str, *args, **kwargs):
        return getattr(self.rpc, f"{stem}_{self.side}_" + kwargs.pop("suffix"))(
            *args, **kwargs
        )

    def measured_pose(self) -> np.ndarray:
        pose = self._call("get", suffix="ee_pose")
        result = np.asarray(pose.parameters(), dtype=float)
        if result.shape != (7,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"invalid measured {self.side} EE pose")
        return result

    def hold_measured(self) -> None:
        pose = mink.SE3(self.measured_pose())
        gripper = float(
            np.asarray(self._call("get", suffix="gripper_exact"), dtype=float).reshape(-1)[0]
        )
        accepted = self._call(
            "set",
            pose,
            gripper_target=gripper,
            preview_time=0.2,
            suffix="ee_target",
        )
        if accepted is False:
            raise RuntimeError(f"{self.side} measured-pose hold was rejected")

    def _observe_torque(self, stage: str) -> None:
        torque = np.abs(
            np.asarray(self._call("get", suffix="joint_torque"), dtype=float)
        )
        exceeded = (
            torque.shape != (6,)
            or not np.all(np.isfinite(torque))
            or bool(np.any(torque > self.torque_limit))
        )
        self._torque_strikes = self._torque_strikes + 1 if exceeded else 0
        if self._torque_strikes >= self.torque_samples:
            self.torque_warning_count += 1
            self.last_torque_warning = {
                "stage": stage,
                "sample_nm": torque.tolist(),
                "limit_nm": self.torque_limit.tolist(),
                "policy": "observe_only",
            }
            self._torque_strikes = 0

    def execute(
        self,
        pose_path,
        *,
        speed_m_s: float,
        gripper_open_ratio: float,
        stage: str,
    ) -> dict:
        samples = sample_pose_path_at_speed(
            pose_path, speed_m_s=speed_m_s, control_hz=self.control_hz
        )
        started = self.clock()
        maximum_position_error = 0.0
        maximum_rotation_error = 0.0
        settle_command_count = 0
        motion_error = None
        try:
            for index, sample in enumerate(samples, start=1):
                self._observe_torque(stage)
                commanded = mink.SE3(sample.pose_wxyz_xyz)
                accepted = self._call(
                    "set",
                    commanded,
                    gripper_target=float(gripper_open_ratio),
                    preview_time=self.preview_time_s,
                    suffix="ee_target",
                )
                if accepted is False:
                    raise RuntimeError(
                        f"{self.side} Cartesian setpoint {index}/{len(samples)} rejected"
                    )
                if index % self.tracking_interval == 0 or index == len(samples):
                    measured = self.measured_pose()
                    position_error = float(
                        np.linalg.norm(measured[4:] - sample.pose_wxyz_xyz[4:])
                    )
                    rotation_error = _quaternion_distance_rad(
                        measured[:4], sample.pose_wxyz_xyz[:4]
                    )
                    maximum_position_error = max(maximum_position_error, position_error)
                    maximum_rotation_error = max(maximum_rotation_error, rotation_error)
                    if (
                        position_error > self.maximum_tracking_position_error_m
                        or rotation_error > self.maximum_tracking_rotation_error_rad
                    ):
                        raise RuntimeError(
                            f"{self.side} stopped following teleop path: "
                            f"position={position_error:.3f}m rotation={rotation_error:.3f}rad"
                        )
                remaining = started + sample.t_s - self.clock()
                if remaining < -2.0 / self.control_hz:
                    raise RuntimeError(f"{self.side} teleop stream missed its deadline")
                if remaining > 0.0:
                    self.sleep(remaining)
            # The Piper target is timestamped preview_time_s into the future.
            # Latching the measured pose immediately after the final sample
            # freezes tracking lag as a real endpoint error.  Keep publishing
            # the exact endpoint through a bounded settle window, then measure
            # and only afterwards convert it to a measured-pose hold.
            settle_count = int(math.ceil(self.final_settle_s * self.control_hz))
            settle_started = self.clock()
            final_pose = mink.SE3(samples[-1].pose_wxyz_xyz)
            for settle_index in range(1, settle_count + 1):
                self._observe_torque(f"{stage}/settle")
                accepted = self._call(
                    "set",
                    final_pose,
                    gripper_target=float(gripper_open_ratio),
                    preview_time=self.preview_time_s,
                    suffix="ee_target",
                )
                if accepted is False:
                    raise RuntimeError(f"{self.side} endpoint settle target rejected")
                settle_command_count += 1
                deadline = settle_started + settle_index / self.control_hz
                remaining = deadline - self.clock()
                if remaining > 0.0:
                    self.sleep(remaining)
            measured = self.measured_pose()
            final_position_error = float(
                np.linalg.norm(measured[4:] - samples[-1].pose_wxyz_xyz[4:])
            )
            final_rotation_error = _quaternion_distance_rad(
                measured[:4], samples[-1].pose_wxyz_xyz[:4]
            )
            maximum_position_error = max(maximum_position_error, final_position_error)
            maximum_rotation_error = max(maximum_rotation_error, final_rotation_error)
            if (
                final_position_error > self.maximum_tracking_position_error_m
                or final_rotation_error > self.maximum_tracking_rotation_error_rad
            ):
                raise RuntimeError(
                    f"{self.side} endpoint did not settle: "
                    f"position={final_position_error:.3f}m "
                    f"rotation={final_rotation_error:.3f}rad"
                )
        except BaseException as error:
            motion_error = error
            raise
        finally:
            try:
                self.hold_measured()
            except BaseException as hold_error:
                if motion_error is None:
                    raise
                add_note = getattr(motion_error, "add_note", None)
                if add_note is not None:
                    add_note(f"measured-pose hold also failed: {hold_error!r}")
        return {
            "physical_arm": self.side,
            "command_path": f"set_{self.side}_ee_target",
            "control_hz": self.control_hz,
            "preview_time_s": self.preview_time_s,
            "sample_count": len(samples),
            "endpoint_settle_s": self.final_settle_s,
            "endpoint_settle_command_count": settle_command_count,
            "stage": stage,
            "maximum_tracking_position_error_m": maximum_position_error,
            "maximum_tracking_rotation_error_rad": maximum_rotation_error,
            "torque_policy": "observe_only",
            "torque_warning_count": self.torque_warning_count,
            "last_torque_warning": self.last_torque_warning,
        }

    def refine_jaw_level(
        self,
        reference: JawLevelReference,
        *,
        gripper_open_ratio: float,
        maximum_attempts: int = 3,
        maximum_correction_deg: float = 5.0,
        maximum_total_correction_deg: float = 8.0,
        correction_duration_s: float = 0.5,
        settle_s: float = 0.6,
        maximum_xyz_drift_m: float = 0.005,
        maximum_xyz_correction_per_attempt_m: float = 0.005,
        maximum_xyz_command_offset_m: float = 0.012,
        hard_xyz_drift_m: float = 0.015,
    ) -> dict:
        """Level the jaw while rejecting coupled XYZ drift with bounded feedback.

        Cartesian IK can leave a repeatable orientation residual even when the
        requested endpoint is exactly level.  Estimate that residual from the
        measured EE rotation, apply its inverse to the next command, and feed
        the measured Cartesian residual back into the next command.  Stop only
        when both the normal level gate and the final XYZ tolerance pass.
        """

        maximum_attempts = int(maximum_attempts)
        if maximum_attempts < 0 or maximum_attempts > 5:
            raise ValueError("maximum_attempts must be within [0, 5]")
        if min(
            maximum_correction_deg,
            maximum_total_correction_deg,
            correction_duration_s,
            settle_s,
            maximum_xyz_drift_m,
            maximum_xyz_correction_per_attempt_m,
            maximum_xyz_command_offset_m,
            hard_xyz_drift_m,
        ) <= 0.0:
            raise ValueError("jaw-level refinement limits must be positive")
        measured = self.measured_pose()
        fixed_xyz = measured[4:].copy()
        command_rotation = Rotation.from_quat(measured[[1, 2, 3, 0]])
        command_xyz = fixed_xyz.copy()
        total_correction_rad = 0.0
        history = []

        def record(attempt: int):
            current = self.measured_pose()
            assessment = assess_jaw_level(current, reference, planned=False)
            drift = float(np.linalg.norm(current[4:] - fixed_xyz))
            item = {
                "attempt": int(attempt),
                "measured_pose_wxyz_xyz": current.tolist(),
                "assessment": assessment.to_dict(),
                "xyz_drift_m": drift,
                "total_command_correction_deg": math.degrees(total_correction_rad),
            }
            history.append(item)
            if drift > hard_xyz_drift_m:
                raise RuntimeError(
                    f"{self.side} level refinement translated {drift:.4f}m"
                )
            return current, assessment, drift

        measured, assessment, xyz_drift = record(0)
        for attempt in range(1, maximum_attempts + 1):
            if assessment.accepted and xyz_drift <= maximum_xyz_drift_m:
                break
            desired = leveled_pose(measured, reference)
            desired_rotation = Rotation.from_quat(desired[[1, 2, 3, 0]])
            measured_rotation = Rotation.from_quat(measured[[1, 2, 3, 0]])
            correction_vector = (
                desired_rotation * measured_rotation.inv()
            ).as_rotvec()
            correction_norm = float(np.linalg.norm(correction_vector))
            allowed_this_attempt = min(
                math.radians(maximum_correction_deg),
                math.radians(maximum_total_correction_deg) - total_correction_rad,
            )
            if allowed_this_attempt <= 0.0:
                correction_vector[:] = 0.0
                correction_norm = 0.0
            elif correction_norm > allowed_this_attempt:
                correction_vector *= allowed_this_attempt / correction_norm
                correction_norm = allowed_this_attempt
            correction = Rotation.from_rotvec(correction_vector)
            next_rotation = correction * command_rotation
            xyz_error = fixed_xyz - measured[4:]
            xyz_error_norm = float(np.linalg.norm(xyz_error))
            if xyz_error_norm > maximum_xyz_correction_per_attempt_m:
                xyz_error *= maximum_xyz_correction_per_attempt_m / xyz_error_norm
            next_command_xyz = command_xyz + xyz_error
            command_offset = next_command_xyz - fixed_xyz
            command_offset_norm = float(np.linalg.norm(command_offset))
            if command_offset_norm > maximum_xyz_command_offset_m:
                next_command_xyz = fixed_xyz + (
                    command_offset
                    * maximum_xyz_command_offset_m
                    / command_offset_norm
                )
            if (
                correction_norm < math.radians(0.02)
                and float(np.linalg.norm(next_command_xyz - command_xyz)) < 0.0002
            ):
                break
            next_xyzw = next_rotation.as_quat()
            next_pose = np.r_[next_xyzw[[3, 0, 1, 2]], next_command_xyz]
            transition = np.asarray([measured, next_pose], dtype=float)
            equivalent_motion_m = max(
                0.12 * correction_norm,
                float(np.linalg.norm(next_command_xyz - measured[4:])),
            )
            samples = sample_pose_path_at_speed(
                transition,
                # Angular duration is encoded as 0.12 m/rad by the sampler.
                speed_m_s=max(
                    1e-6,
                    equivalent_motion_m / float(correction_duration_s),
                ),
                control_hz=self.control_hz,
            )
            started = self.clock()
            for index, sample in enumerate(samples, start=1):
                self._observe_torque(f"jaw_level_refinement_{attempt}")
                accepted = self._call(
                    "set",
                    mink.SE3(sample.pose_wxyz_xyz),
                    gripper_target=float(gripper_open_ratio),
                    preview_time=self.preview_time_s,
                    suffix="ee_target",
                )
                if accepted is False:
                    raise RuntimeError(
                        f"{self.side} jaw-level correction target rejected"
                    )
                remaining = started + sample.t_s - self.clock()
                if remaining > 0.0:
                    self.sleep(remaining)
            settle_count = max(1, int(math.ceil(settle_s * self.control_hz)))
            settle_started = self.clock()
            for index in range(1, settle_count + 1):
                self._observe_torque(f"jaw_level_refinement_{attempt}/settle")
                accepted = self._call(
                    "set",
                    mink.SE3(next_pose),
                    gripper_target=float(gripper_open_ratio),
                    preview_time=self.preview_time_s,
                    suffix="ee_target",
                )
                if accepted is False:
                    raise RuntimeError(
                        f"{self.side} jaw-level settle target rejected"
                    )
                remaining = settle_started + index / self.control_hz - self.clock()
                if remaining > 0.0:
                    self.sleep(remaining)
            total_correction_rad += correction_norm
            command_rotation = next_rotation
            command_xyz = next_command_xyz
            measured, assessment, xyz_drift = record(attempt)
        self.hold_measured()
        accepted = bool(assessment.accepted and xyz_drift <= maximum_xyz_drift_m)
        return {
            "accepted": accepted,
            "physical_arm": self.side,
            "fixed_xyz_m": fixed_xyz.tolist(),
            "attempts_used": len(history) - 1,
            "maximum_attempts": maximum_attempts,
            "maximum_correction_deg": float(maximum_correction_deg),
            "maximum_total_correction_deg": float(maximum_total_correction_deg),
            "maximum_xyz_drift_m": float(maximum_xyz_drift_m),
            "maximum_xyz_correction_per_attempt_m": float(
                maximum_xyz_correction_per_attempt_m
            ),
            "maximum_xyz_command_offset_m": float(maximum_xyz_command_offset_m),
            "hard_xyz_drift_m": float(hard_xyz_drift_m),
            "history": history,
            "final_pose_wxyz_xyz": measured.tolist(),
            "final_assessment": assessment.to_dict(),
        }

"""Closed-loop primitives for autonomous, vision-guided right-arm motion.

The module is deliberately hardware-light: perception, collision checking and
the robot RPC are injected.  This makes the same planner/executor usable in
offline replay, MuJoCo validation and on the physical robot.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import mink
import numpy as np


CONTROL_HZ = 30.0
PREVIEW_TIME_S = 0.05
CHUNK_S = 0.5


class AutonomousStop(RuntimeError):
    """A fail-closed stop that must not be converted into a replan."""


@dataclass(frozen=True)
class Pose:
    """End-effector pose, with a wxyz quaternion and robot-frame xyz."""

    wxyz: tuple[float, float, float, float]
    xyz: tuple[float, float, float]

    @classmethod
    def from_se3(cls, pose) -> "Pose":
        return cls(
            tuple(float(value) for value in pose.rotation().wxyz),
            tuple(float(value) for value in pose.translation()),
        )

    def as_se3(self):
        values = np.asarray((*self.wxyz, *self.xyz), dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise AutonomousStop("refusing a non-finite Cartesian target")
        return mink.SE3(values)


@dataclass(frozen=True)
class TimedWaypoint:
    t_s: float
    pose: Pose
    stage: str


@dataclass
class TrajectoryPlan:
    plan_id: str
    created_at_s: float
    target_xyz_m: tuple[float, float, float]
    waypoints: list[TimedWaypoint]
    minimum_clearance_m: float | None = None
    metadata: dict = field(default_factory=dict)

    def chunks(self, chunk_s: float = CHUNK_S) -> list[list[TimedWaypoint]]:
        if chunk_s <= 0:
            raise ValueError("chunk_s must be positive")
        chunks: list[list[TimedWaypoint]] = []
        current: list[TimedWaypoint] = []
        boundary = chunk_s
        for waypoint in self.waypoints:
            while current and waypoint.t_s > boundary + 1e-9:
                chunks.append(current)
                current = []
                boundary += chunk_s
            current.append(waypoint)
        if current:
            chunks.append(current)
        return chunks


@dataclass(frozen=True)
class SceneSnapshot:
    timestamp_s: float
    target_xyz_m: tuple[float, float, float] | None
    target_instance_id: str | None
    ee_pose: Pose
    lid_visible: bool
    blue_marker_visible: bool
    right_marker_px: tuple[float, float] | None = None
    estimated_gripper_lid_m: float | None = None
    predicted_clearance_m: float | None = None


@dataclass(frozen=True)
class ReplanPolicy:
    maximum_observation_age_s: float = 1.0
    maximum_target_shift_m: float = 0.010
    maximum_position_error_m: float = 0.005
    maximum_rotation_error_deg: float = 3.0
    minimum_clearance_m: float = 0.015


@dataclass(frozen=True)
class ReplanDecision:
    action: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PregraspGate:
    allowed: bool
    reasons: tuple[str, ...]
    pixel_error: float | None


class ESDFGrid:
    """Trilinear robot-frame query over a conservative saved ESDF volume."""

    def __init__(
        self,
        esdf_m,
        origin_xyz_m,
        voxel_size_m,
        *,
        T_esdf_robot=None,
        body_radius_m: float = 0.025,
    ):
        self.esdf = np.asarray(esdf_m, dtype=float)
        self.origin = np.asarray(origin_xyz_m, dtype=float)
        self.voxel = float(np.asarray(voxel_size_m).reshape(()))
        self.T_esdf_robot = np.eye(4) if T_esdf_robot is None else np.asarray(
            T_esdf_robot, dtype=float
        )
        self.body_radius_m = float(body_radius_m)
        if self.esdf.ndim != 3 or self.origin.shape != (3,):
            raise ValueError("invalid ESDF grid")
        if self.voxel <= 0 or self.T_esdf_robot.shape != (4, 4):
            raise ValueError("invalid ESDF transform or voxel size")

    @classmethod
    def from_npz(cls, path, *, T_esdf_robot, body_radius_m=0.025):
        with np.load(path) as data:
            return cls(
                data["esdf_m"],
                data["origin_xyz_m"],
                data["voxel_size_m"],
                T_esdf_robot=T_esdf_robot,
                body_radius_m=body_radius_m,
            )

    def point_clearance(
        self, xyz_robot: Sequence[float], *, radius_m: float = 0.0
    ) -> float | None:
        point_robot = np.asarray((*xyz_robot, 1.0), dtype=float)
        point = (self.T_esdf_robot @ point_robot)[:3]
        # Stored arrays are z,y,x and voxel values live at cell centres.
        xyz_index = (point - self.origin) / self.voxel - 0.5
        lower = np.floor(xyz_index).astype(int)
        fraction = xyz_index - lower
        nx = self.esdf.shape[2]
        ny = self.esdf.shape[1]
        nz = self.esdf.shape[0]
        if np.any(lower < 0) or np.any(lower + 1 >= (nx, ny, nz)):
            return None
        values = np.empty((2, 2, 2), dtype=float)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    values[dz, dy, dx] = self.esdf[
                        lower[2] + dz, lower[1] + dy, lower[0] + dx
                    ]
        if not np.all(np.isfinite(values)):
            return None
        wx, wy, wz = fraction
        interpolated = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    weight = (
                        (wx if dx else 1 - wx)
                        * (wy if dy else 1 - wy)
                        * (wz if dz else 1 - wz)
                    )
                    interpolated += weight * values[dz, dy, dx]
        return float(interpolated - float(radius_m))

    def clearance(self, pose: Pose) -> float | None:
        return self.point_clearance(pose.xyz, radius_m=self.body_radius_m)


class MuJoCoIKValidator:
    """Sequential 6-DOF IK, joint-limit and new-contact validation."""

    def __init__(
        self,
        model_path,
        right_q,
        *,
        left_q=None,
        maximum_joint_step_rad: float = 0.20,
        joint_names: Sequence[str] | None = None,
        left_joint_names: Sequence[str] | None = None,
        ee_frame: str = "left_arm_ee",
        right_q_offset=None,
        left_q_offset=None,
        contact_body_prefix: str = "left_arm_",
        link_capsules=None,
        attached_spheres=None,
    ):
        import mujoco

        from robot.arm.ik_solver import SingleArmIK

        self.mujoco = mujoco
        # ConeE's physical right arm intentionally uses the model's
        # ``left_arm_*`` branch (see ArmNode); matching that production mapping
        # is essential or FK is displaced by roughly the inter-arm baseline.
        self.joint_names = list(joint_names or [
            f"left_arm_joint{index}" for index in range(1, 7)
        ])
        self.right_q_offset = np.zeros(6) if right_q_offset is None else np.asarray(
            right_q_offset, dtype=float
        )
        self.left_q_offset = np.zeros(6) if left_q_offset is None else np.asarray(
            left_q_offset, dtype=float
        )
        if self.right_q_offset.shape != (6,) or self.left_q_offset.shape != (6,):
            raise ValueError("joint-space model offsets must contain six values")
        self.contact_body_prefix = str(contact_body_prefix)
        self.solver = SingleArmIK(
            str(model_path),
            joint_names=self.joint_names,
            ee_frame=ee_frame,
        )
        if left_q is not None:
            left_q = np.asarray(left_q, dtype=float)
            if left_q.shape != (6,) or not np.all(np.isfinite(left_q)):
                raise ValueError("left_q must contain six finite values")
            names = list(left_joint_names or [
                f"right_arm_joint{index}" for index in range(1, 7)
            ])
            left_ids = np.array([
                self.solver.model.joint(name).qposadr[0] for name in names
            ])
            configuration = self.solver.configuration.q.copy()
            configuration[left_ids] = left_q + self.left_q_offset
            self.solver.configuration.update(configuration)
        right_q = np.asarray(right_q, dtype=float)
        if right_q.shape != (6,) or not np.all(np.isfinite(right_q)):
            raise ValueError("right_q must contain six finite values")
        self.solver.init(right_q + self.right_q_offset)
        self.previous_q = right_q.copy()
        self.maximum_joint_step_rad = float(maximum_joint_step_rad)
        self.data = mujoco.MjData(self.solver.model)
        self.baseline_contacts = self._contact_pairs()
        self.last_q = right_q.copy()
        self.q_waypoints: list[list[float]] = []
        # Conservative link capsules in the physical-right branch of the
        # historically crossed ConeE production MJCF. Semantic right/left
        # namespaces are configured explicitly by their caller.
        self.link_capsules = tuple(link_capsules or (
            ("left_arm_link0", "left_arm_link1", 0.058),
            ("left_arm_link1", "left_arm_link2", 0.052),
            ("left_arm_link2", "left_arm_link3", 0.055),
            ("left_arm_link3", "left_arm_link4", 0.050),
            ("left_arm_link4", "left_arm_link5", 0.044),
            ("left_arm_link5", "left_arm_link6", 0.042),
            ("left_arm_link6", "left_arm_gripper_base", 0.035),
        ))
        self.attached_spheres = tuple(attached_spheres or (
            ("left_arm_gripper_base", 0.055),
            ("left_arm_iphone_body", 0.070),
        ))

    def set_right_q(self, right_q: Sequence[float]) -> None:
        """Pose the physical right arm directly for recorded-trajectory replay."""

        right_q = np.asarray(right_q, dtype=float)
        if right_q.shape != (6,) or not np.all(np.isfinite(right_q)):
            raise ValueError("right_q must contain six finite values")
        self.solver.init(right_q + self.right_q_offset)
        self.previous_q = right_q.copy()
        self.last_q = right_q.copy()

    def validate_q(self, right_q: Sequence[float]) -> np.ndarray:
        """Validate and accept one physical-right joint-space waypoint."""

        right_q = np.asarray(right_q, dtype=float)
        if right_q.shape != (6,) or not np.all(np.isfinite(right_q)):
            raise AutonomousStop("refusing a non-finite right-arm waypoint")
        ranges = np.asarray(
            [self.solver.model.joint(name).range for name in self.joint_names],
            dtype=float,
        )
        model_q = right_q + self.right_q_offset
        if np.any(model_q < ranges[:, 0]) or np.any(model_q > ranges[:, 1]):
            raise AutonomousStop("right-arm waypoint exceeds MuJoCo joint limits")
        if np.max(np.abs(right_q - self.previous_q)) > self.maximum_joint_step_rad:
            raise AutonomousStop("right-arm joint waypoint changed discontinuously")
        self.solver.init(right_q + self.right_q_offset)
        current_contacts = self._contact_pairs()
        new_contacts = current_contacts - self.baseline_contacts
        if new_contacts:
            raise AutonomousStop(
                f"MuJoCo predicts new right-arm contacts: {sorted(new_contacts)}"
            )
        self.previous_q = right_q.copy()
        self.last_q = right_q.copy()
        self.q_waypoints.append(right_q.tolist())
        return right_q

    def _contact_pairs(self):
        self.data.qpos[:] = self.solver.configuration.q
        self.mujoco.mj_forward(self.solver.model, self.data)
        pairs = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self.solver.model.geom(int(contact.geom1))
            second = self.solver.model.geom(int(contact.geom2))
            first_body = self.solver.model.body(int(first.bodyid[0])).name
            second_body = self.solver.model.body(int(second.bodyid[0])).name
            if (
                self.contact_body_prefix in first_body
                or self.contact_body_prefix in second_body
            ):
                pairs.add(tuple(sorted((first.name, second.name))))
        return pairs

    def validate(self, pose: Pose) -> np.ndarray:
        q, solved = self.solver.solve_ik(
            pose.as_se3(), max_iter=30, pos_eps=1e-3, rot_eps=1e-3
        )
        q = np.asarray(q, dtype=float) - self.right_q_offset
        if not solved or not np.all(np.isfinite(q)):
            raise AutonomousStop("MuJoCo IK did not solve a trajectory waypoint")
        if np.max(np.abs(q - self.previous_q)) > self.maximum_joint_step_rad:
            raise AutonomousStop("MuJoCo IK changed branch discontinuously")
        current_contacts = self._contact_pairs()
        new_contacts = current_contacts - self.baseline_contacts
        if new_contacts:
            raise AutonomousStop(
                f"MuJoCo predicts new right-arm contacts: {sorted(new_contacts)}"
            )
        self.previous_q = q.copy()
        self.last_q = q.copy()
        self.q_waypoints.append(q.tolist())
        return q

    def robot_proxy_samples(self, *, sample_spacing_m: float = 0.0075):
        self.data.qpos[:] = self.solver.configuration.q
        self.mujoco.mj_forward(self.solver.model, self.data)
        spacing = max(0.005, float(sample_spacing_m))
        samples: list[tuple[np.ndarray, float]] = []

        def body_xyz(name):
            body_id = self.solver.model.body(name).id
            return np.asarray(self.data.xpos[body_id], dtype=float)

        for first_name, second_name, radius in self.link_capsules:
            first = body_xyz(first_name)
            second = body_xyz(second_name)
            length = float(np.linalg.norm(second - first))
            count = max(2, int(math.ceil(length / spacing)) + 1)
            for fraction in np.linspace(0.0, 1.0, count):
                samples.append(
                    (first + fraction * (second - first), float(radius))
                )
        for body_name, radius in self.attached_spheres:
            samples.append((body_xyz(body_name), float(radius)))
        return samples

    def esdf_clearance(
        self, esdf: ESDFGrid, *, sample_spacing_m: float | None = None
    ) -> float | None:
        """Minimum measured clearance for all right-arm link proxies."""

        spacing = (
            max(esdf.voxel * 0.75, 0.005)
            if sample_spacing_m is None
            else float(sample_spacing_m)
        )
        clearances: list[float] = []
        for point, radius in self.robot_proxy_samples(
            sample_spacing_m=spacing
        ):
            value = esdf.point_clearance(point, radius_m=radius)
            if value is None:
                return None
            clearances.append(value)
        return min(clearances) if clearances else None


class AnalyticObstacleSet:
    """Known-object proxies supplementing unreliable transparent depth."""

    def __init__(self, objects: Sequence[dict]):
        self.objects = []
        for item in objects:
            if item.get("status") != "confirmed":
                continue
            if item.get("role") == "target_lid":
                continue
            geometry = item.get("geometry") or {}
            pose = np.asarray(item.get("pose_robot"), dtype=float)
            if geometry.get("type") != "cylinder" or pose.shape != (4, 4):
                continue
            radius = float(geometry["radius_m"])
            height = float(geometry["height_m"])
            margin = float(geometry.get("uncertainty_margin_m", 0.0))
            transform = pose.copy()
            if geometry.get("pose_anchor") == "top_center":
                transform[:3, 3] -= transform[:3, 2] * (height / 2.0)
            self.objects.append(
                (
                    str(item.get("instance_id", "obstacle")),
                    np.linalg.inv(transform),
                    radius,
                    height,
                    margin,
                )
            )

    @staticmethod
    def _cylinder_sdf(local_xyz, radius, height):
        radial = float(np.linalg.norm(local_xyz[:2])) - radius
        axial = abs(float(local_xyz[2])) - height / 2.0
        outside = float(np.linalg.norm(np.maximum([radial, axial], 0.0)))
        inside = min(max(radial, axial), 0.0)
        return outside + inside

    def clearance(self, samples) -> float:
        if not self.objects:
            return math.inf
        result = math.inf
        for point, body_radius in samples:
            homogeneous = np.asarray((*point, 1.0), dtype=float)
            for _, inverse, radius, height, margin in self.objects:
                local = (inverse @ homogeneous)[:3]
                result = min(
                    result,
                    self._cylinder_sdf(local, radius, height)
                    - body_radius
                    - margin,
                )
        return float(result)


class CompositePoseValidator:
    """Require both measured-space clearance and MuJoCo kinematic validity."""

    def __init__(
        self,
        esdf: ESDFGrid,
        mujoco_validator: MuJoCoIKValidator,
        analytic_obstacles: AnalyticObstacleSet | None = None,
    ):
        self.esdf = esdf
        self.mujoco = mujoco_validator
        self.analytic_obstacles = analytic_obstacles

    def __call__(self, pose: Pose) -> float | None:
        self.mujoco.validate(pose)
        measured = self.mujoco.esdf_clearance(self.esdf)
        if measured is None or self.analytic_obstacles is None:
            return measured
        analytic = self.analytic_obstacles.clearance(
            self.mujoco.robot_proxy_samples(
                sample_spacing_m=max(self.esdf.voxel * 0.75, 0.005)
            )
        )
        return min(measured, analytic)


def _normalise_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid quaternion")
    return quaternion / norm


def quaternion_angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    first_q = _normalise_quaternion(first)
    second_q = _normalise_quaternion(second)
    dot = float(np.clip(abs(np.dot(first_q, second_q)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def interpolate_pose(start: Pose, finish: Pose, progress: float) -> Pose:
    progress = float(np.clip(progress, 0.0, 1.0))
    start_q = _normalise_quaternion(start.wxyz)
    finish_q = _normalise_quaternion(finish.wxyz)
    if np.dot(start_q, finish_q) < 0:
        finish_q = -finish_q
    quaternion = _normalise_quaternion((1.0 - progress) * start_q + progress * finish_q)
    xyz = (1.0 - progress) * np.asarray(start.xyz) + progress * np.asarray(finish.xyz)
    return Pose(tuple(quaternion), tuple(float(value) for value in xyz))


def minimum_jerk_segment(
    start: Pose,
    finish: Pose,
    duration_s: float,
    *,
    start_t_s: float = 0.0,
    stage: str,
    control_hz: float = CONTROL_HZ,
) -> list[TimedWaypoint]:
    """Sample a zero-velocity/acceleration minimum-jerk segment."""

    if duration_s <= 0 or control_hz <= 0:
        raise ValueError("duration and control frequency must be positive")
    samples = max(1, int(math.ceil(duration_s * control_hz)))
    waypoints = []
    for index in range(1, samples + 1):
        fraction = index / samples
        smooth = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
        waypoints.append(
            TimedWaypoint(
                start_t_s + fraction * duration_s,
                interpolate_pose(start, finish, smooth),
                stage,
            )
        )
    return waypoints


def segment_duration(
    start: Pose,
    finish: Pose,
    *,
    maximum_linear_speed_m_s: float = 0.035,
    maximum_angular_speed_deg_s: float = 20.0,
    minimum_duration_s: float = 0.5,
) -> float:
    distance = float(np.linalg.norm(np.asarray(finish.xyz) - np.asarray(start.xyz)))
    angle = quaternion_angle_deg(start.wxyz, finish.wxyz)
    return max(
        minimum_duration_s,
        1.875 * distance / maximum_linear_speed_m_s,
        1.875 * angle / maximum_angular_speed_deg_s,
    )


def plan_lift_translate_descend(
    start: Pose,
    target_xyz_m: Sequence[float],
    *,
    grasp_orientation_wxyz: Sequence[float] | None = None,
    lift_m: float = 0.040,
    pregrasp_height_m: float = 0.015,
    final_height_m: float = 0.003,
    validator: Callable[[Pose], float | None] | None = None,
    transit_xy_m: Sequence[float] | None = None,
    now_s: float | None = None,
) -> TrajectoryPlan:
    """Plan the conservative stage order: lift, XY, pregrasp, descend."""

    target = np.asarray(target_xyz_m, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_xyz_m must be a finite 3-vector")
    orientation = tuple(
        _normalise_quaternion(
            start.wxyz if grasp_orientation_wxyz is None else grasp_orientation_wxyz
        )
    )
    safe_z = max(float(start.xyz[2]) + lift_m, float(target[2]) + pregrasp_height_m + lift_m)
    stages = [
        ("lift", Pose(start.wxyz, (start.xyz[0], start.xyz[1], safe_z))),
    ]
    if transit_xy_m is not None:
        transit = np.asarray(transit_xy_m, dtype=float)
        if transit.shape != (2,) or not np.all(np.isfinite(transit)):
            raise ValueError("transit_xy_m must be a finite 2-vector")
        stages.append(
            (
                "translate_detour",
                Pose(orientation, (float(transit[0]), float(transit[1]), safe_z)),
            )
        )
    stages.extend(
        [
            ("translate_xy", Pose(orientation, (float(target[0]), float(target[1]), safe_z))),
            (
            "approach",
            Pose(orientation, (float(target[0]), float(target[1]), float(target[2] + pregrasp_height_m))),
            ),
            (
            "descend",
            Pose(orientation, (float(target[0]), float(target[1]), float(target[2] + final_height_m))),
            ),
        ]
    )
    cursor = start
    elapsed = 0.0
    waypoints: list[TimedWaypoint] = []
    minimum_clearance = math.inf
    for stage, finish in stages:
        duration = segment_duration(cursor, finish)
        segment = minimum_jerk_segment(
            cursor, finish, duration, start_t_s=elapsed, stage=stage
        )
        for waypoint in segment:
            if validator is None:
                continue
            clearance = validator(waypoint.pose)
            if clearance is None or not np.isfinite(clearance):
                raise AutonomousStop(
                    f"unknown collision clearance at {stage}; unknown is not free"
                )
            minimum_clearance = min(minimum_clearance, float(clearance))
        waypoints.extend(segment)
        elapsed += duration
        cursor = finish
    created = time.time() if now_s is None else float(now_s)
    return TrajectoryPlan(
        plan_id=f"plan-{int(created * 1_000_000)}",
        created_at_s=created,
        target_xyz_m=tuple(float(value) for value in target),
        waypoints=waypoints,
        minimum_clearance_m=None if math.isinf(minimum_clearance) else minimum_clearance,
        metadata={"stage_order": [name for name, _ in stages], "control_hz": CONTROL_HZ},
    )


def select_collision_aware_plan(
    start: Pose,
    target_xyz_m: Sequence[float],
    *,
    validator_factory: Callable[[], CompositePoseValidator],
    grasp_orientation_wxyz: Sequence[float] | None = None,
    lift_m: float = 0.040,
    pregrasp_height_m: float = 0.015,
    final_height_m: float = 0.003,
    detour_m: float = 0.070,
) -> TrajectoryPlan:
    """Try direct and both lateral corridors, then select the safest short plan."""

    target = np.asarray(target_xyz_m, dtype=float)
    travel = target[:2] - np.asarray(start.xyz[:2], dtype=float)
    norm = float(np.linalg.norm(travel))
    perpendicular = (
        np.array([0.0, 1.0])
        if norm < 1e-9
        else np.array([-travel[1], travel[0]]) / norm
    )
    midpoint = 0.5 * (target[:2] + np.asarray(start.xyz[:2], dtype=float))
    corridors = [
        ("direct", None),
        ("detour_left", midpoint + perpendicular * float(detour_m)),
        ("detour_right", midpoint - perpendicular * float(detour_m)),
    ]
    accepted: list[tuple[float, TrajectoryPlan]] = []
    rejected: dict[str, str] = {}
    for name, transit in corridors:
        validator = validator_factory()
        try:
            plan = plan_lift_translate_descend(
                start,
                target,
                grasp_orientation_wxyz=grasp_orientation_wxyz,
                lift_m=lift_m,
                pregrasp_height_m=pregrasp_height_m,
                final_height_m=final_height_m,
                validator=validator,
                transit_xy_m=transit,
            )
        except AutonomousStop as error:
            rejected[name] = str(error)
            continue
        duration = plan.waypoints[-1].t_s
        clearance = float(plan.minimum_clearance_m)
        # Prefer clearance, but do not take a very long detour for sub-mm gains.
        score = clearance - 0.0001 * duration
        plan.metadata.update(
            corridor=name,
            rejected_corridors=dict(rejected),
            mujoco_q_waypoints=validator.mujoco.q_waypoints,
        )
        accepted.append((score, plan))
    if not accepted:
        raise AutonomousStop(
            "all approach corridors rejected: "
            + json.dumps(rejected, sort_keys=True)
        )
    accepted.sort(key=lambda item: item[0], reverse=True)
    selected = accepted[0][1]
    selected.metadata["rejected_corridors"] = rejected
    selected.metadata["candidate_count"] = len(corridors)
    selected.metadata["accepted_candidate_count"] = len(accepted)
    return selected


def decide_replan(
    *,
    now_s: float,
    reference: SceneSnapshot,
    current: SceneSnapshot,
    commanded_pose: Pose,
    policy: ReplanPolicy = ReplanPolicy(),
) -> ReplanDecision:
    """Return HOLD, REPLAN, ABORT or CONTINUE after a fresh observation."""

    if now_s - current.timestamp_s > policy.maximum_observation_age_s:
        return ReplanDecision("HOLD", ("observation_stale",))
    if not current.lid_visible or current.target_xyz_m is None:
        return ReplanDecision("REPLAN", ("target_occluded",))
    if (
        reference.target_instance_id is not None
        and current.target_instance_id != reference.target_instance_id
    ):
        return ReplanDecision("REPLAN", ("target_instance_changed",))
    reasons: list[str] = []
    if reference.target_xyz_m is not None:
        shift = np.linalg.norm(
            np.asarray(current.target_xyz_m) - np.asarray(reference.target_xyz_m)
        )
        if shift > policy.maximum_target_shift_m:
            reasons.append("target_shift")
    position_error = np.linalg.norm(
        np.asarray(current.ee_pose.xyz) - np.asarray(commanded_pose.xyz)
    )
    if position_error > policy.maximum_position_error_m:
        reasons.append("trajectory_position_error")
    if (
        quaternion_angle_deg(current.ee_pose.wxyz, commanded_pose.wxyz)
        > policy.maximum_rotation_error_deg
    ):
        reasons.append("trajectory_rotation_error")
    if (
        current.predicted_clearance_m is None
        or current.predicted_clearance_m < policy.minimum_clearance_m
    ):
        reasons.append("clearance_low_or_unknown")
    return ReplanDecision("REPLAN", tuple(reasons)) if reasons else ReplanDecision("CONTINUE")


def check_pregrasp(
    snapshot: SceneSnapshot,
    *,
    goal_px: Sequence[float],
    tolerance_px: float = 8.0,
    maximum_gripper_lid_m: float = 0.010,
) -> PregraspGate:
    reasons: list[str] = []
    if not snapshot.lid_visible:
        reasons.append("lid_not_visible")
    if not snapshot.blue_marker_visible or snapshot.right_marker_px is None:
        reasons.append("blue_marker_not_visible")
        pixel_error = None
    else:
        pixel_error = float(
            np.linalg.norm(np.asarray(snapshot.right_marker_px) - np.asarray(goal_px))
        )
        if pixel_error > tolerance_px:
            reasons.append("right_goal_too_far")
    distance = snapshot.estimated_gripper_lid_m
    if distance is None or not np.isfinite(distance) or distance > maximum_gripper_lid_m:
        reasons.append("gripper_lid_distance_too_large_or_unknown")
    return PregraspGate(not reasons, tuple(reasons), pixel_error)


class AtomicRunState:
    """Durable run journal; every update is an atomic replace plus fsync."""

    def __init__(self, path: str | Path, *, resume: bool = False):
        self.path = Path(path)
        if resume:
            self.payload = json.loads(self.path.read_text())
        else:
            self.payload = {
                "schema": "piper_robot.autonomous_sam_lid_grasp/v1",
                "status": "INITIALIZING",
                "events": [],
            }
            self.write()

    def update(self, status: str, **values) -> None:
        self.payload.update(values)
        self.payload["status"] = status
        self.payload["updated_at_s"] = time.time()
        self.write()

    def event(self, event: str, **values) -> None:
        self.payload.setdefault("events", []).append(
            {"event": event, "timestamp_s": time.time(), **values}
        )
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
        )
        encoded = (json.dumps(self.payload, indent=2, allow_nan=False) + "\n").encode()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise


def validate_calibration(
    calibration: dict,
    *,
    camera_udid: str,
    maximum_age_s: float | None = None,
    now_s: float | None = None,
) -> np.ndarray:
    """Validate the accepted camera->robot transform and camera fingerprint."""

    if calibration.get("accepted") is not True:
        raise AutonomousStop("camera-to-robot calibration is not accepted")
    if calibration.get("record3d_udid") != camera_udid:
        raise AutonomousStop("camera UDID does not match the accepted calibration")
    transform = np.asarray(calibration.get("T_robot_camera"), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise AutonomousStop("T_robot_camera must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise AutonomousStop("T_robot_camera has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise AutonomousStop("T_robot_camera rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.999:
        raise AutonomousStop("T_robot_camera rotation is reflected or singular")
    if maximum_age_s is not None:
        accepted_at = float(calibration.get("accepted_at_s", -math.inf))
        current = time.time() if now_s is None else float(now_s)
        if current - accepted_at > maximum_age_s:
            raise AutonomousStop("camera-to-robot calibration is stale")
    return transform


class AbsoluteCartesianTargetSender:
    """Canonical target path shared by teleoperation and autonomous chunks."""

    def __init__(self, rpc, *, preview_time_s: float = PREVIEW_TIME_S):
        self.rpc = rpc
        self.preview_time_s = float(preview_time_s)

    def send(self, pose, *, gripper_target: float | None = None) -> None:
        if isinstance(pose, Pose):
            pose = pose.as_se3()
        kwargs = {"ee_target": pose, "preview_time": self.preview_time_s}
        if gripper_target is not None:
            kwargs["gripper_target"] = float(gripper_target)
        self.rpc.set_right_ee_target(**kwargs)

    def hold(self) -> None:
        pose = self.rpc.get_right_ee_pose()
        gripper = None
        if hasattr(self.rpc, "get_right_gripper_exact"):
            exact = self.rpc.get_right_gripper_exact()
            gripper = float(np.asarray(exact).reshape(-1)[0])
        self.send(pose, gripper_target=gripper)


class ChunkExecutor:
    """Stream one short chunk at 30 Hz with a sustained torque stop."""

    def __init__(
        self,
        rpc,
        *,
        torque_limit_nm: Sequence[float],
        consecutive_torque_samples: int = 2,
        enforce_torque_stop: bool = True,
        control_hz: float = CONTROL_HZ,
        preview_time_s: float = PREVIEW_TIME_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpc = rpc
        self.sender = AbsoluteCartesianTargetSender(
            rpc, preview_time_s=preview_time_s
        )
        self.torque_limit = np.asarray(torque_limit_nm, dtype=float)
        if self.torque_limit.shape != (6,) or np.any(self.torque_limit <= 0):
            raise ValueError("torque_limit_nm must contain six positive values")
        self.strikes_required = int(consecutive_torque_samples)
        self.enforce_torque_stop = bool(enforce_torque_stop)
        self.torque_warning_count = 0
        self.last_torque_warning = None
        if self.strikes_required <= 0:
            raise ValueError("consecutive_torque_samples must be positive")
        self.period_s = 1.0 / float(control_hz)
        self.clock = clock
        self.sleep = sleep

    def _check_torque(self, strikes: int) -> int:
        sample = np.asarray(self.rpc.get_right_joint_torque(), dtype=float)
        if sample.shape != (6,) or not np.all(np.isfinite(sample)):
            if self.enforce_torque_stop:
                self.sender.hold()
                raise AutonomousStop("invalid right-arm torque sample")
            self.torque_warning_count += 1
            self.last_torque_warning = {
                "reason": "invalid right-arm torque sample",
                "sample": sample.tolist(),
            }
            return 0
        strikes = strikes + 1 if np.any(np.abs(sample) > self.torque_limit) else 0
        if strikes >= self.strikes_required:
            if self.enforce_torque_stop:
                self.sender.hold()
                raise AutonomousStop(
                    "sustained right-arm torque limit exceeded"
                )
            self.torque_warning_count += 1
            self.last_torque_warning = {
                "reason": "sustained right-arm torque limit exceeded",
                "sample": sample.tolist(),
                "limit": self.torque_limit.tolist(),
            }
            return 0
        return strikes

    def execute(self, waypoints: Iterable[TimedWaypoint]) -> Pose:
        waypoints = list(waypoints)
        if not waypoints:
            raise ValueError("cannot execute an empty chunk")
        strikes = 0
        deadline = self.clock()
        for waypoint in waypoints:
            strikes = self._check_torque(strikes)
            self.sender.send(waypoint.pose)
            deadline += self.period_s
            delay = deadline - self.clock()
            if delay > 0:
                self.sleep(delay)
        self._check_torque(strikes)
        return waypoints[-1].pose


def plan_to_dict(plan: TrajectoryPlan) -> dict:
    return asdict(plan)

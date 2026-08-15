"""Branch-local, pose-locked recovery and descent for the physical right arm.

This adapter exists for low work where endpoint IK can trade wrist attitude
for Cartesian position.  It integrates the measured production-CAD Jacobian
in joint space, so every vertical millimetre explicitly requests zero angular
velocity.  A stopped hover rebranch then levels the two fingertips while
moving joint 5 back inside the production model range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .gripper_level import (
    JawLevelReference,
    assess_jaw_level,
    signed_outward_tip_pitch_deg,
)


@dataclass(frozen=True)
class PoseLockedPlan:
    start_q_rad: tuple[float, ...]
    lift_q_path: tuple[tuple[float, ...], ...]
    rebranch_q_path: tuple[tuple[float, ...], ...]
    descent_q_path: tuple[tuple[float, ...], ...]
    hover_assessment: dict
    final_assessment: dict
    hover_tip_pitch_deg: float
    final_tip_pitch_deg: float
    predicted_final_delta_xyz_m: tuple[float, ...]
    chosen_joint5_rebranch_rad: float


class PhysicalRightPoseLockedPlanner:
    """Plan the physical-right branch (production ``left_arm_*``)."""

    def __init__(self, model_path: str | Path, reference: JawLevelReference):
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.names = [f"left_arm_joint{index}" for index in range(1, 7)]
        self.dofs = np.asarray(
            [self.model.joint(name).dofadr[0] for name in self.names], dtype=int
        )
        self.qpos = np.asarray(
            [self.model.joint(name).qposadr[0] for name in self.names], dtype=int
        )
        self.ranges = np.asarray(
            [self.model.jnt_range[self.model.joint(name).id] for name in self.names],
            dtype=float,
        )
        self.site_id = self.model.site("left_arm_ee").id
        self.reference = reference

    def _state(self, q_rad: Sequence[float], *, jacobian: bool = False):
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("right q must be a finite six-vector")
        self.data.qpos[self.qpos] = q
        mujoco.mj_forward(self.model, self.data)
        site = self.data.site(self.site_id)
        rotation = Rotation.from_matrix(site.xmat.reshape(3, 3).copy())
        xyzw = rotation.as_quat()
        pose = np.r_[xyzw[[3, 0, 1, 2]], site.xpos.copy()]
        if not jacobian:
            return pose, rotation
        position = np.zeros((3, self.model.nv))
        angular = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, position, angular, self.site_id
        )
        return pose, rotation, np.vstack(
            (position[:, self.dofs], angular[:, self.dofs])
        )

    def _vertical_path(
        self, q_start: np.ndarray, distance_m: float, *, step_m: float = 0.001
    ) -> list[np.ndarray]:
        count = max(1, int(math.ceil(abs(distance_m) / step_m)))
        dz = float(distance_m) / count
        q = np.asarray(q_start, dtype=float).copy()
        result = []
        for _ in range(count):
            _, _, jacobian = self._state(q, jacobian=True)
            twist = np.asarray([0.0, 0.0, dz, 0.0, 0.0, 0.0])
            damping = 1e-4
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), twist
            )
            if float(np.max(np.abs(delta))) > 0.02:
                raise RuntimeError("pose-locked vertical step is discontinuous")
            q += delta
            result.append(q.copy())
        return result

    def _level_rebranch_target(
        self,
        q_hover: np.ndarray,
        joint5_delta_rad: float,
        *,
        baseline_correction_fraction: float = 1.0,
    ) -> np.ndarray:
        pose, rotation = self._state(q_hover)
        up = np.asarray(self.reference.support_up_robot, dtype=float)
        up /= np.linalg.norm(up)
        approach_ee = np.asarray(self.reference.approach_axis_ee, dtype=float)
        approach_ee /= np.linalg.norm(approach_ee)
        baseline_ee = np.asarray(self.reference.tip_baseline_ee, dtype=float)
        baseline_ee /= np.linalg.norm(baseline_ee)
        matrix = rotation.as_matrix()
        approach0 = float((matrix @ approach_ee) @ up)
        baseline0 = float((matrix @ baseline_ee) @ up)
        position_jacobian = np.zeros((3, 6))
        approach_row = np.zeros(6)
        baseline_row = np.zeros(6)
        epsilon = 2e-4
        for joint in range(6):
            perturbed = q_hover.copy()
            perturbed[joint] += epsilon
            candidate_pose, candidate_rotation = self._state(perturbed)
            candidate_matrix = candidate_rotation.as_matrix()
            position_jacobian[:, joint] = (
                candidate_pose[4:] - pose[4:]
            ) / epsilon
            approach_row[joint] = (
                float((candidate_matrix @ approach_ee) @ up) - approach0
            ) / epsilon
            baseline_row[joint] = (
                float((candidate_matrix @ baseline_ee) @ up) - baseline0
            ) / epsilon
        system = np.vstack(
            (
                position_jacobian,
                approach_row,
                baseline_row,
                np.eye(6)[4],
            )
        )
        fraction = float(baseline_correction_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("baseline correction fraction must be in (0, 1]")
        rhs = np.r_[
            np.zeros(4),
            -baseline0 * fraction,
            float(joint5_delta_rad),
        ]
        delta = np.linalg.solve(system, rhs)
        if float(np.max(np.abs(delta))) > math.radians(15.0):
            raise RuntimeError("jaw-level rebranch exceeds 15 degrees")
        return q_hover + delta

    @staticmethod
    def _minimum_jerk_path(
        start: np.ndarray, target: np.ndarray, *, maximum_step_rad: float
    ) -> list[np.ndarray]:
        largest = float(np.max(np.abs(target - start)))
        count = max(2, int(math.ceil(largest / maximum_step_rad)))
        phase = np.arange(1, count + 1, dtype=float) / count
        blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
        return [start + value * (target - start) for value in blend]

    def _inside_ranges(self, q: np.ndarray, margin: float = 0.0) -> bool:
        return bool(
            np.all(q >= self.ranges[:, 0] + margin)
            and np.all(q <= self.ranges[:, 1] - margin)
        )

    def plan(
        self,
        q_start_rad: Sequence[float],
        *,
        lift_m: float = 0.030,
        descent_m: float = 0.030,
    ) -> PoseLockedPlan:
        start = np.asarray(q_start_rad, dtype=float)
        start_pose, _ = self._state(start)
        lift = self._vertical_path(start, abs(float(lift_m)))
        q_hover = lift[-1]
        # Choose the smallest stopped-hover rebranch that leaves the complete
        # descent inside the model limits with a 5 mrad joint margin.
        accepted = None
        for joint5_delta in np.arange(0.0, 0.121, 0.01):
            target = self._level_rebranch_target(q_hover, float(joint5_delta))
            if not self._inside_ranges(target):
                continue
            descent = self._vertical_path(target, -abs(float(descent_m)))
            if not all(self._inside_ranges(value, margin=0.005) for value in descent):
                continue
            hover_pose, _ = self._state(target)
            final_pose, _ = self._state(descent[-1])
            hover_assessment = assess_jaw_level(
                hover_pose, self.reference, planned=True
            )
            final_assessment = assess_jaw_level(
                final_pose, self.reference, planned=True
            )
            # Intentional tip-down pitch makes the whole jaw plane non-level;
            # only the left/right fingertip baseline must remain horizontal.
            if (
                hover_assessment.tip_height_difference_m
                > self.reference.maximum_tip_height_difference_m
                or final_assessment.tip_height_difference_m
                > self.reference.maximum_tip_height_difference_m
            ):
                continue
            if signed_outward_tip_pitch_deg(hover_pose, self.reference) > 0.0:
                continue
            accepted = (
                float(joint5_delta), target, descent,
                hover_pose, final_pose, hover_assessment, final_assessment,
            )
            break
        if accepted is None:
            raise RuntimeError("no level, joint-limit-safe descent branch exists")
        (
            joint5_delta, target, descent, hover_pose, final_pose,
            hover_assessment, final_assessment,
        ) = accepted
        rebranch = self._minimum_jerk_path(
            q_hover, target, maximum_step_rad=math.radians(1.0)
        )
        return PoseLockedPlan(
            start_q_rad=tuple(float(v) for v in start),
            lift_q_path=tuple(tuple(float(v) for v in q) for q in lift),
            rebranch_q_path=tuple(tuple(float(v) for v in q) for q in rebranch),
            descent_q_path=tuple(tuple(float(v) for v in q) for q in descent),
            hover_assessment=hover_assessment.to_dict(),
            final_assessment=final_assessment.to_dict(),
            hover_tip_pitch_deg=float(
                signed_outward_tip_pitch_deg(hover_pose, self.reference)
            ),
            final_tip_pitch_deg=float(
                signed_outward_tip_pitch_deg(final_pose, self.reference)
            ),
            predicted_final_delta_xyz_m=tuple(
                float(v) for v in final_pose[4:] - start_pose[4:]
            ),
            chosen_joint5_rebranch_rad=joint5_delta,
        )

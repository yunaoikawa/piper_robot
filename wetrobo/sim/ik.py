"""ArmIK — mink Cartesian IK for one arm, operating on a shared MuJoCo model.

Why not reuse robot.arm.ik_solver.SingleArmIK directly: that class hardcodes 6 arm
joints, an ee site named "*_arm_ee", a "home" keyframe, and looks up actuators as
"<joint>_pos". The lab scene arms are 5-DOF, their actuators are named exactly like the
joints ("joint1", "left_gripper"), the sites are "ee"/"left_ee", and the keyframe is
"lab_home". This adapter takes those as explicit arguments and shares the caller's model
(no second model load), so it composes with LabEnv's single physics model.

The arms are 5-DOF, so an arbitrary 6-DOF target orientation is not achievable; the task
weights position above orientation and callers should read back the achieved pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco
import mink


@dataclass
class IKResult:
    q: np.ndarray          # solved joint angles for this arm's DOFs
    pos_err_m: float       # residual position error at the ee site [m]
    rot_err_deg: float     # residual orientation error [deg]
    solved: bool           # both errors within tolerance


class ArmIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: list[str],
        ee_site: str,
        actuator_names: list[str],
        solver_dt: float = 0.01,
        position_cost: float = 1.0,
        orientation_cost: float = 0.05,
    ):
        self.model = model
        self.ee_site = ee_site
        self.solver_dt = solver_dt
        self.dof_ids = np.array([model.joint(n).id for n in joint_names])
        self.qpos_adr = np.array([model.jnt_qposadr[i] for i in self.dof_ids])
        self.actuator_ids = np.array([model.actuator(n).id for n in actuator_names])

        self._cfg = mink.Configuration(model)
        self._ee_task = mink.FrameTask(
            frame_name=ee_site, frame_type="site",
            position_cost=position_cost, orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        self._posture = mink.PostureTask(model, cost=1e-3)
        self._limits = [mink.ConfigurationLimit(model)]

    def forward(self, qpos: np.ndarray) -> mink.SE3:
        """End-effector pose in world for a full qpos vector."""
        self._cfg.update(qpos.copy())
        return self._cfg.get_transform_frame_to_world(self.ee_site, "site")

    def solve(
        self,
        target: mink.SE3,
        seed_qpos: np.ndarray,
        max_iter: int = 200,
        pos_eps: float = 1e-3,
        rot_eps: float = 5e-2,
    ) -> IKResult:
        """Full IK solve seeded from ``seed_qpos`` (the live model qpos).

        Returns the arm's joint angles plus the achieved residual errors. Because the
        arm is 5-DOF, rot_err_deg is often large; ``solved`` requires only pos within
        pos_eps by default (rot_eps is generous)."""
        self._cfg.update(seed_qpos.copy())
        self._posture.set_target_from_configuration(self._cfg)
        self._ee_task.set_target(target)
        for _ in range(max_iter):
            v = mink.solve_ik(
                self._cfg, [self._ee_task, self._posture], self.solver_dt,
                solver="quadprog", damping=1e-5, limits=self._limits,
            )
            self._cfg.integrate_inplace(v, self.solver_dt)
            err = self._ee_task.compute_error(self._cfg)
            if np.linalg.norm(err[:3]) <= pos_eps and np.linalg.norm(err[3:]) <= rot_eps:
                break
        err = self._ee_task.compute_error(self._cfg)
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.degrees(np.linalg.norm(err[3:])))
        return IKResult(
            q=self._cfg.q[self.dof_ids].copy(),
            pos_err_m=pos_err,
            rot_err_deg=rot_err,
            solved=pos_err <= 0.02,
        )

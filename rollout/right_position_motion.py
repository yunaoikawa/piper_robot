"""Local-branch right-arm translation with teleoperation-style streaming."""

from __future__ import annotations

import time

import numpy as np


class RightPositionMotion:
    """Translate the tool without asking IK to reselect its wrist branch."""

    def __init__(self, model_path: str):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.right_joint_names = [
            f"left_arm_joint{index}" for index in range(1, 7)
        ]
        self.other_joint_names = [
            f"right_arm_joint{index}" for index in range(1, 7)
        ]
        self.dofs = np.array(
            [self.model.joint(name).dofadr[0] for name in self.right_joint_names]
        )
        self.site_id = self.model.site("left_arm_ee").id

    def joint_delta(self, rpc, delta_xyz_m) -> np.ndarray:
        requested = np.asarray(delta_xyz_m, dtype=float)
        if requested.shape != (3,) or not np.all(np.isfinite(requested)):
            raise ValueError("delta_xyz_m must be a finite 3-vector")
        right_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        other_q = np.asarray(rpc.get_left_joint_positions(), dtype=float)
        data = self.mujoco.MjData(self.model)
        for name, value in zip(self.right_joint_names, right_q):
            data.qpos[self.model.joint(name).qposadr[0]] = value
        for name, value in zip(self.other_joint_names, other_q):
            data.qpos[self.model.joint(name).qposadr[0]] = value
        self.mujoco.mj_forward(self.model, data)
        jacobian = np.zeros((3, self.model.nv))
        rotation = np.zeros((3, self.model.nv))
        self.mujoco.mj_jacSite(
            self.model, data, jacobian, rotation, self.site_id
        )
        delta = np.linalg.pinv(
            jacobian[:, self.dofs], rcond=1e-4
        ) @ requested
        largest = float(np.max(np.abs(delta)))
        if largest > 0.12:
            delta *= 0.12 / largest
        return delta

    def move(
        self,
        runner,
        delta_xyz_m,
        *,
        duration_s: float = 1.0,
        gross_torque_multiplier: float = 3.0,
    ) -> np.ndarray:
        """Stream one minimum-jerk translation and return measured delta xyz."""

        joint_delta = self.joint_delta(runner.rpc, delta_xyz_m)
        before_xyz = np.asarray(
            runner.rpc.get_right_ee_pose().translation(), dtype=float
        )
        strikes = 0

        def check_torque(stage):
            nonlocal strikes
            torque = np.abs(
                np.asarray(runner.rpc.get_right_joint_torque(), dtype=float)
            )
            limit = runner.torque_limit * float(gross_torque_multiplier)
            strikes = strikes + 1 if np.any(torque > limit) else 0
            if not np.all(np.isfinite(torque)) or strikes >= runner.torque_samples:
                raise RuntimeError(
                    f"gross right torque stop {stage}: {torque.tolist()}"
                )

        motion_error = None
        try:
            runner._prepare_cartesian_motion(check_torque)
            start_q = np.asarray(
                runner.rpc.get_right_joint_positions(), dtype=float
            )
            steps = max(1, int(np.ceil(float(duration_s) * 30.0)))
            started = time.monotonic()
            for index in range(steps):
                check_torque("during streamed position move")
                alpha = (index + 1) / steps
                blend = alpha**3 * (10 + alpha * (-15 + 6 * alpha))
                runner.rpc.set_right_joint_target(
                    start_q + blend * joint_delta,
                    gripper_target=None,
                    preview_time=0.05,
                )
                remaining = (
                    started + (index + 1) * float(duration_s) / steps
                    - time.monotonic()
                )
                if remaining > 0:
                    time.sleep(remaining)
            time.sleep(0.20)
        except BaseException as error:
            motion_error = error
            raise
        finally:
            try:
                runner._finish_cartesian_motion()
            except BaseException:
                if motion_error is None:
                    raise
        after_xyz = np.asarray(
            runner.rpc.get_right_ee_pose().translation(), dtype=float
        )
        return after_xyz - before_xyz

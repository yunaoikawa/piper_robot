import time
import numpy as np
from typing import Optional
from pathlib import Path

from piperlib import PiperJointController, RobotConfigFactory, ControllerConfigFactory, JointState
import mink

from robot.arm.ik_solver import ArmIK

GRIPPER_OPEN = -22.0

class ArmNode:
    def __init__(
        self, can_port: str, mjcf_path: Optional[str] = None, urdf_path: Optional[str] = None, solver_dt: float = 0.01
    ):
        _HERE = Path(__file__).parent
        self.can_port = can_port
        if mjcf_path is None:
            self.mjcf_path = (_HERE / "mujoco/scene_piper.xml").as_posix()
        else:
            self.mjcf_path = mjcf_path
        if urdf_path is None:
            self.urdf_path = (_HERE / "urdf/piper_description_left.xml").as_posix()
        else:
            self.urdf_path = urdf_path
        self.solver_dt = solver_dt

        # initialize arm
        self.robot_config = RobotConfigFactory.get_instance().get_config("piper")
        self.controller_config = ControllerConfigFactory.get_instance().get_config("joint_controller")
        self.robot_config.urdf_path = self.urdf_path
        self.controller_config.controller_dt = 0.005
        self.controller_config.default_kp = np.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
        self.controller_config.default_kd = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.piper = PiperJointController(self.robot_config, self.controller_config, self.can_port)
        self.target: Optional[mink.SE3] = None
        self.gripper_target: Optional[float] = None
        self.target_timestamp: Optional[int] = None
        self.ik_solver = ArmIK(self.mjcf_path, solver_dt=self.solver_dt)
        self.home_q = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78])

        self.init()

    def init(self):
        self.piper.reset_to_home()
        time.sleep(1.0)

        # home
        # q = self.home_q.copy()
        # print(f"q_home: {np.round(q, 4)}")
        # cmd = JointState(self.robot_config.joint_dof)
        # cmd.timestamp = self.piper.get_timestamp() + 1.0
        # cmd.pos = q
        # cmd.gripper_pos = 1.0 * GRIPPER_OPEN
        # self.piper.set_joint_cmd(cmd)
        # time.sleep(2.0)

        q = self.piper.get_joint_state().pos
        print(f"q_reached: {np.round(q, 4)}")
        self.ik_solver.init(q)
        self.target = self.ik_solver.forward_kinematics()

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    def home(self, gripper_target: float = 1.0):
        q = self.home_q.copy()
        cmd = JointState(self.robot_config.joint_dof)
        cmd.timestamp = self.piper.get_timestamp() + 1.0
        cmd.pos = q
        cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        self.piper.set_joint_cmd(cmd)
        time.sleep(2.0)
        self.update_joint_positions()

    def set_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = joint_target
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        if gripper_target is not None:
            cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        self.piper.set_joint_cmd(cmd)

    def set_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None, preview_time: float = 0.1):
        self.target = ee_target
        qd = self.ik_solver.solve_ik(self.target)
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = qd
        if gripper_target is not None:
            cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        self.piper.set_joint_cmd(cmd)

    def get_joint_positions(self) -> np.ndarray:
        return self.piper.get_joint_state().pos

    def get_ee_pose(self) -> mink.SE3:
        self.update_joint_positions()
        return self.ik_solver.forward_kinematics()

    def update_joint_positions(self):
        q = self.piper.get_joint_state().pos
        self.ik_solver.update_configuration(q)

    def stop(self):
        print("called stop")
        pass
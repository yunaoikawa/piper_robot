import time
import numpy as np
# from typing import Any
from pathlib import Path
# import argparse

from piperlib import PiperJointController, RobotConfigFactory, ControllerConfigFactory, JointState
import mink
# from mink import SE3
# from dora import Node

from robot.arm.mink_ik_arm import ArmIK
# from robot.msgs.pose import Pose, ArmCommand
from robot.arm.fps_counter import FPSCounter


# class SE3Filter:
#     def __init__(self, alpha):
#         self.alpha = alpha
#         self.y: SE3 = None
#         self.is_init = False

#     def next(self, x):
#         if not self.is_init:
#             self.y = x
#             self.is_init = True
#             return self.y.copy()
#         self.y = self.y.interpolate(x, self.alpha)
#         return self.y.copy()

#     def reset(self):
#         self.y = None
#         self.is_init = False

GRIPPER_OPEN = -22.0

class ArmNode:
    def __init__(
        self, can_port: str, mjcf_path: str | None = None, urdf_path: str | None = None, solver_dt: float = 0.01
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
        self.controller_config.default_kp = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        self.controller_config.default_kd = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.piper = PiperJointController(self.robot_config, self.controller_config, self.can_port)
        self.target: mink.SE3 | None = None
        self.gripper_target: float | None = None
        self.target_timestamp: int | None = None
        self.ik_solver = ArmIK(self.mjcf_path, solver_dt=self.solver_dt)
        self.home_q = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78])

        # position low-pass filter
        # self.se3_filter = SE3Filter(0.4)

        # fps counter
        self.ik_solver_fps_counter = FPSCounter("ik_solver")

        # communication
        # self.node = Node()
        self.init()

    def init(self):
        self.piper.reset_to_home()
        time.sleep(1.0)
        # self.piper.home_gripper()

        # home
        q = self.home_q.copy()
        print(f"q_home: {np.round(q, 4)}")
        cmd = JointState(self.robot_config.joint_dof)
        cmd.timestamp = self.piper.get_timestamp() + 1.0
        cmd.pos = q
        cmd.gripper_pos = 1.0 * GRIPPER_OPEN
        self.piper.set_joint_cmd(cmd)
        time.sleep(2.0)
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

    # def arm_command_handler(self, event: dict[str, Any]):
    #     arm_command = ArmCommand.decode(event["value"], event["metadata"])
    #     self.target = mink.SE3(arm_command.wxyz_xyz)
    #     qd = self.ik_solver.solve_ik(self.target)
    #     cmd = JointState(self.robot_config.joint_dof)
    #     cmd.pos = qd
    #     if arm_command.gripper is not None:
    #         cmd.gripper_pos = arm_command.gripper * GRIPPER_OPEN
    #     cmd.timestamp = self.piper.get_timestamp() + 0.1
    #     self.piper.set_joint_cmd(cmd)

    def home(self, gripper_target: float = 1.0):
        q = self.home_q.copy()
        cmd = JointState(self.robot_config.joint_dof)
        cmd.timestamp = self.piper.get_timestamp() + 1.0
        cmd.pos = q
        cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        self.piper.set_joint_cmd(cmd)

    def set_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.1):
        self.target = ee_target
        qd = self.ik_solver.solve_ik(self.target)
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = qd
        if gripper_target is not None:
            cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        self.piper.set_joint_cmd(cmd)

    def get_ee_pose(self) -> mink.SE3:
        self.update_joint_positions()
        return self.ik_solver.forward_kinematics()

    # def step(self):
    #     self.update_joint_positions()
    #     ee_pose = self.ik_solver.forward_kinematics()  # update current joint positions

    #     pose_msg = Pose(time.perf_counter_ns(), ee_pose.wxyz_xyz)
    #     # self.node.send_output("ee_pose", *pose_msg.encode())

    def update_joint_positions(self):
        q = self.piper.get_joint_state().pos
        self.ik_solver.update_configuration(q)

    def stop(self):
        print("called stop")
        pass

    # def spin(self):
    #     for event in self.node:
    #         event_type = event["type"]
    #         if event_type == "INPUT":
    #             event_id = event["id"]

    #             if event_id == "arm_command":
    #                 self.arm_command_handler(event)

    #             elif event_id == "tick":
    #                 self.step()

    #         elif event_type == "STOP":
    #             self.stop()


# if __name__ == "__main__":
#     _HERE = Path(__file__).parent
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mjcf_path", type=str, required=True)
#     parser.add_argument("--urdf_path", type=str, required=True)
#     parser.add_argument("--can_port", type=str, required=True)
#     parser.add_argument("--solver_dt", type=float, required=True)
#     args = parser.parse_args()

#     arm_node = ArmNode(
#         can_port=args.can_port,
#         mjcf_path=(_HERE / args.mjcf_path).as_posix(),
#         urdf_path=(_HERE / args.urdf_path).as_posix(),
#         solver_dt=args.solver_dt,
#     )
#     arm_node.spin()

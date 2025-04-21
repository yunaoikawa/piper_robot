import time
import os
import numpy as np
from typing import Any
from pathlib import Path

# from piper_control import piper_control
from piperlib import PiperJointController, RobotConfigFactory, ControllerConfigFactory, JointState
import mink
from dora import Node

from robot.arm.mink_ik_arm import ArmIK
from robot.msgs.pose import Pose

class ArmNode:
    def __init__(self, can_port: str, mjcf_path: str, urdf_path: str, solver_dt: float = 0.03):
        self.can_port = can_port
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # initialize arm
        self.robot_config = RobotConfigFactory.get_instance().get_config("piper")
        self.controller_config = ControllerConfigFactory.get_instance().get_config("joint_controller")
        self.robot_config.urdf_path = urdf_path
        self.piper = PiperJointController(self.robot_config, self.controller_config, self.can_port)
        self.target: mink.SE3 | None = None
        self.ik_solver = ArmIK(mjcf_path, solver_dt=self.solver_dt)

        # communication
        self.node = Node()
        self.init()

    def init(self):
        self.piper.reset_to_home()
        time.sleep(1.0)
        # home
        q = np.array(self.ik_solver.get_home_q())
        print(f"q_home: {q}")
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos()[:] = q
        self.piper.set_joint_cmd(cmd)
        time.sleep(2.0)
        q = np.array(self.piper.get_joint_state().pos())
        print(f"q_reached: {q}")
        self.ik_solver.init(q)
        self.target = self.ik_solver.forward_kinematics()

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    def arm_command_handler(self, event: dict[str, Any]):
        pose = Pose.decode(event["value"], event["metadata"])
        if not self.check_timestamp(pose.timestamp, 0.1):
            return

        target = mink.SE3(pose.wxyz_xyz)
        self.target = target

    def step(self):
        # q = np.array(self.piper.get_joint_positions())
        ee_pose = self.ik_solver.forward_kinematics() # update current joint positions
        if self.target is not None:
            qd = self.ik_solver.solve_ik(self.target)
            print(f"qd: {np.round(qd, 4)}")
            cmd = JointState(self.robot_config.joint_dof)
            cmd.pos()[:] = qd
            self.piper.set_joint_cmd(cmd)

        pose_msg = Pose(time.perf_counter_ns(), ee_pose.wxyz_xyz)
        self.node.send_output("ee_pose", *pose_msg.encode())

    def stop(self):
        print("called stop")
        pass

    def spin(self):
        for event in self.node:
            event_type = event["type"]
            if event_type == "INPUT":
                event_id = event["id"]

                if event_id == "arm_command":
                    self.arm_command_handler(event)

                elif event_id == "tick":
                    self.step()

            elif event_type == "STOP":
                self.stop()


if __name__ == "__main__":
    _HERE = Path(__file__).parent
    _XML_PATH = (_HERE / "mujoco/scene_piper.xml").as_posix()
    _URDF_PATH = (_HERE / "urdf/piper_no_gripper_description.xml").as_posix()
    _CAN_PORT = os.environ.get("CAN_PORT", "can_right")

    arm_node = ArmNode(can_port=_CAN_PORT, mjcf_path=_XML_PATH, urdf_path=_URDF_PATH)
    arm_node.spin()

import time
import numpy as np
from typing import Any, Optional
from pathlib import Path
import argparse

from piperlib import PiperJointController, RobotConfigFactory, ControllerConfigFactory, JointState
import mink
from dora import Node

from robot.arm.ik_solver import BimanualArmIK
from robot.msgs.bimanual_pose import BimanualPose, BimanualArmCommand


class BimanualArmNode:
    def __init__(self, mjcf_path: str, left_urdf_path: str, right_urdf_path: str, solver_dt: float = 0.01):
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # initialize arm
        controller_config = ControllerConfigFactory.get_instance().get_config("joint_controller")

        left_robot_config = RobotConfigFactory.get_instance().get_config("piper")
        left_robot_config.urdf_path = left_urdf_path
        right_robot_config = RobotConfigFactory.get_instance().get_config("piper")
        right_robot_config.urdf_path = right_urdf_path
        self.left_piper = PiperJointController(left_robot_config, controller_config, "can_left")
        self.right_piper = PiperJointController(right_robot_config, controller_config, "can_right")

        # initialize ik
        self.left_target: Optional[mink.SE3] = None
        self.right_target: Optional[mink.SE3] = None
        self.left_gripper: Optional[float] = None
        self.right_gripper: Optional[float] = None
        self.ik_solver = BimanualArmIK(mjcf_path, solver_dt=self.solver_dt)

        # communication
        self.node = Node()
        self.init()

    def init(self):
        self.left_piper.reset_to_home()
        self.right_piper.reset_to_home()
        time.sleep(1.0)
        # home
        home_q = np.array(self.ik_solver.get_home_q())
        left_cmd = JointState(6)
        left_cmd.timestamp = self.left_piper.get_timestamp() + 1.0
        left_cmd.pos = home_q[:6]

        right_cmd = JointState(6)
        right_cmd.timestamp = self.right_piper.get_timestamp() + 1.0
        right_cmd.pos = home_q[6:]

        self.left_piper.set_joint_cmd(left_cmd)
        self.right_piper.set_joint_cmd(right_cmd)

        time.sleep(2.0)
        q = np.concatenate([self.left_piper.get_joint_state().pos, self.right_piper.get_joint_state().pos])
        self.ik_solver.init(q)
        self.left_target, self.right_target = self.ik_solver.forward_kinematics()

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    def bimanual_arm_command_handler(self, event: dict[str, Any]):
        bimanual_arm_command = BimanualArmCommand.decode(event["value"], event["metadata"])

        left_pose = mink.SE3(bimanual_arm_command.left_wxyz_xyz)
        right_pose = mink.SE3(bimanual_arm_command.right_wxyz_xyz)
        self.left_target = left_pose
        self.right_target = right_pose
        self.left_gripper = bimanual_arm_command.left_gripper
        self.right_gripper = bimanual_arm_command.right_gripper

    def step(self):
        # q = np.concatenate([self.left_piper.get_joint_state().pos, self.right_piper.get_joint_state().pos])
        left_ee_pose, right_ee_pose = self.ik_solver.forward_kinematics()
        if self.left_target is not None and self.right_target is not None:
            qd_left, qd_right = self.ik_solver.solve_ik(self.left_target, self.right_target)
            left_cmd = JointState(6)
            left_cmd.pos = qd_left
            if self.left_gripper is not None:
                left_cmd.gripper_pos = self.left_gripper
            right_cmd = JointState(6)
            right_cmd.pos = qd_right
            if self.right_gripper is not None:
                right_cmd.gripper_pos = self.right_gripper
            self.left_piper.set_joint_cmd(left_cmd)
            self.right_piper.set_joint_cmd(right_cmd)

        pose_msg = BimanualPose(time.perf_counter_ns(), left_ee_pose.wxyz_xyz, right_ee_pose.wxyz_xyz)
        self.node.send_output("bimanual_ee_pose", *pose_msg.encode())

    def stop(self):
        pass

    def spin(self):
        for event in self.node:
            event_type = event["type"]
            if event_type == "INPUT":
                event_id = event["id"]

                if event_id == "bimanual_arm_command":
                    self.bimanual_arm_command_handler(event)

                elif event_id == "tick":
                    self.step()

            elif event_type == "STOP":
                self.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf_path", type=str, required=True)
    args = parser.parse_args()

    _HERE = Path(__file__).parent
    _LEFT_URDF_PATH = (_HERE / "urdf/piper_description_left.xml").as_posix()
    _RIGHT_URDF_PATH = (_HERE / "urdf/piper_description_right.xml").as_posix()
    bimanual_arm_node = BimanualArmNode(
        mjcf_path=(_HERE / args.mjcf_path).as_posix(), left_urdf_path=_LEFT_URDF_PATH, right_urdf_path=_RIGHT_URDF_PATH
    )
    bimanual_arm_node.spin()


if __name__ == "__main__":
    main()

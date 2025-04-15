import time
import numpy as np
from typing import Any
from pathlib import Path

from piper_control import piper_control
import mink
from dora import Node

from robot.arm.mink_ik_arm import BimanualArmIK
from robot.msgs.bimanual_pose import BimanualPose


class BimanualArmNode:
    def __init__(self, mjcf_path: str, solver_dt: float = 0.03):
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # initialize arm
        self.left_piper = piper_control.PiperControl(can_port="can_left")
        self.right_piper = piper_control.PiperControl(can_port="can_right")

        # initialize ik
        self.left_target: mink.SE3 | None = None
        self.right_target: mink.SE3 | None = None
        self.ik_solver = BimanualArmIK(mjcf_path, solver_dt=self.solver_dt)

        # communication
        self.node = Node()
        self.init()

    def init(self):
        self.left_piper.reset()
        self.right_piper.reset()
        # home
        home_q = self.ik_solver.get_home_q(home_key="stow")
        self.left_piper.set_joint_positions(home_q[:6])
        self.right_piper.set_joint_positions(home_q[6:])
        time.sleep(1.0)
        q = np.array(self.left_piper.get_joint_positions() + self.right_piper.get_joint_positions())
        self.ik_solver.init(q)
        self.left_target, self.right_target = self.ik_solver.forward_kinematics(q)

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    def bimanual_arm_command_handler(self, event: dict[str, Any]):
        bimanual_pose = BimanualPose.decode(event["value"], event["metadata"])
        if not self.check_timestamp(bimanual_pose.timestamp, 0.1):
            return

        left_pose = mink.SE3(bimanual_pose.left_wxyz_xyz)
        right_pose = mink.SE3(bimanual_pose.right_wxyz_xyz)
        self.left_target = left_pose
        self.right_target = right_pose

    def step(self):
        q = np.array(self.left_piper.get_joint_positions() + self.right_piper.get_joint_positions())
        left_ee_pose, right_ee_pose = self.ik_solver.forward_kinematics(q)
        if self.left_target is not None and self.right_target is not None:  # TODO: check timestamp for the target
            qd_left, qd_right = self.ik_solver.solve_ik(self.left_target, self.right_target)
            self.left_piper.set_joint_positions(qd_left)
            self.right_piper.set_joint_positions(qd_right)

        pose_msg = BimanualPose(time.perf_counter_ns(), left_ee_pose.wxyz_xyz, right_ee_pose.wxyz_xyz)
        self.node.send_output("bimanual_ee_pose", *pose_msg.encode())

    def stop(self):
        self.left_piper._standby()
        self.right_piper._standby()

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

if __name__ == "__main__":
    _HERE = Path(__file__).parent
    _XML_PATH = (_HERE / "mujoco/scene_bimanual.xml").as_posix()

    bimanual_arm_node = BimanualArmNode(mjcf_path=_XML_PATH)
    bimanual_arm_node.spin()

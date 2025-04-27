import time
import numpy as np
from typing import Any
from pathlib import Path
import argparse

from piperlib import PiperJointController, RobotConfigFactory, ControllerConfigFactory, JointState
import mink
from dora import Node

from robot.arm.mink_ik_arm import ArmIK
from robot.msgs.pose import Pose
from robot.arm.fps_counter import FPSCounter

class LowPassFilter:
    def __init__(self, cutoff_freq: float, dt: float):
        self.cutoff_freq = cutoff_freq
        self.dt = dt
        self.alpha = 2 * np.pi * self.cutoff_freq * self.dt / (1 + 2 * np.pi * self.cutoff_freq * self.dt)
        self.prev_val = 0.0

    def filter(self, val: float) -> float:
        self.prev_val = self.prev_val + self.alpha * (val - self.prev_val)
        return self.prev_val

class ArmNode:
    def __init__(self, can_port: str, mjcf_path: str, urdf_path: str, solver_dt: float = 0.01):
        self.can_port = can_port
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # initialize arm
        self.robot_config = RobotConfigFactory.get_instance().get_config("piper")
        self.controller_config = ControllerConfigFactory.get_instance().get_config("joint_controller")
        self.robot_config.urdf_path = urdf_path
        self.controller_config.controller_dt = 0.005
        # self.controller_config.default_kp = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.piper = PiperJointController(self.robot_config, self.controller_config, self.can_port)
        self.target: mink.SE3 | None = None
        self.target_timestamp: int | None = None
        self.ik_solver = ArmIK(mjcf_path, solver_dt=self.solver_dt)

        # position low-pass filter
        self.pos_lpf = LowPassFilter(cutoff_freq=10.0, dt=self.solver_dt)

        # fps counter
        self.ik_solver_fps_counter = FPSCounter("ik_solver")

        # communication
        self.node = Node()
        self.init()

    def init(self):
        self.piper.reset_to_home()
        time.sleep(1.0)
        # home
        q = np.array(self.ik_solver.get_home_q())
        print(f"q_home: {np.round(q, 4)}")
        cmd = JointState(self.robot_config.joint_dof)
        cmd.timestamp = self.piper.get_timestamp() + 1.0
        cmd.pos = q
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

    def arm_command_handler(self, event: dict[str, Any]):
        pose = Pose.decode(event["value"], event["metadata"])
        target = mink.SE3(pose.wxyz_xyz)
        self.target = target
        self.target_timestamp = pose.timestamp

    def step(self):
        # self.update_joint_positions()
        ee_pose = self.ik_solver.forward_kinematics()  # update current joint positions
        if self.target is not None and self.target_timestamp is not None:
            with self.ik_solver_fps_counter:
                qd = self.ik_solver.solve_ik(self.target)
                qd = self.pos_lpf.filter(qd)
            cmd = JointState(self.robot_config.joint_dof)
            cmd.pos = qd
            self.piper.set_joint_cmd(cmd)

        pose_msg = Pose(time.perf_counter_ns(), ee_pose.wxyz_xyz)
        self.node.send_output("ee_pose", *pose_msg.encode())

    def update_joint_positions(self):
        q = self.piper.get_joint_state().pos
        self.ik_solver.update_configuration(q)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf_path", type=str, required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--can_port", type=str, required=True)
    parser.add_argument("--solver_dt", type=float, required=True)
    args = parser.parse_args()

    arm_node = ArmNode(
        can_port=args.can_port,
        mjcf_path=(_HERE / args.mjcf_path).as_posix(),
        urdf_path=(_HERE / args.urdf_path).as_posix(),
        solver_dt=args.solver_dt,
    )
    arm_node.spin()

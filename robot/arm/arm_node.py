import numpy as np
import signal
from typing import cast

import pinocchio as pin

import zmq
from robot.arm.tools import (
    quaternion_from_euler,
    matrix_to_xyzrpy,
    create_transformation_matrix,
)
from robot.network import Subscriber, ARM_COMMAND_PORT
from robot.network.msgs import ArmCommand
from robot.arm.piper_control import PiperControl
from robot.arm.arm_ik import ArmIK

HARDWARE = False

class ArmNode:
    def __init__(self):
        if HARDWARE:
            self.piper_control = PiperControl()
            self.piper_control.enable_piper()

        self.arm_ik = ArmIK()

        ctx = zmq.Context()
        self.arm_command_sub = Subscriber(
            ctx,
            ARM_COMMAND_PORT,
            ["/arm_command"],
            [ArmCommand.deserialize],
            no_block=False,
        )

        self.running = True
        self.start_teleop = False
        self.init_affine = None

    def get_ik_solution(self, x, y, z, roll, pitch, yaw, gripper):
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, _, is_collision = self.arm_ik.ik_fun(target.homogeneous, 0)

        if not is_collision and HARDWARE:
            self.piper_control.joint_control(np.concatenate([sol_q, np.array([gripper])]))

    def get_relative_affine(self, init_affine, current_affine):
        H = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        delta_affine = np.linalg.pinv(init_affine) @ current_affine
        relative_affine = np.linalg.pinv(H) @ delta_affine @ H
        return relative_affine

    def run(self):
        home_pose = create_transformation_matrix(0.19, 0.0, 0.2, 0, 0, 0)
        is_first_frame = True
        target = home_pose.copy()

        while self.running:
            _, arm_command = self.arm_command_sub.receive()
            arm_command = cast(ArmCommand, arm_command)

            if is_first_frame:
                if HARDWARE:
                    self.piper_control.joint_control(np.zeros(7))
                is_first_frame = False

            if arm_command.right_start_teleop:
                self.start_teleop = True
                self.init_affine = arm_command.right_target

            if arm_command.right_pause_teleop:
                self.start_teleop = False
                self.init_affine = None
                home_pose = target.copy()

            # if buttons["rightGrip"][0] > 0.5:
            #     home_pose = create_transformation_matrix(0.19, 0.0, 0.2, 0, 0, 0)

            if self.start_teleop:
                relative_affine = self.get_relative_affine(self.init_affine, arm_command.right_target)
                relative_pos, relative_rot = (
                    relative_affine[:3, 3],
                    relative_affine[:3, :3],
                )

                target_pos = home_pose[:3, 3] + relative_pos
                target_rot = home_pose[:3, :3] @ relative_rot

                target = np.eye(4)
                target[:3, 3] = target_pos
                target[:3, :3] = target_rot
            else:
                target = home_pose.copy()

            RR_ = matrix_to_xyzrpy(target)
            print(
                f"""
                RR: {RR_[0].item():.3f}, {RR_[1].item():.3f}, {RR_[2].item():.3f},
                {RR_[3]:.3f}, {RR_[4]:.3f}, {RR_[5]:.3f}
                """
            )
            r_gripper_value = arm_command.right_gripper_value * 0.07
            self.get_ik_solution(
                RR_[0],
                RR_[1],
                RR_[2],
                RR_[3],
                RR_[4],
                RR_[5],
                r_gripper_value,
            )

    def stop(self):
        self.arm_command_sub.stop()
        del self.arm_ik.vis


def main():
    arm_node = ArmNode()

    def signal_handler(signum, frame):
        arm_node.running = False
        arm_node.stop()

    signal.signal(signal.SIGINT, signal_handler)
    arm_node.run()
if __name__ == "__main__":
    main()
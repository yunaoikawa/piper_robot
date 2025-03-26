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
            self.left_piper_control = PiperControl(can_port="can_left")
            self.right_piper_control = PiperControl(can_port="can_right")
            self.left_piper_control.enable_piper()
            self.right_piper_control.enable_piper()

        self.left_arm_ik = ArmIK(urdf_file="piper-left.urdf")
        self.right_arm_ik = ArmIK(urdf_file="piper-right.urdf")

        ctx = zmq.Context()
        self.arm_command_sub = Subscriber(
            ctx,
            ARM_COMMAND_PORT,
            ["/arm_command"],
            [ArmCommand.deserialize],
            no_block=False,
        )

        self.running = True
        self.left_start_teleop = False
        self.left_init_affine = None
        self.right_start_teleop = False
        self.right_init_affine = None

    def get_left_ik_solution(self, x, y, z, roll, pitch, yaw, gripper):
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, _, is_collision = self.left_arm_ik.ik_fun(target.homogeneous, 0)

        if not is_collision and HARDWARE:
            self.left_piper_control.joint_control(np.concatenate([sol_q, np.array([gripper])]))

    def get_right_ik_solution(self, x, y, z, roll, pitch, yaw, gripper):
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, _, is_collision = self.right_arm_ik.ik_fun(target.homogeneous, 0)

        if not is_collision and HARDWARE:
            self.right_piper_control.joint_control(np.concatenate([sol_q, np.array([gripper])]))

    def get_relative_affine(self, init_affine, current_affine):
        H = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        delta_affine = np.linalg.pinv(init_affine) @ current_affine
        relative_affine = np.linalg.pinv(H) @ delta_affine @ H
        return relative_affine

    def run(self):
        left_home_pose = create_transformation_matrix(0.19, 0.0, 0.2, 0, 0, 0)
        right_home_pose = create_transformation_matrix(0.19, 0.0, 0.2, 0, 0, 0)
        is_first_frame = True
        left_target = left_home_pose.copy()
        right_target = right_home_pose.copy()

        while self.running:
            _, arm_command = self.arm_command_sub.receive()
            arm_command = cast(ArmCommand, arm_command)

            if is_first_frame:
                if HARDWARE:
                    self.left_piper_control.joint_control(np.zeros(7))
                    self.right_piper_control.joint_control(np.zeros(7))
                is_first_frame = False

            if arm_command.left_start_teleop:
                self.left_start_teleop = True
                self.left_init_affine = arm_command.left_target

            if arm_command.right_start_teleop:
                self.right_start_teleop = True
                self.right_init_affine = arm_command.right_target

            if arm_command.left_pause_teleop:
                self.left_start_teleop = False
                self.left_init_affine = None
                left_home_pose = left_target.copy()

            if arm_command.right_pause_teleop:
                self.right_start_teleop = False
                self.right_init_affine = None
                right_home_pose = right_target.copy()

            # if buttons["rightGrip"][0] > 0.5:
            #     home_pose = create_transformation_matrix(0.19, 0.0, 0.2, 0, 0, 0)

            if self.left_start_teleop:
                relative_affine = self.get_relative_affine(self.left_init_affine, arm_command.left_target)
                relative_pos, relative_rot = (
                    relative_affine[:3, 3],
                    relative_affine[:3, :3],
                )

                left_target_pos = left_home_pose[:3, 3] + relative_pos
                left_target_rot = left_home_pose[:3, :3] @ relative_rot

                left_target = np.eye(4)
                left_target[:3, 3] = left_target_pos
                left_target[:3, :3] = left_target_rot
            else:
                left_target = left_home_pose.copy()

            if self.right_start_teleop:
                relative_affine = self.get_relative_affine(self.right_init_affine, arm_command.right_target)
                relative_pos, relative_rot = (
                    relative_affine[:3, 3],
                    relative_affine[:3, :3],
                )

                right_target_pos = right_home_pose[:3, 3] + relative_pos
                right_target_rot = right_home_pose[:3, :3] @ relative_rot

                right_target = np.eye(4)
                right_target[:3, 3] = right_target_pos
                right_target[:3, :3] = right_target_rot
            else:
                right_target = right_home_pose.copy()

            LL_ = matrix_to_xyzrpy(left_target)
            RR_ = matrix_to_xyzrpy(right_target)
            print(f"LL: {LL_[0].item():.3f}, {LL_[1].item():.3f}, {LL_[2].item():.3f}, {LL_[3]:.3f}, {LL_[4]:.3f}, {LL_[5]:.3f}") # noqa
            print(f"RR: {RR_[0].item():.3f}, {RR_[1].item():.3f}, {RR_[2].item():.3f}, {RR_[3]:.3f}, {RR_[4]:.3f}, {RR_[5]:.3f}") # noqa

            l_gripper_value = arm_command.left_gripper_value * 0.07
            r_gripper_value = arm_command.right_gripper_value * 0.07

            self.get_left_ik_solution(
                LL_[0],
                LL_[1],
                LL_[2],
                LL_[3],
                LL_[4],
                LL_[5],
                l_gripper_value,
            )

            self.get_right_ik_solution(
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

def main():
    arm_node = ArmNode()

    def signal_handler(signum, frame):
        arm_node.running = False
        arm_node.stop()

    signal.signal(signal.SIGINT, signal_handler)
    arm_node.run()

if __name__ == "__main__":
    main()
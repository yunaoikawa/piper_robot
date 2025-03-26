import time
from piper_sdk import C_PiperInterface
import numpy as np


class PiperControl:
    def __init__(self, can_port="can_right") -> None:
        self.can_port = can_port
        self.auto_enable = True
        self.gripper_exist = True
        self.gripper_val_mutiple = 1
        self.enable_flag_ = False
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)

    def enable_piper(self):
        enable_flag = False
        timeout = 5
        start_time = time.time()
        while not (enable_flag):
            print("enabling..")
            elapsed_time = time.time() - start_time
            enable_flag = (
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            if enable_flag:
                self.enable_flag_ = True
            if elapsed_time > timeout:
                print("enable timeout")
                exit(0)
            time.sleep(1)

    def joint_control(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0]),
    ):
        factor = 1000 * 180 / np.pi
        joint_0 = round(joint_pos[0] * factor)
        joint_1 = round(joint_pos[1] * factor)
        joint_2 = round(joint_pos[2] * factor)
        joint_3 = round(joint_pos[3] * factor)
        joint_4 = round(joint_pos[4] * factor)
        joint_5 = round(joint_pos[5] * factor)
        gripper_pos = joint_pos[6]

        if len(joint_pos) >= 7:
            gripper_pos = round(gripper_pos * 1e6)
            gripper_pos = gripper_pos * self.gripper_val_mutiple
            gripper_pos = max(0, min(gripper_pos, 60000))
        else:
            gripper_pos = -1
        if self.enable_flag_:
            all_zeros = np.all(joint_vel == 0)
            if not all_zeros:
                if len(joint_vel) == 7:
                    vel_all = round(joint_vel[6])
                    vel_all = max(0, min(vel_all, 100))
                    print("vel_all: %d", vel_all)
                    self.piper.MotionCtrl_2(0x01, 0x01, vel_all, 0x00)
                else:
                    self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            else:
                self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)

            self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            if self.gripper_exist and gripper_pos != -1:
                if abs(gripper_pos) < 200:
                    gripper_pos = 0
                if len(joint_vel) >= 7:
                    gripper_effort = joint_vel[6]
                    gripper_effort = max(0.5, min(gripper_effort, 3))
                    gripper_effort = round(gripper_effort * 1000)
                    self.piper.GripperCtrl(abs(gripper_pos), gripper_effort, 0x01, 0)
                else:
                    self.piper.GripperCtrl(abs(gripper_pos), 1000, 0x01, 0)


if __name__ == "__main__":
    piper_single = PiperControl()

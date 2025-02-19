import time

import numpy as np
from piper_sdk import C_PiperInterface

from utils import scale_and_clip_control


def enable_fun(piper: C_PiperInterface):
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = (
            piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        )
        print("Enable status:", enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        print("--------------------")
        if elapsed_time > timeout:
            print("Timeout....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if elapsed_time_flag:
        print("The program automatically enables timeout, exit the program")
        exit(0)


def setup_arms(left_name="can_left", right_name="can_right"):
    piper_left = C_PiperInterface(left_name)
    piper_left.ConnectPort()
    piper_left.EnableArm(7)
    enable_fun(piper=piper_left)
    piper_right = C_PiperInterface(right_name, can_auto_init=False)
    piper_right.ConnectPort()
    piper_right.EnableArm(7)
    enable_fun(piper=piper_right)
    return piper_left, piper_right


def send_to_arms(
    piper_left: C_PiperInterface, piper_right: C_PiperInterface, q: np.ndarray
):
    right_q, left_q = q[:6].copy(), q[8:14].copy()
    right_joint_values = scale_and_clip_control(right_q).tolist()
    piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    piper_right.JointCtrl(*right_joint_values[:6])
    piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)

    left_joint_values = scale_and_clip_control(left_q).tolist()
    piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    piper_left.JointCtrl(*left_joint_values[:6])
    piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    return True

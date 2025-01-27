#!/usr/bin/env python3
from typing import (
    Optional,
)
import time
from piper_sdk import *

def enable_fun(piper:C_PiperInterface):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("Timeout....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("The program automatically enables timeout, exit the program")
        exit(0)

if __name__ == "__main__":
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    # piper.DisableArm(7)
    # piper.GripperCtrl(0,1000,0x01, 0)
    distance_factor = 100 * 1000 # m
    angle_factor = 57324.840764 #1000*180/3.14 - radians
    position = [0,0,0,0,0,0]
    count = 0
    while True:
        # print(piper.GetArmStatus())
        import time
        count  = count + 1
        # print(count)
        if(count == 0):
            print("1-----------")
            position = [0,0,0,0,0,0]
        elif(count == 200):
            print("2-----------")
            position = [0,0.0,0,0,0,0]
        elif(count == 400):
            print("1-----------")
            position = [0,0,0,0,0,0]
            count = 0

        x = round(position[0]*distance_factor)
        y = round(position[1]*distance_factor)
        z = round(position[2]*distance_factor)
        rx = round(position[3]*angle_factor)
        ry = round(position[4]*angle_factor)
        rz = round(position[5]*angle_factor)
        # piper.MotionCtrl_1()
        piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
        # piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        piper.EndPoseCtrl(x, y, z, rx, ry, rz)
        # piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        # piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        time.sleep(0.01)
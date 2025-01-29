#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 读取机械臂消息并打印,需要先安装piper_sdk
from typing import (
    Optional,
)
from piper_sdk import C_PiperInterface
import time

joint_limits = {}

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface()
    piper.ConnectPort()
    while True:
        # a.SearchMotorMaxAngleSpdAccLimit(1, 1)
        # a.ArmParamEnquiryAndConfig(1,0,2,0,3)
        # a.GripperCtrl(50000,1500,0x01)
        # print(piper.GetArmJointMsgs())
        data = piper.GetArmJointMsgs()
        for line in data.split("\n"):
            if line.startswith("Joint"):
                name, value = line.split(":")
                name = name.split(" ")[1]
                joint = int(name)
                value = int(value.split(",")[0])
                joint_limits[joint] = value
        print(joint_limits)
        print(piper.GetArmGripperMsgs())
        time.sleep(0.005)

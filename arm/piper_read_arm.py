#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 读取机械臂消息并打印,需要先安装piper_sdk
from dataclasses import dataclass
from collections import defaultdict
from typing import (
    Optional,
)
from piper_sdk import C_PiperInterface
import time


@dataclass
class MinMax:
    minVal: int = 1000000000
    maxVal: int = -1000000000


# 1: MinMax(minVal=-156679, maxVal=153666),
# 2: MinMax(minVal=-1301, maxVal=194344),
# 3: MinMax(minVal=-173712, maxVal=1941),
# 4: MinMax(minVal=-102797, maxVal=102566),
# 5: MinMax(minVal=-75731, maxVal=74399),
# 6: MinMax(minVal=611969, maxVal=611969)

joint_limits = defaultdict(MinMax)

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface()
    piper.ConnectPort()
    while True:
        # a.SearchMotorMaxAngleSpdAccLimit(1, 1)
        # a.ArmParamEnquiryAndConfig(1,0,2,0,3)
        # a.GripperCtrl(50000,1500,0x01)
        # print(piper.GetArmJointMsgs())
        try:
            data = str(piper.GetArmJointMsgs())
            for line in data.split("\n"):
                if line.startswith("Joint"):
                    name, value = line.split(":")
                    name = name.split(" ")[1]
                    joint = int(name)
                    value = int(value.split(",")[0])
                    joint_limits[joint].minVal = min(value, joint_limits[joint].minVal)
                    joint_limits[joint].maxVal = max(value, joint_limits[joint].maxVal)
            print(joint_limits)
            print("\033[3A")
            # print(piper.GetArmGripperMsgs())
            time.sleep(0.005)
        except KeyboardInterrupt:
            print("\n" * 10)
            exit()

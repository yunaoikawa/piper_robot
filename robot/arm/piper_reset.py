from piper_sdk import C_PiperInterface_V2

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_left")
    piper.ConnectPort()
    piper.MotionCtrl_1(0x02,0,0)#恢复
    piper.MotionCtrl_2(0, 0, 0, 0x00)#位置速度模式
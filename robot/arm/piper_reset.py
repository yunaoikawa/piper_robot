from piper_sdk import C_PiperInterface_V2

# 测试代码
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ["left", "right", "can0"]:
        print("Usage: python piper_reset.py <left|right>")
        sys.exit(1)

    if "can" in sys.argv[1]:
        can_port = sys.argv[1]
    else:
        can_port = "can_" + sys.argv[1]
    piper = C_PiperInterface_V2(can_port)
    piper.ConnectPort()
    piper.MotionCtrl_1(0x02,0,0)#恢复
    piper.MotionCtrl_2(0, 0, 0, 0x00)#位置速度模式
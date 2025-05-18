from piper_sdk import C_PiperInterface

if __name__ == "__main__":
    piper = C_PiperInterface("can_left")
    piper.ConnectPort()
    piper.MotionCtrl_1(0x02,0,0)
    piper.MotionCtrl_2(0,0,0,0x00)

    del piper

    piper = C_PiperInterface("can_right")
    piper.ConnectPort()
    piper.MotionCtrl_1(0x02,0,0)
    piper.MotionCtrl_2(0,0,0,0x00)

    del piper

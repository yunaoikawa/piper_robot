import time
from dataclasses import dataclass

import numpy as np
from loop_rate_limiters import RateLimiter

from robot.arm.piper_sdk.hardware import PiperHardwareStation

@dataclass
class Gains:
    kp: np.ndarray = np.zeros((6, 1))
    kd: np.ndarray = np.zeros((6, 1))


class ControllerBase:
    def __init__(self, channel="can_left") -> None:
        self.hardware_station = PiperHardwareStation(channel)

    def init_robot(self):
        init_rounds = 10
        for _ in range(init_rounds):
            self.recv_()
            self.check_joint_state_sanity_()
            self.over_current_protection_()

        gains = Gains()

    def recv_(self):
        self.hardware_station.enable_motors()
        time.sleep(0.01) # 10ms
        self.hardware_station.set_ctrl_mode(0x01, 0x04, 0, 0xAD)
        time.sleep(0.01) # 10ms

    def check_joint_state_sanity_(self):
        pass

    def over_current_protection_(self):
        pass


if __name__ == "__main__":
    controller = ControllerBase()
    controller.hardware_station.start()

    rate_limiter = RateLimiter(frequency=200)

    try:
        while True:
            q, qd, tau, timestamp = controller.hardware_station.get_joint_state()
            print(f"at {timestamp}s, q: {q}, qd: {qd}, tau: {tau}")
            rate_limiter.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        controller.hardware_station.stop()






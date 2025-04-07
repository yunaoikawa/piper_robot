from robot.arm.piper_sdk.hardware import PiperHardwareStation
from loop_rate_limiters import RateLimiter

class ControllerBase:
    def __init__(self, channel="can_left") -> None:
        self.hardware_station = PiperHardwareStation(channel)

if __name__ == "__main__":
    controller = ControllerBase()
    controller.hardware_station.start()

    rate_limiter = RateLimiter(frequency=200)

    try:
        while True:
            pos = controller.hardware_station.get_pos()
            print(pos)
            rate_limiter.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        controller.hardware_station.stop()






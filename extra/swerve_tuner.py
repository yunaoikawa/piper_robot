import os

os.environ["CTR_TARGET"] = "Hardware"  # pylint: disable=wrong-import-position
import time

import phoenix6
import phoenix6.unmanaged

from robot.controller.base import SteerMotor, DriveMotor


def test_steer_motor():
    motor = SteerMotor(7)
    last_time = time.time()
    CONTROL_PERIOD = 0.004  # 250Hz

    position_log = []
    target = 0  # -math.pi / 2

    try:
        motor.set_position(target)
        while True:
            phoenix6.BaseStatusSignal.refresh_all(motor.status_signals)
            pos = motor.get_position()
            position_log.append(pos)

            phoenix6.unmanaged.feed_enable(0.1)

            step_time = last_time - time.time()
            if step_time < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD - step_time)
            last_time = time.time()
    finally:
        motor.set_neutral()

        import matplotlib.pyplot as plt

        plt.plot(position_log[::25])
        # draw horizontal line at target
        plt.axhline(y=target, color="r", linestyle="--")
        plt.legend(["TalonFX"])

        plt.savefig("position_plot.png")
        plt.close()


def test_drive_motor():
    motor = DriveMotor(8)
    last_time = time.time()
    CONTROL_PERIOD = 0.004  # 250Hz

    vel_log = []
    target = 0.5

    try:
        motor.set_velocity(target)
        while True:
            phoenix6.BaseStatusSignal.refresh_all(motor.status_signals)
            vel = motor.get_velocity()
            vel_log.append(vel)
            phoenix6.unmanaged.feed_enable(0.1)

            step_time = last_time - time.time()
            if step_time < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD - step_time)
            last_time = time.time()
    finally:
        motor.set_neutral()

        import matplotlib.pyplot as plt

        plt.plot(vel_log[::10])
        # draw horizontal line at target
        plt.axhline(y=target, color="r", linestyle="--")
        plt.legend(["TalonFX"])

        plt.savefig("velocity_plot.png")
        plt.close()


if __name__ == "__main__":
    # test_steer_motor()
    test_drive_motor()

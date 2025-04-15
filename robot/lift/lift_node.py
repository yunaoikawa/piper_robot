import os
import time
from typing import cast

os.environ["CTR_TARGET"] = "Hardware"
import phoenix6
from phoenix6 import configs, controls, hardware
from loop_rate_limiters import RateLimiter

from robot.base.constants import CONTROL_FREQ

class Lift:
    def __init__(self):
        self.lift_motor = hardware.TalonFX(9, canbus="Drivetrain")

        assert self.lift_motor.get_is_pro_licensed()

        self.position_signal = self.lift_motor.get_position()
        self.velocity_signal = self.lift_motor.get_velocity()
        self.status_signals = cast(list[phoenix6.BaseStatusSignal], [self.position_signal, self.velocity_signal])

        self.lift_motor_cfg = configs.TalonFXConfiguration()
        # position control gains
        self.lift_motor_cfg.slot0.k_p = 0.0
        self.lift_motor_cfg.slot0.k_i = 0.0
        self.lift_motor_cfg.slot0.k_d = 0.0
        # self.lift_motor_cfg.slot0.k_s = 10
        # self.lift_motor_cfg.slot0.static_feedforward_sign = signals.StaticFeedforwardSignValue.USE_CLOSED_LOOP_SIGN

        # velocity control gains
        self.lift_motor_cfg.slot1.k_p = 5.0
        self.lift_motor_cfg.slot1.k_i = 0.0
        self.lift_motor_cfg.slot1.k_d = 0.0

        self.lift_motor_cfg.torque_current.peak_forward_torque_current = 120
        self.lift_motor_cfg.torque_current.peak_reverse_torque_current = -120
        self.lift_motor_cfg.audio.beep_on_boot = False

        self.min_pos, self.max_pos = -0.39, 0.0 # [m]

        phoenix6.BaseStatusSignal.set_update_frequency_for_all(CONTROL_FREQ, self.status_signals)

        status = self.lift_motor.configurator.apply(self.lift_motor_cfg)
        if not status.is_ok():
            raise Exception(f"Failed to apply TalonFX configuration: {status}")

        self.position_request = controls.DynamicMotionMagicTorqueCurrentFOC(0, 20, 10, 100, slot=0)
        self.velocity_request = controls.VelocityTorqueCurrentFOC(0, slot=1)
        self.neutral_request = controls.NeutralOut()
        # self.lift_motor.set_position(0)

    def get_position(self) -> float:
        # zero position - fully retracted - 0
        # lead screw pitch = 6mm per rev
        # motor pulley teeth = 16
        # shaft pulley teeth = 24
        # 16/24 * 6 = 4mm per rev
        return -self.position_signal.value * 0.004 # [m]

    def get_velocity(self) -> float:
        return -self.velocity_signal.value * 0.004 # [m/s]

    def set_neutral(self) -> None:
        self.lift_motor.set_control(self.neutral_request)

    def set_control(self, control: float) -> None:
        self.lift_motor.set_control(self.position_request.with_position(-control / 0.004))

    def set_velocity_control(self, velocity: float) -> None:
        self.lift_motor.set_control(self.velocity_request.with_velocity(-velocity / 0.004))

    def update_state(self):
        phoenix6.BaseStatusSignal.refresh_all(self.status_signals)


if __name__ == "__main__":
    lift = Lift()
    lift.set_neutral()
    time.sleep(2.0)
    input("Press Enter to start")
    rate = RateLimiter(100)
    try:
        while True:
            phoenix6.unmanaged.feed_enable(0.1)
            lift.set_velocity_control(0.05)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        lift.set_neutral()

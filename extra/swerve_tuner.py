import os
# os.environ['CTR_TARGET'] = 'Hardware'  # pylint: disable=wrong-import-position
import math
import time

import phoenix6
import phoenix6.unmanaged
from phoenix6 import configs, controls, hardware, signals

from robot.base.constants import ENCODER_MAGNET_OFFSETS, TIRE_RADIUS, DRIVE_GEAR_RATIO, TWO_PI

class SteerMotor:
    def __init__(self, num: int):
        self.num = num
        assert num % 2 == 1, "Steer motors must have odd numbers"
        self.fx = hardware.TalonFX(self.num, canbus="Drivetrain")
        assert self.fx.get_is_pro_licensed()  # Must be Phoenix Pro licensed for FOC

        self.cc = hardware.CANcoder((self.num // 2) + 1, canbus="Drivetrain")

        if self.num == 1:
            time.sleep(0.2)  # First CAN device, wait for CAN bus to be ready
            supply_voltage = self.fx.get_supply_voltage().value
            print(f'Motor supply voltage: {supply_voltage:.2f} V')
            if supply_voltage < 11.5 and os.environ.get("CTR_TARGET", None) == "Hardware":
                raise Exception('Motor supply voltage is too low. Please charge the battery.')

        self.position_signal = self.fx.get_position()
        self.status_signals = [self.position_signal]
        self.position_request = controls.PositionTorqueCurrentFOC(0)
        self.neutral_request = controls.NeutralOut()

        # Motor configuration
        self.fx_cfg = configs.TalonFXConfiguration()
        self.fx_cfg.slot0.k_s = 5
        self.fx_cfg.slot0.static_feedforward_sign = signals.StaticFeedforwardSignValue.USE_CLOSED_LOOP_SIGN
        self.fx_cfg.slot0.k_p = 60
        self.fx_cfg.slot0.k_i = 0.0
        self.fx_cfg.slot0.k_d = 6
        self.fx_cfg.torque_current.peak_forward_torque_current = 40 # Amperes
        self.fx_cfg.torque_current.peak_reverse_torque_current = -40
        self.fx_cfg.audio.beep_on_boot = False
        self.fx_cfg.feedback.feedback_remote_sensor_id = self.cc.device_id
        self.fx_cfg.feedback.feedback_sensor_source = signals.FeedbackSensorSourceValue.FUSED_CANCODER
        self.fx_cfg.feedback.sensor_to_mechanism_ratio = 1.0
        self.fx_cfg.feedback.rotor_to_sensor_ratio = 12.8
        self.fx_cfg.closed_loop_general.continuous_wrap = True

        # CANcoder configuration
        self.cc_cfg = configs.CANcoderConfiguration()
        self.cc_cfg.magnet_sensor.absolute_sensor_discontinuity_point = 0.5
        self.cc_cfg.magnet_sensor.magnet_offset = ENCODER_MAGNET_OFFSETS[self.num//2]
        self.cc_cfg.magnet_sensor.sensor_direction = signals.SensorDirectionValue.COUNTER_CLOCKWISE_POSITIVE

        status = self.fx.configurator.apply(self.fx_cfg)
        if not status.is_ok():
            raise Exception(f'Failed to apply TalonFX configuration: {status}')

        status = self.cc.configurator.apply(self.cc_cfg)
        if not status.is_ok():
            raise Exception(f'Failed to apply CANCoder configuration: {status}')

        self.fx.set_position(0)

    def get_position(self) -> float: # radians (-pi, pi)
        return ((self.position_signal.value + 0.5) % 1 - 0.5) * TWO_PI

    def set_position(self, position: float) -> None:
        assert -math.pi <= position <= math.pi
        self.fx.set_control(self.position_request.with_position(position / TWO_PI))

    def set_neutral(self):
        self.fx.set_control(self.neutral_request)

class DriveMotor:
    def __init__(self, num: int):
        self.num = num
        assert num % 2 == 0, "Drive motors must have even numbers"
        self.fx = hardware.TalonFX(self.num, canbus="Drivetrain")
        assert self.fx.get_is_pro_licensed()  # Must be Phoenix Pro licensed for FOC

        self.velocity_signal = self.fx.get_velocity()
        self.status_signals = [self.velocity_signal]
        self.velocity_request = controls.VelocityTorqueCurrentFOC(0)
        self.neutral_request = controls.NeutralOut()

        # Motor configuration
        self.fx_cfg = configs.TalonFXConfiguration()
        self.fx_cfg.slot0.k_s = 1.5
        self.fx_cfg.slot0.k_p = 3
        self.fx_cfg.slot0.k_i = 0
        self.fx_cfg.slot0.k_d = 0.1
        self.fx_cfg.torque_current.peak_forward_torque_current = 10 # Amperes
        self.fx_cfg.torque_current.peak_reverse_torque_current = -10
        self.fx_cfg.audio.beep_on_boot = False

        status = self.fx.configurator.apply(self.fx_cfg)
        if not status.is_ok():
            raise Exception(f'Failed to apply TalonFX configuration: {status}')

        self.fx.set_position(0)

    def get_velocity(self) -> float:
        return (self.velocity_signal.value / DRIVE_GEAR_RATIO)

    def set_velocity(self, velocity: float) -> None: # m/s
        velocity = DRIVE_GEAR_RATIO * (velocity / (TWO_PI * TIRE_RADIUS))
        self.fx.set_control(self.velocity_request.with_velocity(velocity))

    def set_neutral(self):
        self.fx.set_control(self.neutral_request)

def test_steer_motor():
    motor = SteerMotor(5)
    last_time = time.time()
    CONTROL_PERIOD = 0.004 # 250Hz

    position_log = []
    target = -math.pi / 2

    try:
        motor.set_position(target)
        while True:
            phoenix6.BaseStatusSignal.refresh_all(motor.status_signals)
            pos = motor.get_position()
            position_log.append(pos)

            phoenix6.unmanaged.feed_enable(0.1)

            step_time = last_time - time.time()
            if step_time < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD-step_time)
            last_time = time.time()
    finally:
        motor.set_neutral()

        import matplotlib.pyplot as plt
        plt.plot(position_log[::25])
        # draw horizontal line at target
        plt.axhline(y=target, color='r', linestyle='--')
        plt.legend(["TalonFX"])

        plt.savefig("position_plot.png")
        plt.close()


def test_drive_motor():
    motor = DriveMotor(4)
    last_time = time.time()
    CONTROL_PERIOD = 0.004 # 250Hz

    vel_log = []
    target = 1.0

    try:
        motor.set_velocity(target)
        while True:
            phoenix6.BaseStatusSignal.refresh_all(motor.status_signals)
            vel = motor.get_velocity()
            vel_log.append(vel)
            phoenix6.unmanaged.feed_enable(0.1)

            step_time = last_time - time.time()
            if step_time < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD-step_time)
            last_time = time.time()
    finally:
        motor.set_neutral()

        import matplotlib.pyplot as plt
        plt.plot(vel_log[::10])
        # draw horizontal line at target
        plt.axhline(y=target, color='r', linestyle='--')
        plt.legend(["TalonFX"])

        plt.savefig("velocity_plot_sim.png")
        plt.close()



if __name__ == "__main__":
    # test_steer_motor()
    test_drive_motor()
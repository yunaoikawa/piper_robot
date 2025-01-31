import os
os.environ['CTR_TARGET'] = 'Hardware'  # pylint: disable=wrong-import-position
import math
import time

import phoenix6
import phoenix6.unmanaged
from phoenix6 import configs, controls, hardware, signals

from constants import ENCODER_MAGNET_OFFSETS

TWO_PI = 2 * math.pi

N_r1 = 50.0 / 16.0               # Drive gear ratio (1st stage)
N_r2 = 19.0 / 25.0               # Drive gear ratio (2nd stage)
N_w = 45.0 / 15.0                # Wheel gear ratio
N_r1_r2_w = N_r1 * N_r2 * N_w

class SteerMotor:
    def __init__(self, num):
        self.num = num
        assert num % 2 == 1, "Steer motors must have odd numbers"
        self.fx = hardware.TalonFX(self.num, canbus="Drivetrain")
        assert self.fx.get_is_pro_licensed()  # Must be Phoenix Pro licensed for FOC
        fx_cfg = configs.TalonFXConfiguration()

        self.cc = hardware.CANcoder((self.num // 2) + 1, canbus="Drivetrain")

        if self.num == 1:
            time.sleep(0.2)  # First CAN device, wait for CAN bus to be ready
            supply_voltage = self.fx.get_supply_voltage().value
            print(f'Motor supply voltage: {supply_voltage:.2f} V')
            if supply_voltage < 11.5:
                raise Exception('Motor supply voltage is too low. Please charge the battery.')

        # Status signals
        self.position_signal = self.fx.get_position()
        self.velocity_signal = self.fx.get_velocity()
        self.status_signals = [self.position_signal, self.velocity_signal]

        # Control requests
        self.position_request = controls.PositionTorqueCurrentFOC(0)
        self.neutral_request = controls.NeutralOut()

        # Velocity control gains
        fx_cfg.slot0.k_s = 5
        fx_cfg.slot0.static_feedforward_sign = signals.StaticFeedforwardSignValue.USE_CLOSED_LOOP_SIGN
        fx_cfg.slot0.k_p = 60
        fx_cfg.slot0.k_i = 0.0
        fx_cfg.slot0.k_d = 6

        # Current limits (hard floor with no incline)
        # IMPORTANT: These values limit the force that the base can generate. Proceed very carefully if modifying these values.
        torque_current_limit = 40 # 40 A for steer, 10 A for drive
        fx_cfg.torque_current.peak_forward_torque_current = torque_current_limit
        fx_cfg.torque_current.peak_reverse_torque_current = -torque_current_limit

        # Disable beeps (Note: beep_on_config is not yet supported as of Oct 2024)
        fx_cfg.audio.beep_on_boot = False

        # Fused CANcoder setup
        fx_cfg.feedback.feedback_remote_sensor_id = self.cc.device_id
        fx_cfg.feedback.feedback_sensor_source = signals.FeedbackSensorSourceValue.FUSED_CANCODER
        fx_cfg.feedback.sensor_to_mechanism_ratio = 1.0
        fx_cfg.feedback.rotor_to_sensor_ratio = 12.8

        # Continuous wrap
        fx_cfg.closed_loop_general.continuous_wrap = True

        # CANcoder configuration
        cc_cfg = configs.CANcoderConfiguration()
        cc_cfg.magnet_sensor.absolute_sensor_discontinuity_point = 0.5
        cc_cfg.magnet_sensor.magnet_offset = ENCODER_MAGNET_OFFSETS[(self.num // 2)];
        cc_cfg.magnet_sensor.sensor_direction = signals.SensorDirectionValue.COUNTER_CLOCKWISE_POSITIVE

        # Apply fx configuration
        status = self.fx.configurator.apply(fx_cfg)
        if not status.is_ok():
            raise Exception(f'Failed to apply TalonFX configuration: {status}')

        # Apply cc configuration
        status = self.cc.configurator.apply(cc_cfg)
        if not status.is_ok():
            raise Exception(f'Failed to apply CANCoder configuration: {status}')

        self.fx.set_position(0)

    def get_position(self):
        # return wrapped position
        return ((self.position_signal.value + 0.5) % 1 - 0.5) * TWO_PI

    def set_position(self, position):
        assert -math.pi <= position <= math.pi
        self.fx.set_control(self.position_request.with_position(position / TWO_PI))

    def set_neutral(self):
        self.fx.set_control(self.neutral_request)



class DriveMotor:
    def __init__(self, num):
        self.num = num
        assert num % 2 == 0, "Drive motors must have even numbers"
        self.fx = hardware.TalonFX(self.num, canbus="Drivetrain")
        assert self.fx.get_is_pro_licensed()  # Must be Phoenix Pro licensed for FOC
        fx_cfg = configs.TalonFXConfiguration()

        # Status signals
        self.velocity_signal = self.fx.get_velocity()
        self.status_signals = [self.velocity_signal]

        # Control requests
        self.velocity_request = controls.VelocityTorqueCurrentFOC(0)
        self.neutral_request = controls.NeutralOut()

        # Velocity control gains
        fx_cfg.slot0.k_s = 1.5
        fx_cfg.slot0.k_p = 3
        fx_cfg.slot0.k_i = 0
        fx_cfg.slot0.k_d = 0.1

        # Current limits (hard floor with no incline)
        # IMPORTANT: These values limit the force that the base can generate. Proceed very carefully if modifying these values.
        torque_current_limit = 10 # 40 A for steer, 10 A for drive
        fx_cfg.torque_current.peak_forward_torque_current = torque_current_limit
        fx_cfg.torque_current.peak_reverse_torque_current = -torque_current_limit

        # Disable beeps (Note: beep_on_config is not yet supported as of Oct 2024)
        fx_cfg.audio.beep_on_boot = False

        # Apply fx configuration
        status = self.fx.configurator.apply(fx_cfg)
        if not status.is_ok():
            raise Exception(f'Failed to apply TalonFX configuration: {status}')

        self.fx.set_position(0)

    def get_velocity(self):
        # return reduced velocity
        return (self.velocity_signal.value / N_r1_r2_w)

    def set_velocity(self, velocity):
        self.fx.set_control(self.velocity_request.with_velocity(velocity * N_r1_r2_w))

    def set_neutral(self):
        self.fx.set_control(self.neutral_request)

def test_steer_motor():
    motor = SteerMotor(3)
    last_time = time.time()
    CONTROL_PERIOD = 0.004 # 250Hz

    position_log = []
    target = -math.pi/2

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
    target = 3.0

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

        plt.savefig("velocity_plot.png")
        plt.close()



if __name__ == "__main__":
    test_steer_motor()
    # test_drive_motor()
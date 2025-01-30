import os
os.environ['CTR_TARGET'] = 'Hardware'  # pylint: disable=wrong-import-position
import math
import time

import phoenix6
import phoenix6.unmanaged
from phoenix6 import configs, controls, hardware, CANBus, signals

TWO_PI = 2 * math.pi

N_s = 32.0 / 15.0 * 60.0 / 10.0  # Steer gear ratio
N_r1 = 50.0 / 16.0               # Drive gear ratio (1st stage)
N_r2 = 19.0 / 25.0               # Drive gear ratio (2nd stage)
N_w = 45.0 / 15.0                # Wheel gear ratio
N_r1_r2_w = N_r1 * N_r2 * N_w

class Motor:
    def __init__(self, num):
        self.num = num
        self.is_steer = num % 2 != 0  # Odd num motors are steer motors
        self.fx = hardware.TalonFX(self.num, canbus="Drivetrain")
        assert self.fx.get_is_pro_licensed()  # Must be Phoenix Pro licensed for FOC
        fx_cfg = configs.TalonFXConfiguration()

        self.cc = hardware.CANcoder(1, canbus="Drivetrain")


        if self.num: #  == 1:
            time.sleep(0.2)  # First CAN device, wait for CAN bus to be ready
            supply_voltage = self.fx.get_supply_voltage().value
            print(f'Motor supply voltage: {supply_voltage:.2f} V')
            if supply_voltage < 11.5:
                raise Exception('Motor supply voltage is too low. Please charge the battery.')

        # Status signals
        self.position_signal = self.fx.get_position()
        self.cancoder_position_signal = self.cc.get_absolute_position()
        self.velocity_signal = self.fx.get_velocity()
        self.status_signals = [self.position_signal, self.velocity_signal, self.cancoder_position_signal]

        # Control requests
        self.position_request = controls.PositionTorqueCurrentFOC(0)
        self.neutral_request = controls.NeutralOut()

        # Velocity control gains
        fx_cfg.slot0.k_s = 3
        fx_cfg.slot0.k_p = 0.0
        fx_cfg.slot0.k_i = 0.0
        fx_cfg.slot0.k_d = 0.0 if self.is_steer else 0.0  # Set k_d for steer to prevent caster flutter

        # Current limits (hard floor with no incline)
        # IMPORTANT: These values limit the force that the base can generate. Proceed very carefully if modifying these values.
        torque_current_limit = 40 if self.is_steer else 10  # 40 A for steer, 10 A for drive
        fx_cfg.torque_current.peak_forward_torque_current = torque_current_limit
        fx_cfg.torque_current.peak_reverse_torque_current = -torque_current_limit

        # Disable beeps (Note: beep_on_config is not yet supported as of Oct 2024)
        fx_cfg.audio.beep_on_boot = False

        # Fused CANcoder setup
        fx_cfg.feedback.feedback_remote_sensor_id = self.cc.device_id
        fx_cfg.feedback.feedback_sensor_source = signals.FeedbackSensorSourceValue.FUSED_CANCODER
        fx_cfg.feedback.sensor_to_mechanism_ratio = 1.0
        fx_cfg.feedback.rotor_to_sensor_ratio = 12.8

        # CANcoder configuration
        cc_cfg = configs.CANcoderConfiguration()
        cc_cfg.magnet_sensor.absolute_sensor_discontinuity_point = 0.5
        cc_cfg.magnet_sensor.magnet_offset = -860.0 / 4096;
        cc_cfg.magnet_sensor.sensor_direction = signals.SensorDirectionValue.COUNTER_CLOCKWISE_POSITIVE

        # Apply configuration
        self.fx.configurator.apply(fx_cfg)
        self.cc.configurator.apply(cc_cfg)

    def get_position(self):
        return self.position_signal.value

    def get_velocity(self):
        factor = N_s if self.is_steer else N_r1_r2_w
        return (TWO_PI * self.velocity_signal.value) / factor

    # def set_velocity(self, velocity):
    #     velocity = velocity * N_s if self.is_steer else velocity * N_r1_r2_w
    #     self.fx.set_control(self.velocity_request.with_velocity(velocity / TWO_PI))

    def set_position(self, position):
        self.fx.set_control(self.position_request.with_position(position))

    def set_neutral(self):
        self.fx.set_control(self.neutral_request)

if __name__ == "__main__":
    motor = Motor(1)
    last_time = time.time()
    CONTROL_PERIOD = 0.004 # 250Hz
    position_log = []

    first_frame = True

    try:
        while True:
            phoenix6.BaseStatusSignal.refresh_all(motor.status_signals)
            position_log.append(motor.get_position())

            if first_frame:
                target = motor.get_position() + 0.2
                first_frame = False

            phoenix6.unmanaged.feed_enable(0.1)
            motor.set_position(target)

            step_time = last_time - time.time()
            if step_time < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD-step_time)
            last_time = time.time()
    finally:
        motor.set_neutral()

        import matplotlib.pyplot as plt
        plt.plot(position_log[::25])
        plt.savefig("speed_plot.png")
        plt.close()



import os
import time
import math
import numpy as np
import queue
import threading
from enum import Enum

os.environ["CTR_TARGET"] = "Hardware"  # pylint: disable=wrong-import-position

import phoenix6.unmanaged
from phoenix6 import configs, controls, hardware, signals

from constants import (POLICY_CONTROL_PERIOD, ENCODER_MAGNET_OFFSETS, TWO_PI, N_r1_r2_w, CONTROL_FREQ, CONTROL_PERIOD, NUM_SWERVES, LENGTH, WIDTH, TIRE_RADIUS)

class SteerMotor:
    def __init__(self, num):
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

        # Status signals
        self.position_signal = self.fx.get_position()
        self.status_signals = [self.position_signal]

        # Control requests
        self.position_request = controls.PositionTorqueCurrentFOC(0)
        self.neutral_request = controls.NeutralOut()

        # Velocity control gains
        self.fx_cfg = configs.TalonFXConfiguration()
        self.fx_cfg.slot0.k_s = 5
        self.fx_cfg.slot0.static_feedforward_sign = signals.StaticFeedforwardSignValue.USE_CLOSED_LOOP_SIGN
        self.fx_cfg.slot0.k_p = 60
        self.fx_cfg.slot0.k_i = 0.0
        self.fx_cfg.slot0.k_d = 6

        # Current limits (hard floor with no incline)
        # IMPORTANT: These values limit the force that the base can generate. Proceed very carefully if modifying these values.
        torque_current_limit = 40 # 40 A for steer, 10 A for drive
        self.fx_cfg.torque_current.peak_forward_torque_current = torque_current_limit
        self.fx_cfg.torque_current.peak_reverse_torque_current = -torque_current_limit

        # Disable beeps (Note: beep_on_config is not yet supported as of Oct 2024)
        self.fx_cfg.audio.beep_on_boot = False

        # Fused CANcoder setup
        self.fx_cfg.feedback.feedback_remote_sensor_id = self.cc.device_id
        self.fx_cfg.feedback.feedback_sensor_source = signals.FeedbackSensorSourceValue.FUSED_CANCODER
        self.fx_cfg.feedback.sensor_to_mechanism_ratio = 1.0
        self.fx_cfg.feedback.rotor_to_sensor_ratio = 12.8

        # Continuous wrap
        self.fx_cfg.closed_loop_general.continuous_wrap = True

        # CANcoder configuration
        self.cc_cfg = configs.CANcoderConfiguration()
        self.cc_cfg.magnet_sensor.absolute_sensor_discontinuity_point = 0.5
        self.cc_cfg.magnet_sensor.magnet_offset = ENCODER_MAGNET_OFFSETS[self.num//2]
        self.cc_cfg.magnet_sensor.sensor_direction = signals.SensorDirectionValue.COUNTER_CLOCKWISE_POSITIVE

        # Apply fx configuration
        status = self.fx.configurator.apply(self.fx_cfg)
        if not status.is_ok():
            raise Exception(f'Failed to apply TalonFX configuration: {status}')

        # Apply cc configuration
        status = self.cc.configurator.apply(self.cc_cfg)
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
        """
        velocity: m/s
        """
        velocity = N_r1_r2_w * velocity / (TWO_PI * TIRE_RADIUS)
        self.fx.set_control(self.velocity_request.with_velocity(velocity))

    def set_neutral(self):
        self.fx.set_control(self.neutral_request)


class Swerve:
    def __init__(self, num):
        self.num = num
        self.steer_motor = SteerMotor(2 * self.num - 1)
        self.drive_motor = DriveMotor(2 * self.num)

        # Status signals
        self.status_signals = (
            self.steer_motor.status_signals + self.drive_motor.status_signals
        )

    def get_steer_position(self):
        return self.steer_motor.get_position()

    def get_velocity(self):
        return self.drive_motor.get_velocity()

    def set_steer_position(self, steer_position):
        self.steer_motor.set_position(steer_position)

    def set_velocity(self, drive_velocity):
        self.drive_motor.set_velocity(drive_velocity)

    def set_neutral(self):
        self.steer_motor.set_neutral()
        self.drive_motor.set_neutral()


class CommandType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"

# Currently only used for velocity commands
class FrameType(Enum):
    GLOBAL = "global"
    LOCAL = "local"


class Vehicle:
    def __init__(self, max_vel=(1.0, 1.0, 1.57), max_accel=(0.25, 0.25, 0.79)):
        self.max_vel = max_vel
        self.max_accel = max_accel

        # Vehicle parameters
        self.C_3_to_8 = np.array(
            [
                [1, 0, -LENGTH],
                [1, 0, -LENGTH],
                [1, 0, LENGTH],
                [1, 0, LENGTH],
                [0, 1, -WIDTH],
                [0, 1, WIDTH],
                [0, 1, -WIDTH],
                [0, 1, WIDTH],
            ]
        )

        # TODO: create PID file

        self.swerve_modules = [Swerve(i) for i in range(1, 5)]
        # CAN bus update frequency
        self.status_signals = [
            signal for swerve in self.swerve_modules for signal in swerve.status_signals
        ]
        phoenix6.BaseStatusSignal.set_update_frequency_for_all(
            CONTROL_FREQ, self.status_signals
        )

        # Joint space
        self.steer_pos = np.zeros(NUM_SWERVES)
        self.drive_vel = np.zeros(NUM_SWERVES)

        # Operational space (global frame)
        num_dofs = 3  # (x, y, theta)
        self.x = np.zeros(num_dofs)
        self.dx = np.zeros(num_dofs)

        self.target = None

        # Control loop
        self.command_queue = queue.Queue(1)
        self.control_loop_thread = threading.Thread(
            target=self.control_loop, daemon=True
        )
        self.control_loop_running = False

    def _enqueue_command(self, command_type, target, frame=None):
        if self.command_queue.full():
            print("Warning: Command queue is full. Is control loop running?")
        else:
            command = {"type": command_type, "target": target}
            if frame is not None:
                command["frame"] = FrameType(frame)
            self.command_queue.put(command, block=False)

    def set_target_velocity(self, velocity, frame="global"):
        self._enqueue_command(CommandType.VELOCITY, velocity, frame)

    def set_target_position(self, position):
        self._enqueue_command(CommandType.POSITION, position)

    def update_state(self):
        phoenix6.BaseStatusSignal.refresh_all(self.status_signals)

        for i, swerve in enumerate(self.swerve_modules):
            self.steer_pos[i] = swerve.get_steer_position()
            self.drive_vel[i] = swerve.get_velocity()

    def start_control(self):
        if self.control_loop_thread is None:
            print(
                "To initiate a new control loop, please create a new instance of Vehicle."
            )
            return

        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self):
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None

    def vehicle_velocity_to_angle_and_speed(self, u_3dof):
        wheel_velocities_directional = self.C_3_to_8 @ u_3dof
        vx = wheel_velocities_directional[:4]
        vy = wheel_velocities_directional[4:]

        wheel_speeds = np.sqrt(vx**2 + vy**2)
        wheel_angles = np.arctan2(vy, vx)
        return wheel_speeds, wheel_angles

    def control_loop(self):
        # TODO: Set real-time scheduling policy
        disable_motors = True

        last_command_time = time.time()
        last_step_time = time.time()

        while self.control_loop_running:
            # update state
            self.update_state()

            # Check for new command
            if not self.command_queue.empty():
                command = self.command_queue.get()
                last_command_time = time.time()
                target = command["target"]

                if command["type"] == CommandType.VELOCITY:
                    self.target = target  # np.clip(target, -self.max_vel, self.max_vel) TODO: clip velocity

                elif command["type"] == CommandType.POSITION:
                    raise NotImplementedError("Position control not yet implemented")

                disable_motors = False

            # Maintain current pose if command stream is disrupted
            if time.time() - last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
                disable_motors = True
                # TODO: ruckig control

            if disable_motors:
                for swerve in self.swerve_modules:
                    swerve.set_neutral()
            else:
                phoenix6.unmanaged.feed_enable(0.1)

                wheel_speeds, wheel_angles = self.vehicle_velocity_to_angle_and_speed(self.target)

                for i, swerve in enumerate(self.swerve_modules):
                    swerve.set_steer_position(wheel_angles[i])
                for i, swerve in enumerate(self.swerve_modules):
                    swerve.set_velocity(wheel_speeds[i])

            step_time = time.time() - last_step_time
            if step_time < CONTROL_PERIOD:  # maintain control frequency
                time.sleep(CONTROL_PERIOD - step_time)

            curr_time = time.time()
            step_time = time.time() - last_step_time
            last_step_time = curr_time

            if step_time > 0.005:  # 5 ms
                print(
                    f"Warning: Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} control_loop"
                )

    def get_encoder_offsets(self):
        offsets = []
        for i, swerve in enumerate(self.swerve_modules):
            swerve.steer_motor.cc.configurator.refresh(
                swerve.steer_motor.cc_cfg
            )  # Read current config
            curr_offset = swerve.steer_motor.cc_cfg.magnet_sensor.magnet_offset
            swerve.steer_motor.cc.get_absolute_position().wait_for_update(0.1)
            curr_position = swerve.steer_motor.cc.get_absolute_position().value
            offsets.append(f"{round(4096 * (curr_offset - curr_position))}.0 / 4096")
        print(f"ENCODER_MAGNET_OFFSETS = [{', '.join(offsets)}]")


def circling_profile():
    T_final = 20
    DT = 0.004
    t = np.linspace(0, T_final, int(T_final / DT) + 1)

    R = 1  # turn radius
    w_path = math.pi / 8  # rad/s
    v_path = (R * w_path)  # m/s
    vx = v_path * np.cos(w_path * t)
    vy = v_path * np.sin(w_path * t)
    w = w_path * np.zeros_like(t) * 2
    u_3dof = np.stack([vx, vy, w], axis=0)  # (3, t)
    return u_3dof

def square_profile():
    T_final = 12
    DT = 0.004
    t = np.linspace(0, T_final, int(T_final / DT) + 1)

    v_path = 0.5  # m/s

    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    w = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] < 3:
            vx[i] = v_path
            vy[i] = 0
            w[i] = 0
        elif t[i] < 6:
            vx[i] = 0
            vy[i] = v_path
            w[i] = 0
        elif t[i] < 9:
            vx[i] = -v_path
            vy[i] = 0
            w[i] = 0
        else:
            vx[i] = 0
            vy[i] = -v_path
            w[i] = 0

    u_3dof = np.stack([vx, vy, w], axis=0)  # (3, t)
    return u_3dof

if __name__ == "__main__":
    vehicle = Vehicle()
    # vehicle.get_encoder_offsets(); exit()

    profiles = square_profile()
    vehicle.start_control()

    try:
        for i in range(profiles.shape[1]):
            vehicle.set_target_velocity(profiles[:, i])
            time.sleep(0.004)
        # for _ in range(100):
        #     vehicle.set_target_velocity(np.array([-1, 0, 0]))
        #     time.sleep(0.1)

    finally:
        vehicle.stop_control()

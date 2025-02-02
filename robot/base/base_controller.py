import os
import time
import math
import queue
import threading
from enum import Enum
from typing import Tuple
import numpy as np

os.environ["CTR_TARGET"] = "Hardware"  # pylint: disable=wrong-import-position
import phoenix6.unmanaged
from phoenix6 import configs, controls, hardware, signals
from constants import (POLICY_CONTROL_PERIOD, ENCODER_MAGNET_OFFSETS, TWO_PI, DRIVE_GEAR_RATIO, CONTROL_FREQ, CONTROL_PERIOD, NUM_SWERVES, LENGTH, WIDTH, TIRE_RADIUS)

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

class CommandType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"

# Currently only used for velocity commands
class FrameType(Enum):
    GLOBAL = "global"
    LOCAL = "local"

class Vehicle:
    def __init__(self, max_vel=np.array((1.0, 1.0, 1.57)), max_accel=np.array((0.25, 0.25, 0.79))):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.C = np.array([[1, 0, -WIDTH], [1, 0, WIDTH], [1, 0, -WIDTH], [1, 0, WIDTH],
                           [0, 1, LENGTH], [0, 1, LENGTH], [0, 1, -LENGTH], [0, 1, -LENGTH]])

        self.steer_motors = [SteerMotor(i*2-1) for i in range(NUM_SWERVES)]
        self.drive_motors = [DriveMotor(i*2) for i in range(NUM_SWERVES)]

        self.status_signals = [s for m in self.steer_motors + self.drive_motors for s in m.status_signals]
        phoenix6.BaseStatusSignal.set_update_frequency_for_all(CONTROL_FREQ, self.status_signals)

        self.steer_pos = np.zeros(NUM_SWERVES)
        self.drive_vel = np.zeros(NUM_SWERVES)
        self.target = None

        self.command_queue = queue.Queue(1)
        self.control_loop_thread = threading.Thread(
            target=self.control_loop, daemon=True
        )
        self.control_loop_running = False

    def _enqueue_command(self, command_type: CommandType, target:np.ndarray, frame: FrameType=None) -> None:
        if self.command_queue.full():
            print("Warning: Command queue is full. Is control loop running?")
        else:
            command = {"type": command_type, "target": target}
            if frame is not None:
                command["frame"] = FrameType(frame)
            self.command_queue.put(command, block=False)

    def set_target_velocity(self, velocity: np.ndarray, frame: str="global")->None:
        self._enqueue_command(CommandType.VELOCITY, velocity, frame)

    def set_target_position(self, position: np.ndarray)->None:
        self._enqueue_command(CommandType.POSITION, position)

    def update_state(self) -> None:
        phoenix6.BaseStatusSignal.refresh_all(self.status_signals)
        for i in range(NUM_SWERVES):
            self.steer_pos[i] = self.steer_motors[i].get_position()
            self.drive_vel[i] = self.drive_motors[i].get_velocity()

    def start_control(self) -> None:
        if self.control_loop_thread is None: raise Exception("Control loop thread not initialized. Please create a new instance of Vehicle.")
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self) -> None:
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None

    def vehicle_velocity_to_angle_and_speed(self, u_3dof: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        wheel_velocities_directional = self.C @ u_3dof
        vx, vy = wheel_velocities_directional[:4], wheel_velocities_directional[4:]
        # order: front left, front right, rear left, rear right TODO: fix this with self.C
        # convert to: front right, front left, rear left, rear right
        wheel_speeds = np.sqrt(vx**2 + vy**2)[[1, 0, 2, 3]]
        wheel_angles = np.arctan2(vy, vx)[[1, 0, 2, 3]]
        return wheel_speeds, wheel_angles

    def control_loop(self):
        # TODO: Set real-time scheduling policy
        disable_motors = True
        last_command_time = time.time()
        last_step_time = time.time()

        while self.control_loop_running:
            self.update_state()

            if not self.command_queue.empty():
                command = self.command_queue.get()
                last_command_time = time.time()

                if command["type"] == CommandType.VELOCITY:
                    self.target = np.clip(command["target"], -self.max_vel, self.max_vel) # TODO: clip velocity

                elif command["type"] == CommandType.POSITION:
                    raise NotImplementedError("Position control not yet implemented")

                disable_motors = False

            # Maintain current pose if command stream is disrupted
            if time.time() - last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
                disable_motors = True
                # TODO: ruckig control

            if disable_motors:
                for s in self.steer_motors:
                    s.set_neutral()
                for d in self.drive_motors:
                    d.set_neutral()
            else:
                phoenix6.unmanaged.feed_enable(0.1)

                wheel_speeds, wheel_angles = self.vehicle_velocity_to_angle_and_speed(self.target)

                for i in range(NUM_SWERVES):
                    self.steer_motors[i].set_position(wheel_angles[i])
                    self.drive_motors[i].set_velocity(wheel_speeds[i]) # TODO: cosine error scaling velocity

            step_time = time.time() - last_step_time
            if step_time < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD - step_time)
            if step_time > 0.005: # 5 ms
                print(f"Warning: Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} control_loop")
            last_step_time = time.time()


    def get_encoder_offsets(self):
        offsets = []
        for steer_motor in self.steer_motors:
            steer_motor.cc.configurator.refresh(steer_motor.cc_cfg)
            curr_offset = steer_motor.cc_cfg.magnet_sensor.magnet_offset
            steer_motor.cc.get_absolute_position().wait_for_update(0.1)
            curr_position = steer_motor.cc.get_absolute_position().value
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
        # for i in range(profiles.shape[1]):
        #     vehicle.set_target_velocity(profiles[:, i])
        #     time.sleep(0.004)
        for _ in range(100):
            vehicle.set_target_velocity(np.array([0, 0.1, 0]))
            time.sleep(0.1)
    finally:
        vehicle.stop_control()

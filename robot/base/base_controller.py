import os
import time
import math
import threading
from enum import Enum
from typing import Tuple
import numpy as np

os.environ["CTR_TARGET"] = "Hardware"
import phoenix6.unmanaged
from phoenix6 import configs, controls, hardware, signals

from robot.timer import FrequencyTimer
from robot.constants import (
  POLICY_CONTROL_PERIOD,
  ENCODER_MAGNET_OFFSETS,
  TWO_PI,
  DRIVE_GEAR_RATIO,
  CONTROL_FREQ,
  NUM_SWERVES,
  LENGTH,
  WIDTH,
  TIRE_RADIUS,
)


class SteerMotor:
  def __init__(self, num: int):
    self.num = num
    assert num % 2 == 1, "Steer motors must have odd numbers"
    self.fx = hardware.TalonFX(self.num, canbus="Drivetrain")
    assert self.fx.get_is_pro_licensed()  # Must be Phoenix Pro licensed for FOC

    self.cc = hardware.CANcoder((self.num // 2) + 1, canbus="Drivetrain")

    if self.num == 1:
      time.sleep(0.2)  # wait for CAN bus to be ready
      supply_voltage = self.fx.get_supply_voltage().value
      print(f"Motor supply voltage: {supply_voltage:.2f} V")
      if supply_voltage < 11.5 and os.environ.get("CTR_TARGET", None) == "Hardware":
        raise Exception("Motor supply voltage is too low. Please charge the battery.")

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
    self.fx_cfg.torque_current.peak_forward_torque_current = 40  # Amperes
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
    self.cc_cfg.magnet_sensor.magnet_offset = ENCODER_MAGNET_OFFSETS[self.num // 2]
    self.cc_cfg.magnet_sensor.sensor_direction = signals.SensorDirectionValue.COUNTER_CLOCKWISE_POSITIVE

    status = self.fx.configurator.apply(self.fx_cfg)
    if not status.is_ok():
      raise Exception(f"Failed to apply TalonFX configuration: {status}")

    status = self.cc.configurator.apply(self.cc_cfg)
    if not status.is_ok():
      raise Exception(f"Failed to apply CANCoder configuration: {status}")

    self.fx.set_position(0)

  def get_position(self) -> float:  # radians (-pi, pi)
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
    self.fx_cfg.torque_current.peak_forward_torque_current = 10  # Amperes
    self.fx_cfg.torque_current.peak_reverse_torque_current = -10
    self.fx_cfg.audio.beep_on_boot = False

    for _ in range(3):
      status = self.fx.configurator.apply(self.fx_cfg)
      if status.is_ok():
        break
    if not status.is_ok():
      raise Exception(f"Failed to apply TalonFX configuration: {status}")

    self.fx.set_position(0)

  def get_velocity(self) -> Tuple[float, float]:
    return (TWO_PI * TIRE_RADIUS) * self.velocity_signal.value / DRIVE_GEAR_RATIO, self.velocity_signal.timestamp

  def set_velocity(self, velocity: float) -> None:  # m/s
    velocity = DRIVE_GEAR_RATIO * (velocity / (TWO_PI * TIRE_RADIUS))
    self.fx.set_control(self.velocity_request.with_velocity(velocity))

  def set_neutral(self):
    self.fx.set_control(self.neutral_request)


class ControlMode(Enum):
  POSITION = 1
  VELOCITY = 2


class Base:
  def __init__(self, max_vel=np.array((1.0, 1.0, 1.57)), max_accel=np.array((0.25, 0.25, 0.79))):
    self.max_vel = max_vel
    self.max_accel = max_accel
    self.C = np.array([
      [1, 0, WIDTH],
      [1, 0, -WIDTH],
      [1, 0, -WIDTH],
      [1, 0, WIDTH],
      [0, 1, LENGTH],
      [0, 1, LENGTH],
      [0, 1, -LENGTH],
      [0, 1, -LENGTH],
    ])

    self.steer_motors = [SteerMotor(i * 2 + 1) for i in range(NUM_SWERVES)]
    self.drive_motors = [DriveMotor(i * 2 + 2) for i in range(NUM_SWERVES)]

    self.status_signals = [s for m in self.steer_motors + self.drive_motors for s in m.status_signals]
    phoenix6.BaseStatusSignal.set_update_frequency_for_all(CONTROL_FREQ, self.status_signals)
    self.status_timestamp = self.status_signals[0].timestamp

    self.steer_pos = np.zeros(NUM_SWERVES)
    self.drive_vel = np.zeros(NUM_SWERVES)
    self.x = np.zeros(3) # x, y, theta

    self._command = None
    self.last_command_time = time.time()
    self.timer = FrequencyTimer(frequency=CONTROL_FREQ)

    self.control_loop_thread = threading.Thread(target=self.control_loop, daemon=True)
    self.control_loop_running = True
    self.control_loop_thread.start()

  def set_target_velocity(self, velocity: np.ndarray) -> bytes:
    self._command = {"mode": ControlMode.VELOCITY, "target": velocity}
    self.last_command_time = time.time()
    return b"ok"

  def set_target_position(self, position: np.ndarray) -> bytes:
    self._command = {"mode": ControlMode.POSITION, "target": position}
    self.last_command_time = time.time()
    return b"ok"

  def stop_control(self) -> None:
    self.control_loop_running = False
    self.control_loop_thread.join()
    self.control_loop_thread = None

  def update_state(self) -> None:
    phoenix6.BaseStatusSignal.refresh_all(self.status_signals)
    for i in range(NUM_SWERVES):
      self.steer_pos[i] = self.steer_motors[i].get_position()
      self.drive_vel[i] = self.drive_motors[i].get_velocity()

    dt = self.status_signals[0].timestamp.time - self.status_timestamp.time
    self.status_timestamp = self.status_signals[0].timestamp
    print("dt: ", dt)

    dx = self.angle_and_speed_to_vehicle_velocity(self.drive_vel, self.steer_pos)
    self.x += dx * dt
    self.x[2] = (self.x[2] + np.pi) % (2*np.pi) - np.pi # wrap to (-pi, pi)
    print(self.x)

  def angle_and_speed_to_vehicle_velocity(self, wheel_speeds:np.ndarray, wheel_angles:np.ndarray) -> np.ndarray:
    vx, vy = wheel_speeds * np.cos(wheel_angles), wheel_speeds * np.sin(wheel_angles)
    return np.linalg.lstsq(self.C, np.concatenate((vx, vy)), rcond=None)[0]

  def vehicle_velocity_to_angle_and_speed(self, u_3dof: np.ndarray, cos_error_scaling: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    wheel_velocities_directional = self.C @ u_3dof
    vx, vy = wheel_velocities_directional[:4], wheel_velocities_directional[4:]
    wheel_speeds = np.sqrt(vx**2 + vy**2)
    wheel_angles = np.arctan2(vy, vx)
    if cos_error_scaling:
      wheel_speeds *= np.cos(((wheel_angles - self.steer_pos) + np.pi) % (2 * np.pi) - np.pi)
    return wheel_speeds, wheel_angles

  def control_loop(self):
    # TODO: Set real-time scheduling policy
    while self.control_loop_running:
      with self.timer:
        self.update_state()

        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
          self._command = None

        if self._command is None:
          for s, d in zip(self.steer_motors, self.drive_motors):
            s.set_neutral()
            d.set_neutral()
        else:
          phoenix6.unmanaged.feed_enable(0.1)

          if self._command["mode"] == ControlMode.VELOCITY:
            wheel_speeds, wheel_angles = self.vehicle_velocity_to_angle_and_speed(self._command["target"])
            for i in range(NUM_SWERVES):
              self.steer_motors[i].set_position(wheel_angles[i])
              self.drive_motors[i].set_velocity(wheel_speeds[i])  # TODO: cosine error scaling velocity
          elif self._command["mode"] == ControlMode.POSITION:
            raise NotImplementedError("Position control not implemented yet")

  def get_encoder_offsets(self):
    offsets = []
    for steer_motor in self.steer_motors:
      steer_motor.cc.configurator.refresh(steer_motor.cc_cfg)
      curr_offset = steer_motor.cc_cfg.magnet_sensor.magnet_offset
      steer_motor.cc.get_absolute_position().wait_for_update(0.1)
      curr_position = steer_motor.cc.get_absolute_position().value
      offsets.append(f"{round(4096 * (curr_offset - curr_position))}.0 / 4096")
    print(f"ENCODER_MAGNET_OFFSETS = [{', '.join(offsets)}]")


if __name__ == "__main__":
  vehicle = Base()
  try:
    for _ in range(100):
      vehicle.set_target_velocity(np.array([0, 0, math.pi / 8]))
      time.sleep(0.1)
  finally:
    vehicle.stop_control()

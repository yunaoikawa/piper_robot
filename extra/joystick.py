"""
To run joystick on Jetson Nano, one needs to install xpad drivers. Then, one needs to add the user to input group and create udev rules.
"""
import os
import signal
import numpy as np
import pygame
from pygame.joystick import Joystick

import zmq
from robot.timer import FrequencyTimer
from robot.communications import send_command, CommandType, ROBOT_IP, COMMAND_PORT


pygame.init()

def apply_deadzone(arr, deadzone_size=0.05):
  return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


class GamepadTeleop:
  def __init__(self):
    self.joy = Joystick(0)  # Logitech F710
    self.max_vel = np.array([0.5, 0.5, 0.78])

    self.socket = zmq.Context().socket(zmq.REQ)
    self.socket.connect(f"tcp://{ROBOT_IP}:{COMMAND_PORT}")
    self.control_loop_running = False

    self.timer = FrequencyTimer(frequency=50)

  def run(self):
    last_enabled = False
    print('Press the "Start" button on the gamepad to start control')
    while True:
      with self.timer:
        pygame.event.pump()

        # Start control
        if not self.control_loop_running and self.joy.get_button(7):  # 7 is the "Start" button
          last_enabled = False
          self.control_loop_running = True
          print("Control started")

        # Stop control
        if self.control_loop_running and self.joy.get_button(6):  # 6 is the "Back" button
          self.control_loop_running = False
          print("Control stopped")

        if self.control_loop_running:
          # Hold down left/right bumper to enable control in local/global frame
          right_bumper = self.joy.get_button(5)
          if right_bumper:
            if not last_enabled:
              last_enabled = True

            # Compute unscaled target velocity
            vy = -self.joy.get_axis(3)
            vx = -self.joy.get_axis(4)  # Right analog stick
            w = -self.joy.get_axis(0)  # Left analog stick
            target_velocity = np.array([vx, vy, w])
            # Apply deadzone for joystick drift
            target_velocity = apply_deadzone(target_velocity)

            # Send command to robot
            target_velocity = self.max_vel * target_velocity
            print("Target velocity: ", target_velocity)

            if sum(np.abs(target_velocity)) > 0.0:
              send_command(self.socket, CommandType.SET_TARGET_VELOCITY, target_velocity.tobytes())
              _ = self.socket.recv()

          elif last_enabled:
            print("Robot disabled")
            last_enabled = False

# Handle SIGTERM
def handler(signum, frame):
  os.kill(os.getpid(), signal.SIGINT)

signal.signal(signal.SIGTERM, handler)

if __name__ == "__main__":
  teleop = GamepadTeleop()
  try:
    teleop.run()
  finally:
    teleop.socket.close()

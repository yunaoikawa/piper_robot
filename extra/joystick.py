"""
To run joystick on Jetson Nano, one needs to install xpad drivers. Then, one needs to add the user to input group and create udev rules.
"""
import numpy as np
import pygame
from pygame.joystick import Joystick

import pyarrow as pa
from dora import Node

from robot.controller import CommandType


def apply_deadzone(arr, deadzone_size=0.05):
  return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


def main():
  pygame.init()
  joy = Joystick(0)  # Logitech F710
  max_vels = [np.array([0.5, 0.5, 0.78]), np.array([0.25, 0.25, 0.78]), np.array([0.75, 0.75, 0.78])]
  max_vel_setting = 0
  control_loop_running = False
  last_enabled = False

  node = Node("gamepad_teleop")

  for event in node:
    if event['type'] == "INPUT":
      if event['id'] == 'tick':
        pygame.event.pump()
        if not control_loop_running and joy.get_button(7):
          last_enabled = False
          control_loop_running = True
          print("Control started")

        if control_loop_running and joy.get_button(6):
          control_loop_running = False
          print("Control stopped")

        if control_loop_running:
          left_bumper = joy.get_button(4)
          right_bumper = joy.get_button(5)
          if left_bumper: max_vel_setting = (max_vel_setting + 1) % 3
          if right_bumper:
            if not last_enabled: last_enabled = True
            vy = -joy.get_axis(3)
            vx = -joy.get_axis(4)  # Right analog stick
            w = -joy.get_axis(0)  # Left analog stick
            target_velocity = np.array([vx, vy, w])
            target_velocity = apply_deadzone(target_velocity)

            target_velocity = max_vels[max_vel_setting] * target_velocity
            # print("Target velocity: ", target_velocity)
            if sum(np.abs(target_velocity)) > 0.0:
              node.send_output("command", pa.array(target_velocity.ravel()), metadata={"command_type": CommandType.BASE_VELOCITY.value})
          elif last_enabled:
            print("Robot disabled")
            last_enabled = False


if __name__ == "__main__":
  main()

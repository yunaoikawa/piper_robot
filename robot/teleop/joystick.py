"""
To run joystick on Jetson Nano, one needs to install xpad drivers. Then, one needs to add the user to input group
and create udev rules.
"""

import time
import numpy as np
import pygame
from pygame.joystick import Joystick
from loop_rate_limiters import RateLimiter

from robot.base import Base

XBOX_CONTROLLER_MAP = {
    "start": 7,
    "back": 6,
    "l1": 4,
    "r1": 5,
    "left_horizontal_axis": 0,
    "left_vertical_axis": 1,
    "right_horizontal_axis": 3,
    "right_vertical_axis": 4,
}

PS4_CONTROLLER_MAP = {
    "start": 3,
    "back": 0,  # square button
    "l1": 4,
    "r1": 5,
    "left_horizontal_axis": 0,
    "right_horizontal_axis": 3,
    "right_vertical_axis": 4,
    "pad_y": 7,
}

controller_map = XBOX_CONTROLLER_MAP


def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


class JoystickNode:
    def __init__(self):
        self.joystick = Joystick(0)
        self.max_vels = [np.array([0.5, 0.5, 1.57]), np.array([0.25, 0.25, 1.57]), np.array([0.75, 0.75, 1.57])]
        self.max_vel_setting = 0
        self.vel_alpha = 0.9
        self.control_loop_running = False
        self.base = Base()
        self.base.start_control()
        self.base.home_lift()

    def control_loop(self):
        rate = RateLimiter(60, name="joystick")
        last_target_velocity = np.array([0.0, 0.0, 0.0])
        while True:
            pygame.event.pump()
            if not self.control_loop_running and self.joystick.get_button(controller_map["start"]):
                self.control_loop_running = True
                print("Control started")

            if self.control_loop_running and self.joystick.get_button(controller_map["back"]):
                self.control_loop_running = False
                print("Control stopped")

            if self.control_loop_running:
                left_bumper = self.joystick.get_button(controller_map["l1"])
                if left_bumper:
                    self.max_vel_setting = (self.max_vel_setting + 1) % 3
                    print("Max velocity setting: ", self.max_vel_setting)
                    time.sleep(0.1)

                vy = -self.joystick.get_axis(controller_map["right_horizontal_axis"])  # Right analog stick
                vx = -self.joystick.get_axis(controller_map["right_vertical_axis"])  # Right analog stick
                w = -self.joystick.get_axis(controller_map["left_horizontal_axis"])  # Left analog stick
                target_velocity = np.array([vx, vy, w])
                target_velocity = apply_deadzone(target_velocity)

                target_velocity = self.max_vels[self.max_vel_setting] * target_velocity
                target_velocity = (1 - self.vel_alpha) * target_velocity + self.vel_alpha * last_target_velocity
                last_target_velocity = target_velocity

                lift_target = self.joystick.get_hat(0)[1]  # pad Y axis
                if sum(np.abs(target_velocity)) > 1e-2:
                    self.base.set_target_base_velocity(target_velocity)
                if lift_target != 0:
                    self.base.set_target_lift(np.array([0.39 if lift_target > 0 else 0.0]))

            rate.sleep()


def main():
    pygame.init()
    node = JoystickNode()
    node.control_loop()


if __name__ == "__main__":
    main()

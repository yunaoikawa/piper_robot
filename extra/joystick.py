"""
To run joystick on Jetson Nano, one needs to install xpad drivers. Then, one needs to add the user to input group
and create udev rules.
"""

import time
import numpy as np
import zmq
import math
import pygame
from pygame.joystick import Joystick

from robot.network import Publisher, COMMAND_PORT
from robot.network.timer import FrequencyTimer
from robot.network.msgs import Command, CommandType, LiftCommand

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
    "back": 0,    # square button
    "l1": 4,
    "r1": 5,
    "left_horizontal_axis": 0,
    "right_horizontal_axis": 3,
    "right_vertical_axis": 4,
    "pad_y": 7,
}

controller_map = PS4_CONTROLLER_MAP

def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


def main():
    pygame.init()
    joy = Joystick(0)  # Logitech F710
    max_vels = [np.array([0.5, 0.5, 0.78]), np.array([0.25, 0.25, 0.78]), np.array([0.75, 0.75, 0.78])]
    max_vel_setting = 0
    control_loop_running = False
    last_enabled = False

    ctx = zmq.Context()
    pub = Publisher(ctx, COMMAND_PORT)
    timer = FrequencyTimer("Joystick", 60)
    while True:
        with timer:
            pygame.event.pump()
            if not control_loop_running and joy.get_button(controller_map["start"]):
                last_enabled = False
                control_loop_running = True
                print("Control started")

            if control_loop_running and joy.get_button(controller_map["back"]):
                control_loop_running = False
                print("Control stopped")

            if control_loop_running:
                left_bumper = joy.get_button(controller_map["l1"])
                right_bumper = joy.get_button(controller_map["r1"])
                if left_bumper:
                    max_vel_setting = (max_vel_setting + 1) % 3
                    print("Max velocity setting: ", max_vel_setting)
                    time.sleep(0.1)
                if right_bumper:
                    if not last_enabled:
                        last_enabled = True
                    vy = -joy.get_axis(controller_map["right_horizontal_axis"])  # Right analog stick
                    vx = -joy.get_axis(controller_map["right_vertical_axis"])  # Right analog stick
                    w = -joy.get_axis(controller_map["left_horizontal_axis"])  # Left analog stick
                    target_velocity = np.array([vx, vy, w])
                    target_velocity = apply_deadzone(target_velocity)

                    target_velocity = max_vels[max_vel_setting] * target_velocity
                    # print("Target velocity: ", target_velocity)
                    if sum(np.abs(target_velocity)) > 0.0:
                        pub.publish(
                            "/command",
                            Command(
                                timestamp=time.perf_counter_ns(),
                                type=CommandType.BASE_VELOCITY,
                                target=target_velocity.ravel(),
                            ),
                        )
                    lift_target = joy.get_hat(0)[1] # pad Y axis
                    if lift_target != 0:
                        lift_target = 0.1 if lift_target > 0 else -0.1
                        pub.publish(
                            "/lift_command",
                            LiftCommand(timestamp=time.perf_counter_ns(), target=lift_target),
                        )
                elif last_enabled:
                    print("Robot disabled")
                    last_enabled = False


if __name__ == "__main__":
    main()

import os
import signal
import time
import numpy as np
import pygame
from pygame.joystick import Joystick

from base_server import BaseManager
from constants import BASE_RPC_HOST, BASE_RPC_PORT
from robot_secrets import RPC_AUTH_KEY

pygame.init()

def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))

class GamepadTeleop:
    def __init__(self):
        self.joy = Joystick(0)  # Logitech F710
        manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTH_KEY)
        manager.connect()
        self.max_vel = np.array([0.3, 0.3, 0.78])
        self.base = manager.Base(max_vel=self.max_vel)
        self.control_loop_running = False

    def run(self):
        last_enabled = False
        frame = None
        print('Press the "Start" button on the gamepad to start control')
        while True:
            pygame.event.pump()

            # Start control
            if not self.control_loop_running and self.joy.get_button(7):  # 7 is the "Start" button
                self.base.reset()
                last_enabled = False
                self.control_loop_running = True
                print('Control started')

            # Stop control
            if self.control_loop_running and self.joy.get_button(6):  # 6 is the "Back" button
                self.base.close()
                print('Control stopped')

            if self.control_loop_running:
                # Hold down left/right bumper to enable control in local/global frame
                right_bumper = self.joy.get_button(5)
                if right_bumper:
                    if not last_enabled:
                        print(f'Robot enabled ({frame} frame)')
                        last_enabled = True

                    # Compute unscaled target velocity
                    vy = -self.joy.get_axis(3)
                    vx = -self.joy.get_axis(4)  # Right analog stick
                    w = self.joy.get_axis(0) # Left analog stick
                    target_velocity = np.array([vx, vy, w])
                    # Apply deadzone for joystick drift
                    target_velocity = apply_deadzone(target_velocity)

                    # Send command to robot
                    target_velocity = self.max_vel * target_velocity
                    print("Target velocity: ", target_velocity)
                    self.base.execute_action({"v": target_velocity})
                    # self.vehicle.set_target_position(self.vehicle.x + 1.5 * target_velocity)

                elif last_enabled:
                    print('Robot disabled')
                    last_enabled = False

            time.sleep(0.01)

# Handle SIGTERM
def handler(signum, frame):
    os.kill(os.getpid(), signal.SIGINT)
signal.signal(signal.SIGTERM, handler)

if __name__ == '__main__':
    teleop = GamepadTeleop()
    try:
        teleop.run()
    finally:
        if teleop.control_loop_running:
            teleop.base.close()


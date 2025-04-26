from robot.gripper.DM_CAN import Motor, MotorControl, DM_variable, Control_Type, DM_Motor_Type
import serial
from loop_rate_limiters import RateLimiter

import numpy as np
import time


class TrapezoidalProfile:
    def __init__(self, max_vel, max_accel):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.start_pos = 0.0
        self.target_pos = 0.0
        self.start_time = 0.0
        self.t_accel = 0.0
        self.t_flat = 0.0

    def plan_trajectory(self, current_pos, target_pos):
        self.start_pos = current_pos
        self.target_pos = target_pos
        distance = abs(target_pos - current_pos)

        # Calculate time needed to reach max velocity
        self.t_accel = self.max_vel / self.max_accel

        # Distance covered during acceleration/deceleration
        d_accel = 0.5 * self.max_accel * self.t_accel**2

        if 2 * d_accel > distance:
            # Will not reach max velocity
            self.t_accel = (distance / self.max_accel) ** 0.5
            self.t_flat = 0
            # max_vel = self.max_accel * self.t_accel
        else:
            # Will reach max velocity
            self.t_flat = (distance - 2 * d_accel) / self.max_vel

        self.start_time = time.time()
        return 2*self.t_accel + self.t_flat # Total time for the trajectory

    def get_target(self):
        t = time.time() - self.start_time
        direction = 1 if self.target_pos > self.start_pos else -1

        if t < self.t_accel:
            # Acceleration phase
            return self.start_pos + direction * (0.5 * self.max_accel * t * t)

        elif t < self.t_accel + self.t_flat:
            # Constant velocity phase
            return self.start_pos + direction * (
                0.5 * self.max_accel * self.t_accel**2 + self.max_vel * (t - self.t_accel)
            )

        elif t < self.t_accel + self.t_flat + self.t_accel:
            # Deceleration phase

            return self.target_pos - direction * (0.5 * self.max_accel * (2*self.t_accel + self.t_flat - t)**2)

        else:
            print("Trajectory completed")
            return self.target_pos

Motor1 = Motor(DM_Motor_Type.DM4310, 0x01, 0x11)
serial_device = serial.Serial("/dev/ttyACM0", 921600, timeout=1)
MotorControl1 = MotorControl(serial_device)
MotorControl1.addMotor(Motor1)

MotorControl1.enable(Motor1)

if MotorControl1.switchControlMode(Motor1, Control_Type.MIT):
    print("switch MIT success")

# MotorControl1.disable(Motor1)

profile = TrapezoidalProfile(max_vel=3.0, max_accel=1.0)
rate = RateLimiter(200)  # 1000 Hz
print(f"current position: {Motor1.getPosition():.3f} rad")
pos = float(input("position:"))

print(f"Moving to {pos:.3f} rad")

positions = []
targets = []
ts = []

try:
    duration = profile.plan_trajectory(Motor1.getPosition(), pos)
    print(f"Trajectory planned. Duration: {duration:.3f} seconds")
    i = 0
    while True:
        current_target = profile.get_target()
        MotorControl1.controlMIT(Motor1, kp=10, kd=0.1, q=current_target, dq=0, tau=0)
        if i % 5 == 0:
            positions.append(Motor1.getPosition())
            targets.append(current_target)
            ts.append(time.time() - profile.start_time)
            print(f"Target: {current_target:.3f} Current pos: {Motor1.getPosition():.3f}")
            i = 0

        i += 1
        rate.sleep()
finally:
    import matplotlib.pyplot as plt
    plt.plot(ts, positions)
    plt.plot(ts, targets)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.savefig("trajectory.png")

    MotorControl1.disable(Motor1)
    print("Motor disabled")

from robot.gripper.DM_CAN import Motor, MotorControl, DM_variable, Control_Type, DM_Motor_Type
import serial
from loop_rate_limiters import RateLimiter

import numpy as np
import time


TWO_PI = 2 * np.pi

Motor1 = Motor(DM_Motor_Type.DM4310, 0x01, 0x11)
serial_device = serial.Serial("/dev/ttyACM0", 921600, timeout=1)
MotorControl1 = MotorControl(serial_device)
MotorControl1.addMotor(Motor1)

MotorControl1.enable(Motor1)

if MotorControl1.switchControlMode(Motor1, Control_Type.VEL):
    print("switch VEL success")

if MotorControl1.change_motor_param(Motor1, DM_variable.TMAX, 10.0):
    print("TMAX set to 10.0")

# MotorControl1.disable(Motor1)

MotorControl1.set_zero_position(Motor1)
print(f"current position: {Motor1.getPosition():.3f} rad")
vel = float(input("velocity:"))

positions = []
velocities = []
# targets = []
torques = []
ts = []

try:
    rate = RateLimiter(200)  # 1000 Hz
    while True:
        start_time = time.time()
        for i in range(200 * 3):
            # current_target = profile.get_target()
            # MotorControl1.controlMIT(Motor1, kp=10, kd=0.1, q=current_target, dq=0, tau=0)
            MotorControl1.control_Vel(Motor1, Vel_desired=vel)
            if i % 5 == 0:
                positions.append(Motor1.getPosition())
                velocities.append(Motor1.getVelocity())
                torques.append(Motor1.getTorque())
                # targets.append(current_target)
                ts.append(time.time() - start_time)
                # print(f"Target: {current_target:.3f} Current pos: {Motor1.getPosition():.3f}")

            # i += 1
            rate.sleep()
        vel_s = input("velocity: ")
        if vel_s == "exit":
            break
        else:
            vel = float(vel_s)
finally:
    import matplotlib.pyplot as plt
    plt.plot(ts, positions)
    plt.plot(ts, velocities)
    plt.plot(ts, torques)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.legend(["Position", "Velocity", "Torque"])
    plt.savefig("trajectory.png")

    MotorControl1.disable(Motor1)
    print("Motor disabled")

from DM_SocketCAN import Motor, MotorControl, Control_Type, DM_Motor_Type
import numpy as np
import time

TWO_PI = 2 * np.pi
I_LIMIT = 200
HOMING_VEL = TWO_PI * 100
VEL_DES = 2 * TWO_PI * 100
GRIPPER_MAX_WIDTH = 22  # rad

Motor1 = Motor(DM_Motor_Type.DM4310, 0x01, 0x11)
MotorControl1 = MotorControl(channel="can_left")
MotorControl1.addMotor(Motor1)

MotorControl1.enable(Motor1)

if MotorControl1.switchControlMode(Motor1, Control_Type.Torque_Pos):
    print("switch Torque_Pos success")


def home():
    MotorControl1.set_zero_position(Motor1)
    MotorControl1.control_pos_force(
        Motor1, Pos_des=GRIPPER_MAX_WIDTH + 1, Vel_des=HOMING_VEL, i_des=I_LIMIT
    )  # try to go past the limit
    time.sleep(10)
    MotorControl1.set_zero_position(Motor1)
    print("Motor homed")

home()
time.sleep(0.5)
print(f"current position: {Motor1.getPosition():.3f} rad")
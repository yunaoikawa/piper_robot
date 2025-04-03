import math
from robot.gripper.DM_SocketCAN import Motor, MotorControl, DM_variable, Control_Type, DM_Motor_Type
import time


Motor1=Motor(DM_Motor_Type.DM4310,0x18,0x19)
MotorControl1=MotorControl('can0')
MotorControl1.addMotor(Motor1)

if MotorControl1.change_motor_param(Motor1, DM_variable.ESC_ID, 0x47D):
    print("change ESC_ID success")

if MotorControl1.change_motor_param(Motor1, DM_variable.MST_ID, 0x47E):
    print("change MST_ID success")

exit(0)
# if MotorControl1.switchControlMode(Motor1,Control_Type.MIT):
#     print("switch MIT success")
# print("sub_ver:",MotorControl1.read_motor_param(Motor1,DM_variable.sub_ver))
# print("Gr:",MotorControl1.read_motor_param(Motor1,DM_variable.Gr))

# print("PMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.PMAX))
# print("MST_ID:",MotorControl1.read_motor_param(Motor1,DM_variable.MST_ID))
# print("VMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.VMAX))
# print("TMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.TMAX))

MotorControl1.save_motor_param(Motor1)
input("Press Enter to continue...")

MotorControl1.enable(Motor1)
i=0
while i<10000:
    q=math.sin(time.time())
    i=i+1
    # MotorControl1.control_pos_force(Motor1, 10, 1000,100)
    # MotorControl1.control_Vel(Motor1, q*5)
    # MotorControl1.control_Pos_Vel(Motor1,q*8,30)
    # print("Motor1:","POS:",Motor1.getPosition(),"VEL:",Motor1.getVelocity(),"TORQUE:",Motor1.getTorque())
    MotorControl1.controlMIT(Motor1, 35, 0.1, 8*q, 0, 0)

    # print("Motor2:","POS:",Motor2.getPosition(),"VEL:",Motor2.getVelocity(),"TORQUE:",Motor2.getTorque())
    # print(Motor1.getTorque())
    # print(Motor2.getTorque())
    time.sleep(0.001)
    # MotorControl1.control(Motor3, 50, 0.3, q, 0, 0)

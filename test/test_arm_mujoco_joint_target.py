from robot.rpc import RPCClient
import numpy as np
import time


def main():
    arm_mujoco = RPCClient("localhost", 8081)

    input("Press Enter to initialize ArmMujoco")
    arm_mujoco.start_control()

    input("Press Enter to start pid_tuner")

    start_joint_targets = np.zeros(6)
    end_joint_targets = np.array([0.2,0.2,-0.2,0.3,-0.2,0.5])

    waypoints = np.linspace(start_joint_targets, end_joint_targets, 20)

    while True:
        for waypoint in waypoints:
            arm_mujoco.set_joint_target(waypoint, lift_target=0.0)
            time.sleep(0.05)

        input("Go back to start")
        for waypoint in waypoints[::-1]:
            arm_mujoco.set_joint_target(waypoint, lift_target=0.0)
            time.sleep(0.05)

        input("Go forward")



if __name__ == "__main__":
    main()

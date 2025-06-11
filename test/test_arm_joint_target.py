from robot.rpc import RPCClient
import numpy as np
import time


def main():
    cone_e = RPCClient("localhost", 8081)

    input("Press Enter to start")

    start_joint_targets = np.zeros(6)
    end_joint_targets = np.array([0.2,0.2,-0.2,0.3,-0.2,0.5])

    waypoints = np.linspace(start_joint_targets, end_joint_targets, 20)

    while True:
        for waypoint in waypoints:
            cone_e.set_left_joint_target(waypoint, preview_time=0.05)
            time.sleep(0.05)

        input("Go back to start")
        for waypoint in waypoints[::-1]:
            cone_e.set_left_joint_target(waypoint, preview_time=0.05)
            time.sleep(0.05)

        input("Go forward")


if __name__ == "__main__":
    main()

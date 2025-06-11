from robot.rpc import RPCClient
import numpy as np
import time


def main():
    cone_e = RPCClient("localhost", 8081)

    input("Press Enter to start")

    start_joint_targets = np.zeros(6)
    end_joint_targets = np.array([0.2,0.2,-0.2,0.3,-0.2,0.5])

    # waypoints = np.linspace(start_joint_targets, end_joint_targets, 20)

    # Min jerk interpolation
    def min_jerk(t):
        # Normalized time from 0 to 1
        tau = t
        # Min jerk basis functions
        h1 = tau**3 * (10 - 15*tau + 6*tau**2)
        return h1

    # Generate smoother trajectory using min jerk
    t = np.linspace(0, 1, 20)
    min_jerk_weights = np.array([min_jerk(ti) for ti in t])
    waypoints = start_joint_targets[None,:] + min_jerk_weights[:,None] * (end_joint_targets - start_joint_targets)[None,:]

    while True:
        for waypoint in waypoints:
            cone_e.set_left_joint_target(waypoint, preview_time=0)
            time.sleep(0.05)

        input("Go back to start")
        for waypoint in waypoints[::-1]:
            cone_e.set_left_joint_target(waypoint, preview_time=0)
            time.sleep(0.05)

        input("Go forward")


if __name__ == "__main__":
    main()

from robot.rpc import RPCClient
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
import matplotlib.pyplot as plt


def main():
    arm_mujoco = RPCClient("localhost", 8081)

    input("Press Enter to initialize ArmMujoco")
    arm_mujoco.start_control()

    input("Press Enter to start sine wave")

    t = np.linspace(0.0, 50.0, 501)
    rate_limiter = RateLimiter(10)

    traj_duration = 50.0
    period = 0.1  # m

    y_traj = 0.2 * t / traj_duration
    z_traj = 0.1 * np.sin(2 * np.pi * y_traj / period)

    start_pose = arm_mujoco.get_ee_pose()

    xyz_history = []

    for i in range(len(t)):
        y = y_traj[i]
        z = z_traj[i]
        target_pose = mink.SE3.from_rotation_and_translation(
            rotation=start_pose.rotation(), translation=start_pose.translation() + np.array([0.0, -y, z])
        )
        arm_mujoco.set_ee_target(target_pose)
        xyz = arm_mujoco.get_ee_pose().translation() - start_pose.translation()
        xyz_history.append(xyz)
        rate_limiter.sleep()

    xyz_history = np.array(xyz_history)
    plt.plot(-xyz_history[:, 1], xyz_history[:, 2])
    plt.plot(y_traj, z_traj)
    plt.legend(["actual", "desired"])
    plt.show()

    print("Done")


if __name__ == "__main__":
    main()

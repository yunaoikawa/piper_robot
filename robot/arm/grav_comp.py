import numpy as np
import matplotlib.pyplot as plt
import pathlib
from loop_rate_limiters import RateLimiter
from piper_control import piper_control
import pinocchio as pin


def main():
    piper = piper_control.PiperControl("can_left")
    input("Press Enter to continue. The arm will drop first..")
    # piper.reset()
    # piper.set_joint_positions([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # piper.set_joint_positions([0, 1.5857, -1.4835, 0, 0, 0])

    _HERE = pathlib.Path(__file__).parent
    urdf_file = _HERE / "mujoco" / "urdf" / "piper_no_gripper_description.xml"
    mesh_dir = _HERE / "mujoco" / "assets"

    robot = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[mesh_dir], root_joint=None)
    # robot = robot.buildReducedRobot(
    #     list_of_joints_to_lock=["joint7", "joint8"],
    #     reference_configuration=np.array([0] * robot.model.nq),
    # )

    rate_limiter = RateLimiter(30)
    # Initialize filter parameters
    alpha = 0.2  # Filter coefficient (0 < alpha < 1)
    tau_filtered = None  # Initial filtered value

    important_joints = [1, 2, 4]

    while True:
        q = np.array(piper.get_joint_positions())
        v = np.zeros(robot.model.nv)  # Zero velocity
        a = np.zeros(robot.model.nv)  # Zero acceleration
        # data.qpos = q
        # mujoco.mj_step(model, data)

        # Get joint efforts and gravity compensation torques
        tau = np.array(piper.get_joint_efforts())

        # Apply low-pass filter to tau
        if tau_filtered is None:
            tau_filtered = tau
        else:
            tau_filtered = alpha * tau + (1 - alpha) * tau_filtered
        # tau_filtered = tau

        grav_comp = np.abs(pin.rnea(robot.model, robot.data, q, v, a))

        # Clear plot and update data
        plt.clf()
        # plt.bar(range(6), tau_filtered, alpha=0.5, label='Measured Torque')
        # plt.bar(range(6), grav_comp, alpha=0.5, label='Gravity Compensation')
        plt.bar(
            important_joints,
            grav_comp[important_joints] / tau_filtered[important_joints],
            alpha=1.0,
            label="Ratio",
        )

        # Configure plot
        plt.xlabel("Joint")
        plt.ylabel("Gravity Compensation / Measured Torque")
        plt.title("Joint Efforts vs Gravity Compensation")
        plt.legend()
        plt.grid(True)
        # plt.ylim(-6, 6)  # Adjust y limits as needed

        # Draw plot
        plt.draw()
        plt.pause(0.001)

        rate_limiter.sleep()


if __name__ == "__main__":
    main()

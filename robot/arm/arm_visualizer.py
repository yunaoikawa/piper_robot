import os
import click

import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import FrameTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer


def print_joint_info(robot_model):
    print("\nJoint Information:")
    print("------------------")
    for joint_id in range(robot_model.njoints):
        joint_name = robot_model.names[joint_id]
        print(f"Joint ID: {joint_id}, Name: {joint_name}")


def print_frame_info(robot_model):
    print("\nFrame Information:")
    print("------------------")
    for frame_id in range(robot_model.nframes):
        frame_name = robot_model.frames[frame_id].name
        print(f"Frame ID: {frame_id}, Name: {frame_name}")


# Add this after creating the robot:


@click.command()
@click.argument("urdf_name", type=str, default="bimanual_humanoid")
def main(urdf_name):
    urdf_file = os.path.join(os.path.dirname(__file__), "models", urdf_name + ".urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), "models", "meshes")

    # Build the robot with the package directories
    robot = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[mesh_dir], root_joint=None)

    print(f"URDF description successfully loaded in {robot}")

    viz = start_meshcat_visualizer(robot)

    meshcat_shapes.frame(viz.viewer["lift"], opacity=1.0)
    meshcat_shapes.frame(viz.viewer["lift_target"], opacity=0.5)
    meshcat_shapes.frame(viz.viewer["ee_left"], opacity=1.0)
    meshcat_shapes.frame(viz.viewer["ee_left_target"], opacity=0.5)

    lift_task = FrameTask(
        "lift_top",
        position_cost=0.0,
        orientation_cost=0.0,
    )

    left_ee_task = FrameTask(
        "ee_left",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    tasks = [lift_task, left_ee_task]

    q_ref = custom_configuration_vector(
        robot,
        lift_joint=0.15,
        left_joint2=3.14,
        left_joint3=-2.967,
        right_joint2=3.14,
        right_joint3=-2.967,
    )

    configuration = pink.Configuration(robot.model, robot.data, q_ref)

    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    input("Press Enter to continue...")

    # Select the solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=200, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    while True:
        # lift_target = lift_task.transform_target_to_world
        # lift_target.translation[2] = 0.3 + 0.15 * np.sin(2 * np.pi * 0.5 * t)  # Oscillates between 0 and 0.3 at 0.5 Hz

        left_ee_target = left_ee_task.transform_target_to_world
        left_ee_target.translation[1] = 0.4
        left_ee_target.translation[2] = 0.3 + 0.15 * 1# np.sin(2 * np.pi * 0.5 * t)  # Match the lift target's height

        # viz.viewer["lift_target"].set_transform(lift_target.np)
        viz.viewer["lift"].set_transform(configuration.get_transform_frame_to_world(lift_task.frame).np)

        viz.viewer["ee_left_target"].set_transform(left_ee_target.np)
        viz.viewer["ee_left"].set_transform(configuration.get_transform_frame_to_world(left_ee_task.frame).np)

        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        viz.display(configuration.q)
        rate.sleep()
        t += dt


if __name__ == "__main__":
    main()

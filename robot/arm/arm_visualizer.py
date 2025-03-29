import os
import click
import threading
from queue import Queue, Empty
from typing import Tuple

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


def input_handler(queue: Queue[Tuple[float, float, float]]) -> None:
    print("\nEnter target positions in format: x y z (e.g., 0.3 0.0 0.3)")
    print("Press Ctrl+C to exit")
    while True:
        try:
            user_input = input("Enter new target position: ")
            x, y, z = map(float, user_input.split())
            queue.put((x, y, z))
        except ValueError:
            print("Invalid input. Please enter three numbers separated by spaces.")
        except KeyboardInterrupt:
            break


@click.command()
@click.argument("urdf_name", type=str, default="bimanual_humanoid")
def main(urdf_name):
    urdf_file = os.path.join(os.path.dirname(__file__), "models", urdf_name + ".urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), "models", "meshes")

    # Build the robot with the package directories
    robot = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[mesh_dir], root_joint=None)

    print(f"URDF description successfully loaded in {robot}")

    viz = start_meshcat_visualizer(robot)

    meshcat_shapes.frame(viz.viewer["ee_left"], opacity=1.0)
    meshcat_shapes.frame(viz.viewer["ee_left_target"], opacity=0.5)

    left_ee_task = FrameTask(
        "ee_left",
        position_cost=1.0,
        orientation_cost=0.5,
    )

    tasks = [left_ee_task]

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

    # Select the solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=200, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    # Set up input handling
    input_queue: Queue[Tuple[float, float, float]] = Queue()
    input_thread = threading.Thread(target=input_handler, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

    try:
        while True:
            # Check for new target positions
            try:
                while not input_queue.empty():
                    x, y, z = input_queue.get_nowait()
                    left_ee_target = left_ee_task.transform_target_to_world
                    left_ee_target.translation = np.array([x, y, z])
                    print(f"\nNew target position set: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            except Empty:
                pass

            left_ee_target = left_ee_task.transform_target_to_world
            viz.viewer["ee_left_target"].set_transform(left_ee_target.np)
            viz.viewer["ee_left"].set_transform(configuration.get_transform_frame_to_world(left_ee_task.frame).np)

            velocity = solve_ik(configuration, tasks, dt, solver=solver)
            configuration.integrate_inplace(velocity, dt)

            viz.display(configuration.q)
            rate.sleep()
            t += dt

    except KeyboardInterrupt:
        print("\nExiting...")
        input_thread.join(timeout=0.1)


if __name__ == "__main__":
    main()

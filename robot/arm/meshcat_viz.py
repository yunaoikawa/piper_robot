import os
import click
import threading
from queue import Queue, Empty
from typing import Tuple
import time
import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

from robot.arm.tools import create_transformation_matrix

import pink
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



@click.command()
@click.argument("urdf_name", type=str, default="piper-left")
def main(urdf_name):
    urdf_file = os.path.join(os.path.dirname(__file__), "models", urdf_name + ".urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), "models", "meshes")

    # Build the robot with the package directories
    robot = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[mesh_dir], root_joint=None)

    print(f"URDF description successfully loaded in {robot}")

    viz = start_meshcat_visualizer(robot)

    meshcat_shapes.frame(viz.viewer["ee_left"], opacity=1.0)
    meshcat_shapes.frame(viz.viewer["ee_left_target"], opacity=0.5)

    q_ref = custom_configuration_vector(
        robot,
        joint6=-0.7
    )

    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    viz.display(configuration.q)

    while True:
        time.sleep(1)



if __name__ == "__main__":
    main()

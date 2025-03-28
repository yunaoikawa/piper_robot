import os
import numpy as np
import time
import click

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

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
    urdf_file = os.path.join(os.path.dirname(__file__), "models", urdf_name+".urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), "models", "meshes")

    # Build the robot with the package directories
    robot = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[mesh_dir])

    lift_joint_id = robot.model.getJointId("lift_joint")-1
    left_joint2_id = robot.model.getJointId("left_joint2")-1
    left_joint3_id = robot.model.getJointId("left_joint3")-1
    right_joint2_id = robot.model.getJointId("right_joint2")-1
    right_joint3_id = robot.model.getJointId("right_joint3")-1

    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel("pinocchio")

    neutral_config = pin.neutral(robot.model)
    neutral_config[lift_joint_id] = 0.15
    neutral_config[left_joint2_id] = 3.14
    neutral_config[left_joint3_id] = -2.967
    neutral_config[right_joint2_id] = 3.14
    neutral_config[right_joint3_id] = -2.967
    viz.display(neutral_config)

    while True:
        time.sleep(1)

def main_piper():
    urdf_file = os.path.join(os.path.dirname(__file__), "models", "piper-right.urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), "models", "meshes")

    # Build the robot with the package directories
    robot = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[mesh_dir])
    reduced_robot = robot.buildReducedRobot(
        list_of_joints_to_lock=["joint7", "joint8"],
        reference_configuration=np.array([0] * robot.model.nq),
    )
    print_frame_info(reduced_robot.model)

    viz = MeshcatVisualizer(reduced_robot.model, reduced_robot.collision_model, reduced_robot.visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel("pinocchio")

    neutral_config = pin.neutral(reduced_robot.model)
    viz.display(neutral_config)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
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
@click.argument("urdf_name", type=str, default="bimanual")
def main(urdf_name):
    urdf_file = os.path.join(os.path.dirname(__file__), "models", urdf_name+".urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), "models", "meshes")

    # Build the robot with the package directories
    robot = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[mesh_dir])
    reduced_robot = robot.buildReducedRobot(
        list_of_joints_to_lock=["left_joint7", "left_joint8", "right_joint7", "right_joint8"],
        reference_configuration=np.array([0] * robot.model.nq),
    )

    lift_middle_joint_id = reduced_robot.model.getJointId("lift_middle_joint")-1
    lift_top_joint_id = reduced_robot.model.getJointId("lift_top_joint")-1

    viz = MeshcatVisualizer(reduced_robot.model, reduced_robot.collision_model, reduced_robot.visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel("pinocchio")

    neutral_config = pin.neutral(reduced_robot.model)
    neutral_config[lift_middle_joint_id] = 0.1
    neutral_config[lift_top_joint_id] = 0.1
    viz.display(neutral_config)

    # Display frame for torso_link with a visible axis length and width
    # torso_frame_id = reduced_robot.model.getFrameId("torso_joint")-1
    # print(f"Torso frame ID: {torso_frame_id}")
    # viz.displayFrames(True, frame_ids=[torso_frame_id], axis_length=0.2, axis_width=5)

    while True:
        time.sleep(1)
if __name__ == "__main__":
    main()
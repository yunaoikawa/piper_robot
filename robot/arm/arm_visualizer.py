import os
import numpy as np
import time
import click

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

@click.command()
@click.argument("urdf_name", type=str, default="bimanual")
def main(urdf_name):
    urdf_file = os.path.join(os.path.dirname(__file__), "models", urdf_name+".urdf")
    robot = pin.RobotWrapper.BuildFromURDF(urdf_file)

    reduced_robot = robot.buildReducedRobot(
        list_of_joints_to_lock=["left_joint7", "left_joint8", "right_joint7", "right_joint8"],
        reference_configuration=np.array([0] * robot.model.nq),
    )

    viz = MeshcatVisualizer(reduced_robot.model, reduced_robot.collision_model, reduced_robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel("pinocchio")
    viz.display(pin.neutral(reduced_robot.model))

    # Display frame for torso_link with a visible axis length and width
    torso_frame_id = reduced_robot.model.getFrameId("torso_link")
    print(f"Torso frame ID: {torso_frame_id}")
    viz.displayFrames(True, frame_ids=[torso_frame_id], axis_length=0.2, axis_width=5)

    while True:
        time.sleep(1)
if __name__ == "__main__":
    main()
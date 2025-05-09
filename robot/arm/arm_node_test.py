from robot.arm.arm_node import ArmNode
from pathlib import Path
import numpy as np
import mink

if __name__ == "__main__":
    _HERE = Path(__file__).parent
    arm_node = ArmNode(
        can_port="can_left",
        mjcf_path=(_HERE / "mujoco/scene_piper.xml").as_posix(),
        urdf_path=(_HERE / "urdf/piper_description_left.xml").as_posix(),
        solver_dt=0.01,
    )

    X_EE = arm_node.get_ee_pose()
    print("Current pose:", X_EE)

    # Create initial target pose at origin
    # target = mink.SE3.from_translation(np.array([0.0, 0.0, 0.0]))

    X_EE_EE_forward = mink.SE3.from_translation(np.array([0.00, 0.1, 0.0]))
    X_EE_EE_backward = mink.SE3.from_translation(np.array([-0.00, -0.1, 0.0]))

    # Apply the translation
    X_EE_forward = X_EE.multiply(X_EE_EE_forward)
    print(X_EE_forward)
    input("Press Enter to go forward...")
    arm_node.set_ee_target(X_EE_forward, gripper_target=-22.0, preview_time=1.5)
    input("Press Enter to get current pose...")
    X_EE_forward_current = arm_node.get_ee_pose()
    print("Current pose:", X_EE_forward_current)

    X_EE_backward = X_EE.multiply(X_EE_EE_backward)
    print(X_EE_backward)
    input("Press Enter to go backward...")
    arm_node.set_ee_target(X_EE_backward, gripper_target=-22.0, preview_time=1.5)
    input("Press Enter to get current pose...")
    X_EE_backward_current = arm_node.get_ee_pose()
    print("Current pose:", X_EE_backward_current)

    input("Press Enter to exit...")

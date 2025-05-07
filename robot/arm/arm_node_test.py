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

    target = arm_node.get_ee_pose()
    print("Current pose:", target)

    # Create initial target pose at origin
    # target = mink.SE3.from_translation(np.array([0.0, 0.0, 0.0]))

    delta_forward = mink.SE3.from_translation(np.array([0.25, 0.0, 0.0]))
    delta_forward_large = mink.SE3.from_translation(np.array([0.5, 0.0, 0.0]))
    delta_backward_large = mink.SE3.from_translation(np.array([-0.5, 0.0, 0.0]))

    # Apply the translation
    target = delta_forward.multiply(target)
    print(target)
    input("Press Enter to go forward...")
    arm_node.set_ee_target(target, gripper_target=-22.0, preview_time=1.5)

    target = delta_backward_large.multiply(target)
    print(target)
    input("Press Enter to go backward...")
    arm_node.set_ee_target(target, gripper_target=-22.0, preview_time=1.5)

    target = delta_forward_large.multiply(target)
    print(target)
    input("Press Enter to go forward...")
    arm_node.set_ee_target(target, gripper_target=-22.0, preview_time=1.5)
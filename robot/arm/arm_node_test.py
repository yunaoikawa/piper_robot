from robot.arm.arm_node import ArmNode
from pathlib import Path
import numpy as np
import mink

if __name__ == "__main__":
    _HERE = Path(__file__).parent
    # arm_node = ArmNode(
    #     can_port="can_left",
    #     mjcf_path=(_HERE / "mujoco/scene_piper.xml").as_posix(),
    #     urdf_path=(_HERE / "urdf/piper_description_left.xml").as_posix(),
    #     solver_dt=0.01,
    # )

    # target = arm_node.get_ee_pose()

    target = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3(np.array([0.0, 0.0, 0.0, 1.0])),
        translation=np.array([0.0, 0.0, 0.0]),
    )

    go_backward_delta = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3(np.array([1.0, 0.0, 0.0, 0.0])),
        translation=np.array([-0.25, 0.0, 0.0]),
    )
    go_forward_delta = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3(np.array([0.0, 0.0, 0.0, 1.0])),
        translation=np.array([0.25, 0.0, 0.0]),
    )

    while True:
        target = target.multiply(go_forward_delta)
        print(target.wxyz_xyz)
        # arm_node.set_ee_target(target, gripper_target=-22.0, preview_time=1.5)
        # input("Press Enter to go forward...")
        # target.wxyz_xyz += np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0])
        # arm_node.set_ee_target(target, gripper_target=-22.0, preview_time=1.5)
        # input("Press Enter to go backward...")

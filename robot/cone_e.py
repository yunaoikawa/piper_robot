import functools
import time
import numpy as np
import mink
import atexit
from pathlib import Path

from robot.base import Base
from robot.arm.arm import ArmNode
# from robot.arm.arm_piper_control import ArmNode as ArmNodePiperControl
from robot.rpc import RPCServer


def require_initialization(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            print(f"Warning: {func.__name__} called before ConeE was initialized")
            return None
        return func(self, *args, **kwargs)

    return wrapper


class ConeE:
    def __init__(
        self, base_max_vel=np.array((1.0, 1.0, 1.57)), base_max_accel=np.array((1.0, 1.0, 1.57)), no_arms=False
    ):
        self._initialized = False

        self.base = Base(max_vel=base_max_vel, max_accel=base_max_accel)
        self.no_arms = no_arms
        if not self.no_arms:
            _HERE = Path(__file__).parent
            self.left_arm = ArmNode(
                can_port="can_left",
                mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix()
            )
            self.right_arm = ArmNode(
                can_port="can_right",
                mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix(),
                is_left_arm=False,
            )
            # self.left_arm = ArmNodePiperControl(
            #     can_port="can_left",
            #     mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix()
            # )
            # self.right_arm = ArmNodePiperControl(
            #     can_port="can_right",
            #     mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix(),
            #     is_left_arm=False,
            # )

    def init(self):
        if self._initialized:
            print("Warning: ConeE already initialized")
            return

        self.base.start_control()
        time.sleep(0.5)
        self.base.home_lift()
        time.sleep(0.5)
        # TODO: call gripper homing inside arm_init
        if not self.no_arms:
            self.left_arm.init()
            self.right_arm.init()

        self._initialized = True

    # Base and lift
    @require_initialization
    def set_base_velocity(self, velocity: np.ndarray):
        self.base.set_target_base_velocity(velocity)

    @require_initialization
    def set_base_position(self, position: np.ndarray):
        self.base.set_target_base_position(position)

    @require_initialization
    def set_lift_position(self, position: np.ndarray):
        self.base.set_target_lift(position)

    @require_initialization
    def get_lift_position(self) -> float:
        return self.base.get_lift_position()

    # Left arm
    @require_initialization
    def set_left_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        self.left_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.1):
        # ee_target = mink.SE3(wxyz_xyz=ee_target_wxyz_xyz)
        self.left_arm.set_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    def set_left_gain(self, kp: np.ndarray, kd: np.ndarray):
        self.left_arm.set_gain(kp, kd)

    @require_initialization
    def tuck_left_arm(self):
        self.left_arm.tuck_arms()

    @require_initialization
    def tuck_right_arm(self):
        self.right_arm.tuck_arms()

    @require_initialization
    def open_left_gripper(self):
        self.left_arm.open_gripper()

    @require_initialization
    def open_right_gripper(self):
        self.right_arm.open_gripper()

    @require_initialization
    def close_left_gripper(self):
        self.left_arm.close_gripper()

    @require_initialization
    def close_right_gripper(self):
        self.right_arm.close_gripper()

    @require_initialization
    def home_left_arm(self, gripper_target: float = 1.0):
        self.left_arm.home(gripper_target)

    @require_initialization
    def get_left_ee_pose(self) -> mink.SE3:
        """
        Returns the pose of the left arm's end effector in the world frame (qw, qx, qy, qz, x, y, z).
        """
        return self.left_arm.get_ee_pose()

    @require_initialization
    def get_left_joint_positions(self) -> np.ndarray:
        return self.left_arm.get_joint_positions()

    # Right arm
    @require_initialization
    def set_right_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        self.right_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.1):
        # ee_target = mink.SE3(wxyz_xyz=ee_target_wxyz_xyz)
        self.right_arm.set_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    def set_right_gain(self, kp: np.ndarray, kd: np.ndarray):
        self.right_arm.set_gain(kp, kd)

    @require_initialization
    def home_right_arm(self):
        self.right_arm.home()

    @require_initialization
    def get_right_ee_pose(self) -> mink.SE3:
        return self.right_arm.get_ee_pose()

    @require_initialization
    def get_right_joint_positions(self) -> np.ndarray:
        return self.right_arm.get_joint_positions()


def main():
    cone_e = ConeE(no_arms=True)
    server = RPCServer(cone_e, "localhost", 8081, threaded=False)
    atexit.register(server.stop)
    server.start()


if __name__ == "__main__":
    main()

import warnings
import functools
import time
import numpy as np
import mink
import atexit

from robot.base import Base
from robot.arm.arm import ArmNode
from robot.rpc import RPCServer


def require_initialization(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            warnings.warn(f"{func.__name__} called before ConeE was initialized")
            return None
        return func(self, *args, **kwargs)
    return wrapper


class ConeE:
    def __init__(self, base_max_vel=np.array((1.0, 1.0, 1.57)), base_max_accel=np.array((1.0, 1.0, 1.57))):
        self._initialized = False

        self.base = Base(max_vel=base_max_vel, max_accel=base_max_accel)
        self.left_arm = ArmNode(can_port="can_left")
        # self.right_arm = ArmNode(can_port="can_right")

    def init(self):
        self.base.start_control()
        time.sleep(0.5)
        self.base.home_lift()
        time.sleep(0.5)
        # TODO: call gripper homing inside arm_init
        self.left_arm.init()

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
    def set_left_ee_target(
        self, ee_target_wxyz_xyz: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        ee_target = mink.SE3(wxyz_xyz=ee_target_wxyz_xyz)
        self.left_arm.set_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    def home_left_arm(self):
        self.left_arm.home()

    @require_initialization
    def get_left_ee_pose(self) -> np.ndarray:
        """
        Returns the pose of the left arm's end effector in the world frame (qw, qx, qy, qz, x, y, z).
        """
        return self.left_arm.get_ee_pose().wxyz_xyz

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
    def set_right_ee_target(
        self, ee_target_wxyz_xyz: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        ee_target = mink.SE3(wxyz_xyz=ee_target_wxyz_xyz)
        self.right_arm.set_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    def home_right_arm(self):
        self.right_arm.home()

    @require_initialization
    def get_right_ee_pose(self) -> np.ndarray:
        return self.right_arm.get_ee_pose().wxyz_xyz

    @require_initialization
    def get_right_joint_positions(self) -> np.ndarray:
        return self.right_arm.get_joint_positions()


def main():
    cone_e = ConeE()
    server = RPCServer(cone_e, 'localhost', 5000, threaded=False)
    atexit.register(server.stop)
    server.start()

if __name__ == "__main__":
    main()

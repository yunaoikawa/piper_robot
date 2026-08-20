import argparse
import functools
import numpy as np
import mink
import atexit
from pathlib import Path

from robot.arm.arm import ArmNode
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
        self, base_max_vel=np.array((1.0, 1.0, 1.57)), base_max_accel=np.array((1.0, 1.0, 1.57)),
        no_arms=False, reset_arms_on_init=True,
    ):
        self._initialized = False
        self.no_arms = no_arms
        self.reset_arms_on_init = reset_arms_on_init

        if not self.no_arms:
            _HERE = Path(__file__).parent
            self.left_arm = ArmNode(
                can_port="can_left",
                mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix(),
                use_gripper=True,
            )
            self.right_arm = ArmNode(
                can_port="can_right",
                mjcf_path=(_HERE / "cone-e-description/robot-welded-base-and-lift.mjcf").as_posix(),
                is_left_arm=False,
                use_gripper=True,
            )

    def init(self, reset_arms: bool | None = None):
        if self._initialized:
            print("Warning: ConeE already initialized")
            return

        if not self.no_arms:
            if reset_arms is None:
                reset_arms = self.reset_arms_on_init
            self.left_arm.init(reset=reset_arms)
            self.right_arm.init(reset=reset_arms)

        self._initialized = True

    # ----------------------------------------------------------------------
    # Base and lift (all disabled)
    # ----------------------------------------------------------------------
    @require_initialization
    def set_base_velocity(self, velocity: np.ndarray):
        print("Warning: set_base_velocity() called but base is disabled")

    @require_initialization
    def set_base_position(self, position: np.ndarray):
        print("Warning: set_base_position() called but base is disabled")

    @require_initialization
    def set_lift_position(self, position: np.ndarray):
        print("Warning: set_lift_position() called but base is disabled")

    @require_initialization
    def get_lift_position(self) -> float:
        print("Warning: get_lift_position() called but base is disabled")
        return 0.0

    # ----------------------------------------------------------------------
    # Left arm
    # ----------------------------------------------------------------------
    @require_initialization
    def machine_zero_left_arm(self):
        """Fold the left arm to Piper's true q=0 pose."""
        return self.left_arm.machine_zero()

    @require_initialization
    def machine_zero_right_arm(self):
        """Fold the right arm to Piper's true q=0 pose."""
        return self.right_arm.machine_zero()

    @require_initialization
    def machine_zero_arms(self):
        """Fold both arms to their true q=0 pose, matching server startup.

        Keep the same deterministic left-then-right order used by ``init``.
        This is intentionally distinct from ``home_left_arm`` and
        ``home_right_arm``, which select the upright manipulation pose.
        """
        print("Returning left arm to machine zero (q=0)...", flush=True)
        self.machine_zero_left_arm()
        print("Returning right arm to machine zero (q=0)...", flush=True)
        self.machine_zero_right_arm()
        print("Both arms are at machine zero.", flush=True)

    @require_initialization
    def set_left_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        self.left_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.1):
        return self.left_arm.set_ee_target(
            ee_target, gripper_target, preview_time
        )

    @require_initialization
    def set_left_teleop_ee_target(
        self,
        ee_target: mink.SE3,
        gripper_target: float | None = None,
        preview_time: float = 0.1,
        max_joint_step_rad: float = 4.0 / 30.0,
    ):
        return self.left_arm.set_teleop_ee_target(
            ee_target,
            gripper_target,
            preview_time,
            max_joint_step_rad,
        )

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
        return self.left_arm.get_ee_pose()

    @require_initialization
    def get_left_joint_positions(self) -> np.ndarray:
        return self.left_arm.get_joint_positions()

    @require_initialization
    def get_left_joint_torque(self) -> np.ndarray:
        return self.left_arm.get_joint_torque()

    @require_initialization
    def get_left_gain(self) -> dict[str, np.ndarray]:
        gain = self.left_arm.piper.get_gain()
        return {
            "kp": np.asarray(gain.kp, dtype=float).copy(),
            "kd": np.asarray(gain.kd, dtype=float).copy(),
        }

    @require_initialization
    def get_left_gripper_exact(self) -> float:
        """Get left gripper position as open ratio (0.0=closed, 1.0=open)."""
        if self.left_arm.gripper is not None:
            return self.left_arm.gripper.get_open_ratio()
        return 0.0

    @require_initialization
    def get_left_gripper_status(self) -> dict:
        if self.left_arm.gripper is not None:
            return self.left_arm.gripper.get_status()
        return {}

    # ----------------------------------------------------------------------
    # Right arm
    # ----------------------------------------------------------------------
    @require_initialization
    def set_right_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        self.right_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.1):
        return self.right_arm.set_ee_target(
            ee_target, gripper_target, preview_time
        )

    @require_initialization
    def set_right_teleop_ee_target(
        self,
        ee_target: mink.SE3,
        gripper_target: float | None = None,
        preview_time: float = 0.1,
        max_joint_step_rad: float = 4.0 / 30.0,
    ):
        return self.right_arm.set_teleop_ee_target(
            ee_target,
            gripper_target,
            preview_time,
            max_joint_step_rad,
        )

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

    @require_initialization
    def get_right_joint_torque(self) -> np.ndarray:
        return self.right_arm.get_joint_torque()

    @require_initialization
    def get_right_gain(self) -> dict[str, np.ndarray]:
        gain = self.right_arm.piper.get_gain()
        return {
            "kp": np.asarray(gain.kp, dtype=float).copy(),
            "kd": np.asarray(gain.kd, dtype=float).copy(),
        }

    @require_initialization
    def get_right_gripper_exact(self) -> float:
        """Get right gripper position as open ratio (0.0=closed, 1.0=open)."""
        if self.right_arm.gripper is not None:
            return self.right_arm.gripper.get_open_ratio()
        return 0.0

    @require_initialization
    def get_right_gripper_status(self) -> dict:
        if self.right_arm.gripper is not None:
            return self.right_arm.gripper.get_status()
        return {}

    # ----------------------------------------------------------------------
    # Rest positions (for rollout controller 'h' key)
    # ----------------------------------------------------------------------
    @require_initialization
    def rest_left_arm(self):
        """Move left arm to a safe resting position on the table."""
        self.left_arm.home(gripper_target=1.0)

    @require_initialization
    def rest_right_arm(self):
        """Move right arm to a safe resting position on the table."""
        self.right_arm.home(gripper_target=1.0)

    # ----------------------------------------------------------------------
    # Jacobian (for manipulability calculation)
    # ----------------------------------------------------------------------
    @require_initialization
    def get_left_jacobian(self) -> np.ndarray:
        """Get left arm Jacobian matrix."""
        q = self.left_arm.get_joint_positions()
        self.left_arm.ik_solver.update_configuration(q)
        return self.left_arm.ik_solver.get_jacobian()

    @require_initialization
    def get_right_jacobian(self) -> np.ndarray:
        """Get right arm Jacobian matrix."""
        q = self.right_arm.get_joint_positions()
        self.right_arm.ik_solver.update_configuration(q)
        return self.right_arm.ik_solver.get_jacobian()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a ConeE RPC server so remote clients can teleoperate the robot.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0).")
    parser.add_argument("--port", default=8081, type=int, help="Port to listen on (default: 8081).")
    parser.add_argument(
        "--no-arms",
        action="store_true",
        help="Skip initializing the Piper arms. Useful for smoke tests on machines without hardware.",
    )
    parser.add_argument(
        "--attach-current",
        action="store_true",
        help="Attach to the current joint state without resetting either arm to home.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    cone = ConeE(
        no_arms=args.no_arms,
        reset_arms_on_init=not args.attach_current,
    )
    # Every server mode must expose valid state before a client connects.
    # In normal mode init() performs Piper's true q=0 reset; attach-current
    # latches the measured pose without moving because reset_arms_on_init is
    # false.  Omitting this call leaves every guarded getter uninitialized even
    # though the RPC socket appears healthy.
    cone.init()
    rpc_server = RPCServer(cone, args.host, args.port, threaded=False)
    stop_callback = rpc_server.stop
    atexit.register(stop_callback)

    if args.no_arms:
        mode_desc = "without arm hardware"
    elif args.attach_current:
        mode_desc = "attached to current arm state"
    else:
        mode_desc = "with full hardware control"
    print(f"ConeE RPC server listening on {args.host}:{args.port} ({mode_desc}).")
    print("Press Ctrl+C to exit.")

    try:
        rpc_server.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping ConeE RPC server.")
    finally:
        stop_callback()
        if hasattr(atexit, "unregister"):
            atexit.unregister(stop_callback)


if __name__ == "__main__":
    main()

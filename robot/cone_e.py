import argparse
import functools
import time
import numpy as np
import mink
import atexit
from pathlib import Path

from robot.arm.arm import ArmNode
from robot.rpc import RPCServer


# =============================================================================
# Workspace boundaries (metres)
# =============================================================================
# Derived from every demo we have (23 absolute-action datasets across all 10
# tasks, action + state, both arms) plus a 5 cm margin. Regenerate with:
#     srun ... python src/calibrate_workspace.py
#
# The previous hand-set values were unusable and that is why the clamp below was
# commented out: their z floor of 0.617 sat *above* the 0.599 that petri2bench
# actually reaches, so 4-5% of that task's commanded frames would have been
# clipped -- right at the grasp, where the operational `z-bias -0.025` lives.
# The bias would have been silently swallowed and the task would have regressed
# with no error. The margin is deliberately wider (5 cm) than the largest bias
# in use (3 cm, see outputs/lab/act/horizon/EVAL_RESULTS.md) so a bias can never
# be eaten by the clamp without the arm first leaving the demonstrated envelope.
#
# Note these are not uniformly looser: y_min tightens from -0.498 to -0.176,
# since nothing ever demonstrated reaching that far.
WORKSPACE_MIN = np.array([-0.054, -0.176, 0.549])
WORKSPACE_MAX = np.array([0.589, 0.437, 1.102])


_clamp_log = {"last": 0.0, "suppressed": 0}


def clamp_ee_target(ee_target: mink.SE3) -> mink.SE3:
    """Return a new SE3 with translation clamped to workspace boundaries.

    A clamp that fires silently is the dangerous case -- it turns a bias or a
    policy command into a no-op with no symptom other than the task quietly
    failing. So the first hit always prints; after that we throttle to once a
    second (this runs at 30 Hz) and report how many were folded in.
    """
    p = ee_target.translation()
    p_clamped = np.clip(p, WORKSPACE_MIN, WORKSPACE_MAX)

    if not np.allclose(p, p_clamped):
        now = time.time()
        if now - _clamp_log["last"] >= 1.0:
            extra = (f"  (+{_clamp_log['suppressed']} more in the last second)"
                     if _clamp_log["suppressed"] else "")
            print(f"[Workspace] Position clamped: {np.round(p, 3)} → "
                  f"{np.round(p_clamped, 3)}{extra}", flush=True)
            _clamp_log["last"] = now
            _clamp_log["suppressed"] = 0
        else:
            _clamp_log["suppressed"] += 1

    wxyz = ee_target.rotation().wxyz
    return mink.SE3(np.concatenate([wxyz, p_clamped]))


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
        self.no_arms = no_arms

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

    def init(self):
        if self._initialized:
            print("Warning: ConeE already initialized")
            return

        if not self.no_arms:
            self.left_arm.init()
            self.right_arm.init()

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
    def set_left_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        self.left_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.1):
        ee_target = clamp_ee_target(ee_target)
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
        return self.left_arm.get_ee_pose()

    @require_initialization
    def get_left_joint_positions(self) -> np.ndarray:
        return self.left_arm.get_joint_positions()

    @require_initialization
    def get_left_joint_torque(self) -> np.ndarray:
        return self.left_arm.get_joint_torque()

    @require_initialization
    def get_left_gripper_exact(self) -> float:
        """Get left gripper position as open ratio (0.0=closed, 1.0=open)."""
        if self.left_arm.gripper is not None:
            return self.left_arm.gripper.get_open_ratio()
        return 0.0

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
        ee_target = clamp_ee_target(ee_target)
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

    @require_initialization
    def get_right_joint_torque(self) -> np.ndarray:
        return self.right_arm.get_joint_torque()

    @require_initialization
    def get_right_gripper_exact(self) -> float:
        """Get right gripper position as open ratio (0.0=closed, 1.0=open)."""
        if self.right_arm.gripper is not None:
            return self.right_arm.gripper.get_open_ratio()
        return 0.0

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    cone = ConeE(no_arms=args.no_arms)
    rpc_server = RPCServer(cone, args.host, args.port, threaded=False)
    stop_callback = rpc_server.stop
    atexit.register(stop_callback)

    mode_desc = "without arm hardware" if args.no_arms else "with full hardware control"
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

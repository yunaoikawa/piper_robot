import time
import numpy as np
from typing import Optional
from pathlib import Path
import mink
from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    COMM_SUCCESS,
)
from piperlib import (
    PiperJointController,
    RobotConfigFactory,
    ControllerConfigFactory,
    JointState,
    Gain,
)
from robot.arm.ik_solver import SingleArmIK

# =========================
# Dynamixel constants
# =========================
DXL_PORT = "/dev/ttyUSB0"
DXL_BAUDRATE = 115200
DXL_PROTOCOL_VERSION = 2.0

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_PROFILE_VELOCITY = 112
ADDR_PRESENT_POSITION = 132

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OPERATING_MODE_POSITION = 4

# ⚠️ CHANGE THESE FOR YOUR GRIPPER
DXL_POS_RIGHT_OPEN = 2800
DXL_POS_RIGHT_CLOSE = 6300
DXL_POS_LEFT_CLOSE = 3000
DXL_POS_LEFT_OPEN = 7000

# Servo IDs
DXL_ID_RIGHT = 1
DXL_ID_LEFT = 2


# =========================
# Shared Dynamixel port (singleton)
# =========================
class _SharedDynamixelPort:
    """Singleton to share a single U2D2 port between two grippers."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            port = PortHandler(DXL_PORT)
            if not port.openPort():
                raise RuntimeError(f"Failed to open {DXL_PORT}")
            if not port.setBaudRate(DXL_BAUDRATE):
                raise RuntimeError("Failed to set Dynamixel baudrate")
            packet = PacketHandler(DXL_PROTOCOL_VERSION)
            cls._instance = (port, packet)
            print(f"[Gripper] Shared Dynamixel port opened: {DXL_PORT}")
        return cls._instance


# =========================
# Dynamixel gripper class
# =========================
class DynamixelGripper:
    def __init__(self, dxl_id: int = DXL_ID_RIGHT, inverted: bool = False):
        import struct as _struct
        import time as _time

        self.dxl_id = dxl_id
        self.inverted = inverted

        # 1. Get shared port
        self.port, self.packet = _SharedDynamixelPort.get()

        # 2. Reboot right servo only (left uses fixed values, no reboot needed)
        if self.dxl_id != DXL_ID_LEFT:
            self.packet.reboot(self.port, self.dxl_id)
            _time.sleep(1.5)

        # 3. Disable torque, set mode, enable torque
        self.packet.write1ByteTxRx(self.port, self.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.packet.write1ByteTxRx(self.port, self.dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
        self.packet.write1ByteTxRx(self.port, self.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        self.packet.write2ByteTxRx(self.port, self.dxl_id, 38, 300)

        # 4. Set open/close positions
        if self.dxl_id == DXL_ID_LEFT:
            self.pos_open = DXL_POS_LEFT_OPEN
            self.pos_close = DXL_POS_LEFT_CLOSE
        else:
            self.pos_open = DXL_POS_RIGHT_OPEN
            self.pos_close = DXL_POS_RIGHT_CLOSE

        print(f"[Gripper] Dynamixel ID={self.dxl_id} initialized")

    def set_open_ratio(self, ratio: float):
        """ratio: 0.0 = fully closed, 1.0 = fully open"""
        ratio = float(np.clip(ratio, 0.0, 1.0))
        pos = int(self.pos_close + ratio * (self.pos_open - self.pos_close))
        import struct
        unsigned = struct.unpack('I', struct.pack('i', pos))[0]
        self.packet.write4ByteTxRx(
            self.port, self.dxl_id, ADDR_GOAL_POSITION, unsigned
        )

    def get_open_ratio(self) -> float:
        """Read current position as open ratio (0.0=closed, 1.0=open)."""
        raw, _, _ = self.packet.read4ByteTxRx(
            self.port, self.dxl_id, ADDR_PRESENT_POSITION
        )
        import struct
        pos = struct.unpack("i", struct.pack("I", raw))[0]
        ratio = (pos - self.pos_close) / (self.pos_open - self.pos_close)
        ratio = float(max(0.0, min(1.0, ratio)))
        return ratio

    def close(self):
        self.set_open_ratio(0.0)

    def open(self):
        self.set_open_ratio(1.0)

    def stop(self):
        self.packet.write1ByteTxRx(
            self.port, self.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )


# =========================
# ArmNode (Piper arm + optional Dynamixel gripper)
#
# NOTE: Physical arms are swapped — the arm on the left side was originally
# built as "right" and vice versa. URDF, joint_names, ee_frame, and home_q
# are swapped here to compensate.
# =========================

class ArmNode:
    def __init__(
        self,
        can_port: str,
        mjcf_path: str,
        urdf_path: Optional[str] = None,
        solver_dt: float = 0.01,
        is_left_arm: bool = True,
        use_gripper: bool = True,
    ):
        _HERE = Path(__file__).parent
        self.can_port = can_port
        self.is_left_arm = is_left_arm

        # URDF swapped: left position uses right arm URDF, right position uses left arm URDF
        if urdf_path is None:
            if is_left_arm:
                self.urdf_path = (_HERE / "urdf/piper_description_right.xml").as_posix()
            else:
                self.urdf_path = (_HERE / "urdf/piper_description_left.xml").as_posix()
        else:
            self.urdf_path = urdf_path

        # ----- Piper arm -----
        self.robot_config = RobotConfigFactory.get_instance().get_config("piper")
        self.controller_config = ControllerConfigFactory.get_instance().get_config("joint_controller")
        self.robot_config.urdf_path = self.urdf_path
        self.robot_config.joint_vel_max = np.ones(6) * 5.0
        self.controller_config.controller_dt = 0.003
        self.controller_config.default_kp = np.ones(6) * 2.5
        self.controller_config.default_kd = np.ones(6) * 0.2
        self.controller_config.gravity_compensation = True
        self.controller_config.interpolation_method = "linear"

        self.piper = PiperJointController(
            self.robot_config, self.controller_config, self.can_port
        )

        # ----- Dynamixel gripper (optional) -----
        if use_gripper:
            if is_left_arm:
                self.gripper = DynamixelGripper(dxl_id=DXL_ID_LEFT, inverted=False)
            else:
                self.gripper = DynamixelGripper(dxl_id=DXL_ID_RIGHT, inverted=False)
        else:
            self.gripper = None

        # ----- IK (swapped: left position uses right arm config, vice versa) -----
        if is_left_arm:
            joint_names = [f"right_arm_joint{i}" for i in range(1, 7)]
            ee_frame = "right_arm_ee"
            self.home_q = np.array([0.0, 1.58, -0.58, 0.0, -0.91, 1.40])
        else:
            joint_names = [f"left_arm_joint{i}" for i in range(1, 7)]
            ee_frame = "left_arm_ee"
            self.home_q = np.array([0.0, 1.58, -0.58, 0.0, -0.91, 2.35])

        self.ik_solver = SingleArmIK(
            mjcf_path,
            solver_dt=solver_dt,
            joint_names=joint_names,
            ee_frame=ee_frame,
        )

    # =========================
    # Lifecycle
    # =========================
    def init(self):
        self.reset()
        q = self.piper.get_joint_state().pos
        self.ik_solver.init(q)

    def reset(self):
        self.piper.reset_to_home()
        time.sleep(1.0)
        if self.gripper is not None:
            self.gripper.open()

    def home(self, gripper_target: float = 1.0):
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = self.home_q
        cmd.timestamp = self.piper.get_timestamp() + 1.0
        self.piper.set_joint_cmd(cmd)
        if self.gripper is not None:
            self.gripper.set_open_ratio(gripper_target)
        time.sleep(2.0)

    # =========================
    # Control
    # =========================
    def set_joint_target(
        self,
        joint_target: np.ndarray,
        gripper_target: Optional[float] = None,
        preview_time: float = 0.1,
    ):
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = joint_target
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        self.piper.set_joint_cmd(cmd)
        if gripper_target is not None and self.gripper is not None:
            self.gripper.set_open_ratio(gripper_target)

    def set_ee_target(
        self,
        ee_target: mink.SE3,
        gripper_target: Optional[float] = None,
        preview_time: float = 0.01,
    ):
        qd, _ = self.ik_solver.solve_ik(ee_target)
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = qd
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        self.piper.set_joint_cmd(cmd)
        if gripper_target is not None and self.gripper is not None:
            self.gripper.set_open_ratio(gripper_target)

    # =========================
    # Helpers
    # =========================
    def open_gripper(self):
        if self.gripper is not None:
            self.gripper.open()

    def close_gripper(self):
        if self.gripper is not None:
            self.gripper.close()

    def set_gain(self, kp: np.ndarray, kd: np.ndarray):
        self.piper.set_gain(Gain(kp, kd))

    def get_joint_positions(self) -> np.ndarray:
        return self.piper.get_joint_state().pos

    def get_ee_pose(self) -> mink.SE3:
        q = self.get_joint_positions()
        self.ik_solver.update_configuration(q)
        return self.ik_solver.forward_kinematics()

    def stop(self):
        if self.gripper is not None:
            self.gripper.stop()
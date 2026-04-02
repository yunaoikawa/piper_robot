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
DXL_PORT = "/dev/ttyUSB1"
DXL_BAUDRATE = 115200
DXL_ID = 1
DXL_PROTOCOL_VERSION = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_PROFILE_VELOCITY = 112
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
OPERATING_MODE_POSITION = 3
# ⚠️ CHANGE THESE FOR YOUR GRIPPER
DXL_POS_OPEN = 2800
DXL_POS_CLOSE = 0
# =========================
# Dynamixel gripper class
# =========================
class DynamixelGripper:
    def __init__(self):
        self.port = PortHandler(DXL_PORT)
        self.packet = PacketHandler(DXL_PROTOCOL_VERSION)
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open {DXL_PORT}")
        if not self.port.setBaudRate(DXL_BAUDRATE):
            raise RuntimeError("Failed to set Dynamixel baudrate")
        # Disable torque before changing settings
        self.packet.write1ByteTxRx(
            self.port, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        # Explicitly set Position Control Mode (mode 3)
        dxl_comm_result, dxl_error = self.packet.write1ByteTxRx(
            self.port, DXL_ID, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to set Position Control Mode")
        # Enable torque
        dxl_comm_result, dxl_error = self.packet.write1ByteTxRx(
            self.port, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to enable Dynamixel torque")
        # Set Profile Velocity to 0 (maximum speed)
        self.packet.write4ByteTxRx(
            self.port, DXL_ID, ADDR_PROFILE_VELOCITY, 0
        )
        print("[Gripper] Dynamixel gripper initialized (Position Control Mode, max speed)")
    def set_open_ratio(self, ratio: float):
        """ratio: 0.0 = fully closed, 1.0 = fully open"""
        ratio = float(np.clip(ratio, 0.0, 1.0))
        pos = int(DXL_POS_CLOSE + ratio * (DXL_POS_OPEN - DXL_POS_CLOSE))
        self.packet.write4ByteTxRx(
            self.port, DXL_ID, ADDR_GOAL_POSITION, pos
        )
    def close(self):
        self.set_open_ratio(0.0)
    def open(self):
        self.set_open_ratio(1.0)
    def stop(self):
        self.packet.write1ByteTxRx(
            self.port, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        self.port.closePort()
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
            self.gripper = DynamixelGripper()
        else:
            self.gripper = None
        # ----- IK (swapped: left position uses right arm config, vice versa) -----
        if is_left_arm:
            joint_names = [f"right_arm_joint{i}" for i in range(1, 7)]
            ee_frame = "right_arm_ee"
            self.home_q = np.array([0.0, 1.58, -0.58, 0.0, -0.91, -0.78])
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
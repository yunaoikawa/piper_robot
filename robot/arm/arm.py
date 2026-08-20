import os
import time
import struct
import json
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
from robot.arm.ik_continuity import joint_target_is_continuous
from robot.arm.home import physical_home_q
from robot.arm.ik_solver import SingleArmIK

# =========================
# Dynamixel constants
# =========================
DXL_PORT_ENV = "ROBOT_DYNAMIXEL_PORT"
DXL_BY_ID_GLOB = "usb-FTDI_USB__-__Serial_Converter_*-if00-port0"
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

DXL_POS_RIGHT_OPEN = 2800
DXL_POS_RIGHT_CLOSE = 6300

DXL_ID_RIGHT = 1
DXL_ID_LEFT = 2

# Two physical Piper controllers share the host and CAN/USB workload.  The
# historical 3 ms setting routinely overran at 4--7 ms on Pasteur, which makes
# streamed trajectories repeatedly miss their interpolation deadlines.  A
# 10 ms controller period is the previously verified non-ticking setting and
# still runs much faster than the 30 Hz teleoperation command stream.
PIPER_CONTROLLER_DT_S = 0.01

# Interactive Cartesian commands arrive at 30 Hz.  Keep their default joint
# step below Piper's configured 5 rad/s velocity limit while leaving some
# margin for scheduling jitter.  Callers may provide a different value when
# their stream rate differs.
TELEOP_DEFAULT_MAX_JOINT_STEP_RAD = 4.0 / 30.0

LEFT_CAL_FILE = Path(__file__).parent / "left_gripper_cal.json"


class _SharedDynamixelPort:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            port_name = _resolve_dynamixel_port()
            port = PortHandler(port_name)
            if not port.openPort():
                raise RuntimeError(f"Failed to open {port_name}")
            if not port.setBaudRate(DXL_BAUDRATE):
                raise RuntimeError("Failed to set Dynamixel baudrate")
            packet = PacketHandler(DXL_PROTOCOL_VERSION)
            cls._instance = (port, packet)
            print(f"[Gripper] Shared Dynamixel port opened: {port_name}")
        return cls._instance


def _resolve_dynamixel_port() -> str:
    """Resolve the gripper adapter without relying on ttyUSB enumeration order."""
    configured = os.environ.get(DXL_PORT_ENV)
    if configured:
        return configured

    by_id = Path("/dev/serial/by-id")
    candidates = sorted(by_id.glob(DXL_BY_ID_GLOB)) if by_id.exists() else []
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple Dynamixel adapters found: {candidates}; set {DXL_PORT_ENV}"
        )

    tty_candidates = sorted(Path("/dev").glob("ttyUSB*"))
    if len(tty_candidates) == 1:
        return str(tty_candidates[0])
    raise RuntimeError(
        f"Could not identify the Dynamixel adapter; set {DXL_PORT_ENV}"
    )


def _read_pos(packet, port, dxl_id):
    raw, comm_result, dxl_error = packet.read4ByteTxRx(
        port, dxl_id, ADDR_PRESENT_POSITION
    )
    if comm_result != COMM_SUCCESS or dxl_error != 0:
        raise RuntimeError(
            f"Dynamixel ID={dxl_id} position read failed: "
            f"comm_result={comm_result}, dxl_error={dxl_error}"
        )
    return struct.unpack("i", struct.pack("I", raw))[0]


def _read_signed_4byte(packet, port, dxl_id, address):
    raw, comm_result, dxl_error = packet.read4ByteTxRx(
        port, dxl_id, address
    )
    value = struct.unpack("i", struct.pack("I", raw))[0]
    return value, comm_result, dxl_error


def _write_pos(packet, port, dxl_id, pos):
    unsigned = struct.unpack("I", struct.pack("i", pos))[0]
    return packet.write4ByteTxRx(port, dxl_id, ADDR_GOAL_POSITION, unsigned)


class DynamixelGripper:
    def __init__(self, dxl_id: int = DXL_ID_RIGHT, inverted: bool = False):
        self.dxl_id = dxl_id
        self.inverted = inverted
        self.port, self.packet = _SharedDynamixelPort.get()
        self._last_commanded_pos: int | None = None

        # Reboot right servo only
        if self.dxl_id != DXL_ID_LEFT:
            self.packet.reboot(self.port, self.dxl_id)
            time.sleep(1.5)

        # Setup mode
        self.packet.write1ByteTxRx(self.port, self.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.packet.write1ByteTxRx(self.port, self.dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
        self.packet.write1ByteTxRx(self.port, self.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        self.packet.write2ByteTxRx(self.port, self.dxl_id, 38, 300)

        if self.dxl_id == DXL_ID_LEFT:
            if LEFT_CAL_FILE.exists():
                cal = json.loads(LEFT_CAL_FILE.read_text())
                self.pos_open = cal["open"]
                self.pos_close = cal["close"]
                print(f"[Gripper] ID={self.dxl_id} loaded cal: open={self.pos_open}, close={self.pos_close}")
            else:
                self.pos_open = 5000
                self.pos_close = 12000
                print(f"[Gripper] ID={self.dxl_id} NO CAL FILE! Using defaults. Run: python robot/arm/calibrate_left.py")
        else:
            self.pos_open = DXL_POS_RIGHT_OPEN
            self.pos_close = DXL_POS_RIGHT_CLOSE

        print(f"[Gripper] ID={self.dxl_id} initialized: open={self.pos_open}, close={self.pos_close}")

    def set_open_ratio(self, ratio: float):
        ratio = float(np.clip(ratio, 0.0, 1.0))
        pos = int(self.pos_close + ratio * (self.pos_open - self.pos_close))
        # Position-mode Dynamixels retain and regulate their last goal.  Sending
        # the same close target on every 30 Hz arm pose blocks the arm RPC on a
        # synchronous USB round-trip and makes object transport visibly tick.
        # Latch the goal and write again only when it actually changes.  Failed
        # writes are deliberately not cached, so the next frame retries.
        if pos == self._last_commanded_pos:
            return False
        comm_result, dxl_error = _write_pos(
            self.packet, self.port, self.dxl_id, pos
        )
        if comm_result == COMM_SUCCESS and dxl_error == 0:
            self._last_commanded_pos = pos
            print(
                f"[Gripper] ID={self.dxl_id} goal={pos} "
                f"open_ratio={ratio:.3f}",
                flush=True,
            )
        else:
            print(
                f"[Gripper] ID={self.dxl_id} write failed: goal={pos}, "
                f"comm_result={comm_result}, dxl_error={dxl_error}",
                flush=True,
            )
        return True

    def get_open_ratio(self) -> float:
        pos = _read_pos(self.packet, self.port, self.dxl_id)
        denom = self.pos_open - self.pos_close
        if denom == 0:
            return 0.0
        ratio = (pos - self.pos_close) / denom
        return float(max(0.0, min(1.0, ratio)))

    def get_status(self) -> dict:
        present, present_comm, present_error = _read_signed_4byte(
            self.packet,
            self.port,
            self.dxl_id,
            ADDR_PRESENT_POSITION,
        )
        goal, goal_comm, goal_error = _read_signed_4byte(
            self.packet,
            self.port,
            self.dxl_id,
            ADDR_GOAL_POSITION,
        )
        torque, torque_comm, torque_error = self.packet.read1ByteTxRx(
            self.port,
            self.dxl_id,
            ADDR_TORQUE_ENABLE,
        )
        return {
            "id": self.dxl_id,
            "present_position": present,
            "present_comm_result": present_comm,
            "present_device_error": present_error,
            "goal_position": goal,
            "goal_comm_result": goal_comm,
            "goal_device_error": goal_error,
            "torque_enabled": torque,
            "torque_comm_result": torque_comm,
            "torque_device_error": torque_error,
            "calibrated_open": self.pos_open,
            "calibrated_close": self.pos_close,
            "last_commanded_position": self._last_commanded_pos,
        }

    def close(self):
        self.set_open_ratio(0.0)

    def open(self):
        self.set_open_ratio(1.0)

    def stop(self):
        self.packet.write1ByteTxRx(
            self.port, self.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )


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
        self._last_ik_warning_time = 0.0

        if urdf_path is None:
            if is_left_arm:
                self.urdf_path = (_HERE / "urdf/piper_description_right.xml").as_posix()
            else:
                self.urdf_path = (_HERE / "urdf/piper_description_left.xml").as_posix()
        else:
            self.urdf_path = urdf_path

        self.robot_config = RobotConfigFactory.get_instance().get_config("piper")
        self.controller_config = ControllerConfigFactory.get_instance().get_config("joint_controller")
        self.robot_config.urdf_path = self.urdf_path
        self.robot_config.joint_vel_max = np.ones(6) * 5.0
        self.controller_config.controller_dt = PIPER_CONTROLLER_DT_S
        self.controller_config.default_kp = np.ones(6) * 2.5
        self.controller_config.default_kd = np.ones(6) * 0.2
        self.controller_config.gravity_compensation = True
        self.controller_config.interpolation_method = "linear"

        self.piper = PiperJointController(
            self.robot_config, self.controller_config, self.can_port
        )

        if use_gripper:
            if is_left_arm:
                self.gripper = DynamixelGripper(dxl_id=DXL_ID_LEFT, inverted=False)
            else:
                self.gripper = DynamixelGripper(dxl_id=DXL_ID_RIGHT, inverted=False)
        else:
            self.gripper = None

        if is_left_arm:
            joint_names = [f"right_arm_joint{i}" for i in range(1, 7)]
            ee_frame = "right_arm_ee"
            self.home_q = physical_home_q("left")
        else:
            joint_names = [f"left_arm_joint{i}" for i in range(1, 7)]
            ee_frame = "left_arm_ee"
            self.home_q = physical_home_q("right")

        self.ik_solver = SingleArmIK(
            mjcf_path,
            solver_dt=solver_dt,
            joint_names=joint_names,
            ee_frame=ee_frame,
        )

    def init(self, reset: bool = True):
        if reset:
            self.reset()
        q = self.piper.get_joint_state().pos
        self.ik_solver.init(q)
        if not reset:
            # PiperJointController starts in damping mode (kp=0). Its normal
            # reset_to_home() path enables position gains, but that path also
            # moves the arm. The controller already latched the measured joint
            # state as its fixed command during construction, so enabling the
            # configured gains here holds the current pose without a home move.
            self.piper.set_gain(Gain(
                self.controller_config.default_kp,
                self.controller_config.default_kd,
            ))

    def reset(self):
        self.piper.reset_to_home()
        time.sleep(1.0)
        if self.gripper is not None:
            self.gripper.open()

    def machine_zero(self):
        """Return to Piper's all-zero joint pose and resync Cartesian IK.

        Piper calls this pose ``home`` internally, but it is the folded
        mechanical-zero pose (q = 0), not this repository's upright
        manipulation home stored in ``self.home_q``.  ``reset()`` is also used
        before the IK solver is initialized, while this method is the explicit
        runtime operation exposed to inference/teleoperation clients.
        """
        self.reset()
        q = np.asarray(self.piper.get_joint_state().pos, dtype=float).copy()
        self.ik_solver.update_configuration(q)
        return q

    def home(self, gripper_target: float = 1.0):
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = self.home_q
        cmd.timestamp = self.piper.get_timestamp() + 1.0
        self.piper.set_joint_cmd(cmd)
        if self.gripper is not None:
            self.gripper.set_open_ratio(gripper_target)
        time.sleep(2.0)

    def set_joint_target(self, joint_target, gripper_target=None, preview_time=0.1):
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = joint_target
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        self.piper.set_joint_cmd(cmd)
        if gripper_target is not None and self.gripper is not None:
            self.gripper.set_open_ratio(gripper_target)

    def set_ee_target(self, ee_target, gripper_target=None, preview_time=0.01):
        # Solve from measured joints, not from the previous requested target.
        # This keeps repeated small Cartesian corrections on the same IK branch.
        current_q = np.asarray(self.get_joint_positions(), dtype=float)
        self.ik_solver.update_configuration(current_q)
        qd, is_solved = self.ik_solver.solve_ik(ee_target, max_iter=30)
        continuous, delta = joint_target_is_continuous(current_q, qd)
        if not continuous:
            print(
                f"[IK] discontinuous target rejected: "
                f"max_delta={np.max(np.abs(delta)):.3f}rad",
                flush=True,
            )
            return False
        # Interactive teleoperation historically streamed the continuous
        # best-effort iterate even when Mink had not yet reached its strict
        # 1 mm / 0.001 rad convergence threshold.  Rejecting every such
        # iterate made a healthy arm appear disconnected: the controller
        # received no joint command at all.  Preserve the established teleop
        # behavior while retaining the branch-jump rejection above.
        now = time.monotonic()
        if not is_solved and now - self._last_ik_warning_time >= 1.0:
            print(
                "[IK] target not fully converged; streaming continuous "
                "best-effort target",
                flush=True,
            )
            self._last_ik_warning_time = now
        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = qd
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        self.piper.set_joint_cmd(cmd)
        if gripper_target is not None and self.gripper is not None:
            self.gripper.set_open_ratio(gripper_target)
        return True

    def set_teleop_ee_target(
        self,
        ee_target,
        gripper_target=None,
        preview_time=0.1,
        max_joint_step_rad=TELEOP_DEFAULT_MAX_JOINT_STEP_RAD,
    ):
        """Stream a branch-continuous incremental Cartesian teleop command.

        A hand controller can move farther during a delayed frame than a
        single arm command should.  Solving a distant pose to convergence and
        then rejecting the remote IK solution causes a permanent stop: every
        following frame repeats the same rejection.  Teleoperation instead
        takes one best-effort IK iteration from measured joints and uniformly
        bounds that joint increment.  Repeated 30 Hz calls converge toward the
        live controller pose without ever issuing a branch jump.

        The ordinary ``set_ee_target`` path remains fail-closed for autonomous
        commands; this incremental behavior is deliberately teleop-specific.
        """
        current_q = np.asarray(self.get_joint_positions(), dtype=float)
        self.ik_solver.update_configuration(current_q)
        qd, _ = self.ik_solver.solve_ik(ee_target, max_iter=1)
        qd = np.asarray(qd, dtype=float)
        delta = qd - current_q
        if not np.all(np.isfinite(delta)):
            print("[TELEOP IK] non-finite target rejected", flush=True)
            return False

        max_joint_step_rad = float(max_joint_step_rad)
        if not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be finite and positive")
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_joint_step_rad:
            delta *= max_joint_step_rad / max_delta
            qd = current_q + delta
            now = time.monotonic()
            if now - self._last_ik_warning_time >= 1.0:
                print(
                    "[TELEOP IK] target is ahead; following with bounded "
                    f"joint steps ({max_delta:.3f} -> "
                    f"{max_joint_step_rad:.3f} rad)",
                    flush=True,
                )
                self._last_ik_warning_time = now

        cmd = JointState(self.robot_config.joint_dof)
        cmd.pos = qd
        cmd.timestamp = self.piper.get_timestamp() + preview_time
        self.piper.set_joint_cmd(cmd)
        if gripper_target is not None and self.gripper is not None:
            self.gripper.set_open_ratio(gripper_target)
        return True

    def open_gripper(self):
        if self.gripper is not None:
            self.gripper.open()

    def close_gripper(self):
        if self.gripper is not None:
            self.gripper.close()

    def set_gain(self, kp, kd):
        self.piper.set_gain(Gain(kp, kd))

    def get_joint_positions(self):
        return self.piper.get_joint_state().pos

    def get_joint_torque(self):
        """Return the latest measured joint torques from the Piper controller."""
        return np.asarray(self.piper.get_joint_state().torque, dtype=float).copy()

    def get_ee_pose(self):
        q = self.get_joint_positions()
        self.ik_solver.update_configuration(q)
        return self.ik_solver.forward_kinematics()

    def stop(self):
        if self.gripper is not None:
            self.gripper.stop()

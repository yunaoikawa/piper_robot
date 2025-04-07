import threading
import numpy as np

import can
from can.message import Message

from robot.arm.piper_sdk.can_utils import (
    can_data_to_int32,
    can_data_to_int16,
    can_data_to_uint16,
    can_data_to_uint8,
    float_to_uint,
    POS_MIN,
    POS_MAX,
    VEL_MIN,
    VEL_MAX,
    KP_MIN,
    KP_MAX,
    KD_MIN,
    KD_MAX,
    T_MIN,
    T_MAX,
)

RAD_TO_DEG = 57.295779513082320876798154814105
DEG_TO_RAD = 0.017453292519943295769236907684886

# JOINT_POS_COEFF = [-1, -1, 1, -1, 1, -1]  # TODO: check if needed
TORQUE_COEFF = [1.18125, 1.18125, 1.18125, 0.95844, 0.95844, 0.95844]


class PiperHardwareStation:
    def __init__(self, channel="can0") -> None:
        try:
            self.channel_name = channel
            self.bus = can.interface.Bus(channel=channel, bustype="socketcan")
        except can.CanError as e:
            print(f"Failed to initialize CAN bus: {e}")
            exit(1)

        self.pos = np.zeros((6, 1), dtype=np.float64)
        self.vel = np.zeros((6, 1), dtype=np.float64)
        self.torque = np.zeros((6, 1), dtype=np.float64)
        self.foc_status = np.zeros((6, 1), dtype=np.int8)

        self._state_lock = threading.Lock()
        self._arm_can_stop_event = threading.Event()

    def start(self):
        self._arm_can_stop_event.clear()

        def read_can():
            while not self._arm_can_stop_event.is_set():
                self.read()

        self.can_th = threading.Thread(target=read_can, daemon=True)
        self.can_th.start()

    def stop(self, timeout=0.1):
        self._arm_can_stop_event.set()
        if hasattr(self, "can_th") and self.can_th.is_alive():
            self.can_th.join(timeout=timeout)
            if self.can_th.is_alive():
                print("[WARN] can connection couldn't be stopped")
        self.bus.shutdown()

    def get_pos(self):
        with self._state_lock:
            return self.pos.copy()

    def read(self):
        if self.bus.state == can.BusState.ACTIVE:
            self.rx_message = self.bus.recv()
            if self.rx_message:
                self.can_receive_frame(self.rx_message)
        else:
            print("CAN bus is not OK, skipping message read")

    def send(self, can_id, data):
        msg = Message(channel=self.channel_name, arbitration_id=can_id, data=data, dlc=8, is_extended_id=False)
        if self.bus.state == can.BusState.ACTIVE:
            try:
                self.bus.send(msg)
            except can.CanError:
                print(can.CanError, "message not sent")

    def can_receive_frame(self, rx_message: Message):
        can_id = rx_message.arbitration_id
        can_data = rx_message.data

        if can_id >= 0x251 and can_id <= 0x256:
            with self._state_lock:
                motor_id = can_id - 0x251
                self.vel[motor_id] = DEG_TO_RAD * (can_data_to_int16(can_data[0:2]) / 1000.0)
                self.torque[motor_id] = TORQUE_COEFF[motor_id] * (can_data_to_uint16(can_data[2:4]) / 1000.0)
            # this pos is always 0, so take it from other CAN messages
        elif can_id >= 0x2A5 and can_id <= 0x2A7:
            with self._state_lock:
                motor_id = 2 * (can_id - 0x2A5)
                self.pos[motor_id] = DEG_TO_RAD * (can_data_to_int32(can_data[0:4]) / 1000.0)
                self.pos[motor_id + 1] = DEG_TO_RAD * (can_data_to_int32(can_data[4:8]) / 1000.0)
        elif can_id >= 0x261 and can_id <= 0x266:
            with self._state_lock:
                motor_id = can_id - 0x261
                self.foc_status[motor_id] = can_data_to_uint8(can_data[5:6])

    def enable_motors(self):
        self.send(0x471, [0x07, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    def disable_motors(self):
        self.send(0x471, [0x07, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    def set_ctrl_mode(self, ctrl_mode: int, move_mode: int, speed_rate: int, mit_mode: int):
        self.send(0x151, [ctrl_mode, move_mode, speed_rate, mit_mode, 0x00, 0x00, 0x00, 0x00])

    def send_motor_cmd(self, motor_id: int, pos: float, vel: float, kp: float, kd: float, t: float):
        if not (0 <= motor_id <= 5):
            print(f"Invalid motor ID: {motor_id}")
            return

        data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

        pos_ref = float_to_uint(pos, POS_MIN, POS_MAX, 16)
        vel_ref = float_to_uint(vel, VEL_MIN, VEL_MAX, 12)
        kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12)
        kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12)
        t_ref = float_to_uint(t, T_MIN, T_MAX, 8)

        data[0] = (pos_ref >> 8) & 0xFF # High byte
        data[1] = pos_ref & 0xFF        # Low byte
        data[2] = (vel_ref >> 4) & 0xFF
        data[3] = (((vel_ref & 0xF) << 4) & 0xF0) | ((kp_int >> 8) & 0x0F)
        data[4] = kp_int & 0xFF
        data[5] = (kd_int >> 4) & 0xFF
        data[6] = (((kd_int & 0xF) << 4) & 0xF0) | ((t_ref >> 4) & 0x0F)

        crc = (data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] ^ data[5] ^ data[6]) & 0x0F
        data[7] = ((t_ref << 4) & 0xF0) | crc

        self.send(0x15A + motor_id, data)

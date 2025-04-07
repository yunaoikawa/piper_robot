import threading
import numpy as np

import can
from can.message import Message

from robot.arm.piper_sdk.can_utils import can_data_to_int32, can_data_to_int16, can_data_to_uint16, can_data_to_uint8

RAD_TO_DEG = 57.295779513082320876798154814105
DEG_TO_RAD = 0.017453292519943295769236907684886

JOINT_POS_COEFF = [-1, -1, 1, -1, 1, -1]  # TODO: check if needed
TORQUE_COEFF = [1.18125, 1.18125, 1.18125, 0.95844, 0.95844, 0.95844]


class PiperHardwareStation:
    def __init__(self, channel="can0") -> None:
        try:
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
                self.pos[motor_id] = DEG_TO_RAD * (can_data_to_int32(can_data[0:4]) / 1000.0) * JOINT_POS_COEFF[motor_id]
                self.pos[motor_id + 1] = (
                    DEG_TO_RAD * (can_data_to_int32(can_data[4:8]) / 1000.0) * JOINT_POS_COEFF[motor_id + 1]
                )
        elif can_id >= 0x261 and can_id <= 0x266:
            with self._state_lock:
                motor_id = can_id - 0x261
                self.foc_status[motor_id] = can_data_to_uint8(can_data[5:6])

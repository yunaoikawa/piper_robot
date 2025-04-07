from dataclasses import dataclass
import threading

import can
from can.message import Message

from robot.arm.piper_sdk.can_utils import can_data_to_int32, can_data_to_int16, can_data_to_uint16, can_data_to_uint8

RAD_TO_DEG = 57.295779513082320876798154814105
DEG_TO_RAD = 0.017453292519943295769236907684886

JOINT_POS_COEFF = [-1, -1, 1, -1, 1, -1]  # TODO: check if needed


@dataclass
class MotorState:
    pos: float = 0.0
    vel: float = 0.0
    current: float = 0.0
    foc_status: int = 0


class HardwareStation:
    def __init__(self, channel="can0") -> None:
        try:
            self.bus = can.interface.Bus(channel=channel, bustype="socketcan")
        except can.CanError as e:
            print(f"Failed to initialize CAN bus: {e}")
            exit(1)

        self.state: list[MotorState] = [MotorState() for _ in range(6)]
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
            motor_id = can_id - 0x251
            self.state[motor_id].vel = DEG_TO_RAD * (can_data_to_int16(can_data[0:2]) / 1000.0)
            self.state[motor_id].current = can_data_to_uint16(can_data[2:4]) / 1000.0
            # this pos is always 0, so take it from other CAN messages
        elif can_id >= 0x2A5 and can_id <= 0x2A7:
            motor_id = 2 * (can_id - 0x2A5)
            self.state[motor_id].pos = (
                DEG_TO_RAD * (can_data_to_int32(can_data[0:4]) / 1000.0) * JOINT_POS_COEFF[motor_id]
            )
            self.state[motor_id + 1].pos = (
                DEG_TO_RAD * (can_data_to_int32(can_data[4:8]) / 1000.0) * JOINT_POS_COEFF[motor_id + 1]
            )
        elif can_id >= 0x261 and can_id <= 0x266:
            motor_id = can_id - 0x261
            self.state[motor_id].foc_status = can_data_to_uint8(can_data[5:6])

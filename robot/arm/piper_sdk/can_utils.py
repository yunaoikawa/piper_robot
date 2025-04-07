import struct
import ctypes

KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = -5.0
KD_MAX = 5.0
POS_MIN = -12.5
POS_MAX = 12.5
VEL_MIN = -18.0
VEL_MAX = 18.0
T_MIN = -18.0
T_MAX = 18.0


def can_data_to_uint8(data: bytearray):
    value = int.from_bytes(data, byteorder="big")
    if not (0 <= value <= 255):
        print("Input value exceeds the range [0, 255]")
    value &= 0xFF
    return value


def can_data_to_int32(data: bytearray):
    value = int.from_bytes(data, byteorder="big")
    if not (0 <= value <= 4294967295):
        print("Input value exceeds the range [0, 4294967295]")
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        value -= 0x100000000
    return value


def can_data_to_int16(data: bytearray):
    value = int.from_bytes(data, byteorder="big")
    if not (0 <= value <= 65535):
        print("Input value exceeds the range [0, 65535]")
    value &= 0xFFFF
    if value & 0x8000:
        value -= 0x10000
    return value


def can_data_to_uint16(data: bytearray):
    value = int.from_bytes(data, byteorder="big")
    if not (0 <= value <= 65535):
        print("Input value exceeds the range [0, 65535]")
    value &= 0xFFFF
    return value


## Convert to bytes


def uint16_to_bytes(value: int):
    if not (0 <= value <= 65535):
        raise OverflowError("Input value exceeds the range [0, 65535]")
    return list(struct.unpack("BB", struct.pack(">H", value)))


def int16_to_bytes(value: int):
    if not (-32768 <= value <= 32767):
        raise OverflowError("Input value exceeds the range [-32768, 32767]")
    value = ctypes.c_int16(value).value
    return list(struct.unpack("BB", struct.pack(">h", value)))


def uint8_to_bytes(value: int):
    if not (0 <= value <= 255):
        raise OverflowError("Input value exceeds the range [0, 255]")
    return list(struct.unpack("B", struct.pack(">B", value)))


def float_to_uint(x: float, x_min: float, x_max: float, bits: int):
    span: float = x_max - x_min
    offset: float = x_min
    return int((x - offset) * (float((1 << bits) - 1)) / span)

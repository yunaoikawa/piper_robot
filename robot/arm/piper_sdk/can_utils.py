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

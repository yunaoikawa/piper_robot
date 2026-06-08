#!/usr/bin/env python3
"""Quick left gripper calibration. Run before cone_e when values change.

Usage: python robot/arm/calibrate_left.py
"""
import json, struct, time
from pathlib import Path
from dynamixel_sdk import PortHandler, PacketHandler

PORT = "/dev/ttyUSB0"
CAL_FILE = Path(__file__).parent / "left_gripper_cal.json"

def rp(pkt, port):
    raw, _, _ = pkt.read4ByteTxRx(port, 2, 132)
    return struct.unpack("i", struct.pack("I", raw))[0]

def wp(pkt, port, pos):
    pkt.write4ByteTxRx(port, 2, 116, struct.unpack("I", struct.pack("i", pos))[0])

port = PortHandler(PORT)
port.openPort()
port.setBaudRate(115200)
pkt = PacketHandler(2.0)

pkt.write1ByteTxRx(port, 2, 64, 0)
pkt.write1ByteTxRx(port, 2, 11, 4)
pkt.write2ByteTxRx(port, 2, 38, 300)
pkt.write1ByteTxRx(port, 2, 64, 1)

cur = rp(pkt, port)
print(f"Current pos: {cur}")
print(f"Testing offsets from {cur}. Answer: o=open, c=closed, m=middle")
print()

open_val = close_val = None
for offset in [-4000, -2000, 0, 2000, 4000, 6000, 8000]:
    target = cur + offset
    wp(pkt, port, target)
    time.sleep(1.5)
    actual = rp(pkt, port)
    print(f"  offset={offset:+6d}  pos={actual}")
    ans = input("  o/c/m: ").strip()
    if ans == "o" and open_val is None:
        open_val = actual
    if ans == "c" and close_val is None:
        close_val = actual

if open_val is None or close_val is None:
    print("\nNeed at least one 'o' and one 'c'. Try wider range.")
    port.closePort()
    exit(1)

# Drive back to open
wp(pkt, port, open_val)
time.sleep(1)

cal = {"open": open_val, "close": close_val}
CAL_FILE.write_text(json.dumps(cal))
print(f"\nSaved to {CAL_FILE}: {cal}")
print("Now run: python -m robot.cone_e")
port.closePort()
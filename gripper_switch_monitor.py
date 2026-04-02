#!/usr/bin/env python3
"""
Real-time gripper switch monitor.

Usage:
    python gripper_switch_monitor.py              # auto-detect
    python gripper_switch_monitor.py --port /dev/ttyUSB2
    python gripper_switch_monitor.py --port /dev/ttyACM0
"""

import argparse
import glob
import sys
import time

import serial


def find_arduino_port():
    """Auto-detect Arduino serial port."""
    candidates = (
        glob.glob("/dev/ttyACM*") +
        glob.glob("/dev/ttyUSB*")
    )
    # Filter out known devices (Dynamixel on ttyUSB1, CAN adapters)
    skip = {"/dev/ttyUSB0", "/dev/ttyUSB1"}
    for port in sorted(candidates):
        if port not in skip:
            return port
    # Fallback: try ACM ports first
    for port in sorted(candidates):
        if "ACM" in port:
            return port
    return None


def main():
    parser = argparse.ArgumentParser(description="Gripper switch monitor")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if not specified)")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    port = args.port or find_arduino_port()
    if port is None:
        print("ERROR: No Arduino found. Available ports:")
        for p in glob.glob("/dev/tty[AU]*"):
            print(f"  {p}")
        sys.exit(1)

    print(f"Connecting to {port} at {args.baud} baud...")

    try:
        ser = serial.Serial(port, args.baud, timeout=0.1)
        time.sleep(2)  # Arduino resets on serial connect
        print("Connected! Reading switch state...\n")
        print("  1 = GRIPPING (switch pressed)")
        print("  0 = OPEN (switch released)")
        print("  Press Ctrl+C to exit\n")

        prev_state = None

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            try:
                state = int(line)
            except ValueError:
                continue

            # Visual indicator
            if state == 1:
                bar = "█" * 30
                label = "GRIPPING"
                color = "\033[92m"  # green
            else:
                bar = "░" * 30
                label = "OPEN    "
                color = "\033[91m"  # red

            reset = "\033[0m"
            print(f"\r  {color}{bar}  {label}{reset}", end="", flush=True)

            # Log state changes
            if prev_state is not None and state != prev_state:
                change = "GRABBED" if state == 1 else "RELEASED"
                print(f"\n  >>> {change} at {time.strftime('%H:%M:%S')}")

            prev_state = state

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        if "ser" in locals():
            ser.close()


if __name__ == "__main__":
    main()

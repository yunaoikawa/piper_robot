#!/usr/bin/env python3
"""Replay a recorded teleoperation trajectory on the robot step by step."""

import argparse
import time

import h5py
import mink
import numpy as np
from loop_rate_limiters import RateLimiter

from robot.rpc import RPCClient


def load_trajectory(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        left_ee_pos = f["left_ee_pos"][:]
        left_ee_quat = f["left_ee_quat"][:]  # wxyz
        left_gripper = f["left_gripper"][:]
        right_ee_pos = f["right_ee_pos"][:]
        right_ee_quat = f["right_ee_quat"][:]  # wxyz
        right_gripper = f["right_gripper"][:]
        timestamps = f["timestamps"][:]
        control_freq = float(f.attrs.get("control_frequency_hz", 30))
        num_samples = int(f.attrs.get("num_samples", len(timestamps)))

    print(f"Loaded trajectory: {num_samples} steps at {control_freq} Hz "
          f"({num_samples / control_freq:.1f}s)")
    return {
        "left_ee_pos": left_ee_pos,
        "left_ee_quat": left_ee_quat,
        "left_gripper": left_gripper,
        "right_ee_pos": right_ee_pos,
        "right_ee_quat": right_ee_quat,
        "right_gripper": right_gripper,
        "timestamps": timestamps,
        "control_freq": control_freq,
        "num_samples": num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay a teleoperation trajectory on the robot")
    parser.add_argument("hdf5", type=str, help="Path to trajectory HDF5 file")
    parser.add_argument("--host", type=str, default="localhost", help="Robot RPC host")
    parser.add_argument("--port", type=int, default=8081, help="Robot RPC port")
    parser.add_argument("--hz", type=float, default=None,
                        help="Replay frequency in Hz (default: use trajectory frequency)")
    parser.add_argument("--preview-time", type=float, default=0.05,
                        help="Preview time for arm target (default: 0.05s)")
    parser.add_argument("--no-home", action="store_true",
                        help="Skip homing the robot before replay")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from this step index (default: 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="Stop at this step index (default: end of trajectory)")
    args = parser.parse_args()

    traj = load_trajectory(args.hdf5)
    hz = args.hz if args.hz is not None else traj["control_freq"]
    start_idx = args.start
    end_idx = args.end if args.end is not None else traj["num_samples"]
    num_steps = end_idx - start_idx

    print(f"\nConnecting to robot at {args.host}:{args.port} ...")
    robot = RPCClient(args.host, args.port)
    robot.init()
    print("Connected.")

    if not args.no_home:
        print("\nHoming both arms ...")
        robot.home_left_arm()
        robot.home_right_arm()
        print("Homing complete.")

    print(f"\nReady to replay steps [{start_idx}, {end_idx}) at {hz:.1f} Hz.")
    input("Press Enter to start replay (Ctrl+C to abort) ...")

    rate = RateLimiter(hz)
    print(f"\nStarting replay of {num_steps} steps ...")
    t_start = time.time()

    try:
        for i in range(start_idx, end_idx):
            step = i - start_idx

            left_pose = mink.SE3(np.concatenate([
                traj["left_ee_quat"][i],   # wxyz
                traj["left_ee_pos"][i],    # xyz
            ]))
            right_pose = mink.SE3(np.concatenate([
                traj["right_ee_quat"][i],  # wxyz
                traj["right_ee_pos"][i],   # xyz
            ]))

            robot.set_left_ee_target(
                ee_target=left_pose,
                gripper_target=float(traj["left_gripper"][i]),
                preview_time=args.preview_time,
            )
            robot.set_right_ee_target(
                ee_target=right_pose,
                gripper_target=float(traj["right_gripper"][i]),
                preview_time=args.preview_time,
            )

            if step % int(hz) == 0:
                elapsed = time.time() - t_start
                print(f"  Step {step:4d}/{num_steps}  elapsed={elapsed:.1f}s  "
                      f"left_pos={traj['left_ee_pos'][i]}  "
                      f"lg={traj['left_gripper'][i]:.2f}  rg={traj['right_gripper'][i]:.2f}")

            rate.sleep()

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")

    print(f"\nReplay finished. {min(step + 1, num_steps)}/{num_steps} steps executed.")


if __name__ == "__main__":
    main()

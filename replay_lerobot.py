#!/usr/bin/env python3
"""Replay a LeRobot-format dataset trajectory on the robot step by step."""

import argparse
import time
from pathlib import Path

import mink
import numpy as np
import pandas as pd
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from robot.rpc import RPCClient


def r6d_to_quat_wxyz(r6d):
    """Convert 6D rotation representation back to wxyz quaternion.

    r6d = [a1_x, a1_y, a1_z, a2_x, a2_y, a2_z] where a1, a2 are the first
    two columns of the rotation matrix (as stored by quat_to_r6 in controller.py).
    """
    a1 = r6d[0:3]
    a2 = r6d[3:6]
    # Gram-Schmidt orthonormalization
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    rot_mat = np.column_stack([b1, b2, b3])
    # scipy uses scalar-last (xyzw); mink wants scalar-first (wxyz)
    xyzw = R.from_matrix(rot_mat).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def action_to_poses(action):
    """Split a 20-dim action vector into (left_se3, left_gripper, right_se3, right_gripper)."""
    action = np.asarray(action, dtype=float)
    left_pos = action[0:3]
    left_r6d = action[3:9]
    left_gripper = float(action[9])
    right_pos = action[10:13]
    right_r6d = action[13:19]
    right_gripper = float(action[19])

    left_quat = r6d_to_quat_wxyz(left_r6d)
    right_quat = r6d_to_quat_wxyz(right_r6d)

    left_se3 = mink.SE3(np.concatenate([left_quat, left_pos]))
    right_se3 = mink.SE3(np.concatenate([right_quat, right_pos]))

    return left_se3, left_gripper, right_se3, right_gripper


def load_episode(dataset_dir, episode_index=0):
    dataset_dir = Path(dataset_dir)

    # Find the parquet chunk that contains this episode
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    assert parquet_files, f"No parquet files found under {data_dir}"

    frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        ep_frames = df[df["episode_index"] == episode_index]
        if not ep_frames.empty:
            frames.append(ep_frames)

    assert frames, f"Episode {episode_index} not found in dataset"
    df = pd.concat(frames).sort_values("frame_index").reset_index(drop=True)
    print(f"Loaded episode {episode_index}: {len(df)} frames")
    return df


def main():
    parser = argparse.ArgumentParser(description="Replay a LeRobot dataset episode on the robot")
    parser.add_argument("dataset_dir", type=str, help="Path to LeRobot dataset root directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (default: 0)")
    parser.add_argument("--host", type=str, default="localhost", help="Robot RPC host")
    parser.add_argument("--port", type=int, default=8081, help="Robot RPC port")
    parser.add_argument("--hz", type=float, default=30.0, help="Replay frequency in Hz (default: 30)")
    parser.add_argument("--preview-time", type=float, default=0.05,
                        help="Preview time for arm target (default: 0.05s)")
    parser.add_argument("--no-home", action="store_true", help="Skip homing before replay")
    parser.add_argument("--start", type=int, default=0, help="Start from this frame index")
    parser.add_argument("--end", type=int, default=None, help="Stop at this frame index")
    parser.add_argument("--use-state", action="store_true",
                        help="Replay observation.state instead of action column")
    args = parser.parse_args()

    df = load_episode(args.dataset_dir, args.episode)
    col = "observation.state" if args.use_state else "action"
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(df)
    frames = df.iloc[start_idx:end_idx]
    num_steps = len(frames)
    duration = num_steps / args.hz

    print(f"Replaying column '{col}', steps [{start_idx}, {end_idx}) "
          f"at {args.hz:.1f} Hz (~{duration:.1f}s)")

    print(f"\nConnecting to robot at {args.host}:{args.port} ...")
    robot = RPCClient(args.host, args.port)
    robot.init()
    print("Connected.")

    if not args.no_home:
        print("\nHoming both arms ...")
        robot.home_left_arm()
        robot.home_right_arm()
        print("Homing complete.")

    input(f"\nPress Enter to start replay of {num_steps} steps (Ctrl+C to abort) ...")

    rate = RateLimiter(args.hz)
    t_start = time.time()
    step = 0

    try:
        for _, row in frames.iterrows():
            action_vec = np.asarray(row[col], dtype=float)
            left_se3, left_gripper, right_se3, right_gripper = action_to_poses(action_vec)

            robot.set_left_ee_target(
                ee_target=left_se3,
                gripper_target=left_gripper,
                preview_time=args.preview_time,
            )
            robot.set_right_ee_target(
                ee_target=right_se3,
                gripper_target=right_gripper,
                preview_time=args.preview_time,
            )

            if step % int(args.hz) == 0:
                elapsed = time.time() - t_start
                pos = action_vec[0:3]
                print(f"  Step {step:4d}/{num_steps}  t={elapsed:.1f}s  "
                      f"left_pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]  "
                      f"lg={left_gripper:.2f}  rg={right_gripper:.2f}")

            step += 1
            rate.sleep()

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")

    print(f"\nReplay finished. {step}/{num_steps} steps executed.")


if __name__ == "__main__":
    main()

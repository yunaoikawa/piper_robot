#!/usr/bin/env python3
"""
End-to-end test for the hpc_inference_pi05 pipeline without torch or ZMQ.

Because the mock policy is local, the control loop calls forward() / ActionBuffer /
postprocess_action_chunk / robot directly — no sockets needed.

What this tests
---------------
- ActionBuffer chunk management (empty_overwrite / pop_action / is_empty trigger)
- postprocess_action_chunk converting (chunk_size, 16) → action dicts
- Absolute SE3 pose application on the real robot
"""

import argparse
import time
from pathlib import Path

import mink
import numpy as np
import pandas as pd
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from robot.rpc import RPCClient


# ---------------------------------------------------------------------------
# ActionBuffer  (from hpc_inference_pi05.py, no changes)
# ---------------------------------------------------------------------------

class ActionBuffer:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.actions = []
        self.update_count = 0
        self.total_pops = 0
        self.last_update_time = None

    def empty_overwrite(self, action_chunk):
        if not self.actions:
            self.actions = list(action_chunk)
            self.last_update_time = time.time()
            self.update_count += 1

    def pop_action(self):
        if not self.actions:
            return None
        action = self.actions.pop(0)
        self.total_pops += 1
        age = time.time() - self.last_update_time if self.last_update_time else 0
        action["buffer_remaining"] = len(self.actions)
        action["buffer_age"] = age
        action["is_stale"] = self.total_pops > self.update_count * self.chunk_size
        return action

    @property
    def is_empty(self):
        return len(self.actions) == 0


# ---------------------------------------------------------------------------
# postprocess_action_chunk  (from hpc_inference_pi05.py, torch check removed)
# ---------------------------------------------------------------------------

def postprocess_action_chunk(output: np.ndarray, is_delta_action: bool = False) -> list:
    """
    Convert (chunk_size, 16) numpy array → list of action dicts.

    16-dim layout (Pi05InferencePolicy.forward() output):
        [left_quat_wxyz(4), left_pos(3), right_quat_wxyz(4), right_pos(3),
         left_gripper(1), right_gripper(1)]
    """
    if output.ndim == 3:
        output = output.squeeze(0)          # (1, T, 16) → (T, 16)

    chunk = []
    for t in range(output.shape[0]):
        a = output[t]
        if is_delta_action:
            d = {"left_delta_pose": a[0:7], "right_delta_pose": a[7:14]}
        else:
            d = {"left_ee_pose": a[0:7], "right_ee_pose": a[7:14]}
        d["left_gripper"] = float(a[14])
        d["right_gripper"] = float(a[15])
        d["timestamp"] = time.time()
        d["chunk_index"] = t
        chunk.append(d)
    return chunk


# ---------------------------------------------------------------------------
# Mock policy
# ---------------------------------------------------------------------------

class MockTrajectoryPolicy:
    """
    Reads LeRobot dataset frames and returns action chunks exactly as the real
    Pi05InferencePolicy.forward() would: np.ndarray shape (chunk_size, 16).
    """

    def __init__(self, df: pd.DataFrame, chunk_size: int, col: str = "action"):
        self.df = df
        self.chunk_size = chunk_size
        self.col = col
        self.cursor = 0
        self.is_done = False

    @staticmethod
    def _r6d_to_quat_wxyz(r6d: np.ndarray) -> np.ndarray:
        """(N, 6) → (N, 4) wxyz via Gram-Schmidt."""
        b1 = r6d[:, :3] / np.linalg.norm(r6d[:, :3], axis=1, keepdims=True)
        b2 = r6d[:, 3:6] - np.sum(b1 * r6d[:, 3:6], axis=1, keepdims=True) * b1
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        b3 = np.cross(b1, b2)
        mat = np.stack([b1, b2, b3], axis=-1)   # (N, 3, 3)
        return R.from_matrix(mat).as_quat(scalar_first=True)

    def forward(self, _observation=None) -> np.ndarray | None:
        """Return next (chunk_size, 16) block, or None when exhausted."""
        if self.is_done:
            return None

        start = self.cursor
        if start >= len(self.df):
            self.is_done = True
            return None

        end = min(start + self.chunk_size, len(self.df))
        rows = np.stack(
            [np.asarray(r[self.col], dtype=np.float64)
             for _, r in self.df.iloc[start:end].iterrows()]
        )
        self.cursor = end
        if self.cursor >= len(self.df):
            self.is_done = True

        # Pad last chunk
        if len(rows) < self.chunk_size:
            rows = np.concatenate(
                [rows, np.tile(rows[-1:], (self.chunk_size - len(rows), 1))]
            )

        # LeRobot 20-dim → 16-dim
        lq = self._r6d_to_quat_wxyz(rows[:, 3:9])
        rq = self._r6d_to_quat_wxyz(rows[:, 13:19])
        return np.concatenate(
            [lq, rows[:, 0:3], rq, rows[:, 10:13], rows[:, 9:10], rows[:, 19:20]],
            axis=1,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_episode(dataset_dir: str, episode_index: int = 0) -> pd.DataFrame:
    data_dir = Path(dataset_dir) / "data"
    files = sorted(data_dir.glob("chunk-*/*.parquet"))
    assert files, f"No parquet files under {data_dir}"
    frames = []
    for pf in files:
        df = pd.read_parquet(pf)
        ep = df[df["episode_index"] == episode_index]
        if not ep.empty:
            frames.append(ep)
    assert frames, f"Episode {episode_index} not found"
    df = pd.concat(frames).sort_values("frame_index").reset_index(drop=True)
    print(f"Loaded episode {episode_index}: {len(df)} frames")
    return df


def apply_action(robot: RPCClient, action: dict, preview_time: float):
    if "left_ee_pose" in action and action["left_ee_pose"] is not None:
        robot.set_left_ee_target(
            ee_target=mink.SE3(np.asarray(action["left_ee_pose"], dtype=float)),
            gripper_target=float(action.get("left_gripper", 0.5)),
            preview_time=preview_time,
        )
    if "right_ee_pose" in action and action["right_ee_pose"] is not None:
        robot.set_right_ee_target(
            ee_target=mink.SE3(np.asarray(action["right_ee_pose"], dtype=float)),
            gripper_target=float(action.get("right_gripper", 0.5)),
            preview_time=preview_time,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test hpc_inference_pi05 pipeline locally (no torch, no ZMQ)"
    )
    parser.add_argument("dataset_dir")
    parser.add_argument("--episode",      type=int,   default=0)
    parser.add_argument("--chunk-size",   type=int,   default=16)
    parser.add_argument("--robot-host",   type=str,   default="localhost")
    parser.add_argument("--robot-port",   type=int,   default=8081)
    parser.add_argument("--hz",           type=float, default=30.0)
    parser.add_argument("--preview-time", type=float, default=0.05)
    parser.add_argument("--no-home",      action="store_true")
    parser.add_argument("--col",          type=str,   default="action",
                        choices=["action", "observation.state"])
    args = parser.parse_args()

    df    = load_episode(args.dataset_dir, args.episode)
    total = len(df)

    mock   = MockTrajectoryPolicy(df, chunk_size=args.chunk_size, col=args.col)
    buffer = ActionBuffer(args.chunk_size)

    print(f"\nConnecting to robot at {args.robot_host}:{args.robot_port} ...")
    robot = RPCClient(args.robot_host, args.robot_port)
    robot.init()
    print("Connected.")

    if not args.no_home:
        print("Homing both arms ...")
        robot.home_left_arm()
        robot.home_right_arm()
        print("Homing complete.")

    print(f"\nchunk-size={args.chunk_size}  hz={args.hz}  "
          f"~{total/args.hz:.1f}s  col='{args.col}'")
    input("Press Enter to start (Ctrl+C to abort) ...")

    rate      = RateLimiter(args.hz)
    applied   = 0
    inferences = 0
    t_start   = time.time()

    try:
        while applied < total:
            # Mirror inference_loop trigger: run forward() only when buffer empty
            if buffer.is_empty and not mock.is_done:
                chunk = mock.forward()
                if chunk is not None:
                    action_chunk = postprocess_action_chunk(chunk, is_delta_action=False)
                    buffer.empty_overwrite(action_chunk)
                    inferences += 1
                    print(f"  [inference #{inferences}]  "
                          f"cursor={mock.cursor}/{total}  "
                          f"chunk_size={len(action_chunk)}", flush=True)

            action = buffer.pop_action()
            if action is None:
                # Buffer still empty (first inference not done yet — shouldn't happen
                # in this sync setup, but guard just in case)
                rate.sleep()
                continue

            apply_action(robot, action, args.preview_time)
            applied += 1

            if applied % int(args.hz) == 0:
                elapsed = time.time() - t_start
                print(f"  step={applied:4d}/{total}  t={elapsed:.1f}s  "
                      f"buf={action.get('buffer_remaining', '?')}  "
                      f"inferences={inferences}", flush=True)

            rate.sleep()

    except KeyboardInterrupt:
        print("\nInterrupted.")

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Done: {applied}/{total} steps in {elapsed:.1f}s")
    print(f"Inferences run: {inferences}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

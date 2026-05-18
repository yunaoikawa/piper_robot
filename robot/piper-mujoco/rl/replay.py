"""
保存済み軌跡を MuJoCo viewer で再生する。

Usage (from robot/piper-mujoco/):
    mjpython -m rl.replay --traj rl/trajectories/ckpt_01996800_ep01.npz

キー操作:
    Space    : 一時停止 / 再生
    R        : 先頭に戻る
    →        : 1フレーム進む (一時停止中)
    ←        : 1フレーム戻る (一時停止中)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.env.lab_env import XML_PATH

# GLFW キーコード
_KEY_SPACE = 32
_KEY_RIGHT = 262
_KEY_LEFT  = 263
_KEY_R     = 82


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True, type=Path,
                        help="eval.py --record で保存した .npz ファイル")
    parser.add_argument("--speed", default=1.0, type=float,
                        help="再生速度倍率 (1.0=実時間, 2.0=2倍速, 0.5=スロー)")
    args = parser.parse_args()

    # 軌跡を読み込む
    traj = np.load(args.traj)
    qpos_seq = traj["qpos"]          # (n_frames, nq)
    time_seq = traj["time"]          # (n_frames,)
    success  = bool(traj["success"])
    n_frames = len(qpos_seq)

    total_reward = float(traj["rewards"].sum()) if "rewards" in traj else float("nan")

    print(f"Trajectory : {args.traj}")
    print(f"Frames     : {n_frames}")
    print(f"Duration   : {time_seq[-1]:.2f} s")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Success    : {success}")
    print()
    print("Controls: Space=pause/resume  R=restart  ←/→=frame step (paused)")

    # モデルを読み込む（lab-scene.xml）
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data  = mujoco.MjData(model)

    state = {"frame": 0, "paused": False, "step": 0}

    def key_callback(keycode: int) -> None:
        if keycode == _KEY_SPACE:
            state["paused"] = not state["paused"]
        elif keycode == _KEY_R:
            state["frame"] = 0
            state["paused"] = False
        elif keycode == _KEY_RIGHT and state["paused"]:
            state["step"] = 1
        elif keycode == _KEY_LEFT and state["paused"]:
            state["step"] = -1

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            i = state["frame"]

            # フレームのフル状態を復元
            data.qpos[:] = qpos_seq[i]
            mujoco.mj_forward(model, data)
            viewer.sync()

            if state["paused"]:
                if state["step"] != 0:
                    state["frame"] = int(
                        np.clip(i + state["step"], 0, n_frames - 1)
                    )
                    state["step"] = 0
                time.sleep(0.016)
            else:
                # 実時間に合わせてスリープ
                if i + 1 < n_frames:
                    dt = float(time_seq[i + 1] - time_seq[i]) / args.speed
                else:
                    # 最終フレームに到達 → 先頭に戻る
                    time.sleep(1.0)
                    state["frame"] = 0
                    continue

                time.sleep(max(0.0, dt))
                state["frame"] = i + 1

    print("Replay finished.")


if __name__ == "__main__":
    main()

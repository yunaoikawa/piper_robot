"""
MJCFファイルビューワー

Usage:
    python scripts/viewer/view_mjcf.py <filename>
    python scripts/viewer/view_mjcf.py robot.mjcf
    python scripts/viewer/view_mjcf.py mjcf/scene.mjcf

キー操作:
    Space   : 一時停止 / 再開
    →       : 1ステップ進む (停止中のみ)
    ←       : 1ステップ戻る
"""

import argparse
import collections
import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

HISTORY_SIZE = 1000
# GLFW key codes
KEY_SPACE = 32
KEY_RIGHT = 262
KEY_LEFT = 263


def main():
    parser = argparse.ArgumentParser(description="MJCFファイルをMuJoCoビューワーで表示する")
    parser.add_argument(
        "filename",
        help="MJCFファイルのパス。mjcf/ディレクトリ内のファイル名のみでも可 (例: robot.mjcf)",
    )
    args = parser.parse_args()

    filepath = args.filename

    # ファイルが見つからない場合、mjcf/ディレクトリから探す
    if not os.path.isfile(filepath):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(repo_root, "mjcf", filepath)
        if os.path.isfile(candidate):
            filepath = candidate
        else:
            print(f"Error: ファイルが見つかりません: {args.filename}", file=sys.stderr)
            print("mjcf/ ディレクトリ内のファイル一覧:", file=sys.stderr)
            mjcf_dir = os.path.join(repo_root, "mjcf")
            for f in sorted(os.listdir(mjcf_dir)):
                if f.endswith((".mjcf", ".xml")):
                    print(f"  {f}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading: {filepath}")
    print("Space: 停止/再開  ←: 1ステップ戻る  →: 1ステップ進む (停止中)")

    model = mujoco.MjModel.from_xml_path(filepath)
    data = mujoco.MjData(model)

    print(f"\n--- 関節一覧 ({model.njnt}) ---")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_adr = model.jnt_qposadr[i]
        dof_adr = model.jnt_dofadr[i]
        print(f"{i}: {name}, qpos_adr={qpos_adr}, dof_adr={dof_adr}")
    print()

    paused = False
    # 過去の状態を保存するリングバッファ
    history = collections.deque(maxlen=HISTORY_SIZE)

    def save_state():
        history.append({
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "act":  data.act.copy(),
            "time": data.time,
        })

    def restore_state(state):
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
        data.act[:]  = state["act"]
        data.time    = state["time"]
        mujoco.mj_forward(model, data)

    def key_callback(keycode):
        nonlocal paused
        if keycode == KEY_SPACE:
            paused = not paused
            print("停止中" if paused else "再開")
        elif keycode == KEY_LEFT:
            if len(history) > 1:
                history.pop()  # 現在の状態を捨てる
                restore_state(history[-1])
            else:
                print("これ以上戻れません")
        elif keycode == KEY_RIGHT and paused:
            save_state()
            mujoco.mj_step(model, data)

    # 初期状態を記録
    save_state()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not paused:
                save_state()
                mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()

"""
Evaluation: load a checkpoint, run episodes, optionally record trajectories.

Usage (from robot/piper-mujoco/):
    # 実行のみ（viewer表示）
    mjpython -m rl.eval --checkpoint rl/checkpoints/ckpt_01996800.pt

    # 軌跡を記録して保存
    mjpython -m rl.eval --checkpoint rl/checkpoints/ckpt_01996800.pt --record

    # viewer なしでヘッドレス記録（SSH先など）
    python -m rl.eval --checkpoint rl/checkpoints/ckpt_01996800.pt --record --no_viewer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.agent.a2c import A2C
from rl.env.lab_env import PiperLabEnv


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def run_episode(env, agent, device, record: bool):
    """1エピソード実行。record=True のとき qpos・time・info を収集して返す。"""
    obs, _ = env.reset()
    ep_reward = 0.0
    success = False

    qpos_buf, time_buf, reward_buf = [], [], []

    while True:
        if record:
            qpos_buf.append(env.data.qpos.copy())
            time_buf.append(env.data.time)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.net.get_action_and_value(obs_t)
        action_np = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        ep_reward += reward

        if record:
            reward_buf.append(reward)

        if info.get("flask_in_fridge"):
            success = True
            break
        if terminated or truncated:
            break

    traj = None
    if record:
        traj = {
            "qpos":    np.array(qpos_buf,   dtype=np.float32),
            "time":    np.array(time_buf,   dtype=np.float32),
            "rewards": np.array(reward_buf, dtype=np.float32),
            "success": np.array(success),
        }

    return ep_reward, success, info, traj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config",
        default=Path(__file__).parent / "configs" / "default.yaml",
        type=Path,
    )
    parser.add_argument("--n_episodes", default=5, type=int)
    parser.add_argument("--step_delay", default=0.005, type=float,
                        help="viewer 表示速度調整 (秒/step)")
    parser.add_argument("--record", action="store_true",
                        help="軌跡を npz ファイルに保存する")
    parser.add_argument("--no_viewer", action="store_true",
                        help="viewer を起動せずヘッドレスで実行 (--record と併用)")
    parser.add_argument("--save_dir", default=None, type=Path,
                        help="軌跡の保存先ディレクトリ (デフォルト: rl/trajectories/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]

    env = PiperLabEnv(
        max_episode_steps=env_cfg["max_episode_steps"],
        n_substeps=env_cfg["n_substeps"],
        action_scale=env_cfg.get("action_scale"),
    )
    agent = A2C(env.obs_dim, env.act_dim, agent_cfg)
    agent.load(args.checkpoint)
    agent.net.eval()
    device = agent.device
    print(f"Loaded: {args.checkpoint}")

    # 保存先ディレクトリ
    save_dir = args.save_dir or (Path(__file__).parent / "trajectories")
    if args.record:
        save_dir.mkdir(exist_ok=True)

    # ckpt 名をファイル名プレフィックスに使う
    ckpt_stem = args.checkpoint.stem

    # ------------------------------------------------------------------
    # viewer あり
    # ------------------------------------------------------------------
    if not args.no_viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for ep in range(args.n_episodes):
                obs, _ = env.reset()
                ep_reward = 0.0
                success = False
                qpos_buf, time_buf, reward_buf = [], [], []

                while viewer.is_running():
                    if args.record:
                        qpos_buf.append(env.data.qpos.copy())
                        time_buf.append(env.data.time)

                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action, _, _, _ = agent.net.get_action_and_value(obs_t)
                    action_np = action.squeeze(0).cpu().numpy()

                    obs, reward, terminated, truncated, info = env.step(action_np)
                    ep_reward += reward
                    if args.record:
                        reward_buf.append(reward)

                    viewer.sync()
                    time.sleep(args.step_delay)

                    if info.get("flask_in_fridge"):
                        success = True
                        time.sleep(1.0)
                        break
                    if terminated or truncated:
                        break

                result = "SUCCESS" if success else "failed"
                print(f"  ep{ep+1:02d}: {result} | reward={ep_reward:.2f} | "
                      f"flask_to_target={info.get('flask_to_target', float('nan')):.3f} m")

                if args.record:
                    path = save_dir / f"{ckpt_stem}_ep{ep+1:02d}.npz"
                    np.savez(
                        path,
                        qpos=np.array(qpos_buf, dtype=np.float32),
                        time=np.array(time_buf, dtype=np.float32),
                        rewards=np.array(reward_buf, dtype=np.float32),
                        success=np.array(success),
                    )
                    print(f"    → saved: {path}")

                if not viewer.is_running():
                    break

    # ------------------------------------------------------------------
    # viewer なし（ヘッドレス記録）
    # ------------------------------------------------------------------
    else:
        for ep in range(args.n_episodes):
            ep_reward, success, info, traj = run_episode(
                env, agent, device, record=args.record
            )
            result = "SUCCESS" if success else "failed"
            print(f"  ep{ep+1:02d}: {result} | reward={ep_reward:.2f} | "
                  f"flask_to_target={info.get('flask_to_target', float('nan')):.3f} m")

            if args.record and traj is not None:
                path = save_dir / f"{ckpt_stem}_ep{ep+1:02d}.npz"
                np.savez(path, **traj)
                print(f"    → saved: {path}")


if __name__ == "__main__":
    main()

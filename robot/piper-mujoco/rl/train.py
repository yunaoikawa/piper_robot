"""
A2C training entry point.

Usage (from robot/piper-mujoco/):
    python -m rl.train
    python -m rl.train --config rl/configs/default.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Allow running as `python rl/train.py` from piper-mujoco/
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.agent.a2c import A2C
from rl.env.lab_env import PiperLabEnv


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=Path(__file__).parent / "configs" / "default.yaml",
        type=Path,
    )
    parser.add_argument("--resume", default=None, type=Path, help="Checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["train"]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    env = PiperLabEnv(
        max_episode_steps=env_cfg["max_episode_steps"],
        n_substeps=env_cfg["n_substeps"],
        action_scale=env_cfg.get("action_scale"),
    )
    agent = A2C(env.obs_dim, env.act_dim, agent_cfg)

    if args.resume is not None:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    n_steps = agent_cfg["n_steps"]
    total_episodes = train_cfg["total_episodes"]
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]

    gamma = agent_cfg["gamma"]
    gae_lambda = agent_cfg["gae_lambda"]

    print(f"Training for {total_episodes:,} episodes | obs_dim={env.obs_dim} | act_dim={env.act_dim}")
    print(f"Device: {agent.device}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Ctrl+C 時に中断チェックポイントを保存
    # ------------------------------------------------------------------
    global_step = 0
    update_count = 0
    total_ep_count = 0

    def _save_and_exit(sig, frame):
        path = ckpt_dir / f"interrupted_ep{total_ep_count:06d}.pt"
        agent.save(path)
        print(f"\nInterrupted. Saved checkpoint: {path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)
    all_ep_rewards: list[float] = []
    interval_successes: list[bool] = []
    t_start = time.time()

    while total_ep_count < total_episodes:
        rollout = agent.collect_rollout(env, n_steps)
        global_step += n_steps
        update_count += 1

        completed = rollout["ep_rewards"]
        total_ep_count += len(completed)
        all_ep_rewards.extend(completed)
        interval_successes.extend(rollout["ep_successes"])

        advantages, returns = agent.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            rollout["last_val"],
            gamma,
            gae_lambda,
        )
        losses = agent.update(rollout["obs"], rollout["actions"], advantages, returns)

        if completed and (total_ep_count // log_interval) > ((total_ep_count - len(completed)) // log_interval):
            elapsed = time.time() - t_start
            eps = total_ep_count / elapsed  # episodes per second
            recent_rewards = all_ep_rewards[-20:]
            mean_ep_rew = sum(recent_rewards) / len(recent_rewards)
            any_lifted = any(interval_successes)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{ts}] "
                f"ep={total_ep_count:6d}/{total_episodes} | "
                f"step={global_step:9d} | "
                f"mean_ep_rew={mean_ep_rew:7.2f} | "
                f"actor={losses['actor_loss']:7.4f} | "
                f"critic={losses['critic_loss']:7.4f} | "
                f"entropy={losses['entropy']:.4f} | "
                f"ep/s={eps:.2f} | "
                f"lifted={any_lifted}"
            )
            interval_successes.clear()

        if completed and (total_ep_count // save_interval) > ((total_ep_count - len(completed)) // save_interval):
            ckpt_path = ckpt_dir / f"ckpt_ep{total_ep_count:06d}.pt"
            agent.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    agent.save(ckpt_dir / "final.pt")
    print(f"\nTraining complete. Final checkpoint saved to {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()

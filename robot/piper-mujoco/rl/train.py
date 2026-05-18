"""
A2C training entry point.

Usage (from robot/piper-mujoco/):
    python -m rl.train
    python -m rl.train --config rl/configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
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
    )
    agent = A2C(env.obs_dim, env.act_dim, agent_cfg)

    if args.resume is not None:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    n_steps = agent_cfg["n_steps"]
    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]

    gamma = agent_cfg["gamma"]
    gae_lambda = agent_cfg["gae_lambda"]

    print(f"Training for {total_steps:,} steps | obs_dim={env.obs_dim} | act_dim={env.act_dim}")
    print(f"Device: {agent.device}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    update_count = 0
    all_ep_rewards: list[float] = []
    t_start = time.time()

    while global_step < total_steps:
        rollout = agent.collect_rollout(env, n_steps)
        global_step += n_steps
        update_count += 1

        all_ep_rewards.extend(rollout["ep_rewards"])

        advantages, returns = agent.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            rollout["last_val"],
            gamma,
            gae_lambda,
        )
        losses = agent.update(rollout["obs"], rollout["actions"], advantages, returns)

        if update_count % log_interval == 0:
            elapsed = time.time() - t_start
            sps = global_step / elapsed  # steps per second
            recent_rewards = all_ep_rewards[-20:] if all_ep_rewards else [float("nan")]
            mean_ep_rew = sum(recent_rewards) / len(recent_rewards)
            print(
                f"step={global_step:8d} | "
                f"updates={update_count:5d} | "
                f"mean_ep_rew={mean_ep_rew:7.2f} | "
                f"actor={losses['actor_loss']:7.4f} | "
                f"critic={losses['critic_loss']:7.4f} | "
                f"entropy={losses['entropy']:.4f} | "
                f"sps={sps:.0f}"
            )

        if update_count % save_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_{global_step:08d}.pt"
            agent.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    agent.save(ckpt_dir / "final.pt")
    print(f"\nTraining complete. Final checkpoint saved to {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()

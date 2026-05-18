"""
Evaluation: load a checkpoint and visualise in the MuJoCo viewer.

Usage (from robot/piper-mujoco/):
    mjpython -m rl.eval --checkpoint rl/checkpoints/final.pt
    mjpython -m rl.eval --checkpoint rl/checkpoints/final.pt --n_episodes 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.agent.a2c import A2C
from rl.env.lab_env import PiperLabEnv


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--config",
        default=Path(__file__).parent / "configs" / "default.yaml",
        type=Path,
    )
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument(
        "--step_delay",
        default=0.005,
        type=float,
        help="Seconds to sleep between steps (for viewing speed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]

    env = PiperLabEnv(
        max_episode_steps=env_cfg["max_episode_steps"],
        n_substeps=env_cfg["n_substeps"],
    )
    agent = A2C(env.obs_dim, env.act_dim, agent_cfg)
    agent.load(args.checkpoint)
    agent.net.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    device = agent.device

    # ------------------------------------------------------------------
    # Launch passive viewer and run episodes
    # ------------------------------------------------------------------
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            success = False

            while viewer.is_running():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _, _ = agent.net.get_action_and_value(obs_t)
                action_np = action.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action_np)
                ep_reward += reward

                viewer.sync()
                time.sleep(args.step_delay)

                if info.get("flask_in_fridge"):
                    success = True
                    print(f"  Episode {ep+1}: SUCCESS! reward={ep_reward:.2f}")
                    time.sleep(1.0)  # pause to see success
                    break

                if terminated or truncated:
                    break

            if not success:
                print(
                    f"  Episode {ep+1}: failed | reward={ep_reward:.2f} | "
                    f"flask_to_target={info.get('flask_to_target', float('nan')):.3f} m"
                )

            if not viewer.is_running():
                break


if __name__ == "__main__":
    main()

"""
Advantage Actor-Critic (A2C) agent.

Workflow per update:
  1. collect_rollout()  → gather n_steps of (obs, act, rew, done, val)
  2. compute_gae()      → generalised advantage estimation
  3. update()           → one gradient step on actor + critic losses
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .actor_critic import ActorCritic


class A2C:
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(
            obs_dim, act_dim, hidden_dim=cfg.get("hidden_dim", 256)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, env, n_steps: int) -> dict:
        """
        Run the environment for *n_steps* steps and return a dict of arrays.
        Handles episode resets internally; returns final obs for bootstrapping.
        """
        obs_buf = np.zeros((n_steps, env.obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, env.act_dim), dtype=np.float32)
        rew_buf = np.zeros(n_steps, dtype=np.float32)
        done_buf = np.zeros(n_steps, dtype=np.float32)
        val_buf = np.zeros(n_steps, dtype=np.float32)
        logp_buf = np.zeros(n_steps, dtype=np.float32)

        ep_rewards: list[float] = []
        current_ep_reward: float = 0.0

        obs, _ = env.reset()

        for t in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, _, value = self.net.get_action_and_value(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            obs_buf[t] = obs
            act_buf[t] = action_np
            rew_buf[t] = reward
            done_buf[t] = float(done)
            val_buf[t] = value.item()
            logp_buf[t] = log_prob.item()

            current_ep_reward += reward
            if done:
                ep_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap value of the state after the last step
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_val = self.net.get_value(obs_t).item()

        return {
            "obs": obs_buf,
            "actions": act_buf,
            "rewards": rew_buf,
            "dones": done_buf,
            "values": val_buf,
            "log_probs": logp_buf,
            "last_val": last_val,
            "ep_rewards": ep_rewards,
        }

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_val: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = last_val
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Network update
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(actions).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        _, log_prob, entropy, value = self.net.get_action_and_value(obs_t, act_t)

        actor_loss = -(adv_t * log_prob).mean()
        critic_loss = F.mse_loss(value, ret_t)
        entropy_loss = -entropy.mean()

        cfg = self.cfg
        total_loss = (
            actor_loss
            + cfg["value_coef"] * critic_loss
            + cfg["entropy_coef"] * entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), cfg["max_grad_norm"]
        )
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item(),
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        torch.save(
            {"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()},
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

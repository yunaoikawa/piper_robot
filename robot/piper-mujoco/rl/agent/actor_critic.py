"""
Actor-Critic network for continuous action spaces.

Architecture:
  Shared MLP backbone → actor head (Gaussian policy) + critic head (state value)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        # Learnable log standard deviation (shared across batch)
        self.actor_log_std = nn.Parameter(
            torch.full((act_dim,), log_std_init)
        )
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small gain for policy head → near-zero initial actions
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action_mean, action_std, state_value)."""
        features = self.backbone(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return mean, std, value

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or evaluate) an action and return
        (action, log_prob, entropy, value).
        """
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.critic(features).squeeze(-1)

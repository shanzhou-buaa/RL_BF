from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class TanhGaussianActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        initial_log_std: float = -0.7,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(states)
        mean = self.actor_mean(features)
        log_std = self.log_std.expand_as(mean).clamp(-5.0, 2.0)
        value = self.critic(features).squeeze(-1)
        return mean, log_std, value

    def distribution(self, states: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean, log_std, value = self.forward(states)
        return Normal(mean, log_std.exp()), value

    def _squashed_log_prob(self, dist: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + 1.0e-6).sum(dim=-1)
        return log_prob - correction

    def _effective_squashed_entropy(
        self,
        dist: Normal,
        raw_reference: torch.Tensor,
    ) -> torch.Tensor:
        action_reference = torch.tanh(raw_reference)
        normal_entropy = dist.entropy()
        squash_correction = torch.log(1.0 - action_reference.pow(2) + 1.0e-6)
        return (normal_entropy + squash_correction).sum(dim=-1)

    def act(self, states: torch.Tensor, deterministic: bool = False):
        dist, value = self.distribution(states)
        raw_action = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        entropy = self._effective_squashed_entropy(dist, dist.mean)
        return action, raw_action, log_prob, entropy, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        dist, value = self.distribution(states)
        raw_action = atanh(actions)
        log_prob = self._squashed_log_prob(dist, raw_action, actions)
        entropy = self._effective_squashed_entropy(dist, dist.mean)
        return log_prob, entropy, value

    def value(self, states: torch.Tensor) -> torch.Tensor:
        _, _, value = self.forward(states)
        return value

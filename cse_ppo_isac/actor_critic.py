from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        initial_log_std: float = -1.1,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(states)
        mean = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std, value

    def distribution(self, states: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean, log_std, value = self.forward(states)
        return Normal(mean, log_std.exp()), value

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False):
        dist, value = self.distribution(state.unsqueeze(0))
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(entropy.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def act_batch(self, states: torch.Tensor, deterministic: bool = False):
        dist, values = self.distribution(states)
        actions = dist.mean if deterministic else dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropies = dist.entropy().sum(dim=-1)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            entropies.cpu().numpy(),
            values.cpu().numpy(),
        )

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        dist, values = self.distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropies = dist.entropy().sum(dim=-1)
        return log_probs, entropies, values

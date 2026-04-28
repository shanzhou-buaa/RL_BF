from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from .buffer import RolloutBatch, compute_gae, normalize
from .config import PPOConfig
from .policy import TanhGaussianActorCritic


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: PPOConfig,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.policy = TanhGaussianActorCritic(
            state_dim, action_dim, hidden_dim=cfg.hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.action_dim = int(action_dim)

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        advantages, returns = compute_gae(batch, self.cfg.gamma, self.cfg.gae_lambda)
        advantages = normalize(advantages)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        n = states.shape[0]
        stats = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "alpha_entropy": float(self.cfg.entropy_coef),
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "entropy_threshold": float("nan"),
            "high_entropy_rate": 1.0,
            "macro_step_rate": 0.0,
            "average_macro_length": 1.0,
            "num_macro_segments": 0.0,
        }
        updates = 0
        for _ in range(self.cfg.ppo_epochs):
            permutation = torch.randperm(n, device=self.device)
            for start in range(0, n, self.cfg.minibatch_size):
                idx = permutation[start : start + self.cfg.minibatch_size]
                log_probs, entropies, values = self.policy.evaluate_actions(
                    states[idx], actions[idx]
                )
                ratio = torch.exp(log_probs - old_log_probs[idx])
                clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                )
                actor_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
                critic_loss = ((values - ret[idx]) ** 2).mean()
                entropy = entropies.mean() / max(self.action_dim, 1)
                loss = (
                    actor_loss
                    + self.cfg.value_coef * critic_loss
                    - self.cfg.entropy_coef * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = (old_log_probs[idx] - log_probs).mean().abs()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_ratio).float().mean()
                    )
                stats["actor_loss"] += float(actor_loss.item())
                stats["critic_loss"] += float(critic_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_fraction"] += float(clip_fraction.item())
                stats["grad_norm"] += float(grad_norm)
                updates += 1
            if stats["approx_kl"] / max(updates, 1) > self.cfg.target_kl:
                break
        for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"):
            stats[key] /= max(updates, 1)
        return stats

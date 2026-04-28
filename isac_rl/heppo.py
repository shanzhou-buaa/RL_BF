from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from .buffer import RolloutBatch, compute_gae
from .config import PPOConfig
from .entropy_macro import EntropyMacroBuilder, MacroSegment
from .ppo import PPOAgent


class HEPPOAgent(PPOAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: PPOConfig,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, cfg, device)
        self.macro_builder = EntropyMacroBuilder(
            quantile=cfg.entropy_quantile,
            ema=cfg.entropy_threshold_ema,
            max_macro_len=cfg.max_macro_len,
            gamma=cfg.gamma,
        )
        self.alpha_he = float(cfg.alpha_he_init)

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        advantages, _ = compute_gae(batch, self.cfg.gamma, self.cfg.gae_lambda)
        _, segments, macro_stats = self.macro_builder.build(batch, advantages)
        segments = self.macro_builder.apply_group_correction(segments)
        self._update_entropy_coef(macro_stats["high_entropy_rate"])

        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        n = len(segments)
        stats = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "alpha_entropy": float(self.alpha_he),
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            **macro_stats,
        }
        if n == 0:
            return stats
        updates = 0
        for _ in range(self.cfg.ppo_epochs):
            permutation = np.random.permutation(n)
            for start in range(0, n, self.cfg.minibatch_size):
                selected = [segments[int(i)] for i in permutation[start : start + self.cfg.minibatch_size]]
                loss_parts = self._segment_loss(states, actions, selected)
                loss = (
                    loss_parts["actor_loss"]
                    + self.cfg.value_coef * loss_parts["critic_loss"]
                    - self.alpha_he * loss_parts["entropy"]
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()
                stats["actor_loss"] += float(loss_parts["actor_loss"].item())
                stats["critic_loss"] += float(loss_parts["critic_loss"].item())
                stats["entropy"] += float(loss_parts["entropy"].item())
                stats["approx_kl"] += float(loss_parts["approx_kl"].item())
                stats["clip_fraction"] += float(loss_parts["clip_fraction"].item())
                stats["grad_norm"] += float(grad_norm)
                updates += 1
            if stats["approx_kl"] / max(updates, 1) > self.cfg.target_kl:
                break
        for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"):
            stats[key] /= max(updates, 1)
        return stats

    def _segment_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        segments: list[MacroSegment],
    ) -> Dict[str, torch.Tensor]:
        policy_terms = []
        value_terms = []
        entropy_terms = []
        kls = []
        clipped_flags = []
        for segment in segments:
            idx = torch.as_tensor(segment.indices, dtype=torch.long, device=self.device)
            log_probs, entropies, values = self.policy.evaluate_actions(
                states.index_select(0, idx), actions.index_select(0, idx)
            )
            new_log_prob = log_probs.sum()
            old_log_prob = torch.as_tensor(
                segment.old_log_prob,
                dtype=torch.float32,
                device=self.device,
            )
            ratio = torch.exp(new_log_prob - old_log_prob)
            adv = torch.as_tensor(segment.advantage, dtype=torch.float32, device=self.device)
            clipped = torch.clamp(
                ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
            )
            policy_terms.append(torch.min(ratio * adv, clipped * adv))
            target = torch.as_tensor(segment.ret, dtype=torch.float32, device=self.device)
            value_terms.append((values[0] - target) ** 2)
            entropy_terms.append(entropies.mean() / max(self.action_dim, 1))
            segment_kl = torch.abs(old_log_prob - new_log_prob) / max(segment.macro_len, 1)
            kls.append(segment_kl)
            clipped_flags.append((torch.abs(ratio - 1.0) > self.cfg.clip_ratio).float())
        return {
            "actor_loss": -torch.stack(policy_terms).mean(),
            "critic_loss": torch.stack(value_terms).mean(),
            "entropy": torch.stack(entropy_terms).mean(),
            "approx_kl": torch.stack(kls).mean(),
            "clip_fraction": torch.stack(clipped_flags).mean(),
        }

    def _update_entropy_coef(self, high_entropy_rate: float) -> None:
        if high_entropy_rate < self.cfg.target_high_entropy_rate:
            self.alpha_he *= 1.02
        else:
            self.alpha_he *= 0.98
        self.alpha_he = float(
            np.clip(self.alpha_he, self.cfg.alpha_he_min, self.cfg.alpha_he_max)
        )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    entropies: np.ndarray
    episode_steps: np.ndarray
    next_values: np.ndarray


def compute_gae(batch: RolloutBatch, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(batch.rewards, dtype=np.float32)
    gae = 0.0
    for idx in reversed(range(batch.rewards.size)):
        nonterminal = 0.0 if batch.dones[idx] else 1.0
        delta = (
            batch.rewards[idx]
            + gamma * batch.next_values[idx] * nonterminal
            - batch.values[idx]
        )
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[idx] = gae
        if batch.dones[idx]:
            gae = 0.0
    returns = advantages + batch.values
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return values.astype(np.float32)
    return ((values - values.mean()) / (values.std() + 1.0e-8)).astype(np.float32)

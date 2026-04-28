from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import List

import numpy as np

from .buffer import RolloutBatch, normalize


@dataclass(frozen=True)
class MacroSegment:
    indices: list[int]
    old_log_prob: float
    advantage: float
    ret: float
    start_step: int
    macro_len: int
    entropy: float
    reward: float


class EntropyMacroBuilder:
    def __init__(
        self,
        quantile: float = 0.4,
        ema: float = 0.8,
        max_macro_len: int = 3,
        gamma: float = 0.97,
    ):
        self.quantile = float(quantile)
        self.ema = float(ema)
        self.max_macro_len = int(max_macro_len)
        self.gamma = float(gamma)
        self.threshold: float | None = None

    def update_threshold(self, entropies: np.ndarray) -> float:
        target = float(np.quantile(entropies, np.clip(self.quantile, 0.0, 1.0)))
        if self.threshold is None:
            self.threshold = target
        else:
            self.threshold = self.ema * self.threshold + (1.0 - self.ema) * target
        return float(self.threshold)

    def build(
        self,
        batch: RolloutBatch,
        advantages: np.ndarray,
    ) -> tuple[float, List[MacroSegment], dict[str, float]]:
        threshold = self.update_threshold(batch.entropies)
        low = batch.entropies < threshold
        segments: list[MacroSegment] = []
        idx = 0
        macro_steps = 0
        macro_lengths = []
        while idx < batch.rewards.size:
            if not low[idx]:
                indices = [idx]
            else:
                indices = []
                while (
                    idx + len(indices) < batch.rewards.size
                    and low[idx + len(indices)]
                    and len(indices) < self.max_macro_len
                ):
                    pos = idx + len(indices)
                    indices.append(pos)
                    if batch.dones[pos]:
                        break
            segment = self._make_segment(batch, advantages, indices)
            segments.append(segment)
            if len(indices) > 1:
                macro_steps += len(indices)
                macro_lengths.append(len(indices))
            idx = indices[-1] + 1

        stats = {
            "entropy_mean": float(np.mean(batch.entropies)),
            "entropy_threshold": float(threshold),
            "high_entropy_rate": float(np.mean(~low)),
            "macro_step_rate": float(macro_steps / max(batch.rewards.size, 1)),
            "average_macro_length": float(np.mean(macro_lengths) if macro_lengths else 1.0),
            "num_macro_segments": float(len(macro_lengths)),
        }
        return threshold, segments, stats

    def _make_segment(
        self,
        batch: RolloutBatch,
        advantages: np.ndarray,
        indices: list[int],
    ) -> MacroSegment:
        discounts = np.asarray([self.gamma**j for j in range(len(indices))], dtype=np.float32)
        rewards = batch.rewards[indices]
        discounted_reward = float(np.sum(discounts * rewards))
        last = indices[-1]
        bootstrap = 0.0 if batch.dones[last] else (self.gamma ** len(indices)) * batch.next_values[last]
        ret = float(discounted_reward + bootstrap)
        advantage = float(np.sum(discounts * advantages[indices]))
        return MacroSegment(
            indices=list(indices),
            old_log_prob=float(np.sum(batch.log_probs[indices])),
            advantage=advantage,
            ret=ret,
            start_step=int(batch.episode_steps[indices[0]]),
            macro_len=len(indices),
            entropy=float(np.mean(batch.entropies[indices])),
            reward=discounted_reward,
        )

    def apply_group_correction(self, segments: list[MacroSegment]) -> list[MacroSegment]:
        if not segments:
            return []
        gae_values = normalize(np.asarray([segment.advantage for segment in segments], dtype=np.float32))
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, segment in enumerate(segments):
            grouped[(segment.start_step, segment.macro_len)].append(idx)
        group_scores = np.zeros(len(segments), dtype=np.float32)
        rewards = np.asarray([segment.reward for segment in segments], dtype=np.float32)
        for indices in grouped.values():
            vals = rewards[indices]
            if len(indices) > 1:
                group_scores[indices] = (vals - vals.mean()) / (vals.std() + 1.0e-8)
        corrected = []
        for idx, segment in enumerate(segments):
            corrected.append(
                replace(
                    segment,
                    advantage=float(0.8 * gae_values[idx] + 0.2 * group_scores[idx]),
                )
            )
        return corrected

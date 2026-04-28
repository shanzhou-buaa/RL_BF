import numpy as np

from isac_rl.buffer import RolloutBatch
from isac_rl.entropy_macro import EntropyMacroBuilder


def test_entropy_macro_builder_groups_consecutive_low_entropy_steps():
    batch = RolloutBatch(
        states=np.zeros((6, 3), dtype=np.float32),
        actions=np.zeros((6, 2), dtype=np.float32),
        rewards=np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        dones=np.array([False, False, False, False, False, True]),
        values=np.zeros(6, dtype=np.float32),
        log_probs=np.zeros(6, dtype=np.float32),
        entropies=np.array([0.8, 0.1, 0.2, 0.9, 0.1, 0.1], dtype=np.float32),
        episode_steps=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        next_values=np.zeros(6, dtype=np.float32),
    )
    advantages = np.arange(6, dtype=np.float32)
    builder = EntropyMacroBuilder(quantile=0.5, ema=0.0, max_macro_len=3, gamma=0.9)

    threshold, segments, stats = builder.build(batch, advantages)

    assert np.isclose(threshold, 0.15)
    assert [segment.indices for segment in segments] == [[0], [1], [2], [3], [4, 5]]
    assert stats["num_macro_segments"] == 1
    assert stats["average_macro_length"] == 2.0
    assert stats["macro_step_rate"] == 2 / 6


def test_group_normalized_macro_advantage_is_finite():
    batch = RolloutBatch(
        states=np.zeros((4, 2), dtype=np.float32),
        actions=np.zeros((4, 1), dtype=np.float32),
        rewards=np.ones(4, dtype=np.float32),
        dones=np.array([False, True, False, True]),
        values=np.zeros(4, dtype=np.float32),
        log_probs=np.zeros(4, dtype=np.float32),
        entropies=np.zeros(4, dtype=np.float32),
        episode_steps=np.array([0, 1, 0, 1], dtype=np.int32),
        next_values=np.zeros(4, dtype=np.float32),
    )
    builder = EntropyMacroBuilder(quantile=1.0, ema=0.0, max_macro_len=2, gamma=0.99)

    _, segments, _ = builder.build(batch, np.ones(4, dtype=np.float32))
    corrected = builder.apply_group_correction(segments)

    assert all(np.isfinite(segment.advantage) for segment in corrected)

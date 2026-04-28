import numpy as np
import torch

from isac_rl.buffer import RolloutBatch
from isac_rl.config import PPOConfig
from isac_rl.entropy_macro import EntropyMacroBuilder, MacroSegment
from isac_rl.heppo import HEPPOAgent


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


def test_heppo_macro_ratio_uses_average_log_ratio():
    agent = HEPPOAgent(
        state_dim=1,
        action_dim=1,
        cfg=PPOConfig(clip_ratio=0.5, hidden_dim=4),
        device="cpu",
    )

    def fake_evaluate_actions(_states, _actions):
        return (
            torch.tensor([0.1, 0.1], dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
        )

    agent.policy.evaluate_actions = fake_evaluate_actions
    segment = MacroSegment(
        indices=[0, 1],
        old_log_prob=0.0,
        advantage=1.0,
        ret=0.0,
        start_step=0,
        macro_len=2,
        entropy=0.0,
        reward=0.0,
    )

    loss = agent._segment_loss(
        states=torch.zeros((2, 1), dtype=torch.float32),
        actions=torch.zeros((2, 1), dtype=torch.float32),
        segments=[segment],
    )

    expected_ratio = torch.exp(torch.tensor(0.1))
    torch.testing.assert_close(loss["actor_loss"], -expected_ratio)
    torch.testing.assert_close(loss["approx_kl"], torch.tensor(0.1))

import numpy as np

from isac_rl.buffer import RolloutBatch, compute_gae


def test_terminal_reward_propagates_to_previous_step_in_episode():
    batch = RolloutBatch(
        states=np.zeros((2, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
        rewards=np.array([0.0, 10.0], dtype=np.float32),
        dones=np.array([False, True]),
        values=np.zeros(2, dtype=np.float32),
        log_probs=np.zeros(2, dtype=np.float32),
        entropies=np.zeros(2, dtype=np.float32),
        episode_steps=np.array([0, 1], dtype=np.int32),
        next_values=np.zeros(2, dtype=np.float32),
    )

    advantages, returns = compute_gae(batch, gamma=1.0, gae_lambda=1.0)

    assert np.allclose(advantages, np.array([10.0, 10.0], dtype=np.float32))
    assert np.allclose(returns, np.array([10.0, 10.0], dtype=np.float32))

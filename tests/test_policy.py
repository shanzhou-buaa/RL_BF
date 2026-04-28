import torch

from isac_rl.policy import TanhGaussianActorCritic


def test_tanh_gaussian_policy_bounds_actions_and_scores_them():
    torch.manual_seed(0)
    policy = TanhGaussianActorCritic(state_dim=7, action_dim=5, hidden_dim=16)
    states = torch.randn(4, 7)

    action, raw_action, log_prob, entropy, value = policy.act(states)
    eval_log_prob, eval_entropy, eval_value = policy.evaluate_actions(states, action)

    assert action.shape == (4, 5)
    assert raw_action.shape == (4, 5)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(eval_log_prob).all()
    assert torch.isfinite(eval_entropy).all()
    assert eval_value.shape == (4,)


def test_deterministic_policy_uses_tanh_mean():
    torch.manual_seed(0)
    policy = TanhGaussianActorCritic(state_dim=3, action_dim=2, hidden_dim=8)
    states = torch.zeros(1, 3)

    action, _, _, _, _ = policy.act(states, deterministic=True)
    mean, _, _ = policy.forward(states)

    torch.testing.assert_close(action, torch.tanh(mean))

import inspect

import numpy as np

from isac_rl.config import EnvConfig, SystemConfig
from isac_rl.env import ISACBeamformingEnv
from isac_rl.metrics import per_antenna_power_normalize


def test_env_uses_random_reset_and_direct_action_update():
    sys_cfg = SystemConfig(M=4, K=2, target_angles_deg=(-40.0, 0.0, 40.0))
    env_cfg = EnvConfig(episode_steps=2, action_scale=0.03)
    env = ISACBeamformingEnv(sys_cfg, env_cfg, seed=3)

    state = env.reset()
    H0 = env.H.copy()
    W0 = env.W.copy()
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[: env.action_dim // 2] = 0.25

    next_state, reward, done, info = env.step(action)

    assert state.shape == (env.state_dim,)
    assert next_state.shape == (env.state_dim,)
    assert not done
    assert np.isfinite(reward)
    assert info["t"] == 1
    assert not np.allclose(env.W, W0)
    np.testing.assert_allclose(H0, env.H)
    row_power = np.sum(np.abs(env.W) ** 2, axis=1)
    np.testing.assert_allclose(row_power, sys_cfg.total_power / sys_cfg.M, rtol=1e-6)


def test_env_source_does_not_contain_baseline_or_candidate_logic():
    import isac_rl.env as env_module

    source = inspect.getsource(env_module)
    forbidden = (
        "baselines",
        "line_search",
        "candidate",
        "no-op",
        "component_guard",
        "zf_beamformer",
        "sdr",
    )

    for token in forbidden:
        assert token not in source


def test_state_contains_six_feature_groups():
    sys_cfg = SystemConfig(M=3, K=1)
    env = ISACBeamformingEnv(sys_cfg, EnvConfig(episode_steps=4), seed=4)
    env.reset()

    groups = env.build_state_groups()

    assert set(groups) == {
        "channel",
        "beamformer",
        "sinr",
        "radar",
        "beampattern",
        "progress",
    }
    assert sum(value.size for value in groups.values()) == env.state_dim


def test_reward_is_bounded_for_large_objective():
    env = ISACBeamformingEnv(SystemConfig(M=3, K=1), EnvConfig(), seed=5)
    env.reset()
    env.prev_info = {"objective": 1.0}

    reward = env.compute_reward(
        {
            "objective": 10_000.0,
            "min_sinr_gap_db": -100.0,
            "feasible": False,
        },
        done=True,
    )

    assert -10.0 <= reward <= 10.0


def test_reward_depends_on_objective_progress_not_feasible_or_margin():
    env = ISACBeamformingEnv(SystemConfig(M=3, K=1), EnvConfig(), seed=6)
    env.reset()
    env.prev_info = {"objective": 5.0}

    base = {"objective": 4.0, "min_sinr_gap_db": -100.0, "feasible": False}
    changed = {"objective": 4.0, "min_sinr_gap_db": 100.0, "feasible": True}

    assert env.compute_reward(base, done=False) == env.compute_reward(changed, done=False)

import csv
import json
from pathlib import Path

import numpy as np
import torch

from isac_rl.config import EnvConfig, PPOConfig, SystemConfig, TrainConfig
from isac_rl.trainer import collect_rollout, train_algorithms


def test_smoke_train_writes_expected_logs(tmp_path: Path):
    sys_cfg = SystemConfig(M=3, K=1, angle_grid_step_deg=5.0)
    env_cfg = EnvConfig(episode_steps=2, action_scale=0.03)
    ppo_cfg = PPOConfig(
        updates=1,
        episodes_per_update=2,
        ppo_epochs=1,
        minibatch_size=4,
        hidden_dim=16,
        lr=1e-3,
    )
    train_cfg = TrainConfig(algos=("ppo", "heppo"), seeds=(1,), device="cpu")

    train_algorithms(sys_cfg, env_cfg, ppo_cfg, train_cfg, tmp_path, show_progress=False)

    for filename in ("config.json", "summary.json", "train_history.csv", "eval_history.csv", "entropy_history.csv"):
        assert (tmp_path / filename).exists()
    assert (tmp_path / "checkpoints" / "ppo_seed1.pt").exists()
    assert (tmp_path / "checkpoints" / "heppo_seed1.pt").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["algorithms"]) == {"ppo", "heppo"}
    assert set(summary["checkpoints"]) == {"ppo_seed1", "heppo_seed1"}

    with (tmp_path / "entropy_history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["algo"] for row in rows} == {"ppo", "heppo"}

    with (tmp_path / "eval_history.csv").open(newline="", encoding="utf-8") as handle:
        eval_rows = list(csv.DictReader(handle))
    assert "episode" in eval_rows[0]
    assert "eval_C_target" in eval_rows[0]
    assert "eval_C_offset" in eval_rows[0]
    assert "eval_target_peak_offset_error" in eval_rows[0]


def test_training_core_modules_do_not_import_baselines():
    for path in (
        Path("isac_rl/env.py"),
        Path("isac_rl/trainer.py"),
        Path("isac_rl/ppo.py"),
        Path("isac_rl/heppo.py"),
    ):
        if path.exists():
            assert "baselines" not in path.read_text(encoding="utf-8")


def test_baseline_module_is_not_part_of_pure_rl_package():
    assert not Path("isac_rl/baselines.py").exists()


def test_collect_rollout_seed_depends_on_update(monkeypatch):
    seen_seeds = []

    class FakeEnv:
        state_dim = 1
        action_dim = 1

        def __init__(self, _sys_cfg, _env_cfg, seed):
            seen_seeds.append(seed)
            self.current_info = None

        def reset(self):
            return np.zeros(1, dtype=np.float32)

        def step(self, _action):
            self.current_info = {
                "objective": float(seen_seeds[-1]),
                "feasible": True,
                "min_sinr_db": 0.0,
            }
            return np.zeros(1, dtype=np.float32), 0.0, True, self.current_info

    class FakePolicy:
        def act(self, state_t):
            batch = state_t.shape[0]
            zeros = torch.zeros((batch, 1), dtype=torch.float32)
            scalar = torch.zeros(batch, dtype=torch.float32)
            return zeros, zeros, scalar, scalar, scalar

        def value(self, state_t):
            return torch.zeros(state_t.shape[0], dtype=torch.float32)

    class FakeAgent:
        policy = FakePolicy()

    monkeypatch.setattr("isac_rl.trainer.ISACBeamformingEnv", FakeEnv)

    sys_cfg = SystemConfig(M=3, K=1, angle_grid_step_deg=5.0)
    env_cfg = EnvConfig(episode_steps=1)

    collect_rollout(FakeAgent(), sys_cfg, env_cfg, seed=7, update=1, episodes=2, device="cpu")
    collect_rollout(FakeAgent(), sys_cfg, env_cfg, seed=7, update=2, episodes=2, device="cpu")

    assert seen_seeds == [7_000_002, 7_000_003, 7_000_004, 7_000_005]


def test_collect_rollout_batches_policy_forward_per_step(monkeypatch):
    batch_sizes = []

    class FakeEnv:
        state_dim = 1
        action_dim = 1

        def __init__(self, _sys_cfg, _env_cfg, seed):
            self.seed = seed
            self.step_count = 0
            self.current_info = None

        def reset(self):
            return np.asarray([self.seed], dtype=np.float32)

        def step(self, _action):
            self.step_count += 1
            done = self.step_count >= 3
            self.current_info = {
                "objective": float(self.seed + self.step_count),
                "feasible": done,
                "min_sinr_db": float(self.step_count),
            }
            return np.asarray([self.seed + self.step_count], dtype=np.float32), 1.0, done, self.current_info

    class FakePolicy:
        def act(self, state_t):
            batch_sizes.append(int(state_t.shape[0]))
            batch = state_t.shape[0]
            zeros = torch.zeros((batch, 1), dtype=torch.float32)
            scalar = torch.zeros(batch, dtype=torch.float32)
            return zeros, zeros, scalar, scalar, scalar

        def value(self, state_t):
            return torch.zeros(state_t.shape[0], dtype=torch.float32)

    class FakeAgent:
        policy = FakePolicy()

    monkeypatch.setattr("isac_rl.trainer.ISACBeamformingEnv", FakeEnv)

    sys_cfg = SystemConfig(M=3, K=1, angle_grid_step_deg=5.0)
    env_cfg = EnvConfig(episode_steps=3)
    batch, stats = collect_rollout(
        FakeAgent(),
        sys_cfg,
        env_cfg,
        seed=1,
        update=1,
        episodes=4,
        device="cpu",
    )

    assert batch_sizes == [4, 4, 4]
    assert batch.states.shape[0] == 12
    assert stats["reward"] == 3.0

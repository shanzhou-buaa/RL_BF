import csv
import json
from pathlib import Path

from isac_rl.config import EnvConfig, PPOConfig, SystemConfig, TrainConfig
from isac_rl.trainer import train_algorithms


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
    assert (tmp_path / "checkpoints" / "ppo.pt").exists()
    assert (tmp_path / "checkpoints" / "heppo.pt").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["algorithms"]) == {"ppo", "heppo"}

    with (tmp_path / "entropy_history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["algo"] for row in rows} == {"ppo", "heppo"}


def test_training_core_modules_do_not_import_baselines():
    for path in (
        Path("isac_rl/env.py"),
        Path("isac_rl/trainer.py"),
        Path("isac_rl/ppo.py"),
        Path("isac_rl/heppo.py"),
    ):
        if path.exists():
            assert "baselines" not in path.read_text(encoding="utf-8")

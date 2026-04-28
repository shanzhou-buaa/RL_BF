from pathlib import Path

import numpy as np

import run_plot
from isac_rl.plotting import _group_mean_std


def test_run_plot_auto_eval_uses_current_eval_without_baselines(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))
        (tmp_path / "patterns.npz").write_bytes(b"placeholder")

    plotted = []
    monkeypatch.setattr(run_plot.subprocess, "run", fake_run)
    monkeypatch.setattr(run_plot, "plot_all", lambda log_dir: plotted.append(Path(log_dir)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_plot.py",
            "--log-dir",
            str(tmp_path),
            "--eval-channels",
            "7",
            "--plot-seed",
            "2030",
            "--device",
            "cpu",
        ],
    )

    run_plot.main()

    assert len(calls) == 1
    cmd, check = calls[0]
    assert check is True
    assert any(Path(part).name == "run_eval.py" for part in cmd)
    assert "--baselines" not in cmd
    assert "--eval-channels" in cmd
    assert "7" in cmd
    assert plotted == [tmp_path]


def test_convergence_grouping_uses_episode_when_available():
    grouped = _group_mean_std(
        [
            {"algo": "ppo", "update": "1", "episode": "100", "eval_objective": "3.0"},
            {"algo": "ppo", "update": "1", "episode": "100", "eval_objective": "5.0"},
            {"algo": "ppo", "update": "2", "episode": "200", "eval_objective": "1.0"},
        ],
        "eval_objective",
    )

    xs, means, stds = grouped["ppo"]
    np.testing.assert_allclose(xs, np.asarray([100.0, 200.0]))
    np.testing.assert_allclose(means, np.asarray([4.0, 1.0]))
    np.testing.assert_allclose(stds, np.asarray([1.0, 0.0]))

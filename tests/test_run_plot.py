from pathlib import Path

import run_plot


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

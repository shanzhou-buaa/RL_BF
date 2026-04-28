from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_convergence(log_dir: Path) -> None:
    set_ieee_style()
    figures = log_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(log_dir / "eval_history.csv")
    if not rows:
        return
    for metric, ylabel, name in (
        ("eval_reward", "Evaluation reward", "convergence_reward"),
        ("eval_objective", "Evaluation objective J", "convergence_objective"),
    ):
        plt.figure(figsize=(5.8, 3.4))
        for algo in sorted({row["algo"] for row in rows}):
            algo_rows = [row for row in rows if row["algo"] == algo]
            updates = np.asarray([float(row["update"]) for row in algo_rows])
            values = np.asarray([float(row[metric]) for row in algo_rows])
            plt.plot(updates, values, marker="o", label=algo.upper())
        plt.xlabel("Update")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / f"{name}.png", dpi=220)
        plt.savefig(figures / f"{name}.pdf")
        plt.close()


def plot_beampattern_npz(log_dir: Path) -> None:
    set_ieee_style()
    path = log_dir / "patterns.npz"
    if not path.exists():
        return
    data = np.load(path)
    angle_grid = data["angle_grid"]
    desired = data["desired"]
    pattern_keys = [key for key in data.files if key.startswith("pattern_")]
    if not pattern_keys:
        return
    ref = max(float(np.max(data[key])) for key in pattern_keys)
    ref = max(ref, 1.0e-9)
    figures = log_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.8, 3.4))
    plt.plot(angle_grid, np.where(desired > 0, 0.0, -40.0), "k--", label="Desired")
    for key in pattern_keys:
        label = key.replace("pattern_", "").upper()
        y = 10.0 * np.log10(np.maximum(data[key] / ref, 1.0e-12))
        plt.plot(angle_grid, y, label=label)
    plt.xlabel("Angle (degree)")
    plt.ylabel("Normalized beampattern (dB)")
    plt.ylim(-40, 5)
    plt.xlim(float(np.min(angle_grid)), float(np.max(angle_grid)))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "beampattern.png", dpi=220)
    plt.savefig(figures / "beampattern.pdf")
    plt.close()


def plot_entropy_stats(log_dir: Path) -> None:
    set_ieee_style()
    rows = read_csv_rows(log_dir / "entropy_history.csv")
    if not rows:
        return
    figures = log_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.8, 3.4))
    for algo in sorted({row["algo"] for row in rows}):
        algo_rows = [row for row in rows if row["algo"] == algo]
        updates = np.asarray([float(row["update"]) for row in algo_rows])
        macro = np.asarray([float(row["macro_step_rate"]) for row in algo_rows])
        high = np.asarray([float(row["high_entropy_rate"]) for row in algo_rows])
        plt.plot(updates, high, label=f"{algo.upper()} high entropy")
        plt.plot(updates, macro, linestyle="--", label=f"{algo.upper()} macro")
    plt.xlabel("Update")
    plt.ylabel("Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "entropy_macro_stats.png", dpi=220)
    plt.savefig(figures / "entropy_macro_stats.pdf")
    plt.close()


def plot_runtime_bar(log_dir: Path) -> None:
    set_ieee_style()
    path = log_dir / "runtime.json"
    if not path.exists():
        return
    runtime = json.loads(path.read_text(encoding="utf-8"))
    if not runtime:
        return
    figures = log_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = list(runtime.keys())
    values = [float(runtime[name]) for name in methods]
    x = np.arange(len(methods))
    plt.figure(figsize=(5.8, 3.4))
    plt.bar(x, values, width=0.6)
    plt.xticks(x, [name.upper() for name in methods])
    plt.ylabel("Runtime (s)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "runtime_bar.png", dpi=220)
    plt.savefig(figures / "runtime_bar.pdf")
    plt.close()


def plot_all(log_dir: str | Path) -> None:
    path = Path(log_dir)
    plot_convergence(path)
    plot_beampattern_npz(path)
    plot_entropy_stats(path)
    plot_runtime_bar(path)

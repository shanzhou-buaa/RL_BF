from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import EnvConfig
from .env import ISACBeamformingEnv


def plot_beampattern(
    cfg: EnvConfig,
    patterns: Dict[str, np.ndarray],
    output: str,
    seed: int = 0,
) -> None:
    env = ISACBeamformingEnv(cfg, seed=seed)
    angle_grid = env.angle_grid
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.0))
    for name, pattern in patterns.items():
        plt.plot(angle_grid, 10.0 * np.log10(np.maximum(pattern, 1.0e-9)), label=name)
    desired = env.desired * max(max(np.max(p) for p in patterns.values()), 1.0e-6)
    plt.plot(angle_grid, 10.0 * np.log10(np.maximum(desired, 1.0e-9)), "k--", label="desired")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Beampattern (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.axis([-100, 100, -15, 10])
    plt.savefig(output, dpi=180)
    plt.close()


def plot_training_rewards(
    histories: Mapping[str, Sequence[Mapping[str, float]]],
    output_dir: str | Path,
    episodes_per_update: int,
    dpi: int = 180,
    smooth_window: int = 5,
) -> None:
    """Plot mean episodic reward recorded at every PPO update."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.0))
    has_curve = False
    for method, history in histories.items():
        if not history:
            continue
        updates = np.asarray([float(row["update"]) for row in history], dtype=np.float64)
        rewards = np.asarray([float(row["reward"]) for row in history], dtype=np.float64)
        episodes = updates * max(episodes_per_update, 1)
        if smooth_window > 1 and rewards.size >= smooth_window:
            kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
            padded = np.pad(rewards, (smooth_window - 1, 0), mode="edge")
            rewards_to_plot = np.convolve(padded, kernel, mode="valid")
        else:
            rewards_to_plot = rewards
        plt.plot(episodes, rewards_to_plot, linewidth=1.9, label=method)
        has_curve = True

    if not has_curve:
        plt.close()
        return

    plt.xlabel("Training episode")
    plt.ylabel("Mean episode reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=dpi)
    plt.savefig(output_dir / "reward_curve.pdf")
    plt.close()

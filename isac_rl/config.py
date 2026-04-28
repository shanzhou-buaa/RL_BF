from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    """Fixed Liu2020-style ISAC setup."""

    M: int = 10
    K: int = 2
    target_angles_deg: Tuple[float, ...] = (-40.0, 0.0, 40.0)
    beam_width_deg: float = 10.0
    total_power: float = 1.0
    noise_power: float = 0.01
    sinr_threshold_db: float = 12.0
    angle_grid_min_deg: float = -90.0
    angle_grid_max_deg: float = 90.0
    angle_grid_step_deg: float = 0.1
    coarse_grid_step_deg: float = 2.0
    cross_corr_weight: float = 1.0
    tau_sinr: float = 2.0
    Lr1_ref: float = 1.0
    Lr2_ref: float = 1.0
    Lr_ref: float = 1.0
    w_bp: float = 1.0
    w_cc: float = 0.3
    w_sinr: float = 8.0
    w_side: float = 0.8
    w_band: float = 1.0
    w_balance: float = 0.2

    @property
    def angle_grid(self) -> np.ndarray:
        return np.arange(
            self.angle_grid_min_deg,
            self.angle_grid_max_deg + 0.5 * self.angle_grid_step_deg,
            self.angle_grid_step_deg,
            dtype=np.float64,
        )

    @property
    def coarse_grid(self) -> np.ndarray:
        return np.arange(
            self.angle_grid_min_deg,
            self.angle_grid_max_deg + 0.5 * self.coarse_grid_step_deg,
            self.coarse_grid_step_deg,
            dtype=np.float64,
        )

    @property
    def sinr_threshold(self) -> float:
        return 10.0 ** (self.sinr_threshold_db / 10.0)


@dataclass(frozen=True)
class EnvConfig:
    episode_steps: int = 8
    action_scale: float = 0.03
    beta_progress: float = 2.0
    progress_scale: float = 0.05
    beta_margin: float = 0.5
    beta_feasible: float = 0.5
    terminal_weight: float = 2.0
    feasible_terminal_bonus: float = 2.0
    ema_reward_decay: float = 0.9
    reward_objective_scale: float = 10.0
    reward_clip: float = 10.0


@dataclass(frozen=True)
class PPOConfig:
    updates: int = 300
    episodes_per_update: int = 256
    ppo_epochs: int = 5
    minibatch_size: int = 512
    lr: float = 3.0e-4
    hidden_dim: int = 128
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 1.0e-3
    max_grad_norm: float = 1.0
    target_kl: float = 0.05
    max_macro_len: int = 3
    entropy_quantile: float = 0.4
    entropy_threshold_ema: float = 0.8
    alpha_he_init: float = 1.0e-3
    alpha_he_min: float = 1.0e-5
    alpha_he_max: float = 5.0e-2
    target_high_entropy_rate: float = 0.45


@dataclass(frozen=True)
class TrainConfig:
    algos: Tuple[str, ...] = ("ppo", "heppo")
    seeds: Tuple[int, ...] = (1,)
    eval_channels: int = 64
    eval_seed: int = 2026
    eval_interval: int = 1
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    extra: dict = field(default_factory=dict)

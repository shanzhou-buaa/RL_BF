from __future__ import annotations

from typing import Dict

import numpy as np

from .config import EnvConfig, SystemConfig
from .metrics import (
    EPS,
    MetricCache,
    compute_all_metrics,
    compute_beampattern,
    desired_beampattern,
    per_antenna_power_normalize,
    steering_matrix,
)
from .utils import complex_normal


class ISACBeamformingEnv:
    """Pure residual-control environment for Liu2020 ISAC beamforming."""

    def __init__(self, sys_cfg: SystemConfig, env_cfg: EnvConfig, seed: int = 0):
        self.sys = sys_cfg
        self.cfg = env_cfg
        self.rng = np.random.default_rng(seed)
        self.H: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.t = 0
        self.prev_info: Dict[str, object] | None = None
        self.prev_reward = 0.0
        self.ema_reward = 0.0
        self.last_objective_delta = 0.0
        self.last_sinr_cost_delta = 0.0
        self.current_info: Dict[str, object] | None = None
        self.metric_cache = MetricCache.from_config(self.sys)
        self._coarse_steering = steering_matrix(self.sys.M, self.sys.coarse_grid)
        self._coarse_desired = desired_beampattern(
            self.sys.coarse_grid, self.sys.target_angles_deg, self.sys.beam_width_deg
        )

    @property
    def action_dim(self) -> int:
        return 2 * self.sys.M * (self.sys.K + self.sys.M)

    @property
    def state_dim(self) -> int:
        if self.H is None or self.W is None:
            self.reset()
        return int(sum(value.size for value in self.build_state_groups().values()))

    def reset(self) -> np.ndarray:
        self.H = complex_normal((self.sys.K, self.sys.M), self.rng)
        self.W = per_antenna_power_normalize(
            complex_normal((self.sys.M, self.sys.K + self.sys.M), self.rng),
            self.sys.total_power,
        )
        self.t = 0
        self.current_info = compute_all_metrics(
            self.W,
            self.H,
            self.sys,
            self.metric_cache,
        )
        self.prev_info = dict(self.current_info)
        self.prev_reward = 0.0
        self.ema_reward = 0.0
        self.last_objective_delta = 0.0
        self.last_sinr_cost_delta = 0.0
        return self.build_state()

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.size != self.action_dim:
            raise ValueError(f"expected action size {self.action_dim}, got {action.size}")
        action = np.clip(action, -1.0, 1.0)
        half = action.size // 2
        real = action[:half].reshape(self.sys.M, self.sys.K + self.sys.M)
        imag = action[half:].reshape(self.sys.M, self.sys.K + self.sys.M)
        return real + 1j * imag

    def step(self, action: np.ndarray):
        assert self.H is not None and self.W is not None and self.prev_info is not None
        delta_W = self.decode_action(action)
        W_next = self.W + self.cfg.action_scale * delta_W
        W_next = per_antenna_power_normalize(W_next, self.sys.total_power)
        info = compute_all_metrics(W_next, self.H, self.sys, self.metric_cache)
        next_t = self.t + 1
        done = next_t >= self.cfg.episode_steps
        reward = self.compute_reward(info, done)

        self.last_objective_delta = float(
            self.prev_info["objective"] - info["objective"]
        )
        self.last_sinr_cost_delta = float(self.prev_info["C_sinr"] - info["C_sinr"])
        self.prev_reward = float(reward)
        self.ema_reward = float(
            self.cfg.ema_reward_decay * self.ema_reward
            + (1.0 - self.cfg.ema_reward_decay) * reward
        )
        self.W = W_next
        self.t = next_t
        self.prev_info = dict(info)
        self.current_info = info
        info = dict(info)
        info["reward"] = float(reward)
        info["t"] = self.t
        return self.build_state(), float(reward), bool(done), info

    def compute_reward(self, info: Dict[str, object], done: bool) -> float:
        assert self.prev_info is not None
        previous = float(self.prev_info["objective"])
        current = float(info["objective"])
        progress = (previous - current) / max(abs(previous), EPS)
        objective_penalty = -np.tanh(current / max(self.cfg.reward_objective_scale, EPS))
        reward = (
            objective_penalty
            + self.cfg.beta_progress * np.tanh(progress / self.cfg.progress_scale)
            + self.cfg.beta_margin * np.tanh(float(info["min_sinr_gap_db"]) / 5.0)
            + self.cfg.beta_feasible * float(info["feasible"])
        )
        if done:
            reward += -0.5 * self.cfg.terminal_weight * np.tanh(
                current / max(self.cfg.reward_objective_scale, EPS)
            )
            reward += self.cfg.feasible_terminal_bonus * float(info["feasible"])
        return float(np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip))

    def build_state_groups(self) -> Dict[str, np.ndarray]:
        assert self.H is not None and self.W is not None and self.current_info is not None
        H_scale = np.sqrt(np.mean(np.abs(self.H) ** 2) + EPS)
        H_norm = self.H / H_scale
        channel = np.concatenate([np.real(H_norm).ravel(), np.imag(H_norm).ravel()])

        W_norm = self.W / np.sqrt(self.sys.total_power / self.sys.M)
        beamformer = np.concatenate([np.real(W_norm).ravel(), np.imag(W_norm).ravel()])

        sinr_db = np.asarray(self.current_info["sinr_db"], dtype=np.float64)
        gap_db = sinr_db - self.sys.sinr_threshold_db
        violation = np.log1p(
            np.exp(np.clip((self.sys.sinr_threshold_db - sinr_db) / self.sys.tau_sinr, -60.0, 60.0))
        )
        sinr = np.concatenate(
            [
                gap_db / 20.0,
                violation,
                np.asarray(
                    [
                        float(np.min(gap_db) / 20.0),
                        float(np.mean(violation)),
                        float(self.current_info["feasible"]),
                    ],
                    dtype=np.float64,
                ),
            ]
        )

        target_gains = np.asarray(self.current_info["target_gains"], dtype=np.float64)
        target_mean = max(float(self.current_info["target_mean_gain"]), EPS)
        radar = np.asarray(
            [
                np.log1p(float(self.current_info["Lr1"]) / self.sys.Lr1_ref),
                np.log1p(float(self.current_info["Lr2"]) / self.sys.Lr2_ref),
                np.log1p(float(self.current_info["Lr"]) / self.sys.Lr_ref),
                np.log1p(float(self.current_info["peak_sidelobe_ratio"])),
                np.log1p(float(self.current_info["mean_sidelobe_ratio"])),
                float(self.current_info["target_min_gain"]) / target_mean,
                *[float(value / target_mean) for value in target_gains],
            ],
            dtype=np.float64,
        )

        P_coarse = compute_beampattern(self.W, self._coarse_steering)
        alpha = max(float(self.current_info["alpha"]), EPS)
        beampattern = np.clip(P_coarse / alpha - self._coarse_desired, -5.0, 5.0)

        T = max(self.cfg.episode_steps, 1)
        progress = np.asarray(
            [
                self.prev_reward,
                self.last_objective_delta,
                self.last_sinr_cost_delta,
                self.ema_reward,
                self.t / T,
                (T - self.t) / T,
            ],
            dtype=np.float64,
        )

        return {
            "channel": channel.astype(np.float32),
            "beamformer": beamformer.astype(np.float32),
            "sinr": sinr.astype(np.float32),
            "radar": radar.astype(np.float32),
            "beampattern": beampattern.astype(np.float32),
            "progress": progress.astype(np.float32),
        }

    def build_state(self) -> np.ndarray:
        return np.concatenate(list(self.build_state_groups().values())).astype(np.float32)

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import EnvConfig
from .math_utils import (
    action_dim,
    action_to_residual,
    complex_randn,
    compute_sinr,
    desired_beampattern,
    make_angle_grid,
    radar_loss,
    random_beamformer,
    row_power_normalize,
    steering_matrix,
    zf_beamformer,
)
from .selection import beam_quality_objective


class ISACBeamformingEnv:
    """Finite-horizon continuous-control environment for Liu-style ISAC design."""

    def __init__(self, cfg: EnvConfig, structured_action: bool = True, seed: int = 0):
        self.cfg = cfg
        self.structured_action = structured_action
        self.rng = np.random.default_rng(seed)
        self.angle_grid = make_angle_grid(
            cfg.angle_grid_min_deg, cfg.angle_grid_max_deg, cfg.angle_grid_step_deg
        )
        self.grid_steering = steering_matrix(cfg.num_antennas, self.angle_grid)
        self.target_steering = steering_matrix(
            cfg.num_antennas, np.asarray(cfg.target_angles_deg, dtype=np.float64)
        )
        self.desired = desired_beampattern(
            self.angle_grid, cfg.target_angles_deg, cfg.beam_width_deg
        )
        self.H: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self._cached_metrics: Optional[Dict[str, object]] = None
        self.step_count = 0

    @property
    def action_size(self) -> int:
        return action_dim(
            self.cfg.num_antennas,
            self.cfg.num_users,
            self.structured_action,
        )

    @property
    def state_size(self) -> int:
        m = self.cfg.num_antennas
        k = self.cfg.num_users
        return (
            2 * k * m
            + 2 * m * (k + m)
            + k
            + 1
            + len(self.cfg.target_angles_deg)
            + 1
        )

    def _loss_fn(self, W: np.ndarray) -> float:
        return float(self.objective_for_W(W)["beam_objective"])

    def _beam_diagnostics(self, pattern: np.ndarray, alpha: float) -> Dict[str, object]:
        cfg = self.cfg
        target_indices = [
            int(np.argmin(np.abs(self.angle_grid - theta))) for theta in cfg.target_angles_deg
        ]
        target_power = np.maximum(pattern[target_indices], 1.0e-9)
        target_mean = float(np.mean(target_power))
        target_min = float(np.min(target_power))
        target_min_ratio = float(target_min / max(target_mean, 1.0e-9))

        side_mask = self.desired <= 0.0
        side_ratio = np.asarray(pattern[side_mask], dtype=np.float64) / target_mean
        if side_ratio.size:
            sidelobe_ratio = float(np.max(side_ratio))
        else:
            sidelobe_ratio = 0.0

        target_band_errors = []
        half_width = cfg.beam_width_deg / 2.0
        alpha_safe = max(float(alpha), 1.0e-9)
        normalized_pattern = np.asarray(pattern, dtype=np.float64) / alpha_safe
        for theta, center_idx in zip(cfg.target_angles_deg, target_indices):
            mask = np.abs(self.angle_grid - theta) <= half_width
            target_band_errors.append(float(np.mean((normalized_pattern[mask] - 1.0) ** 2)))
        if side_mask.any():
            sidelobe_leakage = float(np.max(normalized_pattern[side_mask]))
        else:
            sidelobe_leakage = 0.0
        target_band_error_mean = (
            float(np.mean(target_band_errors)) if target_band_errors else 0.0
        )

        return {
            "target_mean": target_mean,
            "target_min": target_min,
            "target_min_ratio": target_min_ratio,
            "sidelobe_ratio": sidelobe_ratio,
            "sidelobe_leakage": sidelobe_leakage,
            "target_band_errors": target_band_errors,
            "target_band_error_mean": target_band_error_mean,
        }

    def objective_for_W(self, W: np.ndarray) -> Dict[str, object]:
        loss, alpha, cross, pattern = self.metrics_for_W(W)
        beampattern_loss = max(
            0.0, float(loss) - self.cfg.cross_corr_weight * float(cross)
        )
        diagnostics = self._beam_diagnostics(pattern, alpha)
        return {
            "radar_loss": float(loss),
            "beam_objective": float(loss),
            "beampattern_loss": float(beampattern_loss),
            "alpha": float(alpha),
            "cross_corr": float(cross),
            "weighted_cross_corr": float(self.cfg.cross_corr_weight * cross),
            "pattern": pattern,
            **diagnostics,
        }

    def reset(self) -> np.ndarray:
        cfg = self.cfg
        self.H = complex_randn((cfg.num_users, cfg.num_antennas), self.rng)
        self._cached_metrics = None
        if cfg.init_mode == "random":
            self.W = random_beamformer(
                cfg.num_antennas,
                cfg.num_users,
                cfg.total_power,
                self.rng,
            )
        elif cfg.init_mode in {"policy", "zf"}:
            self.W, _ = zf_beamformer(
                self.H,
                self.target_steering,
                cfg.total_power,
                cfg.noise_power,
                cfg.sinr_threshold,
                self.rng,
                loss_fn=self._loss_fn,
                comm_safety=cfg.init_comm_safety,
            )
        else:
            raise ValueError(
                f"unknown init_mode={cfg.init_mode!r}; expected random, policy or zf"
            )
        self.step_count = 0
        metrics = self.current_metrics()
        return self._state()

    def metrics_for_W(self, W: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        return radar_loss(
            W,
            self.grid_steering,
            self.desired,
            self.target_steering,
            self.cfg.cross_corr_weight,
        )

    def current_metrics(self) -> Dict[str, object]:
        assert self.W is not None and self.H is not None
        radar_metrics = self.objective_for_W(self.W)
        sinr = compute_sinr(self.W, self.H, self.cfg.num_users, self.cfg.noise_power)
        margin = sinr - self.cfg.sinr_threshold
        cost = float(np.mean(np.maximum(-margin, 0.0) / self.cfg.sinr_threshold))
        min_sinr = float(np.min(sinr))
        metrics = {
            **radar_metrics,
            "sinr": sinr,
            "min_sinr": min_sinr,
            "min_margin": float(np.min(margin)),
            "cost": cost,
            "feasible": bool(np.all(margin >= 0.0)),
        }
        self._cached_metrics = {key: value for key, value in metrics.items() if key != "pattern"}
        return metrics

    def _candidate_info(
        self,
        W: np.ndarray,
        step_scale: float,
        action_norm: float,
        accepted: bool,
    ) -> Dict[str, object]:
        cfg = self.cfg
        new_metrics = self.objective_for_W(W)
        new_loss = float(new_metrics["beam_objective"])
        sinr = compute_sinr(W, self.H, cfg.num_users, cfg.noise_power)
        margin = sinr - cfg.sinr_threshold
        cost = float(np.mean(np.maximum(-margin, 0.0) / cfg.sinr_threshold))
        feasible = bool(np.all(margin >= 0.0))
        sinr_violations = int(np.sum(margin < 0.0))
        sinr_db = 10.0 * np.log10(np.maximum(sinr, 1.0e-12))
        sinr_gap_db = np.maximum(cfg.sinr_threshold_db - sinr_db, 0.0)
        sinr_gap_scale = max(abs(float(cfg.sinr_threshold_db)), 1.0)
        sinr_penalty = float(
            cfg.sinr_violation_penalty * np.sum(sinr_gap_db / sinr_gap_scale)
        )
        row_power = np.sum(np.abs(W) ** 2, axis=1)
        power_error = float(np.linalg.norm(row_power - cfg.total_power / cfg.num_antennas))
        power_penalty = float(cfg.power_penalty_weight * power_error**2)
        quality_objective = beam_quality_objective(
            {
                "radar_loss": float(new_metrics["radar_loss"]),
                "beam_objective": float(new_loss),
                "beampattern_loss": float(new_metrics["beampattern_loss"]),
                "sidelobe_leakage": float(new_metrics["sidelobe_leakage"]),
                "sidelobe_ratio": float(new_metrics["sidelobe_ratio"]),
                "target_band_error_mean": float(new_metrics["target_band_error_mean"]),
                "target_min_ratio": float(new_metrics["target_min_ratio"]),
            },
            beampattern_weight=cfg.quality_beampattern_weight,
            sidelobe_leakage_weight=cfg.quality_sidelobe_leakage_weight,
            sidelobe_ratio_weight=cfg.quality_sidelobe_ratio_weight,
            target_band_weight=cfg.quality_target_band_weight,
            target_balance_weight=cfg.quality_target_balance_weight,
        )
        score = float(quality_objective + cfg.constraint_score_weight * cost)

        return {
            "radar_loss": float(new_metrics["radar_loss"]),
            "beam_objective": float(new_loss),
            "beampattern_loss": float(new_metrics["beampattern_loss"]),
            "cross_corr": float(new_metrics["cross_corr"]),
            "weighted_cross_corr": float(
                cfg.cross_corr_weight * new_metrics["cross_corr"]
            ),
            "target_mean": float(new_metrics["target_mean"]),
            "target_min": float(new_metrics["target_min"]),
            "target_min_ratio": float(new_metrics["target_min_ratio"]),
            "sidelobe_ratio": float(new_metrics["sidelobe_ratio"]),
            "sidelobe_leakage": float(new_metrics["sidelobe_leakage"]),
            "target_band_errors": [
                float(value) for value in new_metrics["target_band_errors"]
            ],
            "target_band_error_mean": float(new_metrics["target_band_error_mean"]),
            "sinr": sinr,
            "sinr_db": sinr_db,
            "sinr_gap_db": sinr_gap_db,
            "mean_sinr_gap_db": float(np.mean(sinr_gap_db)),
            "min_sinr": float(np.min(sinr)),
            "min_margin": float(np.min(margin)),
            "cost": cost,
            "feasible": feasible,
            "sinr_violations": sinr_violations,
            "sinr_penalty": sinr_penalty,
            "power_error": power_error,
            "power_penalty": power_penalty,
            "action_norm": action_norm,
            "accepted": accepted,
            "step_scale": float(step_scale),
            "quality_objective": quality_objective,
            "score": score,
        }

    def _line_search_scales(self) -> list[float]:
        if not self.cfg.use_action_line_search:
            return [1.0]
        scales = []
        scale = 1.0
        min_scale = max(float(self.cfg.min_action_step_scale), 0.0)
        while scale >= min_scale:
            scales.append(float(scale))
            scale *= 0.5
        return scales

    def _component_guard_penalty(
        self, info: Dict[str, object], previous: Dict[str, object]
    ) -> float:
        cfg = self.cfg
        penalty = 0.0
        for key in (
            "beampattern_loss",
            "cross_corr",
            "sidelobe_leakage",
            "target_band_error_mean",
        ):
            old_value = float(previous.get(key, info.get(key, 0.0)))
            new_value = float(info.get(key, old_value))
            penalty += max(0.0, new_value - old_value) / max(abs(old_value), 1.0e-9)
        return float(cfg.quality_component_guard_weight * penalty)

    def step(self, action: np.ndarray):
        assert self.W is not None and self.H is not None
        cfg = self.cfg
        prev_metrics = self.current_metrics()
        prev_loss = float(prev_metrics["beam_objective"])
        prev_quality = beam_quality_objective(
            prev_metrics,
            beampattern_weight=cfg.quality_beampattern_weight,
            sidelobe_leakage_weight=cfg.quality_sidelobe_leakage_weight,
            sidelobe_ratio_weight=cfg.quality_sidelobe_ratio_weight,
            target_band_weight=cfg.quality_target_band_weight,
            target_balance_weight=cfg.quality_target_balance_weight,
        )
        delta_W = action_to_residual(
            action,
            self.H,
            cfg.num_antennas,
            cfg.num_users,
            self.structured_action,
        )
        action_norm = float(np.mean(action**2))
        delta_W = cfg.action_scale * delta_W

        candidates: list[tuple[np.ndarray, Dict[str, object]]] = []
        if cfg.use_action_line_search:
            candidates.append(
                (
                    self.W.copy(),
                    self._candidate_info(
                        self.W,
                        step_scale=0.0,
                        action_norm=action_norm,
                        accepted=False,
                    ),
                )
            )
        for scale in self._line_search_scales():
            candidate_W = row_power_normalize(
                self.W + scale * delta_W,
                cfg.total_power,
            )
            candidates.append(
                (
                    candidate_W,
                    self._candidate_info(
                        candidate_W,
                        step_scale=scale,
                        action_norm=action_norm,
                        accepted=True,
                    ),
                )
            )

        for _, info in candidates:
            penalty = self._component_guard_penalty(info, prev_metrics)
            info["component_guard_penalty"] = penalty
            info["score"] = float(info["score"] + penalty)

        new_W, info = min(candidates, key=lambda item: item[1]["score"])
        new_loss = float(info["beam_objective"])
        relative_loss_improvement = float(
            (prev_loss - new_loss) / max(abs(prev_loss), 1.0e-9)
        )
        relative_quality_improvement = float(
            (prev_quality - float(info["quality_objective"]))
            / max(abs(prev_quality), 1.0e-9)
        )
        loss_reward = float(cfg.loss_reward_weight * relative_quality_improvement)
        constraint_reward = float(-cfg.constraint_reward_weight * info["cost"])
        action_reward = float(-cfg.action_penalty * action_norm)
        reward = float(loss_reward + constraint_reward + action_reward)

        info.update(
            {
                "loss_reward": loss_reward,
                "quality_reward": reward,
                "relative_loss_improvement": relative_loss_improvement,
                "relative_quality_improvement": relative_quality_improvement,
                "improvement_reward": loss_reward,
                "unconstrained_reward": float(loss_reward + action_reward),
                "constraint_reward": constraint_reward,
                "reward": reward,
            }
        )
        self.W = new_W
        self.step_count += 1
        done = self.step_count >= cfg.max_steps
        self._cached_metrics = info
        return self._state(), reward, done, info

    def _state(self) -> np.ndarray:
        assert self.W is not None and self.H is not None
        metrics = self._cached_metrics if self._cached_metrics is not None else self.current_metrics()
        target_band_errors = list(metrics.get("target_band_errors", []))
        if len(target_band_errors) < len(self.cfg.target_angles_deg):
            target_band_errors.extend(
                [0.0] * (len(self.cfg.target_angles_deg) - len(target_band_errors))
            )
        parts = [
            np.real(self.H).ravel(),
            np.imag(self.H).ravel(),
            np.real(self.W).ravel(),
            np.imag(self.W).ravel(),
            np.asarray(metrics["sinr"], dtype=np.float64) / self.cfg.sinr_threshold,
            np.asarray([metrics["beam_objective"]], dtype=np.float64),
            np.asarray(target_band_errors[: len(self.cfg.target_angles_deg)], dtype=np.float64),
            np.asarray([metrics.get("sidelobe_leakage", 0.0)], dtype=np.float64),
        ]
        return np.concatenate(parts).astype(np.float32)

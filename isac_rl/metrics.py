from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Tuple

import numpy as np

from .config import SystemConfig

EPS = 1.0e-9


def steering_matrix(num_antennas: int, angles_deg: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    m = np.arange(num_antennas, dtype=np.float64)
    return np.exp(-1j * np.pi * np.outer(np.sin(angles), m))


def desired_beampattern(
    angle_grid_deg: np.ndarray,
    target_angles_deg: Tuple[float, ...],
    beam_width_deg: float,
) -> np.ndarray:
    desired = np.zeros_like(angle_grid_deg, dtype=np.float64)
    half_width = beam_width_deg / 2.0
    for theta in target_angles_deg:
        desired[np.abs(angle_grid_deg - theta) <= half_width] = 1.0
    return desired


@dataclass(frozen=True)
class MetricCache:
    angle_grid: np.ndarray
    grid_steering: np.ndarray
    target_steering: np.ndarray
    desired: np.ndarray

    @classmethod
    def from_config(cls, cfg: SystemConfig) -> "MetricCache":
        angle_grid = cfg.angle_grid
        grid_steering = steering_matrix(cfg.M, angle_grid)
        target_steering = steering_matrix(
            cfg.M,
            np.asarray(cfg.target_angles_deg, dtype=np.float64),
        )
        desired = desired_beampattern(
            angle_grid,
            cfg.target_angles_deg,
            cfg.beam_width_deg,
        )
        return cls(
            angle_grid=angle_grid,
            grid_steering=grid_steering,
            target_steering=target_steering,
            desired=desired,
        )


def per_antenna_power_normalize(W: np.ndarray, total_power: float) -> np.ndarray:
    Wn = np.asarray(W, dtype=np.complex128).copy()
    target = np.sqrt(total_power / Wn.shape[0])
    row_norms = np.linalg.norm(Wn, axis=1)
    empty = row_norms < EPS
    if np.any(empty):
        Wn[empty, 0] = target
        row_norms = np.linalg.norm(Wn, axis=1)
    Wn *= (target / np.maximum(row_norms, EPS))[:, None]
    return Wn


def compute_covariance(W: np.ndarray) -> np.ndarray:
    return W @ W.conj().T


def compute_beampattern(W: np.ndarray, grid_steering: np.ndarray) -> np.ndarray:
    R = compute_covariance(W)
    pattern = np.real(np.einsum("lm,mn,ln->l", grid_steering.conj(), R, grid_steering))
    return np.maximum(pattern, 0.0)


def optimal_alpha(pattern: np.ndarray, desired: np.ndarray) -> float:
    denom = float(np.dot(desired, desired)) + EPS
    return max(0.0, float(np.dot(desired, pattern) / denom))


def compute_Lr1(pattern: np.ndarray, desired: np.ndarray, alpha: float) -> float:
    return float(np.mean((alpha * desired - pattern) ** 2))


def compute_Lr2(R: np.ndarray, target_steering: np.ndarray) -> float:
    total = 0.0
    count = 0
    for p, q in combinations(range(target_steering.shape[0]), 2):
        value = target_steering[q].conj() @ R @ target_steering[p]
        total += float(np.abs(value) ** 2)
        count += 1
    return float(total / max(count, 1))


def compute_sinr(
    W: np.ndarray,
    H: np.ndarray,
    num_users: int,
    noise_power: float,
) -> np.ndarray:
    F = H @ W
    Fc = F[:, :num_users]
    Fr = F[:, num_users:]
    signal = np.abs(np.diag(Fc)) ** 2
    multi_user = np.sum(np.abs(Fc) ** 2, axis=1) - signal
    radar_interference = np.sum(np.abs(Fr) ** 2, axis=1)
    return signal / np.maximum(multi_user + radar_interference + noise_power, EPS)


def compute_sidelobe_metrics(
    pattern: np.ndarray,
    desired: np.ndarray,
    angle_grid: np.ndarray,
    target_angles_deg: Tuple[float, ...],
    beam_width_deg: float,
    alpha: float,
) -> Dict[str, object]:
    target_indices = [
        int(np.argmin(np.abs(angle_grid - theta))) for theta in target_angles_deg
    ]
    target_gains = np.maximum(pattern[target_indices], EPS)
    target_mean_gain = float(np.mean(target_gains))
    target_min_gain = float(np.min(target_gains))
    side_mask = desired <= 0.0
    side_values = np.maximum(pattern[side_mask], 0.0)
    if side_values.size:
        peak_side = float(np.max(side_values))
        mean_side = float(np.mean(side_values))
    else:
        peak_side = 0.0
        mean_side = 0.0
    peak_sidelobe_ratio = peak_side / max(target_mean_gain, EPS)
    mean_sidelobe_ratio = mean_side / max(target_mean_gain, EPS)

    band_errors = []
    half_width = beam_width_deg / 2.0
    alpha_safe = max(alpha, EPS)
    normalized = pattern / alpha_safe
    for theta in target_angles_deg:
        band = np.abs(angle_grid - theta) <= half_width
        band_errors.append(float(np.mean((normalized[band] - 1.0) ** 2)))

    return {
        "target_gains": target_gains.astype(np.float64),
        "target_mean_gain": target_mean_gain,
        "target_min_gain": target_min_gain,
        "target_min_ratio": target_min_gain / max(target_mean_gain, EPS),
        "peak_sidelobe_ratio": float(peak_sidelobe_ratio),
        "mean_sidelobe_ratio": float(mean_sidelobe_ratio),
        "target_band_error": float(np.mean(band_errors) if band_errors else 0.0),
    }


def compute_all_metrics(
    W: np.ndarray,
    H: np.ndarray,
    cfg: SystemConfig,
    cache: MetricCache | None = None,
) -> Dict[str, object]:
    if cache is None:
        cache = MetricCache.from_config(cfg)
    angle_grid = cache.angle_grid
    grid_steering = cache.grid_steering
    target_steering = cache.target_steering
    desired = cache.desired
    pattern = compute_beampattern(W, grid_steering)
    alpha = optimal_alpha(pattern, desired)
    R = compute_covariance(W)
    Lr1 = compute_Lr1(pattern, desired, alpha)
    Lr2 = compute_Lr2(R, target_steering)
    Lr = float(Lr1 + cfg.cross_corr_weight * Lr2)
    sinr = compute_sinr(W, H, cfg.K, cfg.noise_power)
    sinr_db = 10.0 * np.log10(np.maximum(sinr, EPS))
    gap_db = sinr_db - cfg.sinr_threshold_db
    violation = np.log1p(np.exp(np.clip((cfg.sinr_threshold_db - sinr_db) / cfg.tau_sinr, -60.0, 60.0)))
    C_sinr = float(np.mean(violation**2))
    side = compute_sidelobe_metrics(
        pattern, desired, angle_grid, cfg.target_angles_deg, cfg.beam_width_deg, alpha
    )
    C_side = float(np.log1p(side["peak_sidelobe_ratio"]))
    C_band = float(side["target_band_error"])
    C_balance = float(max(0.0, 1.0 - side["target_min_ratio"]))
    objective = float(
        cfg.w_bp * np.log1p(Lr1 / cfg.Lr1_ref)
        + cfg.w_cc * np.log1p(Lr2 / cfg.Lr2_ref)
        + cfg.w_sinr * C_sinr
        + cfg.w_side * C_side
        + cfg.w_band * C_band
        + cfg.w_balance * C_balance
    )
    return {
        "objective": objective,
        "Lr": Lr,
        "Lr1": float(Lr1),
        "Lr2": float(Lr2),
        "alpha": float(alpha),
        "pattern": pattern,
        "desired": desired,
        "sinr": sinr,
        "sinr_db": sinr_db,
        "sinr_gap_db": gap_db,
        "min_sinr": float(np.min(sinr)),
        "min_sinr_db": float(np.min(sinr_db)),
        "min_sinr_gap_db": float(np.min(gap_db)),
        "feasible": bool(np.all(gap_db >= 0.0)),
        "C_sinr": C_sinr,
        "C_side": C_side,
        "C_band": C_band,
        "C_balance": C_balance,
        **side,
    }

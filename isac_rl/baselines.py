from __future__ import annotations

from time import perf_counter
from typing import Dict, Tuple

import numpy as np

from .config import SystemConfig
from .metrics import compute_all_metrics, compute_sinr, per_antenna_power_normalize, steering_matrix
from .utils import complex_normal

EPS = 1.0e-9


def channel_pinv(H: np.ndarray) -> np.ndarray:
    gram = H @ H.conj().T
    return H.conj().T @ np.linalg.pinv(gram + EPS * np.eye(gram.shape[0]))


def nullspace_basis(H: np.ndarray, tol: float = 1.0e-8) -> np.ndarray:
    _, s, vh = np.linalg.svd(H, full_matrices=True)
    rank = int(np.sum(s > tol * max(H.shape) * (s[0] if s.size else 1.0)))
    return vh.conj().T[:, rank:]


def zf_baseline(H: np.ndarray, cfg: SystemConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    Hplus = channel_pinv(H)
    required_gain = np.sqrt(max(cfg.sinr_threshold * cfg.noise_power * 1.05, EPS))
    Wc = Hplus @ np.diag(np.full(cfg.K, required_gain, dtype=np.float64))
    target_steering = steering_matrix(
        cfg.M, np.asarray(cfg.target_angles_deg, dtype=np.float64)
    )
    N = nullspace_basis(H)
    Wr = np.zeros((cfg.M, cfg.M), dtype=np.complex128)
    for col in range(cfg.M):
        steering = target_steering[col % target_steering.shape[0]]
        if N.size:
            Wr[:, col] = N @ (N.conj().T @ steering)
        else:
            Wr[:, col] = steering
    best_W = None
    best_metrics = None
    for cs in np.geomspace(0.7, 18.0, 18):
        for rs in np.geomspace(0.15, 3.0, 10):
            W = per_antenna_power_normalize(np.hstack([cs * Wc, rs * Wr]), cfg.total_power)
            metrics = compute_all_metrics(W, H, cfg)
            if best_metrics is None:
                best_W, best_metrics = W, metrics
                continue
            if metrics["feasible"] and not best_metrics["feasible"]:
                best_W, best_metrics = W, metrics
            elif metrics["feasible"] == best_metrics["feasible"]:
                key = "objective" if metrics["feasible"] else "min_sinr_db"
                better = metrics[key] < best_metrics[key] if key == "objective" else metrics[key] > best_metrics[key]
                if better:
                    best_W, best_metrics = W, metrics
    assert best_W is not None and best_metrics is not None
    return best_W, best_metrics


def random_search_sdr_surrogate(
    H: np.ndarray,
    cfg: SystemConfig,
    rng: np.random.Generator,
    candidates: int = 256,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Offline SDR placeholder used when no convex SDR solver is wired in."""

    best_W = None
    best_metrics = None
    for _ in range(candidates):
        W = per_antenna_power_normalize(
            complex_normal((cfg.M, cfg.K + cfg.M), rng), cfg.total_power
        )
        metrics = compute_all_metrics(W, H, cfg)
        if best_metrics is None or metrics["objective"] < best_metrics["objective"]:
            best_W, best_metrics = W, metrics
    assert best_W is not None and best_metrics is not None
    return best_W, best_metrics


def evaluate_baseline(name: str, H: np.ndarray, cfg: SystemConfig, seed: int = 0):
    start = perf_counter()
    if name == "zf":
        W, metrics = zf_baseline(H, cfg)
    elif name == "sdr":
        W, metrics = random_search_sdr_surrogate(H, cfg, np.random.default_rng(seed))
        metrics = {**metrics, "note": "random-search SDR surrogate"}
    else:
        raise ValueError(f"unknown baseline: {name}")
    metrics = dict(metrics)
    metrics["runtime_sec"] = perf_counter() - start
    return W, metrics

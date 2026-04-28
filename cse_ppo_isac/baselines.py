from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import EnvConfig
from .env import ISACBeamformingEnv
from .math_utils import (
    EPS,
    compute_sinr,
    radar_loss_from_covariance,
    radar_loss,
    zf_beamformer,
)


def _mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    return {
        "radar_loss": float(np.mean([row["radar_loss"] for row in rows])),
        "beam_objective": float(
            np.mean([row.get("beam_objective", row["radar_loss"]) for row in rows])
        ),
        "beampattern_loss": float(
            np.mean([row.get("beampattern_loss", row["radar_loss"]) for row in rows])
        ),
        "cross_corr": float(np.mean([row.get("cross_corr", 0.0) for row in rows])),
        "min_sinr": float(np.mean([row["min_sinr"] for row in rows])),
        "feasible_rate": float(np.mean([row["feasible"] for row in rows])),
        "cost": float(np.mean([row["cost"] for row in rows])),
        "solve_time_sec": float(np.mean([row.get("solve_time_sec", 0.0) for row in rows])),
    }


def _parallel_map(fn, jobs: List[Tuple], num_workers: int):
    if num_workers <= 1 or len(jobs) <= 1:
        return [fn(job) for job in jobs]
    chunksize = max(1, len(jobs) // (4 * num_workers))
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(fn, jobs, chunksize=chunksize))


def _zf_episode(job: Tuple[EnvConfig, int]) -> Dict[str, float]:
    cfg, seed = job
    env = ISACBeamformingEnv(cfg, structured_action=True, seed=seed)
    env.reset()
    assert env.H is not None
    start = perf_counter()
    W, _ = zf_beamformer(
        env.H,
        env.target_steering,
        cfg.total_power,
        cfg.noise_power,
        cfg.sinr_threshold,
        env.rng,
        loss_fn=env._loss_fn,
    )
    env.W = W
    metrics = env.current_metrics()
    return {
        "radar_loss": float(metrics["radar_loss"]),
        "beam_objective": float(metrics.get("beam_objective", metrics["radar_loss"])),
        "beampattern_loss": float(metrics.get("beampattern_loss", metrics["radar_loss"])),
        "cross_corr": float(metrics.get("cross_corr", 0.0)),
        "min_sinr": float(metrics["min_sinr"]),
        "feasible": float(metrics["feasible"]),
        "cost": float(metrics["cost"]),
        "solve_time_sec": perf_counter() - start,
    }


def evaluate_zf(
    cfg: EnvConfig,
    episodes: int,
    seed: int = 0,
    num_workers: int = 1,
) -> Dict[str, float]:
    jobs = [(cfg, seed + ep) for ep in range(episodes)]
    return _mean_metrics(_parallel_map(_zf_episode, jobs, num_workers))


def _radar_objective_cvxpy(cp, R, alpha, env: ISACBeamformingEnv):
    pattern_terms = []
    for a in env.grid_steering:
        A = np.outer(a, a.conj())
        pattern_terms.append(cp.real(cp.trace(R @ A)))
    pattern = cp.hstack(pattern_terms)
    mse = cp.sum_squares(alpha * env.desired - pattern) / len(env.desired)

    cross_terms = []
    for p in range(env.target_steering.shape[0]):
        for q in range(p + 1, env.target_steering.shape[0]):
            ap = env.target_steering[p]
            aq = env.target_steering[q]
            A = np.outer(ap, aq.conj())
            cross_terms.append(cp.square(cp.abs(cp.trace(R @ A))))
    if cross_terms:
        cross = cp.sum(cross_terms) / len(cross_terms)
    else:
        cross = 0.0
    return mse + env.cfg.cross_corr_weight * cross


def _row_quadratic_cvxpy(cp, row: np.ndarray, X):
    return cp.real((row.reshape(1, -1) @ X @ row.conj().reshape(-1, 1))[0, 0])


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs * eigvals) @ eigvecs.conj().T


def recover_sdr_precoder(
    H: np.ndarray,
    R_value: np.ndarray,
    Rk_values: List[np.ndarray],
    cfg: EnvConfig,
) -> np.ndarray:
    """Recover W=[Wc,Wr] from Liu2020 SDR matrices."""

    m = cfg.num_antennas
    k = cfg.num_users
    Wc = np.zeros((m, k), dtype=np.complex128)
    used_cov = np.zeros((m, m), dtype=np.complex128)
    for user_idx, Rk in enumerate(Rk_values):
        Rk = _project_psd(np.asarray(Rk, dtype=np.complex128))
        hk = H[user_idx, :]
        h_col = hk.conj().reshape(-1, 1)
        denom = float(np.real(hk.reshape(1, -1) @ Rk @ h_col))
        if denom <= EPS:
            eigvals, eigvecs = np.linalg.eigh(Rk)
            w = eigvecs[:, int(np.argmax(eigvals))] * np.sqrt(max(np.max(eigvals), 0.0))
        else:
            w = (Rk @ h_col).ravel() / np.sqrt(denom)
        Wc[:, user_idx] = w
        used_cov += np.outer(w, w.conj())

    residual = _project_psd(R_value - used_cov)
    eigvals, eigvecs = np.linalg.eigh(residual)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    Wr = eigvecs @ np.diag(np.sqrt(eigvals))
    if Wr.shape[1] < m:
        Wr = np.pad(Wr, ((0, 0), (0, m - Wr.shape[1])), mode="constant")
    else:
        Wr = Wr[:, :m]
    return np.hstack([Wc, Wr])


def solve_sdr_solution(
    cfg: EnvConfig,
    H: np.ndarray,
    env: ISACBeamformingEnv,
    solver: str = "SCS",
    max_iters: int = 2500,
    eps: float = 1.0e-4,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Solve Liu2020 SDR relaxation with cvxpy and recover a precoder."""

    # cvxpy 1.4.x may still access scipy sparse `.A`, which was removed in
    # newer SciPy releases. Patch it locally before cvxpy canonicalization.
    import scipy.sparse as sp

    for sparse_cls in (sp.csc_matrix, sp.csr_matrix, sp.coo_matrix):
        if not hasattr(sparse_cls, "A"):
            sparse_cls.A = property(lambda self: self.toarray())

    import cvxpy as cp

    m = cfg.num_antennas
    k = cfg.num_users
    gamma = cfg.sinr_threshold
    R = cp.Variable((m, m), hermitian=True)
    Rks = [cp.Variable((m, m), hermitian=True) for _ in range(k)]
    alpha = cp.Variable(nonneg=True)

    objective = _radar_objective_cvxpy(cp, R, alpha, env)
    constraints = [
        R >> 0,
        R - sum(Rks) >> 0,
        cp.real(cp.diag(R)) == cfg.total_power / m,
    ]
    for user_idx in range(k):
        Rk = Rks[user_idx]
        hk = H[user_idx, :]
        constraints.append(Rk >> 0)
        signal = _row_quadratic_cvxpy(cp, hk, Rk)
        total = _row_quadratic_cvxpy(cp, hk, R)
        constraints.append((1.0 + 1.0 / gamma) * signal >= total + cfg.noise_power)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    solver_name = solver.upper()
    solve_kwargs = {"solver": solver_name, "verbose": False}
    if solver_name == "SCS":
        solve_kwargs.update({"max_iters": max_iters, "eps": eps, "normalize": True})
    elif solver_name == "CLARABEL":
        solve_kwargs.update({"max_iter": max_iters, "tol_gap_abs": eps, "tol_feas": eps})

    start = perf_counter()
    problem.solve(**solve_kwargs)
    solve_time = perf_counter() - start
    if problem.status not in ("optimal", "optimal_inaccurate") or R.value is None:
        raise RuntimeError(f"SDR solve failed with status={problem.status}")

    R_value = np.asarray(R.value, dtype=np.complex128)
    R_value = 0.5 * (R_value + R_value.conj().T)
    Rk_values = [np.asarray(Rk.value, dtype=np.complex128) for Rk in Rks]
    loss, _, cross, _ = radar_loss_from_covariance(
        R_value,
        env.grid_steering,
        env.desired,
        env.target_steering,
        cfg.cross_corr_weight,
    )
    beampattern_loss = max(0.0, float(loss) - cfg.cross_corr_weight * float(cross))
    sinrs = []
    for user_idx, Rk_value in enumerate(Rk_values):
        hk = H[user_idx, :]
        signal = float(np.real(hk @ Rk_value @ hk.conj()))
        total = float(np.real(hk @ R_value @ hk.conj()))
        interference = max(total - signal, 0.0)
        sinrs.append(signal / max(interference + cfg.noise_power, EPS))
    sinrs = np.asarray(sinrs, dtype=np.float64)
    margin = sinrs - gamma
    cost = float(np.mean(np.maximum(-margin, 0.0) / gamma))
    W = recover_sdr_precoder(H, R_value, Rk_values, cfg)
    beamformer_loss, _, beamformer_cross, _ = radar_loss(
        W,
        env.grid_steering,
        env.desired,
        env.target_steering,
        cfg.cross_corr_weight,
    )
    beamformer_beampattern_loss = max(
        0.0, float(beamformer_loss) - cfg.cross_corr_weight * float(beamformer_cross)
    )
    beamformer_sinrs = compute_sinr(W, H, cfg.num_users, cfg.noise_power)
    metrics = {
        "radar_loss": float(loss),
        "beam_objective": float(loss),
        "beampattern_loss": float(beampattern_loss),
        "cross_corr": float(cross),
        "min_sinr": float(np.min(sinrs)),
        "feasible": float(np.all(margin >= -1.0e-3)),
        "cost": cost,
        "solve_time_sec": solve_time,
        "status": problem.status,
        "objective": float(problem.value),
        "beamformer_radar_loss": float(beamformer_loss),
        "beamformer_beampattern_loss": float(beamformer_beampattern_loss),
        "beamformer_min_sinr": float(np.min(beamformer_sinrs)),
    }
    return metrics, W


def solve_sdr_beamformer(
    cfg: EnvConfig,
    H: np.ndarray,
    env: ISACBeamformingEnv,
    solver: str = "SCS",
    max_iters: int = 2500,
    eps: float = 1.0e-4,
) -> Dict[str, float]:
    metrics, _ = solve_sdr_solution(
        cfg, H, env, solver=solver, max_iters=max_iters, eps=eps
    )
    return metrics


def _sdr_episode(job: Tuple[EnvConfig, int, str, int, float]) -> Dict[str, float]:
    cfg, seed, solver, max_iters, eps = job
    env = ISACBeamformingEnv(cfg, structured_action=True, seed=seed)
    env.reset()
    assert env.H is not None
    return solve_sdr_beamformer(cfg, env.H, env, solver=solver, max_iters=max_iters, eps=eps)


def evaluate_sdr_if_available(
    cfg: EnvConfig,
    episodes: int,
    seed: int = 0,
    num_workers: int = 1,
    solver: str = "SCS",
    max_iters: int = 2500,
    eps: float = 1.0e-4,
) -> Dict[str, float]:
    try:
        import cvxpy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "SDR baseline requires cvxpy. Run with /home/user/anaconda3/envs/YH_RL/bin/python."
        ) from exc

    jobs = [(cfg, seed + ep, solver, max_iters, eps) for ep in range(episodes)]
    rows = _parallel_map(_sdr_episode, jobs, num_workers)
    output = _mean_metrics(rows)
    output["status"] = ",".join(sorted({str(row.get("status", "")) for row in rows}))
    output["objective"] = float(np.mean([row.get("objective", np.nan) for row in rows]))
    return output


def sinr_db(value: float) -> float:
    return 10.0 * np.log10(max(value, 1.0e-12))

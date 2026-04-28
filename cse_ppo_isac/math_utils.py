from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


EPS = 1.0e-9


def complex_randn(shape, rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def steering_matrix(num_antennas: int, angles_deg: np.ndarray) -> np.ndarray:
    """ULA steering rows for half-wavelength spacing."""

    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    m = np.arange(num_antennas, dtype=np.float64)
    return np.exp(-1j * np.pi * np.outer(np.sin(angles_rad), m))


def make_angle_grid(min_deg: float, max_deg: float, step_deg: float) -> np.ndarray:
    return np.arange(min_deg, max_deg + 0.5 * step_deg, step_deg, dtype=np.float64)


def desired_beampattern(
    angle_grid_deg: np.ndarray,
    target_angles_deg: Tuple[float, ...],
    beam_width_deg: float,
) -> np.ndarray:
    desired = np.zeros_like(angle_grid_deg, dtype=np.float64)
    half = beam_width_deg / 2.0
    for theta in target_angles_deg:
        desired[np.abs(angle_grid_deg - theta) <= half] = 1.0
    return desired


def row_power_normalize(W: np.ndarray, total_power: float) -> np.ndarray:
    """Enforce [W W^H]_{m,m}=Pt/M for every transmit antenna."""

    Wn = W.copy()
    target = np.sqrt(total_power / W.shape[0])
    row_norms = np.linalg.norm(Wn, axis=1)
    empty = row_norms < EPS
    if np.any(empty):
        Wn[empty, 0] = target
        row_norms = np.linalg.norm(Wn, axis=1)
    Wn *= (target / np.maximum(row_norms, EPS))[:, None]
    return Wn


def covariance(W: np.ndarray) -> np.ndarray:
    return W @ W.conj().T


def random_beamformer(
    num_antennas: int,
    num_users: int,
    total_power: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random W=[Wc,Wr] initialization with per-antenna power normalization."""

    W = complex_randn((num_antennas, num_users + num_antennas), rng)
    return row_power_normalize(W, total_power)


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


def radar_loss_from_covariance(
    R: np.ndarray,
    grid_steering: np.ndarray,
    desired: np.ndarray,
    target_steering: np.ndarray,
    cross_corr_weight: float,
) -> Tuple[float, float, float, np.ndarray]:
    """Return Liu-style radar loss, optimal alpha, cross term and beampattern."""

    pattern = np.real(np.einsum("lm,mn,ln->l", grid_steering.conj(), R, grid_steering))
    denom = float(np.dot(desired, desired)) + EPS
    alpha = max(0.0, float(np.dot(desired, pattern) / denom))
    mse = float(np.mean((alpha * desired - pattern) ** 2))

    cross = 0.0
    pairs = 0
    for p in range(target_steering.shape[0]):
        for q in range(p + 1, target_steering.shape[0]):
            pc = target_steering[q].conj() @ R @ target_steering[p]
            cross += float(np.abs(pc) ** 2)
            pairs += 1
    if pairs:
        cross /= pairs
    return mse + cross_corr_weight * cross, alpha, cross, pattern


def radar_loss(
    W: np.ndarray,
    grid_steering: np.ndarray,
    desired: np.ndarray,
    target_steering: np.ndarray,
    cross_corr_weight: float,
) -> Tuple[float, float, float, np.ndarray]:
    return radar_loss_from_covariance(
        covariance(W), grid_steering, desired, target_steering, cross_corr_weight
    )


def channel_pinv(H: np.ndarray) -> np.ndarray:
    gram = H @ H.conj().T
    reg = EPS * np.eye(gram.shape[0], dtype=np.complex128)
    return H.conj().T @ np.linalg.pinv(gram + reg)


def nullspace_basis(H: np.ndarray, tol: float = 1.0e-8) -> np.ndarray:
    _, s, vh = np.linalg.svd(H, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        rank = int(np.sum(s > tol * max(H.shape) * s[0]))
    return vh.conj().T[:, rank:]


def action_dim(
    num_antennas: int,
    num_users: int,
    structured: bool,
) -> int:
    if structured:
        return 2 * (
            num_users
            + (num_antennas - num_users) * (num_users + num_antennas)
        )
    return 2 * num_antennas * (num_users + num_antennas)


def _as_complex_vector(action: np.ndarray) -> np.ndarray:
    half = action.size // 2
    return action[:half] + 1j * action[half:]


def action_to_beamformer(
    action: np.ndarray,
    num_antennas: int,
    num_users: int,
    total_power: float,
) -> np.ndarray:
    """Map a real policy action directly to W=[Wc,Wr]."""

    expected = 2 * num_antennas * (num_users + num_antennas)
    if action.size != expected:
        raise ValueError(f"expected action size {expected}, got {action.size}")
    W = _as_complex_vector(action.astype(np.float64, copy=False)).reshape(
        num_antennas, num_users + num_antennas
    )
    return row_power_normalize(W, total_power)


def action_to_residual(
    action: np.ndarray,
    H: np.ndarray,
    num_antennas: int,
    num_users: int,
    structured: bool,
) -> np.ndarray:
    """Map a real policy action to a residual update for W=[Wc,Wr]."""

    expected = action_dim(num_antennas, num_users, structured)
    if action.size != expected:
        raise ValueError(f"expected action size {expected}, got {action.size}")

    if not structured:
        return _as_complex_vector(action.astype(np.float64, copy=False)).reshape(
            num_antennas, num_users + num_antennas
        )

    values = _as_complex_vector(action.astype(np.float64, copy=False))
    gain_count = num_users
    gains = values[:gain_count]
    coeffs = values[gain_count:].reshape(
        num_antennas - num_users, num_users + num_antennas
    )

    Hplus = channel_pinv(H)
    N = nullspace_basis(H)
    delta = np.zeros((num_antennas, num_users + num_antennas), dtype=np.complex128)
    delta[:, :num_users] += Hplus @ np.diag(gains)
    if N.size:
        delta += N @ coeffs
    return delta


def zf_beamformer(
    H: np.ndarray,
    target_steering: np.ndarray,
    total_power: float,
    noise_power: float,
    sinr_threshold: float,
    rng: np.random.Generator,
    loss_fn: Optional[Callable[[np.ndarray], float]] = None,
    comm_safety: float = 1.25,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """ZF-inspired feasible initializer/baseline with null-space radar columns."""

    num_users, num_antennas = H.shape
    Hplus = channel_pinv(H)
    N = nullspace_basis(H)
    required_gain = np.sqrt(max(sinr_threshold * noise_power * comm_safety, EPS))
    Wc0 = Hplus @ np.diag(np.full(num_users, required_gain, dtype=np.float64))

    Wr0 = np.zeros((num_antennas, num_antennas), dtype=np.complex128)
    if N.size:
        for j in range(num_antennas):
            if j < target_steering.shape[0]:
                col = target_steering[j]
                Wr0[:, j] = N @ (N.conj().T @ col)
    else:
        for j in range(min(num_antennas, target_steering.shape[0])):
            Wr0[:, j] = target_steering[j]

    best_W = None
    best_loss = np.inf
    best_margin = -np.inf
    comm_scales = np.geomspace(0.7, 18.0, 18)
    radar_scales = np.geomspace(0.15, 3.0, 10)
    for cs in comm_scales:
        for rs in radar_scales:
            W = row_power_normalize(np.hstack([cs * Wc0, rs * Wr0]), total_power)
            sinr = compute_sinr(W, H, num_users, noise_power)
            margin = float(np.min(sinr - sinr_threshold))
            score = loss_fn(W) if loss_fn is not None else -margin
            feasible = margin >= 0.0
            if (feasible and score < best_loss) or (best_W is None and margin > best_margin):
                best_W = W
                best_loss = float(score)
                best_margin = margin

    assert best_W is not None
    return best_W, {"radar_loss": best_loss, "min_margin": best_margin}

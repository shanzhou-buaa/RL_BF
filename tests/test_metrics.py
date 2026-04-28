import numpy as np

from isac_rl.config import SystemConfig
from isac_rl.metrics import (
    MetricCache,
    center_weight_vector,
    compute_all_metrics,
    compute_beampattern,
    compute_Lr1,
    compute_sinr,
    desired_beampattern,
    per_antenna_power_normalize,
    steering_matrix,
)


def test_power_normalization_enforces_per_antenna_constraint():
    cfg = SystemConfig(M=4, K=2, total_power=1.0)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(cfg.M, cfg.K + cfg.M)) + 1j * rng.normal(
        size=(cfg.M, cfg.K + cfg.M)
    )

    Wn = per_antenna_power_normalize(W, cfg.total_power)

    row_power = np.sum(np.abs(Wn) ** 2, axis=1)
    np.testing.assert_allclose(row_power, cfg.total_power / cfg.M, rtol=1e-10, atol=1e-10)


def test_metrics_follow_liu_shapes_and_return_objective_terms():
    cfg = SystemConfig(M=5, K=2)
    rng = np.random.default_rng(1)
    H = rng.normal(size=(cfg.K, cfg.M)) + 1j * rng.normal(size=(cfg.K, cfg.M))
    W = per_antenna_power_normalize(
        rng.normal(size=(cfg.M, cfg.K + cfg.M))
        + 1j * rng.normal(size=(cfg.M, cfg.K + cfg.M)),
        cfg.total_power,
    )

    grid_steering = steering_matrix(cfg.M, cfg.angle_grid)
    desired = desired_beampattern(cfg.angle_grid, cfg.target_angles_deg, cfg.beam_width_deg)
    pattern = compute_beampattern(W, grid_steering)
    sinr = compute_sinr(W, H, cfg.K, cfg.noise_power)
    metrics = compute_all_metrics(W, H, cfg)

    assert pattern.shape == cfg.angle_grid.shape
    assert sinr.shape == (cfg.K,)
    assert np.all(pattern >= -1e-10)
    assert np.all(sinr >= 0.0)
    for key in (
        "objective",
        "Lr",
        "Lr1",
        "Lr1_plain",
        "Lr2",
        "C_sinr",
        "C_side",
        "C_band",
        "C_balance",
        "target_center_error",
        "target_peak_offset_error",
        "target_peak_offsets_deg",
        "min_sinr_db",
        "feasible",
    ):
        assert key in metrics
    assert desired.shape == cfg.angle_grid.shape


def test_compute_all_metrics_accepts_cached_fixed_matrices(monkeypatch):
    cfg = SystemConfig(M=4, K=1, angle_grid_step_deg=5.0)
    rng = np.random.default_rng(2)
    H = rng.normal(size=(cfg.K, cfg.M)) + 1j * rng.normal(size=(cfg.K, cfg.M))
    W = per_antenna_power_normalize(
        rng.normal(size=(cfg.M, cfg.K + cfg.M))
        + 1j * rng.normal(size=(cfg.M, cfg.K + cfg.M)),
        cfg.total_power,
    )
    cache = MetricCache.from_config(cfg)

    def fail_steering_matrix(*_args, **_kwargs):
        raise AssertionError("steering_matrix should not be called when cache is supplied")

    monkeypatch.setattr("isac_rl.metrics.steering_matrix", fail_steering_matrix)

    metrics = compute_all_metrics(W, H, cfg, cache)

    assert metrics["pattern"].shape == cache.angle_grid.shape
    assert metrics["desired"] is cache.desired


def test_center_weighted_lri_penalizes_target_center_error_more():
    cfg = SystemConfig(
        M=4,
        K=1,
        target_angles_deg=(0.0,),
        beam_width_deg=10.0,
        angle_grid_min_deg=-10.0,
        angle_grid_max_deg=10.0,
        angle_grid_step_deg=1.0,
        center_weight=8.0,
        center_sigma_deg=1.0,
    )
    desired = desired_beampattern(cfg.angle_grid, cfg.target_angles_deg, cfg.beam_width_deg)
    weights = center_weight_vector(
        cfg.angle_grid,
        cfg.target_angles_deg,
        cfg.center_weight,
        cfg.center_sigma_deg,
    )
    pattern = desired.copy()
    pattern[np.argmin(np.abs(cfg.angle_grid - 0.0))] = 0.0

    plain = compute_Lr1(pattern, desired, alpha=1.0)
    weighted = compute_Lr1(pattern, desired, alpha=1.0, weights=weights)

    assert weighted > plain


def test_objective_uses_only_radar_and_sinr_terms():
    cfg = SystemConfig(M=4, K=1, angle_grid_step_deg=5.0)
    rng = np.random.default_rng(3)
    H = rng.normal(size=(cfg.K, cfg.M)) + 1j * rng.normal(size=(cfg.K, cfg.M))
    W = per_antenna_power_normalize(
        rng.normal(size=(cfg.M, cfg.K + cfg.M))
        + 1j * rng.normal(size=(cfg.M, cfg.K + cfg.M)),
        cfg.total_power,
    )

    metrics = compute_all_metrics(W, H, cfg)
    expected = (
        cfg.w_radar * np.log1p(metrics["Lr"] / cfg.Lr_ref)
        + cfg.w_sinr * metrics["C_sinr"]
    )

    assert np.isclose(metrics["objective"], expected)

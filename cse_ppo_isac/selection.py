from __future__ import annotations

from typing import Mapping


def radar_objective(metrics: Mapping[str, object]) -> float:
    return float(metrics.get("beam_objective", metrics["radar_loss"]))


def beam_quality_objective(
    metrics: Mapping[str, object],
    *,
    beampattern_weight: float = 2.0,
    sidelobe_leakage_weight: float = 1.0,
    sidelobe_ratio_weight: float = 0.25,
    target_band_weight: float = 2.0,
    target_balance_weight: float = 0.25,
) -> float:
    """Merit score aligned with visually acceptable Liu beampatterns.

    The complete Liu objective remains in the score, but a candidate cannot
    win only by reducing the target cross-correlation term while degrading
    the main beampattern shape.
    """

    beampattern = float(metrics.get("beampattern_loss", radar_objective(metrics)))
    sidelobe_leakage = float(metrics.get("sidelobe_leakage", 0.0))
    sidelobe_ratio = float(metrics.get("sidelobe_ratio", 0.0))
    target_band = float(metrics.get("target_band_error_mean", 0.0))
    target_balance = max(0.0, 1.0 - float(metrics.get("target_min_ratio", 1.0)))
    return float(
        radar_objective(metrics)
        + beampattern_weight * beampattern
        + sidelobe_leakage_weight * sidelobe_leakage
        + sidelobe_ratio_weight * sidelobe_ratio
        + target_band_weight * target_band
        + target_balance_weight * target_balance
    )


def is_better_candidate(
    candidate: Mapping[str, object],
    incumbent: Mapping[str, object],
    rel_tol: float = 0.0,
) -> bool:
    """Rank feasible beamformer candidates.

    Feasible candidates must not materially degrade either Liu loss term or
    the peak sidelobe diagnostics. Among candidates passing those component
    guards, the ranking uses the visual-quality merit score and then
    deterministic tie-breakers.
    """

    if bool(candidate["feasible"]) != bool(incumbent["feasible"]):
        return bool(candidate["feasible"])

    candidate_l1 = float(candidate.get("beampattern_loss", radar_objective(candidate)))
    incumbent_l1 = float(incumbent.get("beampattern_loss", radar_objective(incumbent)))
    candidate_l2 = float(candidate.get("cross_corr", 0.0))
    incumbent_l2 = float(incumbent.get("cross_corr", 0.0))
    candidate_sidelobe = float(candidate.get("sidelobe_leakage", float("inf")))
    incumbent_sidelobe = float(incumbent.get("sidelobe_leakage", float("inf")))
    candidate_target_band = float(candidate.get("target_band_error_mean", 0.0))
    incumbent_target_band = float(incumbent.get("target_band_error_mean", 0.0))
    guard_tol = min(max(float(rel_tol), 0.0), 0.02)

    if candidate["feasible"]:
        def too_much_worse(new_value: float, old_value: float) -> bool:
            return new_value > old_value + guard_tol * max(abs(new_value), abs(old_value), 1.0e-9)

        if too_much_worse(candidate_l1, incumbent_l1):
            return False
        if too_much_worse(candidate_l2, incumbent_l2):
            return False
        if too_much_worse(candidate_sidelobe, incumbent_sidelobe):
            return False
        if too_much_worse(candidate_target_band, incumbent_target_band):
            return False

        candidate_quality = beam_quality_objective(candidate)
        incumbent_quality = beam_quality_objective(incumbent)
        tol = max(float(rel_tol), 0.0) * max(
            abs(candidate_quality), abs(incumbent_quality), 1.0e-9
        )
        if candidate_quality < incumbent_quality - tol:
            return True
        if candidate_quality > incumbent_quality + tol:
            return False

        if candidate_l1 != incumbent_l1:
            return candidate_l1 < incumbent_l1

        if candidate_sidelobe != incumbent_sidelobe:
            return candidate_sidelobe < incumbent_sidelobe

        candidate_mainlobe = float(candidate.get("target_mean", 0.0))
        incumbent_mainlobe = float(incumbent.get("target_mean", 0.0))
        if candidate_mainlobe != incumbent_mainlobe:
            return candidate_mainlobe > incumbent_mainlobe

        if candidate_l2 != incumbent_l2:
            return candidate_l2 < incumbent_l2
        return radar_objective(candidate) < radar_objective(incumbent)

    if candidate["cost"] != incumbent["cost"]:
        return float(candidate["cost"]) < float(incumbent["cost"])
    candidate_quality = beam_quality_objective(candidate)
    incumbent_quality = beam_quality_objective(incumbent)
    return candidate_quality < incumbent_quality

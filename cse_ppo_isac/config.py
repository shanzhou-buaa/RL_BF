from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    """Simulation parameters following the Liu 2020 ISAC setup."""

    num_antennas: int = 10
    num_users: int = 2
    total_power: float = 1.0
    noise_power: float = 0.01
    sinr_threshold_db: float = 12.0
    target_angles_deg: Tuple[float, ...] = (-40.0, 0.0, 40.0)
    beam_width_deg: float = 10.0
    angle_grid_min_deg: float = -90.0
    angle_grid_max_deg: float = 90.0
    angle_grid_step_deg: float = 0.1
    cross_corr_weight: float = 1.0
    max_steps: int = 8
    init_mode: str = "policy"
    action_scale: float = 0.04
    action_penalty: float = 0.0
    init_comm_safety: float = 1.05
    loss_reward_weight: float = 1.0
    constraint_reward_weight: float = 0.0
    constraint_score_weight: float = 2.0
    quality_beampattern_weight: float = 2.0
    quality_sidelobe_leakage_weight: float = 1.0
    quality_sidelobe_ratio_weight: float = 0.25
    quality_target_band_weight: float = 2.0
    quality_target_balance_weight: float = 0.25
    quality_component_guard_weight: float = 10.0
    sinr_violation_penalty: float = 10.0
    power_penalty_weight: float = 1.0
    use_action_line_search: bool = True
    min_action_step_scale: float = 1.0 / 256.0

    @property
    def sinr_threshold(self) -> float:
        return 10.0 ** (self.sinr_threshold_db / 10.0)


@dataclass
class PPOConfig:
    """PPO and CSE-PPO training knobs."""

    updates: int = 40
    episodes_per_update: int = 8
    rollout_workers: int = 1
    rollout_backend: str = "process"
    eval_batch_size: int = 64
    eval_episodes: int = 64
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 6.0e-4
    hidden_dim: int = 128
    ppo_epochs: int = 3
    value_coef: float = 0.5
    entropy_coef: float = 1.0e-3
    max_grad_norm: float = 1.0
    dual_lr: float = 0.05
    cost_limit: float = 0.0
    entropy_threshold: float = 0.18
    entropy_threshold_quantile: float = 0.35
    entropy_threshold_ema: float = 0.8
    max_macro_steps: int = 4
    feasibility_beta: float = 6.0
    feasibility_entropy_floor: float = 0.1
    selection_rel_tolerance: float = 0.15
    initial_log_std: float = -1.1
    seed: int = 7
    device: str = "cpu"
    use_nullspace_action: bool = False
    use_macro_consolidation: bool = True
    use_feasibility_weighted_entropy: bool = True
    use_dual_control: bool = True

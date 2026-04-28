#!/usr/bin/env python
from __future__ import annotations
import os

# 按照总线顺序枚举 GPU。不要在代码中固定 CUDA_VISIBLE_DEVICES；
# 如需指定 A100/V100，请在命令行环境变量中设置，避免误把任务锁死到某一块卡。
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if "CUDA_VISIBLE_DEVICES" not in os.environ and "CSE_PPO_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CSE_PPO_CUDA_VISIBLE_DEVICES"]
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from dataclasses import asdict
from datetime import datetime
import numpy as np
import torch

from cse_ppo_isac.baselines import (
    evaluate_sdr_if_available,
    evaluate_zf,
    sinr_db,
    solve_sdr_solution,
)
from cse_ppo_isac.config import EnvConfig, PPOConfig
from cse_ppo_isac.env import ISACBeamformingEnv
from cse_ppo_isac.math_utils import radar_loss, zf_beamformer
from cse_ppo_isac.methods import build_ppo_config
from cse_ppo_isac.plotting import plot_training_rewards
from cse_ppo_isac.selection import is_better_candidate
from cse_ppo_isac.trainer import CSEPPOTrainer


PPO_METHODS = {"vanilla", "structured", "cse"}


def parse_args():
    parser = argparse.ArgumentParser(description="CSE-PPO for multiuser ISAC beamforming")
    parser.add_argument("--algorithm", choices=("cse", "vanilla"), default="cse")
    parser.add_argument(
        "--methods",
        default="vanilla,structured,cse",
        help="Comma list of methods used to generate beamformers: zf,sdr,vanilla,structured,cse",
    )
    parser.add_argument("--K", type=int, default=2, help="Number of communication users")
    parser.add_argument("--sinr-db", type=float, default=12.0, help="SINR threshold in dB")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--episodes-per-update", type=int, default=128)
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=16,
        help="CPU workers for PPO environment reset/step; 0 auto-uses available CPUs",
    )
    parser.add_argument(
        "--rollout-backend",
        choices=("serial", "thread", "process"),
        default="process",
        help="Backend for PPO environment reset/step parallelism",
    )
    # 一次 PPO update 的流程是：
    # 收集一批 rollout 数据，例如 episodes_per_update * max_steps 个 transition。
    # 用这同一批 on-policy 数据重复优化策略网络若干轮。
    # --ppo-epochs 就是第 2 步重复训练的轮数。
    # ppo_epochs 太小，样本利用率低；太大，容易过拟合同一批 rollout，破坏 PPO 的 on-policy 假设。
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=6.0e-4)
    parser.add_argument("--entropy-coef", type=float, default=1.0e-3)
    parser.add_argument("--initial-log-std", type=float, default=-1.1)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--selection-rel-tol",
        type=float,
        default=0.15,
        help=(
            "Relative Lr near-tie tolerance for fixed-channel PPO beamformer "
            "selection; near ties prefer lower Lr1, lower sidelobes, stronger "
            "mainlobes, then lower Lr2."
        ),
    )
    parser.add_argument(
        "--inference-candidates",
        type=int,
        default=16,
        help="Policy-guided candidate rollouts used to select the saved PPO beamformer",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--init-mode",
        choices=("random", "policy", "zf"),
        default="policy",
        help="Initial beamformer for PPO episodes; policy/zf uses the feasible ZF-inspired initializer",
    )
    parser.add_argument("--action-scale", type=float, default=0.04)
    parser.add_argument("--action-penalty", type=float, default=0.0)
    parser.add_argument("--init-comm-safety", type=float, default=1.05)
    parser.add_argument("--loss-reward-weight", type=float, default=1.0)
    parser.add_argument("--constraint-reward-weight", type=float, default=0.0)
    parser.add_argument("--constraint-score-weight", type=float, default=2.0)
    parser.add_argument("--quality-beampattern-weight", type=float, default=2.0)
    parser.add_argument("--quality-sidelobe-leakage-weight", type=float, default=1.0)
    parser.add_argument("--quality-sidelobe-ratio-weight", type=float, default=0.25)
    parser.add_argument("--quality-target-band-weight", type=float, default=2.0)
    parser.add_argument("--quality-target-balance-weight", type=float, default=0.25)
    parser.add_argument("--quality-component-guard-weight", type=float, default=10.0)
    parser.add_argument("--sinr-violation-penalty", type=float, default=10.0)
    parser.add_argument("--power-penalty-weight", type=float, default=1.0)
    parser.add_argument(
        "--cross-corr-weight",
        type=float,
        default=1.0,
        help=(
            "Liu-style target cross-correlation weight in Lr = Lr1 + wc Lr2."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel-seed", type=int, default=42)
    parser.add_argument("--dual-lr", type=float, default=0.05)
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--use-nullspace", action="store_true")
    parser.add_argument("--no-nullspace", action="store_true")
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--no-feasibility-entropy", action="store_true")
    parser.add_argument("--no-dual", action="store_true")
    parser.add_argument("--no-action-line-search", action="store_true")
    parser.add_argument("--min-action-step-scale", type=float, default=1.0 / 256.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Torch CPU intra-op threads; 0 keeps PyTorch default",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=0,
        help="Torch CPU inter-op threads; 0 keeps PyTorch default",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=64,
        help="Process workers for ZF/SDR baseline evaluation only; PPO training uses episodes-per-update",
    )
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate baselines")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--log-root", default="log", help="Root directory for timestamped logs")
    parser.add_argument("--try-sdr", action="store_true", help="Compatibility flag; SDR runs by default")
    parser.add_argument("--no-sdr", action="store_true", help="Skip cvxpy SDR baseline")
    parser.add_argument("--sdr-solver", default="SCS", choices=("SCS", "CLARABEL"))
    parser.add_argument("--sdr-max-iters", type=int, default=3000)
    parser.add_argument("--sdr-eps", type=float, default=1.0e-5)
    return parser.parse_args()


def configure_runtime(args) -> None:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads > 0:
        try:
            torch.set_num_interop_threads(args.torch_interop_threads)
        except RuntimeError:
            # PyTorch 只允许在并行工作开始前设置 inter-op 线程；若外部已初始化则保持默认。
            pass
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "requested CUDA device but torch.cuda.is_available() is False. "
            "请检查 YH_RL 环境中的 CUDA 版 PyTorch 和 CUDA_VISIBLE_DEVICES 设置。"
        )


def hardware_record(args) -> dict:
    record = {
        "device_arg": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
    }
    if torch.cuda.is_available():
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        record["cuda_devices"] = devices
    return record


def parse_methods(value: str):
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    valid = {"zf", "sdr", "vanilla", "structured", "cse"}
    invalid = [method for method in methods if method not in valid]
    if invalid:
        raise ValueError(f"unknown methods: {invalid}")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")
    return methods

def fresh_env(cfg: EnvConfig, seed: int, structured_action: bool = True) -> ISACBeamformingEnv:
    env = ISACBeamformingEnv(cfg, structured_action=structured_action, seed=seed)
    env.reset()
    return env


def current_metrics(env: ISACBeamformingEnv, W):
    env.W = W
    metrics = env.current_metrics()
    return {
        "radar_loss": float(metrics["radar_loss"]),
        "beam_objective": float(metrics.get("beam_objective", metrics["radar_loss"])),
        "beampattern_loss": float(metrics.get("beampattern_loss", metrics["radar_loss"])),
        "cross_corr": float(metrics.get("cross_corr", 0.0)),
        "weighted_cross_corr": float(env.cfg.cross_corr_weight * metrics.get("cross_corr", 0.0)),
        "target_mean": float(metrics.get("target_mean", 0.0)),
        "sidelobe_ratio": float(metrics.get("sidelobe_ratio", 0.0)),
        "sidelobe_leakage": float(metrics.get("sidelobe_leakage", 0.0)),
        "target_min": float(metrics.get("target_min", 0.0)),
        "target_min_ratio": float(metrics.get("target_min_ratio", 0.0)),
        "target_band_errors": [
            float(value) for value in metrics.get("target_band_errors", [])
        ],
        "target_band_error_mean": float(metrics.get("target_band_error_mean", 0.0)),
        "min_sinr": float(metrics["min_sinr"]),
        "min_sinr_db": sinr_db(float(metrics["min_sinr"])),
        "feasible": bool(metrics["feasible"]),
        "cost": float(metrics["cost"]),
    }


def best_policy_beamformer(
    trainer: CSEPPOTrainer,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    channel_seed: int,
    candidates: int = 16,
):
    best_W = None
    best_metrics = None
    num_candidates = max(1, candidates)

    for candidate_idx in range(num_candidates):
        env = fresh_env(
            env_cfg,
            channel_seed,
            structured_action=ppo_cfg.use_nullspace_action,
        )
        state = env._state()
        candidate_best_W = None
        candidate_best_metrics = None

        for _ in range(env_cfg.max_steps):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=ppo_cfg.device)
            action, _, _, _ = trainer.policy.act(
                state_t, deterministic=(candidate_idx == 0)
            )
            state, _, done, _ = env.step(action)
            step_W = env.W.copy()
            step_metrics = current_metrics(env, step_W)
            if candidate_best_metrics is None or is_better_candidate(
                step_metrics,
                candidate_best_metrics,
                rel_tol=ppo_cfg.selection_rel_tolerance,
            ):
                candidate_best_W = step_W
                candidate_best_metrics = step_metrics
            if done:
                break

        if candidate_best_metrics is not None and (
            best_metrics is None or is_better_candidate(
                candidate_best_metrics,
                best_metrics,
                rel_tol=ppo_cfg.selection_rel_tolerance,
            )
        ):
            best_W = candidate_best_W
            best_metrics = candidate_best_metrics

    assert best_W is not None and best_metrics is not None
    return best_W, best_metrics


def pattern_for(env: ISACBeamformingEnv, W):
    loss, alpha, cross, pattern = radar_loss(
        W,
        env.grid_steering,
        env.desired,
        env.target_steering,
        env.cfg.cross_corr_weight,
    )
    beampattern_loss = max(
        0.0, float(loss) - env.cfg.cross_corr_weight * float(cross)
    )
    return {
        "radar_loss": float(loss),
        "beam_objective": float(loss),
        "beampattern_loss": float(beampattern_loss),
        "alpha": float(alpha),
        "cross_corr": float(cross),
        "weighted_cross_corr": float(env.cfg.cross_corr_weight * cross),
    }, pattern

def make_log_dir(root: str) -> Path:
    base = Path(root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    path = base
    suffix = 1
    while path.exists():
        path = Path(f"{base}_{suffix:02d}")
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_complex_beamformer_csv(path: Path, method: str, W: np.ndarray, num_users: int) -> None:
    """Save W=[Wc,Wr] as one CSV row per complex beamformer entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "block",
        "antenna_index",
        "column_index",
        "global_column_index",
        "real",
        "imag",
        "abs",
        "phase_rad",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for antenna_idx in range(W.shape[0]):
            for global_col_idx in range(W.shape[1]):
                value = W[antenna_idx, global_col_idx]
                is_comm = global_col_idx < num_users
                writer.writerow(
                    {
                        "method": method,
                        "block": "Wc" if is_comm else "Wr",
                        "antenna_index": antenna_idx,
                        "column_index": global_col_idx if is_comm else global_col_idx - num_users,
                        "global_column_index": global_col_idx,
                        "real": float(np.real(value)),
                        "imag": float(np.imag(value)),
                        "abs": float(np.abs(value)),
                        "phase_rad": float(np.angle(value)),
                    }
                )


def save_ppo_beamformer_csvs(
    log_dir: Path, beamformers: dict[str, np.ndarray], num_users: int
) -> dict[str, str]:
    csv_files = {}
    csv_dir = log_dir / "ppo_beamformers_csv"
    for method in ("vanilla", "structured", "cse"):
        if method not in beamformers:
            continue
        path = csv_dir / f"{method}_beamformer.csv"
        save_complex_beamformer_csv(path, method, beamformers[method], num_users)
        csv_files[method] = str(path)
    return csv_files


def save_training_history_csv(
    path: Path, method: str, history: list[dict[str, float]], episodes_per_update: int
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "update",
        "episode_start",
        "episode_end",
        "reward",
        "cost",
        "dual",
        "entropy_threshold",
        "low_entropy_rate",
        "quality_reward",
        "improvement_reward",
        "relative_loss_improvement",
        "relative_quality_improvement",
        "radar_loss",
        "beam_objective",
        "beampattern_loss",
        "cross_corr",
        "sidelobe_ratio",
        "sidelobe_leakage",
        "target_mean",
        "target_min",
        "target_min_ratio",
        "target_band_error_mean",
        "feasible_rate",
        "accepted_rate",
        "mean_step_scale",
        "rollout_time_sec",
        "reset_time_sec",
        "policy_time_sec",
        "env_step_time_sec",
        "transition_time_sec",
        "segment_time_sec",
        "ppo_update_time_sec",
        "update_time_sec",
        "transitions_per_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            update = int(row["update"])
            out = {
                "method": method,
                "update": update,
                "episode_start": (update - 1) * episodes_per_update + 1,
                "episode_end": update * episodes_per_update,
            }
            for key in (
                "reward",
                "cost",
                "dual",
                "entropy_threshold",
                "low_entropy_rate",
                "quality_reward",
                "improvement_reward",
                "relative_loss_improvement",
                "relative_quality_improvement",
                "radar_loss",
                "beam_objective",
                "beampattern_loss",
                "cross_corr",
                "sidelobe_ratio",
                "sidelobe_leakage",
                "target_mean",
                "target_min",
                "target_min_ratio",
                "target_band_error_mean",
                "feasible_rate",
                "accepted_rate",
                "mean_step_scale",
                "rollout_time_sec",
                "reset_time_sec",
                "policy_time_sec",
                "env_step_time_sec",
                "transition_time_sec",
                "segment_time_sec",
                "ppo_update_time_sec",
                "update_time_sec",
                "transitions_per_sec",
            ):
                out[key] = float(row.get(key, np.nan))
            writer.writerow(out)
    return str(path)


def main():
    args = parse_args()
    configure_runtime(args)
    env_cfg = EnvConfig(
        num_users=args.K,
        sinr_threshold_db=args.sinr_db,
        max_steps=args.max_steps,
        init_mode=args.init_mode,
        action_scale=args.action_scale,
        action_penalty=args.action_penalty,
        init_comm_safety=args.init_comm_safety,
        loss_reward_weight=args.loss_reward_weight,
        constraint_reward_weight=args.constraint_reward_weight,
        constraint_score_weight=args.constraint_score_weight,
        quality_beampattern_weight=args.quality_beampattern_weight,
        quality_sidelobe_leakage_weight=args.quality_sidelobe_leakage_weight,
        quality_sidelobe_ratio_weight=args.quality_sidelobe_ratio_weight,
        quality_target_band_weight=args.quality_target_band_weight,
        quality_target_balance_weight=args.quality_target_balance_weight,
        quality_component_guard_weight=args.quality_component_guard_weight,
        sinr_violation_penalty=args.sinr_violation_penalty,
        power_penalty_weight=args.power_penalty_weight,
        cross_corr_weight=args.cross_corr_weight,
        use_action_line_search=not args.no_action_line_search,
        min_action_step_scale=args.min_action_step_scale,
    )
    ppo_cfg = build_ppo_config(args, args.algorithm)
    methods = parse_methods(args.methods)
    if args.no_sdr and "sdr" in methods:
        methods = [method for method in methods if method != "sdr"]

    log_dir = make_log_dir(args.log_root)
    config_record = {
        "args": vars(args),
        "methods": methods,
        "env_config": asdict(env_cfg),
        "ppo_config": asdict(ppo_cfg),
        "hardware": hardware_record(args),
        "log_dir": str(log_dir),
    }
    config_path = log_dir / "config.json"
    save_json(config_path, config_record)
    print(f"日志目录: {log_dir}", flush=True)
    print(f"参数配置已保存: {config_path}", flush=True)
    print("硬件配置:", json.dumps(config_record["hardware"], ensure_ascii=False), flush=True)

    summary = {
        "config": {
            "algorithm": args.algorithm,
            "K": args.K,
            "sinr_db": args.sinr_db,
            "max_steps": args.max_steps,
            "init_mode": env_cfg.init_mode,
            "use_nullspace_action": ppo_cfg.use_nullspace_action,
            "use_macro_consolidation": ppo_cfg.use_macro_consolidation,
            "use_feasibility_weighted_entropy": ppo_cfg.use_feasibility_weighted_entropy,
            "use_dual_control": ppo_cfg.use_dual_control,
            "eval_batch_size": ppo_cfg.eval_batch_size,
            "inference_candidates": args.inference_candidates,
            "selection_rel_tolerance": ppo_cfg.selection_rel_tolerance,
            "rollout_workers": args.rollout_workers,
            "rollout_backend": args.rollout_backend,
            "num_workers": args.num_workers,
            "log_dir": str(log_dir),
            "channel_seed": args.channel_seed,
            "methods": methods,
            "hardware": config_record["hardware"],
        },
        "baselines": {},
        "methods": {},
    }

    t0 = perf_counter()
    beamformers = {}
    patterns = {}
    beamformer_metrics = {}
    training_histories = {}
    training_history_files = {}
    base_env = fresh_env(env_cfg, args.channel_seed, structured_action=True)
    assert base_env.H is not None
    fixed_H = base_env.H.copy()

    if "zf" in methods:
        zf = evaluate_zf(
            env_cfg,
            args.eval_episodes,
            seed=args.seed + 20_000,
            num_workers=args.num_workers,
        )
        zf["min_sinr_db"] = sinr_db(zf["min_sinr"])
        summary["baselines"]["zf"] = zf
        print("ZF:", json.dumps(zf, indent=2), flush=True)

        zf_env = fresh_env(env_cfg, args.channel_seed, structured_action=True)
        W_zf, _ = zf_beamformer(
            zf_env.H,
            zf_env.target_steering,
            env_cfg.total_power,
            env_cfg.noise_power,
            env_cfg.sinr_threshold,
            zf_env.rng,
            loss_fn=zf_env._loss_fn,
        )
        beamformers["zf"] = W_zf
        beamformer_metrics["zf"] = current_metrics(zf_env, W_zf)
        pattern_metrics, patterns["zf"] = pattern_for(zf_env, W_zf)
        beamformer_metrics["zf"].update(pattern_metrics)
        summary["methods"]["zf"] = {"eval": zf, "beamformer": beamformer_metrics["zf"]}

    if "sdr" in methods:
        try:
            sdr_eval = evaluate_sdr_if_available(
                env_cfg,
                args.eval_episodes,
                seed=args.seed + 30_000,
                num_workers=args.num_workers,
                solver=args.sdr_solver,
                max_iters=args.sdr_max_iters,
                eps=args.sdr_eps,
            )
            summary["baselines"]["sdr"] = sdr_eval

            sdr_env = fresh_env(env_cfg, args.channel_seed, structured_action=True)
            sdr_metrics, W_sdr = solve_sdr_solution(
                env_cfg,
                sdr_env.H,
                sdr_env,
                solver=args.sdr_solver,
                max_iters=args.sdr_max_iters,
                eps=args.sdr_eps,
            )
            beamformers["sdr"] = W_sdr
            beamformer_metrics["sdr"] = {**sdr_metrics, **current_metrics(sdr_env, W_sdr)}
            pattern_metrics, patterns["sdr"] = pattern_for(sdr_env, W_sdr)
            beamformer_metrics["sdr"].update(pattern_metrics)
            summary["methods"]["sdr"] = {
                "eval": sdr_eval,
                "beamformer": beamformer_metrics["sdr"],
            }
        except Exception as exc:
            summary["baselines"]["sdr"] = {"skipped": str(exc)}
            print("SDR skipped:", exc, flush=True)

    for method in [method for method in methods if method in PPO_METHODS]:
        if args.skip_train:
            summary["methods"][method] = {"skipped": "skip-train was set"}
            continue
        method_ppo_cfg = build_ppo_config(args, method)
        trainer = CSEPPOTrainer(env_cfg, method_ppo_cfg)
        train_start = perf_counter()
        history = trainer.train()
        train_time = perf_counter() - train_start
        training_histories[method] = history
        training_history_files[method] = save_training_history_csv(
            log_dir / "training_history" / f"{method}_history.csv",
            method,
            history,
            method_ppo_cfg.episodes_per_update,
        )
        plot_training_rewards(
            training_histories,
            log_dir / "training_history",
            episodes_per_update=args.episodes_per_update,
        )
        eval_start = perf_counter()
        eval_metrics = trainer.evaluate(args.eval_episodes)
        eval_time = perf_counter() - eval_start
        eval_metrics["min_sinr_db"] = sinr_db(eval_metrics["min_sinr"])
        inference_start = perf_counter()
        W_policy, policy_metrics = best_policy_beamformer(
            trainer,
            env_cfg,
            method_ppo_cfg,
            args.channel_seed,
            candidates=args.inference_candidates,
        )
        inference_time = perf_counter() - inference_start
        policy_env = fresh_env(
            env_cfg, args.channel_seed, structured_action=method_ppo_cfg.use_nullspace_action
        )
        pattern_metrics, patterns[method] = pattern_for(policy_env, W_policy)
        policy_metrics.update(pattern_metrics)
        beamformers[method] = W_policy
        beamformer_metrics[method] = policy_metrics
        method_summary = {
            "algorithm": method,
            "metrics": eval_metrics,
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "inference_time_sec": inference_time,
            "last_history": history[-1] if history else {},
            "history_file": training_history_files[method],
            "inference_candidates": args.inference_candidates,
            "rollout_workers": trainer.rollout_workers,
            "rollout_backend": trainer.rollout_backend,
            "beamformer": policy_metrics,
        }
        summary["methods"][method] = method_summary
        if method == args.algorithm:
            summary["rl"] = method_summary
        print(f"{method.upper()}:", json.dumps(method_summary, indent=2), flush=True)
        trainer.close()

    if beamformers:
        npz_payload = {"H": fixed_H, **beamformers}
        np.savez_compressed(log_dir / "beamformers.npz", **npz_payload)
        np.savez_compressed(
            log_dir / "patterns.npz",
            angle_grid=base_env.angle_grid,
            desired=base_env.desired,
            **patterns,
        )
        save_json(log_dir / "beamformer_metrics.json", beamformer_metrics)
        ppo_csv_files = save_ppo_beamformer_csvs(log_dir, beamformers, env_cfg.num_users)
        summary["beamformer_files"] = {
            "beamformers": str(log_dir / "beamformers.npz"),
            "patterns": str(log_dir / "patterns.npz"),
            "metrics": str(log_dir / "beamformer_metrics.json"),
        }
        if ppo_csv_files:
            summary["beamformer_files"]["ppo_csv"] = ppo_csv_files
        print(f"波束矩阵已保存: {log_dir / 'beamformers.npz'}", flush=True)
        if ppo_csv_files:
            print(f"PPO波束CSV已保存: {log_dir / 'ppo_beamformers_csv'}", flush=True)

    if training_history_files:
        summary["training_files"] = {
            "history_csv": training_history_files,
            "reward_curve_png": str(log_dir / "training_history" / "reward_curve.png"),
            "reward_curve_pdf": str(log_dir / "training_history" / "reward_curve.pdf"),
        }
        print(f"训练曲线已保存: {log_dir / 'training_history' / 'reward_curve.png'}", flush=True)

    summary["elapsed_sec"] = perf_counter() - t0
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    summary_path = log_dir / "summary.json"
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(f"结果摘要已保存: {summary_path}", flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"结果摘要已保存到指定路径: {out}", flush=True)


if __name__ == "__main__":
    main()

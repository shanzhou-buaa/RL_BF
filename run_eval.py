#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from isac_rl.config import EnvConfig, PPOConfig, SystemConfig
from isac_rl.env import ISACBeamformingEnv
from isac_rl.metrics import compute_all_metrics, desired_beampattern
from isac_rl.policy import TanhGaussianActorCritic
from isac_rl.plotting import plot_all
from isac_rl.utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO/HE-PPO checkpoints")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--eval-channels", type=int, default=256)
    parser.add_argument("--plot-seed", type=int, default=2026)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _dataclass_from_config(cls, payload):
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in names})


def load_configs(log_dir: Path):
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    return (
        _dataclass_from_config(SystemConfig, config["system"]),
        _dataclass_from_config(EnvConfig, config["env"]),
        _dataclass_from_config(PPOConfig, config["ppo"]),
    )


def load_policy(checkpoint: Path, device: str) -> tuple[str, int, TanhGaussianActorCritic]:
    payload = torch.load(checkpoint, map_location=device)
    policy = TanhGaussianActorCritic(
        payload["state_dim"],
        payload["action_dim"],
        hidden_dim=payload["ppo"]["hidden_dim"],
    ).to(device)
    policy.load_state_dict(payload["model"])
    policy.eval()
    return str(payload["algo"]), int(payload["seed"]), policy


def summarize_metrics(metrics_list: list[dict[str, object]]) -> dict[str, float]:
    scalar_keys = [
        "objective",
        "Lr",
        "Lr1",
        "Lr2",
        "C_sinr",
        "C_side",
        "C_band",
        "C_balance",
        "min_sinr_db",
        "peak_sidelobe_ratio",
        "mean_sidelobe_ratio",
        "target_band_error",
    ]
    summary: dict[str, float] = {}
    for key in scalar_keys:
        values = np.asarray([float(item[key]) for item in metrics_list], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    summary["feasible_rate"] = float(
        np.mean([float(item["feasible"]) for item in metrics_list])
    )
    return summary


@torch.no_grad()
def rollout_policy_on_seed(
    policy: TanhGaussianActorCritic,
    sys_cfg: SystemConfig,
    env_cfg: EnvConfig,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], float]:
    start = perf_counter()
    env = ISACBeamformingEnv(sys_cfg, env_cfg, seed=seed)
    state = env.reset()

    for _ in range(env_cfg.episode_steps):
        state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
        action, _, _, _, _ = policy.act(state_t, deterministic=True)
        state, _, done, _ = env.step(action.squeeze(0).cpu().numpy())
        if done:
            break

    assert env.H is not None and env.W is not None
    metrics = compute_all_metrics(env.W, env.H, sys_cfg)
    return env.H.copy(), env.W.copy(), metrics, perf_counter() - start


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    sys_cfg: SystemConfig,
    env_cfg: EnvConfig,
    eval_channels: int,
    plot_seed: int,
    device: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, float], float]:
    algo, seed, policy = load_policy(checkpoint, device)
    method_name = f"{algo}_seed{seed}"

    metrics_list = []
    runtimes = []
    for idx in range(eval_channels):
        _, _, metrics, runtime = rollout_policy_on_seed(
            policy,
            sys_cfg,
            env_cfg,
            seed=plot_seed + idx,
            device=device,
        )
        metrics_list.append(metrics)
        runtimes.append(runtime)

    H_plot, W_plot, plot_metrics, _ = rollout_policy_on_seed(
        policy,
        sys_cfg,
        env_cfg,
        seed=plot_seed,
        device=device,
    )

    summary = summarize_metrics(metrics_list)
    summary["runtime_sec"] = float(np.mean(runtimes))
    summary["runtime_sec_std"] = float(np.std(runtimes))
    summary["checkpoint"] = str(checkpoint)
    return method_name, H_plot, W_plot, plot_metrics["pattern"], summary, summary["runtime_sec"]


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    sys_cfg, env_cfg, _ = load_configs(log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")

    beamformers = {}
    patterns = {}
    metrics = {}
    runtime = {}
    H_plot = None

    for checkpoint in checkpoints:
        method_name, H, W, pattern, summary, runtime_sec = evaluate_checkpoint(
            checkpoint=checkpoint,
            sys_cfg=sys_cfg,
            env_cfg=env_cfg,
            eval_channels=args.eval_channels,
            plot_seed=args.plot_seed,
            device=args.device,
        )
        if H_plot is None:
            H_plot = H
        beamformers[f"W_{method_name}"] = W
        patterns[f"pattern_{method_name}"] = pattern
        metrics[method_name] = summary
        runtime[method_name] = runtime_sec

    assert H_plot is not None
    desired = desired_beampattern(
        sys_cfg.angle_grid,
        sys_cfg.target_angles_deg,
        sys_cfg.beam_width_deg,
    )

    np.savez_compressed(log_dir / "beamformers.npz", H_plot=H_plot, **beamformers)
    np.savez_compressed(
        log_dir / "patterns.npz",
        angle_grid=sys_cfg.angle_grid,
        desired=desired,
        **patterns,
    )
    save_json(log_dir / "metrics.json", metrics)
    save_json(log_dir / "runtime.json", runtime)

    if args.save_plots:
        plot_all(log_dir)

    print(f"evaluation complete: {log_dir}")


if __name__ == "__main__":
    main()

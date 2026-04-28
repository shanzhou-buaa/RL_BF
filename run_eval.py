#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from isac_rl.baselines import evaluate_baseline
from isac_rl.config import EnvConfig, PPOConfig, SystemConfig
from isac_rl.env import ISACBeamformingEnv
from isac_rl.metrics import compute_all_metrics, steering_matrix
from isac_rl.policy import TanhGaussianActorCritic
from isac_rl.plotting import plot_all
from isac_rl.utils import parse_str_tuple, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO/HE-PPO and offline baselines")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--baselines", default="zf,sdr")
    parser.add_argument("--eval-channels", type=int, default=256)
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


@torch.no_grad()
def rollout_policy(checkpoint: Path, sys_cfg: SystemConfig, env_cfg: EnvConfig, device: str):
    start = perf_counter()
    payload = torch.load(checkpoint, map_location=device)
    env = ISACBeamformingEnv(sys_cfg, env_cfg, seed=2026)
    state = env.reset()
    policy = TanhGaussianActorCritic(
        payload["state_dim"], payload["action_dim"], hidden_dim=payload["ppo"]["hidden_dim"]
    ).to(device)
    policy.load_state_dict(payload["model"])
    for _ in range(env_cfg.episode_steps):
        state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
        action, _, _, _, _ = policy.act(state_t, deterministic=True)
        state, _, done, _ = env.step(action.squeeze(0).cpu().numpy())
        if done:
            break
    metrics = compute_all_metrics(env.W, env.H, sys_cfg)
    metrics["runtime_sec"] = perf_counter() - start
    return env.H.copy(), env.W.copy(), metrics


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    sys_cfg, env_cfg, _ = load_configs(log_dir)
    beamformers = {}
    patterns = {}
    metrics = {}
    runtime = {}
    H_test = None

    for checkpoint in sorted((log_dir / "checkpoints").glob("*.pt")):
        algo = checkpoint.stem
        H, W, item = rollout_policy(checkpoint, sys_cfg, env_cfg, args.device)
        H_test = H if H_test is None else H_test
        beamformers[f"W_{algo}"] = W
        metrics[algo] = {key: value for key, value in item.items() if key not in {"pattern", "desired", "sinr", "sinr_db", "sinr_gap_db", "target_gains"}}
        runtime[algo] = float(item["runtime_sec"])
        patterns[f"pattern_{algo}"] = item["pattern"]

    assert H_test is not None
    for baseline in parse_str_tuple(args.baselines):
        W, item = evaluate_baseline(baseline, H_test, sys_cfg)
        beamformers[f"W_{baseline}"] = W
        metrics[baseline] = {key: value for key, value in item.items() if key not in {"pattern", "desired", "sinr", "sinr_db", "sinr_gap_db", "target_gains"}}
        runtime[baseline] = float(item["runtime_sec"])
        patterns[f"pattern_{baseline}"] = item["pattern"]

    np.savez_compressed(log_dir / "beamformers.npz", H_test=H_test, **beamformers)
    np.savez_compressed(
        log_dir / "patterns.npz",
        angle_grid=sys_cfg.angle_grid,
        desired=next(iter(patterns.values())) * 0 + compute_all_metrics(next(iter(beamformers.values())), H_test, sys_cfg)["desired"],
        **patterns,
    )
    save_json(log_dir / "metrics.json", metrics)
    save_json(log_dir / "runtime.json", runtime)
    if args.save_plots:
        plot_all(log_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from isac_rl.config import EnvConfig, PPOConfig, SystemConfig, TrainConfig
from isac_rl.trainer import train_algorithms
from isac_rl.utils import parse_float_tuple, parse_int_list, parse_str_tuple, timestamped_log_dir


def _normalize_negative_angle_arg(argv: list[str]) -> list[str]:
    """Allow: --target-angles -40,0,40.

    argparse treats a value starting with '-' as a possible option. Converting
    to --target-angles=-40,0,40 keeps the CLI convenient for the Liu2020 target
    set while preserving normal option parsing.
    """

    normalized: list[str] = []
    idx = 0
    while idx < len(argv):
        if (
            argv[idx] == "--target-angles"
            and idx + 1 < len(argv)
            and argv[idx + 1].startswith("-")
            and not argv[idx + 1].startswith("--")
        ):
            normalized.append(f"--target-angles={argv[idx + 1]}")
            idx += 2
            continue
        normalized.append(argv[idx])
        idx += 1
    return normalized


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train PPO and HE-PPO for pure-RL ISAC beamforming")
    parser.add_argument("--algos", default="ppo,heppo")
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument(
        "--target-angles",
        default="-40,0,40",
        help="Comma-separated target angles. Use --target-angles=-40,0,40 for negative first values.",
    )
    parser.add_argument("--sinr-db", type=float, default=12.0)
    parser.add_argument("--episode-steps", type=int, default=8)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--action-scale", type=float, default=0.03)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--eval-channels", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--log-root", default="log")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    argv = sys.argv[1:] if argv is None else argv
    return parser.parse_args(_normalize_negative_angle_arg(argv))


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_dir = Path(args.log_dir) if args.log_dir else timestamped_log_dir(args.log_root)
    sys_cfg = SystemConfig(
        M=args.M,
        K=args.K,
        target_angles_deg=parse_float_tuple(args.target_angles),
        sinr_threshold_db=args.sinr_db,
    )
    env_cfg = EnvConfig(episode_steps=args.episode_steps, action_scale=args.action_scale)
    ppo_cfg = PPOConfig(
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
    )
    train_cfg = TrainConfig(
        algos=parse_str_tuple(args.algos),
        seeds=parse_int_list(args.seeds),
        eval_channels=args.eval_channels,
        eval_seed=args.eval_seed,
        eval_interval=args.eval_interval,
        device=args.device,
    )
    print(f"log_dir: {log_dir}", flush=True)
    train_algorithms(
        sys_cfg,
        env_cfg,
        ppo_cfg,
        train_cfg,
        log_dir,
        show_progress=not args.no_progress,
    )
    print(f"training complete: {log_dir}", flush=True)


if __name__ == "__main__":
    main()

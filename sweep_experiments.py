#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from cse_ppo_isac.baselines import evaluate_zf, sinr_db
from cse_ppo_isac.config import EnvConfig
from cse_ppo_isac.methods import build_ppo_config
from cse_ppo_isac.trainer import CSEPPOTrainer


def parse_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep CSE-PPO ISAC experiments")
    parser.add_argument("--algorithms", default="cse,structured,vanilla")
    parser.add_argument("--K-list", default="2,4,6")
    parser.add_argument("--sinr-db-list", default="4,8,12,16,20,24")
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--selection-rel-tol", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--cross-corr-weight",
        type=float,
        default=1.0,
        help="Liu-style target cross-correlation weight in Lr = Lr1 + wc Lr2.",
    )
    parser.add_argument("--sinr-violation-penalty", type=float, default=10.0)
    parser.add_argument("--power-penalty-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers for baselines")
    parser.add_argument("--include-sdr", action="store_true", help="Compatibility flag; SDR runs by default")
    parser.add_argument("--no-sdr", action="store_true", help="Skip cvxpy SDR baseline")
    parser.add_argument("--sdr-solver", default="SCS", choices=("SCS", "CLARABEL"))
    parser.add_argument("--sdr-max-iters", type=int, default=3000)
    parser.add_argument("--sdr-eps", type=float, default=1.0e-5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--log-root", default="log", help="Root directory for timestamped logs")
    return parser.parse_args()

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


def main():
    args = parse_args()
    algorithms = parse_list(args.algorithms, str)
    k_list = parse_list(args.K_list, int)
    sinr_list = parse_list(args.sinr_db_list, float)
    log_dir = make_log_dir(args.log_root)
    output_dir = Path(args.output_dir) if args.output_dir else log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    jsonl_path = output_dir / "records.jsonl"
    config_path = log_dir / "config.json"
    save_json(
        config_path,
        {
            "args": vars(args),
            "algorithms": algorithms,
            "K_list": k_list,
            "sinr_db_list": sinr_list,
            "log_dir": str(log_dir),
            "output_dir": str(output_dir),
        },
    )
    print(f"日志目录: {log_dir}", flush=True)
    print(f"参数配置已保存: {config_path}", flush=True)
    print(f"实验结果目录: {output_dir}", flush=True)

    fieldnames = [
        "algorithm",
        "K",
        "sinr_db",
        "radar_loss",
        "beam_objective",
        "beampattern_loss",
        "cross_corr",
        "sidelobe_ratio",
        "sidelobe_leakage",
        "target_min_ratio",
        "target_band_error_mean",
        "min_sinr",
        "min_sinr_db",
        "feasible_rate",
        "cost",
        "train_time_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for K in k_list:
            for sinr_db_value in sinr_list:
                env_cfg = EnvConfig(
                    num_users=K,
                    sinr_threshold_db=sinr_db_value,
                    max_steps=args.max_steps,
                    sinr_violation_penalty=args.sinr_violation_penalty,
                    power_penalty_weight=args.power_penalty_weight,
                    cross_corr_weight=args.cross_corr_weight,
                )
                base_config = {
                    "K": K,
                    "sinr_db": sinr_db_value,
                    "env_config": asdict(env_cfg),
                }
                zf = evaluate_zf(
                    env_cfg,
                    args.eval_episodes,
                    seed=args.seed + 100_000 + K,
                    num_workers=args.num_workers,
                )
                zf_row = {
                    "algorithm": "zf",
                    "K": K,
                    "sinr_db": sinr_db_value,
                    "radar_loss": zf["radar_loss"],
                    "beam_objective": zf.get("beam_objective", zf["radar_loss"]),
                    "beampattern_loss": zf.get("beampattern_loss", 0.0),
                    "cross_corr": zf.get("cross_corr", 0.0),
                    "sidelobe_ratio": zf.get("sidelobe_ratio", 0.0),
                    "sidelobe_leakage": zf.get("sidelobe_leakage", 0.0),
                    "target_min_ratio": zf.get("target_min_ratio", 0.0),
                    "target_band_error_mean": zf.get("target_band_error_mean", 0.0),
                    "min_sinr": zf["min_sinr"],
                    "min_sinr_db": sinr_db(zf["min_sinr"]),
                    "feasible_rate": zf["feasible_rate"],
                    "cost": zf["cost"],
                    "train_time_sec": 0.0,
                }
                writer.writerow(zf_row)
                jsonl_file.write(
                    json.dumps(
                        {"config": {"algorithm": "zf", **base_config}, "result": zf_row}
                    )
                    + "\n"
                )
                jsonl_file.flush()
                print("done", zf_row, flush=True)

                if not args.no_sdr:
                    from cse_ppo_isac.baselines import evaluate_sdr_if_available

                    sdr = evaluate_sdr_if_available(
                        env_cfg,
                        args.eval_episodes,
                        seed=args.seed + 120_000 + K,
                        num_workers=args.num_workers,
                        solver=args.sdr_solver,
                        max_iters=args.sdr_max_iters,
                        eps=args.sdr_eps,
                    )
                    sdr_row = {
                        "algorithm": "sdr",
                        "K": K,
                        "sinr_db": sinr_db_value,
                        "radar_loss": sdr["radar_loss"],
                        "beam_objective": sdr.get("beam_objective", sdr["radar_loss"]),
                        "beampattern_loss": sdr.get("beampattern_loss", 0.0),
                        "cross_corr": sdr.get("cross_corr", 0.0),
                        "sidelobe_ratio": sdr.get("sidelobe_ratio", 0.0),
                        "sidelobe_leakage": sdr.get("sidelobe_leakage", 0.0),
                        "target_min_ratio": sdr.get("target_min_ratio", 0.0),
                        "target_band_error_mean": sdr.get("target_band_error_mean", 0.0),
                        "min_sinr": sdr["min_sinr"],
                        "min_sinr_db": sinr_db(sdr["min_sinr"]),
                        "feasible_rate": sdr["feasible_rate"],
                        "cost": sdr["cost"],
                        "train_time_sec": sdr["solve_time_sec"],
                    }
                    writer.writerow(sdr_row)
                    jsonl_file.write(
                        json.dumps(
                            {
                                "config": {
                                    "algorithm": "sdr",
                                    "solver": args.sdr_solver,
                                    "max_iters": args.sdr_max_iters,
                                    "eps": args.sdr_eps,
                                    **base_config,
                                },
                                "result": sdr_row,
                            }
                        )
                        + "\n"
                    )
                    jsonl_file.flush()
                    print("done", sdr_row, flush=True)

                for algorithm in algorithms:
                    ppo_cfg = build_ppo_config(
                        args,
                        algorithm,
                        seed=args.seed + 1000 * K + int(10 * sinr_db_value),
                    )
                    per_run_config = {
                        "algorithm": algorithm,
                        "K": K,
                        "sinr_db": sinr_db_value,
                        "env_config": asdict(env_cfg),
                        "ppo_config": asdict(ppo_cfg),
                    }
                    trainer = CSEPPOTrainer(env_cfg, ppo_cfg)
                    start = perf_counter()
                    trainer.train()
                    train_time = perf_counter() - start
                    metrics = trainer.evaluate(args.eval_episodes)
                    row = {
                        "algorithm": algorithm,
                        "K": K,
                        "sinr_db": sinr_db_value,
                        "radar_loss": metrics["radar_loss"],
                        "beam_objective": metrics.get("beam_objective", metrics["radar_loss"]),
                        "beampattern_loss": metrics.get("beampattern_loss", 0.0),
                        "cross_corr": metrics.get("cross_corr", 0.0),
                        "sidelobe_ratio": metrics.get("sidelobe_ratio", 0.0),
                        "sidelobe_leakage": metrics.get("sidelobe_leakage", 0.0),
                        "target_min_ratio": metrics.get("target_min_ratio", 0.0),
                        "target_band_error_mean": metrics.get("target_band_error_mean", 0.0),
                        "min_sinr": metrics["min_sinr"],
                        "min_sinr_db": sinr_db(metrics["min_sinr"]),
                        "feasible_rate": metrics["feasible_rate"],
                        "cost": metrics["cost"],
                        "train_time_sec": train_time,
                    }
                    writer.writerow(row)
                    jsonl_file.write(json.dumps({"config": per_run_config, "result": row}) + "\n")
                    jsonl_file.flush()
                    print("done", row, flush=True)

    print(f"wrote {csv_path}")
    print(f"wrote {jsonl_path}")
    save_json(log_dir / "finished.json", {"summary_csv": str(csv_path), "records_jsonl": str(jsonl_path)})
    print(f"完成信息已保存: {log_dir / 'finished.json'}")


if __name__ == "__main__":
    main()

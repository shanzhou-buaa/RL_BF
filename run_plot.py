#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from isac_rl.plotting import plot_all


def parse_args():
    parser = argparse.ArgumentParser(description="Plot HE-PPO ISAC experiment outputs")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--eval-channels", type=int, default=256)
    parser.add_argument("--plot-seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-auto-eval",
        action="store_true",
        help="Do not run run_eval.py when patterns.npz is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    patterns_path = log_dir / "patterns.npz"

    if not patterns_path.exists():
        if args.no_auto_eval:
            raise FileNotFoundError(
                f"{patterns_path} does not exist. Run run_eval.py first or omit --no-auto-eval."
            )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().with_name("run_eval.py")),
                "--log-dir",
                str(log_dir),
                "--eval-channels",
                str(args.eval_channels),
                "--plot-seed",
                str(args.plot_seed),
                "--device",
                args.device,
                "--save-plots",
            ],
            check=True,
        )

    if not patterns_path.exists():
        raise FileNotFoundError(
            f"{patterns_path} still does not exist after evaluation."
        )

    plot_all(log_dir)
    print(f"figures saved to: {log_dir / 'figures'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse

from isac_rl.plotting import plot_all


def parse_args():
    parser = argparse.ArgumentParser(description="Plot HE-PPO ISAC experiment outputs")
    parser.add_argument("--log-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_all(args.log_dir)


if __name__ == "__main__":
    main()

"""Create a new side-by-side 0/full fine-tuning transfer plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from battery_weighted_maml.evaluation.plots import plot_transfer_zero_vs_full


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine transfer 0-step and full fine-tuning trajectories"
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/baseline/cx2_37_gru_l100_soh_transfer_samefam_s42",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    output = plot_transfer_zero_vs_full(
        Path(args.run_dir).resolve(),
        Path(args.output).resolve() if args.output else None,
    )
    print(f"Combined transfer 0/full plot: {output}")


if __name__ == "__main__":
    main()

"""One-command training followed by held-out-cell evaluation."""

from __future__ import annotations

import argparse

from .config import load_config, resolve_data_root
from .evaluate import evaluate_checkpoint
from .train import train_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate partial V-Q forecaster")
    parser.add_argument("--config", default="configs/matr_partial_vq_forecasting.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    data_root = resolve_data_root(config, args.data_root)
    run_dir = train_run(
        config,
        args.fold,
        data_root,
        resume=args.resume,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    evaluation = evaluate_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        data_root=data_root,
        device_name=config.device,
    )
    print(f"Run directory: {run_dir}")
    print(f"Evaluation directory: {evaluation}")


if __name__ == "__main__":
    main()

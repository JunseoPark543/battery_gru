"""Train one fold and optionally evaluate its best checkpoint."""

from __future__ import annotations

import argparse

from .config import load_config, resolve_data_root
from .evaluate import evaluate_checkpoint
from .train import train_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MATR lifetime I-V ANP")
    parser.add_argument("--config", default="configs/matr_horizon_lifetime_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--task-batch-size", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    data_root = resolve_data_root(config, args.data_root)
    run_dir = train_run(
        config, args.fold, data_root,
        resume=args.resume, max_steps=args.max_steps,
        task_batch_size=args.task_batch_size,
    )
    print(f"Lifetime I-V ANP run: {run_dir}")
    if args.evaluate:
        destination = evaluate_checkpoint(
            config, run_dir / "checkpoints/best.pt", data_root
        )
        print(f"Lifetime I-V evaluation: {destination}")


if __name__ == "__main__":
    main()

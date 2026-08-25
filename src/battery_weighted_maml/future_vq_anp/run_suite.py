"""One command for training followed by held-out future V-Q evaluation."""

from __future__ import annotations

import argparse

from .config import load_config, resolve_data_root
from .evaluate import evaluate_checkpoint
from .train import train_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate future V-Q latent ANP")
    parser.add_argument("--config", default="configs/matr_future_vq_anp.yaml")
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
    root = resolve_data_root(config, args.data_root)
    run_dir = train_run(
        config,
        args.fold,
        root,
        resume=args.resume,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    output = evaluate_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        data_root=root,
        device_name=config.device,
    )
    print(f"Run directory: {run_dir}")
    print(f"Evaluation directory: {output}")


if __name__ == "__main__":
    main()

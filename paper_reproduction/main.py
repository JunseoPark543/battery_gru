"""Command-line entry point for training, Optuna search, and meta-testing."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_reproduction.adapt_and_test import evaluate_test_tasks
from paper_reproduction.config import ExperimentConfig, load_config, save_config
from paper_reproduction.data import load_tasks
from paper_reproduction.maml_train import (
    load_meta_checkpoint,
    run_optuna_search,
    train_meta_model,
)
from paper_reproduction.model import GRUEncoderDecoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return device


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("paper_reproduction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def create_model(config: ExperimentConfig) -> GRUEncoderDecoder:
    model = GRUEncoderDecoder(
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if config.model.hidden_size == 64 and config.model.num_layers == 1 and parameter_count != 25793:
        raise RuntimeError(f"paper model must have 25,793 parameters, found {parameter_count}")
    return model


def _rooted(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _new_run_dir(config: ExperimentConfig, root: Path, mode: str) -> Path:
    output = _rooted(config.paths.output_dir, root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output / f"{timestamp}_{mode}_L{config.data.history_length}_seed{config.seed}"


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    root = Path.cwd().resolve()
    config = load_config(args.config)
    if args.history_length is not None:
        config.data.history_length = args.history_length
    if args.max_epochs is not None:
        config.maml.max_epochs = args.max_epochs
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    calce_dir = _rooted(config.paths.calce_dir, root)

    if args.resume is not None and args.mode in {"train", "all"}:
        run_dir = Path(args.resume).resolve().parent.parent
    else:
        run_dir = _new_run_dir(config, root, args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "logs/run.log")
    save_config(config, run_dir / "config_resolved.yaml")
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "device": str(device),
        "seed": config.seed,
        "history_length": config.data.history_length,
        "train_cells": config.data.train_cells,
        "test_cells": config.data.test_cells,
        "algorithm": "full_second_order_maml",
        "weighted_meta_learning": False,
        "config": config.to_dict(),
    }
    _write_manifest(run_dir / "run_manifest.json", manifest)
    logger.info(
        "mode=%s device=%s L=%d algorithm=full_second_order_maml train=%s test=%s",
        args.mode, device, config.data.history_length,
        config.data.train_cells, config.data.test_cells,
    )

    if args.mode == "optuna":
        train_tasks = load_tasks(calce_dir, config.data.train_cells)
        result = run_optuna_search(
            config, train_tasks, device, run_dir / "optuna", logger
        )
        manifest.update({"status": "completed", "optuna": result})
        _write_manifest(run_dir / "run_manifest.json", manifest)
        return run_dir

    model = create_model(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("model_parameters=%d expected_default=25793", parameter_count)
    best_checkpoint: Path
    if args.mode in {"train", "all"}:
        train_tasks = load_tasks(calce_dir, config.data.train_cells)
        result = train_meta_model(
            model,
            train_tasks,
            config,
            device,
            run_dir,
            logger,
            resume=args.resume,
        )
        model = result.model
        best_checkpoint = run_dir / "checkpoints/best_meta_model.pt"
        manifest.update(
            {
                "best_epoch": result.best_epoch,
                "best_meta_loss": result.best_meta_loss,
                "last_epoch": result.last_epoch,
            }
        )
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in test mode")
        best_checkpoint = Path(args.checkpoint).resolve()
        payload = load_meta_checkpoint(best_checkpoint, model, device)
        if int(payload["history_length"]) != config.data.history_length:
            raise ValueError("test config L differs from checkpoint L")
        if list(payload["train_cells"]) != list(config.data.train_cells):
            raise ValueError("test config training split differs from checkpoint")
        model.to(device).eval()

    if args.mode in {"test", "all"}:
        # Meta-test cells are loaded only after meta-training/checkpoint selection.
        test_tasks = load_tasks(calce_dir, config.data.test_cells)
        summary = evaluate_test_tasks(
            model,
            test_tasks,
            config,
            device,
            run_dir / "meta_test",
            logger,
        )
        manifest["meta_test_rows"] = len(summary)
        manifest["checkpoint_used_for_test"] = str(best_checkpoint)

    manifest.update(
        {"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    )
    _write_manifest(run_dir / "run_manifest.json", manifest)
    logger.info("Completed paper reproduction run: %s", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full second-order GRU MAML reproduction for CALCE SOH/RUL"
    )
    parser.add_argument("--config", default="paper_reproduction/config.yaml")
    parser.add_argument("--mode", choices=["train", "test", "all", "optuna"], default="all")
    parser.add_argument("--checkpoint", help="best_meta_model.pt for test-only mode")
    parser.add_argument("--resume", help="last.pt for resumable train/all mode")
    parser.add_argument("--history-length", type=int, choices=[100, 200, 300, 400, 500])
    parser.add_argument("--max-epochs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


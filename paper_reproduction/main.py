"""Command-line entry point for training, Optuna search, and meta-testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
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
from paper_reproduction.data import load_tasks, preprocessing_summary
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


def _slug(value: object, maximum_length: int = 24) -> str:
    text = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value)
    ).strip("-")
    return (text or "unnamed")[:maximum_length]


def _number_tag(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace("-", "m").replace(".", "p").replace("+", "")


def _config_fingerprint(config: ExperimentConfig) -> str:
    canonical = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _new_run_dir(
    config: ExperimentConfig,
    root: Path,
    mode: str,
    target_cell: str | None = None,
) -> Path:
    output = _rooted(config.paths.output_dir, root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label = _slug(config.maml.experiment_label, maximum_length=18)
    fields = [timestamp, mode, f"L{config.data.history_length}", label]
    if target_cell is not None:
        target = Path(target_cell).stem.removeprefix("CALCE_")
        fields.append(_slug(target, maximum_length=16))
    fields.extend([f"s{config.seed}", f"c{_config_fingerprint(config)}"])
    return output / "_".join(fields)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _parse_multi_step_weights(value: str) -> dict[int, float]:
    try:
        pairs = [item.split(":", maxsplit=1) for item in value.split(",")]
        return {int(step): float(weight) for step, weight in pairs}
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "weights must look like 1:0.2,3:0.3,5:0.5"
        ) from exc


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "off"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number or null") from exc


def run(args: argparse.Namespace) -> Path:
    root = Path.cwd().resolve()
    config = load_config(args.config)
    if args.device is not None:
        config.device = args.device
    if args.history_length is not None:
        config.data.history_length = args.history_length
    if args.max_epochs is not None:
        config.maml.max_epochs = args.max_epochs
    if args.forecast_mode is not None:
        config.evaluation.forecast_mode = args.forecast_mode
    if args.max_prediction_length is not None:
        config.evaluation.max_prediction_length = args.max_prediction_length
    if args.fast_learning_rate is not None:
        config.adaptation.learning_rate = None
        config.adaptation.fast_learning_rate = args.fast_learning_rate
    if args.complete_learning_rate is not None:
        config.adaptation.learning_rate = None
        config.adaptation.complete_learning_rate = args.complete_learning_rate
    if args.complete_max_steps is not None:
        config.adaptation.complete_max_steps = args.complete_max_steps
    if args.complete_patience is not None:
        config.adaptation.complete_patience = args.complete_patience
    if args.scheduler is not None:
        config.adaptation.scheduler = args.scheduler
    if args.loss_reduction is not None:
        config.loss.recursive_reduction = args.loss_reduction
        config.adaptation.recursive_loss_reduction = args.loss_reduction
    if args.sampling_mode is not None:
        config.adaptation.sampling_mode = args.sampling_mode
    if args.fast_sampling_mode is not None:
        config.adaptation.fast_sampling_mode = args.fast_sampling_mode
    if hasattr(args, "gradient_clip_norm"):
        config.adaptation.gradient_clip_norm = args.gradient_clip_norm
    if args.inner_steps is not None:
        config.maml.inner_steps = args.inner_steps
        if args.multi_step_query_weights is None:
            config.maml.multi_step_query_weights = {args.inner_steps: 1.0}
    if args.multi_step_query_weights is not None:
        config.maml.multi_step_query_weights = args.multi_step_query_weights
    if args.experiment_label is not None:
        config.maml.experiment_label = args.experiment_label
    if args.oracle_diagnostics is not None:
        config.adaptation.oracle_diagnostics = args.oracle_diagnostics
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    calce_dir = _rooted(config.paths.calce_dir, root)

    if args.resume is not None and args.mode in {"train", "all"}:
        run_dir = Path(args.resume).resolve().parent.parent
    else:
        name_target = (
            args.test_cell
            if args.test_cell is not None
            else (config.data.test_cells[0] if args.mode == "adapt" else None)
        )
        run_dir = _new_run_dir(config, root, args.mode, name_target)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "logs/run.log")
    save_config(config, run_dir / "config.yaml")
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
        "run_name": run_dir.name,
        "run_name_schema_version": 3,
        "config_fingerprint": _config_fingerprint(config),
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
            raise ValueError("--checkpoint is required in test/adapt mode")
        best_checkpoint = Path(args.checkpoint).resolve()
        payload = load_meta_checkpoint(best_checkpoint, model, device)
        if int(payload["history_length"]) != config.data.history_length:
            raise ValueError("test config L differs from checkpoint L")
        if list(payload["train_cells"]) != list(config.data.train_cells):
            raise ValueError("test config training split differs from checkpoint")
        model.to(device).eval()

    canonical_meta_checkpoint = run_dir / "checkpoints/meta_model.pt"
    canonical_meta_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if best_checkpoint.resolve() != canonical_meta_checkpoint.resolve():
        shutil.copy2(best_checkpoint, canonical_meta_checkpoint)

    # This report intentionally runs only after the meta checkpoint is fixed.
    # Loading test-cell labels for diagnostics therefore cannot affect training
    # or model selection.
    all_tasks = load_tasks(
        calce_dir,
        [*config.data.train_cells, *config.data.test_cells],
    )
    preprocessing_summary(
        all_tasks,
        config.data.history_length,
        config.evaluation.eol_threshold,
    ).to_csv(run_dir / "preprocessing_summary.csv", index=False)

    if args.mode in {"test", "adapt", "all"}:
        # Meta-test cells are loaded only after meta-training/checkpoint selection.
        selected_test_cells = (
            [args.test_cell]
            if args.test_cell is not None
            else (
                [config.data.test_cells[0]]
                if args.mode == "adapt"
                else config.data.test_cells
            )
        )
        unknown = set(selected_test_cells) - set(config.data.test_cells)
        if unknown:
            raise ValueError(f"--test-cell is not in configured test cells: {sorted(unknown)}")
        by_name = {task.name: task for task in all_tasks}
        test_tasks = [by_name[name] for name in selected_test_cells]
        flat_output = args.mode == "adapt" and len(test_tasks) == 1
        test_output = run_dir if flat_output else run_dir / "meta_test"
        summary = evaluate_test_tasks(
            model,
            test_tasks,
            config,
            device,
            test_output,
            logger,
            flat_output=flat_output,
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
    parser.add_argument(
        "--mode", choices=["train", "test", "adapt", "all", "optuna"], default="all"
    )
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--checkpoint", help="best_meta_model.pt for test-only mode")
    parser.add_argument("--resume", help="last.pt for resumable train/all mode")
    parser.add_argument("--history-length", type=int, choices=[100, 200, 300, 400, 500])
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--forecast-mode", choices=["paper", "deployment"])
    parser.add_argument("--max-prediction-length", type=int)
    parser.add_argument("--test-cell", help="adapt only this configured meta-test cell")
    parser.add_argument("--fast-learning-rate", type=float)
    parser.add_argument("--complete-learning-rate", type=float)
    parser.add_argument("--complete-max-steps", type=int)
    parser.add_argument("--complete-patience", type=int)
    parser.add_argument("--scheduler", choices=["constant", "step", "plateau"])
    parser.add_argument(
        "--loss-reduction", choices=["point_balanced", "sample_balanced"]
    )
    parser.add_argument(
        "--sampling-mode", choices=["random", "length_stratified", "full_support"]
    )
    parser.add_argument(
        "--fast-sampling-mode", choices=["random", "length_stratified", "full_support"]
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=_parse_optional_float,
        default=argparse.SUPPRESS,
        help="adaptation clip norm, or null to disable",
    )
    parser.add_argument("--inner-steps", type=int, choices=[1, 3, 5])
    parser.add_argument(
        "--multi-step-query-weights",
        type=_parse_multi_step_weights,
        help="for example 1:0.2,3:0.3,5:0.5",
    )
    parser.add_argument("--experiment-label")
    parser.add_argument(
        "--oracle-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

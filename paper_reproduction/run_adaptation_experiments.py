"""Run the requested CX2_37/L=500 adaptation ablation matrix.

The query trajectory is diagnostic only.  Complete checkpoints are selected
solely by recursive MAE on the chronological tail of the observed support.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_reproduction.adapt_and_test import (
    _validation_length,
    run_adaptation_trajectory,
)
from paper_reproduction.config import ExperimentConfig, load_config, save_config
from paper_reproduction.data import load_cell_task, preprocessing_summary
from paper_reproduction.main import resolve_device, seed_everything
from paper_reproduction.maml_train import load_meta_checkpoint
from paper_reproduction.model import GRUEncoderDecoder


RESULT_COLUMNS = [
    "experiment",
    "learning_rate",
    "loss_reduction",
    "sampling_mode",
    "gradient_clip",
    "best_step",
    "support_validation_mae_percent",
    "query_mae_percent",
    "query_rmse_percent",
    "query_r2",
    "predicted_eol",
    "predicted_rul",
]


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("paper_reproduction.adaptation_comparison")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _write_progress(
    output: Path,
    rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    """Persist completed variants so an interrupted long run is unambiguous."""
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
        output / "experiment_comparison.partial.csv", index=False
    )
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _result_row(
    experiment: str,
    learning_rate: float,
    reduction: str,
    sampling: str,
    clip: float | None,
    step: int,
    diagnostics: pd.DataFrame,
) -> dict[str, object]:
    selected = diagnostics.loc[diagnostics["step"] == step]
    if len(selected) != 1:
        raise RuntimeError(f"{experiment}: diagnostics do not contain step {step}")
    row = selected.iloc[0]
    return {
        "experiment": experiment,
        "learning_rate": learning_rate,
        "loss_reduction": reduction,
        "sampling_mode": sampling,
        "gradient_clip": clip,
        "best_step": step,
        "support_validation_mae_percent": row["support_validation_mae_percent"],
        "query_mae_percent": row["query_mae_percent"],
        "query_rmse_percent": row["query_rmse_percent"],
        "query_r2": row["query_r2"],
        "predicted_eol": row["predicted_eol"],
        "predicted_rul": row["predicted_rul"],
    }


def _run_fast(
    label: str,
    model: GRUEncoderDecoder,
    task,
    base: ExperimentConfig,
    device: torch.device,
    reduction: str,
    output: Path,
) -> list[dict[str, object]]:
    config = copy.deepcopy(base)
    config.adaptation.recursive_loss_reduction = reduction
    config.adaptation.fast_learning_rate = 0.05
    config.adaptation.learning_rate = None
    config.adaptation.fast_sampling_mode = "random"
    config.adaptation.gradient_clip_norm = None
    support, _ = task.split(config.data.history_length)
    trajectory = run_adaptation_trajectory(
        model,
        task,
        config,
        device,
        training_soh=support,
        validation_soh=None,
        learning_rate=0.05,
        max_steps=5,
        sampling_mode="random",
        seed_offset=1000,
        patience=None,
        capture_steps=(0, 1, 3, 5),
        query_diagnostics=True,
    )
    trajectory.diagnostics.to_csv(output / f"{label}_diagnostics.csv", index=False)
    return [
        _result_row(
            f"{label}_fast_{step}", 0.05, reduction, "random", None, step,
            trajectory.diagnostics,
        )
        for step in (0, 1, 3, 5)
    ]


def _run_complete(
    label: str,
    model: GRUEncoderDecoder,
    task,
    base: ExperimentConfig,
    device: torch.device,
    learning_rate: float,
    reduction: str,
    sampling: str,
    clip: float | None,
    output: Path,
    max_steps: int,
) -> dict[str, object]:
    config = copy.deepcopy(base)
    config.adaptation.learning_rate = None
    config.adaptation.complete_learning_rate = learning_rate
    config.adaptation.complete_max_steps = max_steps
    config.adaptation.recursive_loss_reduction = reduction
    config.adaptation.sampling_mode = sampling
    config.adaptation.gradient_clip_norm = clip
    support, _ = task.split(config.data.history_length)
    validation_length = _validation_length(config.data.history_length, config)
    trajectory = run_adaptation_trajectory(
        model,
        task,
        config,
        device,
        training_soh=support[:-validation_length],
        validation_soh=support[-validation_length:],
        learning_rate=learning_rate,
        max_steps=max_steps,
        sampling_mode=sampling,
        seed_offset=1000,
        patience=config.adaptation.complete_patience,
        capture_steps=(0, 1, 2, 3, 5, 10),
        query_diagnostics=True,
    )
    trajectory.diagnostics.to_csv(output / f"{label}_diagnostics.csv", index=False)
    torch.save(trajectory.deployment_best_state, output / f"{label}_best_model.pt")
    torch.save(trajectory.final_state, output / f"{label}_final_model.pt")
    return _result_row(
        label,
        learning_rate,
        reduction,
        sampling,
        clip,
        trajectory.deployment_best_step,
        trajectory.diagnostics,
    )


def run(args: argparse.Namespace) -> Path:
    root = Path.cwd().resolve()
    config = load_config(args.config)
    config.data.history_length = 500
    config.device = args.device or config.device
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    checkpoint = Path(args.checkpoint).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else root
        / config.paths.output_dir
        / "comparisons"
        / f"{datetime.now():%Y%m%d_%H%M%S_%f}_CX2_37_L500"
    )
    output.mkdir(parents=True, exist_ok=True)
    logger = _logger(output / "comparison.log")
    save_config(config, output / "config.yaml")
    model = GRUEncoderDecoder(
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
    )
    payload = load_meta_checkpoint(checkpoint, model, device)
    if int(payload["history_length"]) != 500:
        raise ValueError("comparison checkpoint must have history_length=500")
    model.to(device).eval()
    calce_dir = Path(config.paths.calce_dir)
    if not calce_dir.is_absolute():
        calce_dir = root / calce_dir
    task = load_cell_task(calce_dir / "CALCE_CX2_37.pkl")
    preprocessing_summary([task], 500, config.evaluation.eol_threshold).to_csv(
        output / "preprocessing_summary.csv", index=False
    )

    rows: list[dict[str, object]] = []
    complete_max_steps = (
        args.complete_max_steps or config.adaptation.complete_max_steps
    )
    completed_experiments: list[str] = []
    manifest: dict[str, object] = {
        "status": "running",
        "checkpoint": str(checkpoint),
        "device": str(device),
        "target": task.name,
        "history_length": 500,
        "complete_max_steps": complete_max_steps,
        "completed_experiments": completed_experiments,
        "query_is_diagnostic_only": True,
        "complete_selection": "chronological_support_recursive_validation_mae",
    }
    _write_progress(output, rows, manifest)
    logger.info("starting A: point-balanced fast trajectory")
    rows.extend(_run_fast("A", model, task, config, device, "point_balanced", output))
    completed_experiments.append("A")
    _write_progress(output, rows, manifest)
    logger.info("starting B: sample-balanced fast trajectory")
    rows.extend(_run_fast("B", model, task, config, device, "sample_balanced", output))
    completed_experiments.append("B")
    _write_progress(output, rows, manifest)
    # C-E retain the original point-balanced/random/unclipped path so the LR
    # comparison is isolated. F applies the complete stabilization bundle.
    variants = [
        ("C", 0.05, "point_balanced", "random", None),
        ("D", 0.005, "point_balanced", "random", None),
        ("E", 0.001, "point_balanced", "random", None),
        ("F", 0.005, "sample_balanced", "length_stratified", 1.0),
    ]
    for label, learning_rate, reduction, sampling, clip in variants:
        logger.info(
            "starting %s lr=%g reduction=%s sampling=%s clip=%s",
            label, learning_rate, reduction, sampling, clip,
        )
        rows.append(
            _run_complete(
                label,
                model,
                task,
                config,
                device,
                learning_rate,
                reduction,
                sampling,
                clip,
                output,
                complete_max_steps,
            )
        )
        completed_experiments.append(label)
        _write_progress(output, rows, manifest)
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    frame.to_csv(output / "experiment_comparison.csv", index=False)
    manifest["status"] = "completed"
    _write_progress(output, rows, manifest)
    logger.info("comparison complete: %s", output / "experiment_comparison.csv")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CX2_37 L500 adaptation comparison A-F")
    parser.add_argument("--config", default="paper_reproduction/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument(
        "--complete-max-steps",
        type=int,
        help="diagnostic override; omit for the configured 500-step experiment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""CLI for four-fold CALCE direct-RUL BOIL domain generalization."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .config import DOMAIN_NAMES, ExperimentConfig, load_config, save_config
from .data import CellSample, load_calce_samples, split_domain
from .metrics import regression_metrics, save_json, save_prediction_figure
from .trainer import SourceOnlyTrainer, resolve_device, set_global_seed


LOGGER = logging.getLogger(__name__)


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _selected_folds(config: ExperimentConfig, override: str | None) -> list[str]:
    if override is not None:
        return list(DOMAIN_NAMES) if override == "all" else [override]
    configured = config.evaluation.held_out_domains
    return list(DOMAIN_NAMES) if configured == "all" else list(configured)


def _new_run_root(config: ExperimentConfig, seeds: Sequence[int]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed_text = "-".join(str(seed) for seed in seeds)
    return Path(config.output.output_dir) / f"{stamp}_direct-rul-boil_L100_seed{seed_text}"


def _parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "domain_invariant": sum(
            parameter.numel() for parameter in model.domain_invariant.parameters()
        ),
        "domain_specific": sum(
            parameter.numel() for parameter in model.domain_specific.parameters()
        ),
        "prediction_head": sum(
            parameter.numel() for parameter in model.predictor.parameters()
        ),
    }


def _evaluate_unseen_domain(
    trainer: SourceOnlyTrainer,
    targets: Sequence[CellSample],
    clip_negative_rul: bool,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Pure DG evaluation: direct forward only, with no target adaptation."""
    if not targets:
        raise ValueError("held-out target evaluation requires at least one cell")
    if any(target.domain != trainer.held_out_domain for target in targets):
        raise RuntimeError("target evaluation partition contains the wrong domain")
    model = trainer.model
    model.eval()
    normalized = np.stack(
        [trainer.feature_normalizer.transform(sample.features) for sample in targets]
    )
    features = torch.as_tensor(normalized, dtype=torch.float32, device=trainer.device)
    with torch.no_grad():
        standardized_prediction = model(features, grl_strength=0.0).prediction
    raw_prediction = trainer.rul_normalizer.inverse(
        standardized_prediction.cpu().numpy()
    )
    prediction = np.maximum(raw_prediction, 0.0) if clip_negative_rul else raw_prediction
    rows: list[dict[str, Any]] = []
    for sample, raw, predicted in zip(targets, raw_prediction, prediction):
        rows.append(
            {
                "seed": seed,
                "held_out_domain": trainer.held_out_domain,
                "file_name": sample.file_name,
                "cell_id": sample.cell_id,
                "history_cycles": sample.features.shape[0],
                "actual_eol_cycle": sample.eol_cycle,
                "actual_rul_cycles": sample.rul_cycles,
                "raw_predicted_rul_cycles": float(raw),
                "predicted_rul_cycles": float(predicted),
                "predicted_eol_cycle": float(predicted + sample.features.shape[0]),
                "absolute_error_cycles": float(abs(predicted - sample.rul_cycles)),
                "target_adaptation_steps": 0,
            }
        )
    frame = pd.DataFrame(rows)
    metrics = regression_metrics(
        frame["actual_rul_cycles"], frame["predicted_rul_cycles"]
    )
    metrics.update(
        {
            "seed": seed,
            "held_out_domain": trainer.held_out_domain,
            "target_adaptation": False,
            "checkpoint_selection_used_target_labels": False,
            "best_source_cv_iteration": trainer.best_iteration,
            "best_source_cv_mae_cycles": trainer.best_score,
        }
    )
    return frame, metrics


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.device is not None:
        config.device = args.device
        config.validate()
    seeds = [args.seed] if args.seed is not None else list(config.seeds)
    folds = _selected_folds(config, args.fold)
    if args.resume is not None and (len(folds) != 1 or len(seeds) != 1):
        raise ValueError("--resume requires exactly one --fold and one --seed")
    device = resolve_device(config.device)
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        fold_run_dir = resume_path.parent.parent
        run_root = fold_run_dir.parent
    else:
        run_root = _new_run_root(config, seeds).resolve()
        run_root.mkdir(parents=True, exist_ok=False)
        save_config(config, run_root / "resolved_config.yaml")
    _configure_logging(run_root / "experiment.log")
    LOGGER.info(
        "Direct-RUL BOIL experiment: folds=%s seeds=%s device=%s config=%s",
        folds,
        seeds,
        device,
        config_path,
    )
    samples = load_calce_samples(
        Path(config.data.calce_dir), Path(config.data.label_path), config.data
    )
    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        for held_out in folds:
            set_global_seed(seed)
            source, target = split_domain(samples, held_out)
            if resume_path is None:
                fold_run_dir = run_root / f"fold_{held_out}_seed{seed}"
            fold_run_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                {
                    "protocol": "pure leave-one-domain-out domain generalization",
                    "held_out_domain": held_out,
                    "seed": seed,
                    "history_length": config.data.history_length,
                    "label_definition": "EOL cycle - 100",
                    "source_domains": sorted({sample.domain for sample in source}),
                    "source_files": [sample.file_name for sample in source],
                    "target_files": [sample.file_name for sample in target],
                    "target_data_used_during_training": False,
                    "target_adaptation": False,
                    "boil_inner_updated_modules": ["domain_specific"],
                    "boil_inner_frozen_modules": [
                        "domain_invariant",
                        "predictor",
                        "domain_classifier",
                    ],
                },
                fold_run_dir / "protocol.json",
            )
            trainer = SourceOnlyTrainer(
                source,
                held_out,
                config,
                device,
                fold_run_dir,
                seed,
            )
            LOGGER.info(
                "Starting fold=%s source_cells=%d target_cells=%d parameters=%s",
                held_out,
                len(source),
                len(target),
                _parameter_summary(trainer.model),
            )
            result = trainer.train(resume=resume_path)
            predictions, metrics = _evaluate_unseen_domain(
                trainer, target, config.evaluation.clip_negative_rul, seed
            )
            predictions.to_csv(fold_run_dir / "target_predictions.csv", index=False)
            save_json(metrics, fold_run_dir / "target_metrics.json")
            save_prediction_figure(
                predictions,
                fold_run_dir / "target_prediction_scatter.png",
                f"Pure DG: unseen {held_out} (seed {seed})",
            )
            save_json(
                {
                    "checkpoint": str(result.checkpoint_path.resolve()),
                    "best_iteration": result.best_iteration,
                    "stopped_iteration": result.stopped_iteration,
                    "feature_normalizer": result.feature_normalizer.state_dict(),
                    "rul_normalizer": result.rul_normalizer.state_dict(),
                    "parameter_counts": _parameter_summary(result.model),
                },
                fold_run_dir / "model_manifest.json",
            )
            all_predictions.append(predictions)
            fold_metrics.append(metrics)
            LOGGER.info(
                "Completed fold=%s seed=%d target_MAE=%.3f target_RMSE=%.3f",
                held_out,
                seed,
                metrics["mae_cycles"],
                metrics["rmse_cycles"],
            )
    # Rebuild the summary from all completed fold directories. This preserves
    # earlier fold results when one interrupted fold is resumed separately.
    completed_prediction_files = sorted(
        run_root.glob("fold_*_seed*/target_predictions.csv")
    )
    if completed_prediction_files:
        aggregate = pd.concat(
            [pd.read_csv(path) for path in completed_prediction_files],
            ignore_index=True,
        )
    else:
        aggregate = pd.concat(all_predictions, ignore_index=True)
    aggregate.to_csv(run_root / "all_target_predictions.csv", index=False)
    completed_metric_files = sorted(run_root.glob("fold_*_seed*/target_metrics.json"))
    if completed_metric_files:
        fold_metrics = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in completed_metric_files
        ]
    pd.DataFrame(fold_metrics).to_csv(run_root / "fold_metrics.csv", index=False)
    seed_metric_rows: list[dict[str, Any]] = []
    for completed_seed, group in aggregate.groupby("seed", sort=True):
        row = regression_metrics(
            group["actual_rul_cycles"], group["predicted_rul_cycles"]
        )
        row["seed"] = int(completed_seed)
        row["completed_folds"] = int(group["held_out_domain"].nunique())
        seed_metric_rows.append(row)
    seed_metric_frame = pd.DataFrame(seed_metric_rows)
    seed_metric_frame.to_csv(run_root / "seed_metrics.csv", index=False)
    aggregate_metrics = regression_metrics(
        aggregate["actual_rul_cycles"], aggregate["predicted_rul_cycles"]
    )
    summary_columns = ["mae_cycles", "rmse_cycles", "mape_percent"]
    seed_mean = {
        column: float(seed_metric_frame[column].mean()) for column in summary_columns
    }
    seed_std = {
        column: float(seed_metric_frame[column].std(ddof=0)) for column in summary_columns
    }
    aggregate_metrics.update(
        {
            "completed_folds": sorted(aggregate["held_out_domain"].unique().tolist()),
            "completed_seeds": sorted(int(value) for value in aggregate["seed"].unique()),
            "protocol": "pure leave-one-domain-out domain generalization",
            "target_adaptation": False,
            "seed_level_metric_mean": seed_mean,
            "seed_level_metric_population_std": seed_std,
        }
    )
    save_json(aggregate_metrics, run_root / "aggregate_metrics.json")
    save_prediction_figure(
        aggregate,
        run_root / "aggregate_target_prediction_scatter.png",
        "CALCE direct RUL: pure unseen-domain evaluation",
    )
    LOGGER.info("Completed experiment: %s", run_root)
    return run_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate standalone CALCE direct-RUL MMDGN-inspired BOIL"
    )
    parser.add_argument(
        "--config", default="calce_direct_rul_boil/config.yaml", help="YAML config"
    )
    parser.add_argument(
        "--fold",
        choices=["all", *DOMAIN_NAMES],
        default=None,
        help="override held-out domain(s) from config",
    )
    parser.add_argument("--seed", type=int, default=None, help="run one seed")
    parser.add_argument(
        "--device", default=None, help="override device: auto, cpu, cuda, cuda:0"
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="last.pt path; requires one explicit --fold and one --seed",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

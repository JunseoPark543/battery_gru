"""CLI for HUST raw-RUL hierarchical BOIL protocol generalization."""

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

from .config import ExperimentConfig, load_config, save_config
from .data import (
    CellSample,
    load_hust_samples,
    protocol_sort_key,
    split_protocol,
)
from .metrics import (
    regression_metrics,
    save_json,
    save_key_results_dashboard,
    save_prediction_figure,
)
from .model import HUSTDirectRULModel
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


def _selected_folds(
    config: ExperimentConfig, available: Sequence[str], override: str | None
) -> list[str]:
    requested: list[str]
    if override is not None:
        requested = list(available) if override == "all" else [override]
    elif config.evaluation.held_out_protocols == "all":
        requested = list(available)
    else:
        requested = list(config.evaluation.held_out_protocols)
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(
            f"unknown HUST held-out protocols {sorted(unknown)}; available={list(available)}"
        )
    return sorted(requested, key=protocol_sort_key)


def _new_run_root(config: ExperimentConfig, seeds: Sequence[int]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed_text = "-".join(str(seed) for seed in seeds)
    name = f"{stamp}_hust-direct-rul-boil_L100_raw-rul_seed{seed_text}"
    return (Path(config.output.output_dir) / name).resolve()


def _parameter_summary(model: HUSTDirectRULModel) -> dict[str, int]:
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


def _evaluate_target(
    trainer: SourceOnlyTrainer,
    targets: Sequence[CellSample],
    seed: int,
    clip_negative: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if any(sample.protocol != trainer.held_out_protocol for sample in targets):
        raise RuntimeError("target partition contains an unexpected protocol")
    waveforms = torch.as_tensor(
        np.stack(
            [trainer.normalizer.transform_waveforms(sample.waveforms) for sample in targets]
        ),
        dtype=torch.float32,
        device=trainer.device,
    )
    scalars = torch.as_tensor(
        np.stack(
            [trainer.normalizer.transform_scalars(sample.scalars) for sample in targets]
        ),
        dtype=torch.float32,
        device=trainer.device,
    )
    trainer.model.eval()
    with torch.no_grad():
        raw_prediction = (
            trainer.model(waveforms, scalars, grl_strength=0.0)
            .prediction.cpu()
            .numpy()
        )
    prediction = np.maximum(raw_prediction, 0.0) if clip_negative else raw_prediction
    rows: list[dict[str, Any]] = []
    for sample, raw, predicted in zip(targets, raw_prediction, prediction):
        rows.append(
            {
                "seed": seed,
                "held_out_protocol": trainer.held_out_protocol,
                "file_name": sample.file_name,
                "cell_id": sample.cell_id,
                "replicate": sample.replicate,
                "history_cycles": sample.waveforms.shape[0],
                "actual_eol_cycle": sample.eol_cycle,
                "actual_rul_cycles": sample.rul_cycles,
                "raw_predicted_rul_cycles": float(raw),
                "predicted_rul_cycles": float(predicted),
                "predicted_eol_cycle": float(predicted + sample.waveforms.shape[0]),
                "absolute_error_cycles": float(abs(predicted - sample.rul_cycles)),
                "target_adaptation_steps": 0,
            }
        )
    frame = pd.DataFrame(rows)
    metrics: dict[str, Any] = regression_metrics(
        frame.actual_rul_cycles, frame.predicted_rul_cycles
    )
    source_mean = float(np.mean([sample.rul_cycles for sample in trainer.train_samples]))
    source_mean_prediction = np.full(len(frame), source_mean, dtype=np.float64)
    baseline = regression_metrics(frame.actual_rul_cycles, source_mean_prediction)
    metrics.update(
        {
            "seed": seed,
            "held_out_protocol": trainer.held_out_protocol,
            "target_adaptation": False,
            "target_labels_used_for_checkpoint_selection": False,
            "label_normalization": "none; raw RUL cycles",
            "best_source_validation_iteration": trainer.best_iteration,
            "best_source_validation_mae_cycles": trainer.best_score,
            "source_mean_baseline_rul_cycles": source_mean,
            "source_mean_baseline_mae_cycles": baseline["mae_cycles"],
        }
    )
    return frame, metrics


def _data_summary(samples: Sequence[CellSample]) -> dict[str, Any]:
    protocols = sorted({sample.protocol for sample in samples}, key=protocol_sort_key)
    return {
        "cells": len(samples),
        "protocols": protocols,
        "cells_per_protocol": {
            protocol: sum(sample.protocol == protocol for sample in samples)
            for protocol in protocols
        },
        "eol_cycle_range": [
            min(sample.eol_cycle for sample in samples),
            max(sample.eol_cycle for sample in samples),
        ],
        "raw_rul_at_cycle_100_range": [
            min(sample.rul_cycles for sample in samples),
            max(sample.rul_cycles for sample in samples),
        ],
        "files": [sample.summary() for sample in samples],
    }


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.device is not None:
        config.device = args.device
        config.validate()
    samples = load_hust_samples(
        Path(config.data.hust_dir), Path(config.data.label_path), config.data
    )
    protocols = sorted({sample.protocol for sample in samples}, key=protocol_sort_key)
    summary = _data_summary(samples)
    if args.inspect_data_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return Path.cwd()
    seeds = [args.seed] if args.seed is not None else list(config.seeds)
    folds = _selected_folds(config, protocols, args.fold)
    if args.resume is not None and len(seeds) != 1:
        raise ValueError("--resume requires exactly one seed")
    device = resolve_device(config.device)
    resume_path = Path(args.resume).resolve() if args.resume else None
    resume_fold: str | None = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_fold_dir = resume_path.parent.parent
        run_root = resume_fold_dir.parent
        matching = [
            protocol
            for protocol in protocols
            if resume_fold_dir.name == f"fold_{protocol}_seed{seeds[0]}"
        ]
        if len(matching) != 1:
            raise ValueError("could not infer HUST protocol fold from checkpoint path")
        resume_fold = matching[0]
        if resume_fold not in folds:
            raise ValueError(f"checkpoint fold {resume_fold} is not selected")
    else:
        run_root = _new_run_root(config, seeds)
        run_root.mkdir(parents=True, exist_ok=False)
        save_config(config, run_root / "resolved_config.yaml")
    _configure_logging(run_root / "experiment.log")
    save_json(summary, run_root / "dataset_summary.json")
    LOGGER.info(
        "HUST raw-RUL BOIL: cells=%d protocols=%s folds=%s seeds=%s device=%s",
        len(samples),
        protocols,
        folds,
        seeds,
        device,
    )
    produced_predictions: list[pd.DataFrame] = []
    produced_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        for held_out in folds:
            set_global_seed(seed)
            source, target = split_protocol(samples, held_out)
            fold_dir = run_root / f"fold_{held_out}_seed{seed}"
            fold_resume: Path | None = None
            if resume_path is not None:
                completed = fold_dir / "target_predictions.csv"
                if completed.is_file():
                    LOGGER.info("Skipping completed fold=%s seed=%d", held_out, seed)
                    continue
                candidate = fold_dir / "checkpoints" / "last.pt"
                if held_out == resume_fold:
                    fold_resume = resume_path
                elif candidate.is_file():
                    fold_resume = candidate
            fold_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                {
                    "protocol": "pure leave-one-HUST-protocol-out domain generalization",
                    "held_out_protocol": held_out,
                    "source_protocols": sorted(
                        {sample.protocol for sample in source}, key=protocol_sort_key
                    ),
                    "source_files": [sample.file_name for sample in source],
                    "target_files": [sample.file_name for sample in target],
                    "target_data_used_during_training": False,
                    "target_adaptation": False,
                    "history_cycles": config.data.history_length,
                    "reference_cycle": config.data.reference_cycle,
                    "label": "raw RUL cycles = EOL - 100",
                    "boil_inner_updated_modules": ["domain_specific"],
                    "boil_inner_frozen_modules": [
                        "domain_invariant", "predictor", "domain_classifier"
                    ],
                },
                fold_dir / "protocol.json",
            )
            trainer = SourceOnlyTrainer(
                source,
                held_out,
                config,
                device,
                fold_dir,
                seed,
            )
            LOGGER.info(
                "Starting fold=%s source=%d target=%d train=%d source_val=%d params=%s",
                held_out,
                len(source),
                len(target),
                len(trainer.train_samples),
                len(trainer.validation_samples),
                _parameter_summary(trainer.model),
            )
            result = trainer.train(resume=fold_resume)
            predictions, metrics = _evaluate_target(
                trainer, target, seed, config.evaluation.clip_negative_rul
            )
            predictions.to_csv(fold_dir / "target_predictions.csv", index=False)
            save_json(metrics, fold_dir / "target_metrics.json")
            save_prediction_figure(
                predictions,
                fold_dir / "target_prediction_scatter.png",
                f"HUST pure DG: unseen {held_out} (seed {seed})",
            )
            save_json(
                {
                    "checkpoint": str(result.checkpoint_path.resolve()),
                    "best_iteration": result.best_iteration,
                    "stopped_iteration": result.stopped_iteration,
                    "label_normalization": "none",
                    "raw_rul_output": True,
                    "normalizer": result.normalizer.state_dict(),
                    "parameter_counts": _parameter_summary(result.model),
                },
                fold_dir / "model_manifest.json",
            )
            produced_predictions.append(predictions)
            produced_metrics.append(metrics)
            LOGGER.info(
                "Completed fold=%s target_MAE=%.3f RMSE=%.3f baseline_MAE=%.3f",
                held_out,
                metrics["mae_cycles"],
                metrics["rmse_cycles"],
                metrics["source_mean_baseline_mae_cycles"],
            )

    prediction_files = sorted(run_root.glob("fold_*_seed*/target_predictions.csv"))
    aggregate = (
        pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
        if prediction_files
        else pd.concat(produced_predictions, ignore_index=True)
    )
    metric_files = sorted(run_root.glob("fold_*_seed*/target_metrics.json"))
    fold_metrics = (
        [json.loads(path.read_text(encoding="utf-8")) for path in metric_files]
        if metric_files
        else produced_metrics
    )
    aggregate.to_csv(run_root / "all_target_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(run_root / "fold_metrics.csv", index=False)
    aggregate_metrics: dict[str, Any] = regression_metrics(
        aggregate.actual_rul_cycles, aggregate.predicted_rul_cycles
    )
    aggregate_metrics.update(
        {
            "completed_protocols": sorted(
                aggregate.held_out_protocol.unique().tolist(), key=protocol_sort_key
            ),
            "completed_seeds": sorted(int(seed) for seed in aggregate.seed.unique()),
            "protocol": "pure leave-one-HUST-protocol-out domain generalization",
            "target_adaptation": False,
            "label_normalization": "none; raw RUL cycles",
        }
    )
    save_json(aggregate_metrics, run_root / "aggregate_metrics.json")
    save_prediction_figure(
        aggregate,
        run_root / "aggregate_target_prediction_scatter.png",
        "HUST direct raw-RUL: unseen-protocol evaluation",
    )
    save_key_results_dashboard(
        aggregate,
        pd.DataFrame(fold_metrics),
        run_root / "key_results_dashboard.png",
    )
    LOGGER.info("Completed HUST experiment: %s", run_root)
    return run_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HUST hierarchical raw-RUL BOIL protocol generalization"
    )
    parser.add_argument(
        "--config", default="hust_direct_rul_boil/config.yaml", help="YAML config"
    )
    parser.add_argument(
        "--fold", default=None, help="all or one protocol ID such as protocol_1"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--resume", default=None, help="path to a fold last.pt")
    parser.add_argument(
        "--inspect-data-only",
        action="store_true",
        help="validate HUST files/features and print summary without creating a model",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

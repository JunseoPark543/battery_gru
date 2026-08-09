"""CLI and experiment orchestration for the hybrid ANIL/BOIL study."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from hust_direct_rul_boil.data import (
    CellSample,
    load_hust_samples,
    protocol_sort_key,
    split_protocol,
)
from hust_direct_rul_boil.metrics import (
    regression_metrics,
    save_json,
    save_prediction_figure,
)

from .analysis import (
    save_adaptation_curve,
    save_feature_visualization,
    save_method_comparison_figure,
)
from .config import METHODS, ExperimentConfig, load_config, save_config
from .evaluator import evaluate_unseen_protocol
from .meta import parameter_policy
from .trainer import HybridTrainer, resolve_device, set_global_seed


LOGGER = logging.getLogger(__name__)
DISPLAY_NAMES = {
    "supervised": "Supervised",
    "maml": "Full MAML",
    "anil": "ANIL",
    "boil": "BOIL",
    "hybrid": "General-ANIL + Specific-BOIL",
}


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


def _data_summary(samples: Sequence[CellSample]) -> dict[str, Any]:
    protocols = sorted({sample.protocol for sample in samples}, key=protocol_sort_key)
    return {
        "cells": len(samples),
        "protocols": protocols,
        "cells_per_protocol": {
            protocol: sum(sample.protocol == protocol for sample in samples)
            for protocol in protocols
        },
        "input_shape_per_cell": {
            "waveforms": list(samples[0].waveforms.shape),
            "scalars": list(samples[0].scalars.shape),
        },
        "target": "raw RUL cycles = EOL - 100",
        "eol_cycle_range": [
            min(sample.eol_cycle for sample in samples),
            max(sample.eol_cycle for sample in samples),
        ],
        "rul_cycle_range": [
            min(sample.rul_cycles for sample in samples),
            max(sample.rul_cycles for sample in samples),
        ],
        "files": [sample.summary() for sample in samples],
    }


def _selected_folds(
    config: ExperimentConfig, available: Sequence[str], override: str | None
) -> list[str]:
    if override is not None:
        requested = list(available) if override == "all" else [override]
    elif config.evaluation.held_out_protocols == "all":
        requested = list(available)
    else:
        requested = list(config.evaluation.held_out_protocols)
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"unknown held-out protocols {sorted(unknown)}")
    return sorted(requested, key=protocol_sort_key)


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.device is not None:
        config.device = args.device
    if args.inner_steps is not None:
        config.train.inner_steps = args.inner_steps
    if args.inner_lr_general is not None:
        config.train.inner_lr_general_head = args.inner_lr_general
    if args.inner_lr_specific is not None:
        config.train.inner_lr_specific_encoder = args.inner_lr_specific
    if args.first_order:
        config.train.first_order = True
    if args.second_order:
        config.train.first_order = False
    if args.prediction_mode is not None:
        config.ablation.prediction_mode = args.prediction_mode
    for argument, attribute in (
        ("lambda_total", "lambda_total_prediction"),
        ("lambda_general_prediction", "lambda_general_prediction"),
        ("lambda_general_domain", "lambda_general_domain"),
        ("lambda_specific_domain", "lambda_specific_domain"),
        ("lambda_specific_contrastive", "lambda_specific_contrastive"),
        ("lambda_reconstruction", "lambda_reconstruction"),
        ("lambda_consistency", "lambda_consistency"),
        ("lambda_orthogonal", "lambda_orthogonal"),
        ("lambda_residual", "lambda_residual"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(config.loss, attribute, value)
    for argument, attribute in (
        ("no_grl", "use_grl"),
        ("no_specific_domain", "use_specific_domain_classifier"),
        ("no_reconstruction", "use_reconstruction"),
        ("no_consistency", "use_consistency"),
        ("no_orthogonality", "use_orthogonality"),
        ("no_general_prediction_loss", "use_general_prediction_loss"),
        ("no_residual_regularization", "use_residual_regularization"),
    ):
        if getattr(args, argument):
            setattr(config.ablation, attribute, False)
    config.validate()


def _run_name(methods: Sequence[str], seeds: Sequence[int], config: ExperimentConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    method_text = "all-methods" if len(methods) > 1 else methods[0]
    seed_text = "-".join(str(seed) for seed in seeds)
    order = "fo" if config.train.first_order else "so"
    validation = (
        "valdomain"
        if config.train.validation_strategy == "held_out_source_protocol"
        else "valcells"
    )
    contrastive = str(config.loss.lambda_specific_contrastive).replace(".", "p")
    return (
        f"{stamp}_hust_{method_text}_L100_{order}_inner{config.train.inner_steps}_"
        f"{validation}_sc{contrastive}_rep{config.evaluation.target_support_repeats}_"
        f"seed{seed_text}"
    )


def _fold_prediction_figure(frame: pd.DataFrame, path: Path, title: str) -> None:
    primary = frame.rename(
        columns={"target_y": "actual_rul_cycles", "y_hat": "predicted_rul_cycles"}
    )
    save_prediction_figure(primary, path, title)


def _aggregate_outputs(run_root: Path, config: ExperimentConfig) -> None:
    prediction_files = sorted(run_root.glob("method_*/fold_*_seed*/target_predictions.csv"))
    metric_files = sorted(run_root.glob("method_*/fold_*_seed*/adaptation_metrics.csv"))
    if not prediction_files or not metric_files:
        raise RuntimeError("no completed fold outputs were found for aggregation")
    predictions = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    fold_metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    predictions.to_csv(run_root / "all_target_predictions.csv", index=False)
    fold_metrics.to_csv(run_root / "all_fold_adaptation_metrics.csv", index=False)
    aggregate_rows: list[dict[str, Any]] = []
    for (method, step), group in predictions.groupby(["method", "adaptation_step"], sort=False):
        metrics = regression_metrics(group.target_y, group.y_hat)
        fold_group = fold_metrics[
            (fold_metrics.method == method) & (fold_metrics.adaptation_step == step)
        ]
        aggregate_rows.append(
            {
                "method": method,
                "adaptation_step": int(step),
                **metrics,
                "general_domain_accuracy": float(
                    fold_group.source_validation_general_domain_accuracy.mean()
                ),
                "specific_domain_accuracy": float(
                    fold_group.source_validation_specific_domain_accuracy.mean()
                ),
                "general_representation_change": float(
                    fold_group.general_representation_cosine_distance.mean()
                ),
                "specific_representation_change": float(
                    fold_group.specific_representation_cosine_distance.mean()
                ),
                "mean_absolute_specific_residual": float(
                    fold_group.mean_absolute_specific_residual.mean()
                ),
                "y_general_mae_cycles": float(fold_group.y_general_mae_cycles.mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(run_root / "aggregate_adaptation_metrics.csv", index=False)
    comparison_rows: list[dict[str, Any]] = []
    for method, group in aggregate.groupby("method", sort=False):
        best = group.loc[group.mae_cycles.idxmin()]
        primary_step = 0 if method == "supervised" else config.evaluation.primary_adaptation_step
        primary_match = group[group.adaptation_step == primary_step]
        if primary_match.empty:
            raise RuntimeError(f"method={method} has no primary step={primary_step}")
        primary = primary_match.iloc[0]
        policy_file = next(run_root.glob(f"method_{method}/fold_*_seed*/inner_policy.json"))
        policy = json.loads(policy_file.read_text(encoding="utf-8"))
        comparison_rows.append(
            {
                "method": DISPLAY_NAMES[method],
                "method_key": method,
                "mae_cycles": primary.mae_cycles,
                "rmse_cycles": primary.rmse_cycles,
                "mape_percent": primary.mape_percent,
                "r2": primary.r2,
                "params_adapted": ", ".join(policy["inner_updated_modules"]) or "none",
                "adapted_parameter_count": int(policy["inner_updated_parameter_count"]),
                "primary_adaptation_step": primary_step,
                "diagnostic_best_adaptation_step": int(best.adaptation_step),
                "diagnostic_best_mae_cycles": best.mae_cycles,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(run_root / "method_comparison.csv", index=False)
    save_method_comparison_figure(comparison, run_root / "method_comparison.png")
    for method in predictions.method.unique():
        feature_files = sorted(run_root.glob(f"method_{method}/fold_*_seed*/primary_features.npz"))
        if not feature_files:
            continue
        payloads = [np.load(path, allow_pickle=False) for path in feature_files]
        general = np.concatenate([payload["general"] for payload in payloads])
        specific = np.concatenate([payload["specific"] for payload in payloads])
        domains = np.concatenate([payload["domain"] for payload in payloads])
        normalized_rul = np.concatenate([payload["normalized_rul"] for payload in payloads])
        if len(general) >= 3:
            save_feature_visualization(
                general,
                specific,
                domains,
                normalized_rul,
                run_root / f"method_{method}" / "representation_tsne.png",
                seed=int(config.seeds[0]),
            )
        for payload in payloads:
            payload.close()
    proposed = aggregate[
        (aggregate.method == "hybrid")
        & (aggregate.adaptation_step == config.evaluation.primary_adaptation_step)
    ]
    if not proposed.empty:
        row = proposed.iloc[0].to_dict()
        save_json(
            {
                "method": "General-ANIL + Specific-BOIL",
                "primary_adaptation_step": config.evaluation.primary_adaptation_step,
                "general_domain_accuracy_source_validation": row["general_domain_accuracy"],
                "specific_domain_accuracy_source_validation": row["specific_domain_accuracy"],
                "general_representation_cosine_change": row["general_representation_change"],
                "specific_representation_cosine_change": row["specific_representation_change"],
                "mean_absolute_delta_y_S": row["mean_absolute_specific_residual"],
                "y_G_only_mae_cycles": row["y_general_mae_cycles"],
                "final_mae_cycles": row["mae_cycles"],
                "best_step_is_oracle_diagnostic_only": True,
            },
            run_root / "proposed_method_summary.json",
        )


def run(args: argparse.Namespace) -> Path:
    config = load_config(Path(args.config).resolve())
    _apply_overrides(config, args)
    samples = load_hust_samples(config.data.hust_dir, config.data.label_path, config.data)
    summary = _data_summary(samples)
    if args.inspect_data_only:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return Path.cwd()
    protocols = sorted({sample.protocol for sample in samples}, key=protocol_sort_key)
    folds = _selected_folds(config, protocols, args.fold)
    seeds = [args.seed] if args.seed is not None else list(config.seeds)
    methods = list(METHODS) if args.method == "all" else [args.method or config.method]
    if args.resume is not None and (len(methods) != 1 or len(seeds) != 1 or len(folds) != 1):
        raise ValueError("--resume requires one method, one seed, and one --fold")
    device = resolve_device(config.device)
    if args.resume is None:
        run_root = (Path(config.output.output_dir) / _run_name(methods, seeds, config)).resolve()
        run_root.mkdir(parents=True, exist_ok=False)
    else:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        run_root = resume_path.parents[3]
    _configure_logging(run_root / "experiment.log")
    save_config(config, run_root / "resolved_config.yaml")
    save_json(summary, run_root / "dataset_summary.json")
    (run_root / "command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "HUST hybrid study methods=%s folds=%s seeds=%s device=%s target_support=%d",
        methods,
        folds,
        seeds,
        device,
        config.evaluation.target_support_cells,
    )
    resume_path = Path(args.resume).resolve() if args.resume else None
    for method in methods:
        method_config = copy.deepcopy(config)
        method_config.method = method
        method_config.validate()
        method_dir = run_root / f"method_{method}"
        method_dir.mkdir(parents=True, exist_ok=True)
        save_config(method_config, method_dir / "resolved_config.yaml")
        for seed in seeds:
            for held_out in folds:
                set_global_seed(seed)
                source, target = split_protocol(samples, held_out)
                fold_dir = method_dir / f"fold_{held_out}_seed{seed}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                completed = fold_dir / "target_predictions.csv"
                if completed.is_file() and resume_path is not None:
                    LOGGER.info("Skipping completed method=%s fold=%s seed=%d", method, held_out, seed)
                    continue
                trainer = HybridTrainer(
                    source,
                    held_out,
                    method_config,
                    device,
                    fold_dir,
                    seed,
                )
                save_json(
                    {
                        "method": method,
                        "held_out_protocol": held_out,
                        "source_protocols": trainer.source_protocols,
                        "target_files": [sample.file_name for sample in target],
                        "target_protocol_used_for_training_or_checkpoint_selection": False,
                        "target_evaluation": (
                            "few-shot: labeled target support cells adapt the model; "
                            "disjoint target query cells are metrics-only"
                        ),
                        "parameter_policy": parameter_policy(trainer.model, method),
                    },
                    fold_dir / "protocol.json",
                )
                LOGGER.info(
                    "Starting method=%s fold=%s source=%d target=%d params=%s",
                    method,
                    held_out,
                    len(source),
                    len(target),
                    trainer.model.module_parameter_counts(),
                )
                training = trainer.train(resume=resume_path)
                evaluation = evaluate_unseen_protocol(
                    training, target, method_config, device, seed
                )
                evaluation.predictions.to_csv(fold_dir / "target_predictions.csv", index=False)
                evaluation.metrics.to_csv(fold_dir / "adaptation_metrics.csv", index=False)
                np.savez_compressed(fold_dir / "primary_features.npz", **evaluation.feature_payload)
                save_json(
                    {
                        "target_support_splits": evaluation.support_splits,
                        "target_support_repeats": method_config.evaluation.target_support_repeats,
                        "target_support_labels_used_for_adaptation": method != "supervised",
                        "target_query_labels_used_for_adaptation_or_model_selection": False,
                        "best_checkpoint": str(training.checkpoint_path.resolve()),
                        "normalizer": training.normalizer.state_dict(),
                        "parameter_counts": training.model.module_parameter_counts(),
                    },
                    fold_dir / "evaluation_manifest.json",
                )
                save_adaptation_curve(
                    evaluation.metrics,
                    fold_dir / "adaptation_curve.png",
                    f"{DISPLAY_NAMES[method]} | unseen {held_out} | seed {seed}",
                )
                primary = evaluation.predictions[
                    evaluation.predictions.adaptation_step
                    == method_config.evaluation.primary_adaptation_step
                ]
                _fold_prediction_figure(
                    primary,
                    fold_dir / "target_prediction_scatter.png",
                    f"{DISPLAY_NAMES[method]}: unseen {held_out}",
                )
                LOGGER.info("Completed method=%s fold=%s seed=%d", method, held_out, seed)
                resume_path = None
    _aggregate_outputs(run_root, config)
    LOGGER.info("Completed HUST hybrid study: %s", run_root)
    return run_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HUST General-ANIL + Specific-BOIL direct-RUL study"
    )
    parser.add_argument("--config", default="hust_hybrid_anil_boil/config.yaml")
    parser.add_argument("--method", choices=(*METHODS, "all"), default=None)
    parser.add_argument("--fold", default=None, help="all or protocol_1 ... protocol_10")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--inspect-data-only", action="store_true")
    parser.add_argument("--inner-steps", type=int, default=None)
    parser.add_argument("--inner-lr-general", type=float, default=None)
    parser.add_argument("--inner-lr-specific", type=float, default=None)
    order = parser.add_mutually_exclusive_group()
    order.add_argument("--first-order", action="store_true")
    order.add_argument("--second-order", action="store_true")
    parser.add_argument("--prediction-mode", choices=("residual", "concat"), default=None)
    parser.add_argument("--lambda-total", type=float, default=None)
    parser.add_argument("--lambda-general-prediction", type=float, default=None)
    parser.add_argument("--lambda-general-domain", type=float, default=None)
    parser.add_argument("--lambda-specific-domain", type=float, default=None)
    parser.add_argument("--lambda-specific-contrastive", type=float, default=None)
    parser.add_argument("--lambda-reconstruction", type=float, default=None)
    parser.add_argument("--lambda-consistency", type=float, default=None)
    parser.add_argument("--lambda-orthogonal", type=float, default=None)
    parser.add_argument("--lambda-residual", type=float, default=None)
    parser.add_argument("--no-grl", action="store_true")
    parser.add_argument("--no-specific-domain", action="store_true")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--no-consistency", action="store_true")
    parser.add_argument("--no-orthogonality", action="store_true")
    parser.add_argument("--no-general-prediction-loss", action="store_true")
    parser.add_argument("--no-residual-regularization", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

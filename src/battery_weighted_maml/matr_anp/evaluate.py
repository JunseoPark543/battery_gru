"""Held-out cell evaluation for MATR ANP checkpoints."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, load_config, resolve_data_root, save_config
from .data import load_matr_dataset
from .episodes import EpisodeSampler
from .features import EpisodeUnavailable, FoldScalers, PartialIVProcessor
from .inference import (
    measure_forward_latency,
    predict_episode,
    prediction_frame,
    trajectory_metrics,
)
from .model import build_model
from .plotting import plot_metric_vs_beta, plot_trajectory_betas
from .runtime import parameter_checksum, resolve_device, seed_everything, write_json


METRICS = (
    "future_rmse",
    "current_soh_abs_error",
    "nll",
    "coverage_95",
    "interval_width_95",
    "inference_latency_ms",
)


def _validate_checkpoint_config(config: ExperimentConfig, payload: dict[str, Any]) -> None:
    saved = payload.get("config", {})
    current = config.to_dict()
    for section in ("seed", "data", "q_grid", "episode", "model"):
        if saved.get(section) != current.get(section):
            raise ValueError(
                f"evaluation config section '{section}' differs from the training checkpoint"
            )


def _aggregate(per_cell: pd.DataFrame) -> pd.DataFrame:
    valid = per_cell[per_cell["status"] == "ok"]
    rows: list[dict[str, Any]] = []
    group_columns = ["fold", "seed", "model", "alpha", "beta"]
    for keys, group in valid.groupby(group_columns, dropna=False):
        base = dict(zip(group_columns, keys))
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "num_cells": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_run(
    config: ExperimentConfig,
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    mc_samples: int | None = None,
) -> Path:
    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MATR ANP checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("dataset") != "MATR" or payload.get("algorithm") != "attentive_neural_process":
        raise ValueError("checkpoint is not a MATR Attentive Neural Process checkpoint")
    _validate_checkpoint_config(config, payload)
    sample_count = mc_samples or config.evaluation.mc_samples
    if sample_count <= 0:
        raise ValueError("mc_samples must be positive")
    seed_everything(config.seed, config.training.deterministic)
    device = resolve_device(config.device)
    spec = payload["model_spec"]
    model_name = str(spec["model_name"])
    model, rebuilt_spec = build_model(
        model_name,
        config.model,
        resolved_hidden_dim=int(spec["hidden_dim"]),
    )
    if rebuilt_spec.parameter_count != int(spec["parameter_count"]):
        raise ValueError("checkpoint/model parameter count mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    checksum_before = parameter_checksum(model)

    cells, audit = load_matr_dataset(data_root, config.data, tolerate_invalid_cells=True)
    by_id = {cell.cell_id: cell for cell in cells}
    split = payload["fold_split"]
    missing = sorted(set(split["test_cells"]) - set(by_id))
    if missing:
        raise ValueError(f"checkpoint test cells are missing from MATR data: {missing}")
    scalers = FoldScalers.from_dict(payload["scalers"])
    if set(scalers.fit_cell_ids) != set(split["train_cells"]):
        raise ValueError("checkpoint scalers were not fit on exactly the fold training cells")
    if set(scalers.fit_cell_ids) & set(split["test_cells"]):
        raise ValueError("test-cell leakage detected in checkpoint scalers")
    processor = PartialIVProcessor(config.q_grid, config.data)
    sampler = EpisodeSampler(config.episode, processor, scalers)
    fold = int(split["fold"])

    destination = Path(output_dir).resolve() if output_dir else source.parent.parent / "evaluation" / source.stem
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "plots").mkdir(exist_ok=True)
    save_config(config, destination / "resolved_config.yaml")
    audit.to_csv(destination / "data_audit.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    valid_plot_keys: list[tuple[str, float, int]] = []
    for cell_index, cell_id in enumerate(split["test_cells"]):
        cell = by_id[cell_id]
        for alpha_index, alpha in enumerate(config.episode.evaluation_alphas):
            baseline_cache = None
            for beta_index, beta in enumerate(config.episode.beta_values):
                base = {
                    "fold": fold,
                    "seed": config.seed,
                    "model": model_name,
                    "cell_id": cell_id,
                    "alpha": float(alpha),
                    "beta": float(beta),
                }
                try:
                    episode = sampler.evaluation(cell, alpha, beta)
                except EpisodeUnavailable as exc:
                    metric_rows.append(
                        {**base, "status": "skipped", "reason": str(exc), **{name: np.nan for name in METRICS}}
                    )
                    continue
                # The same seed at all betas makes SOH-only baselines exactly horizontal.
                prediction_seed = config.seed + fold * 1_000_003 + cell_index * 10_007 + alpha_index * 101
                if not rebuilt_spec.conditional_iv and baseline_cache is not None:
                    result, latency_mean, latency_median, latency_std = baseline_cache
                else:
                    result = predict_episode(
                        model,
                        episode,
                        scalers,
                        device,
                        mc_samples=sample_count,
                        interval_level=config.evaluation.interval_level,
                        seed=prediction_seed,
                    )
                    latency_mean, latency_median, latency_std = measure_forward_latency(
                        model,
                        episode,
                        device,
                        warmup=config.evaluation.inference_warmup,
                        repeats=config.evaluation.inference_repeats,
                    )
                    if not rebuilt_spec.conditional_iv:
                        baseline_cache = (
                            result, latency_mean, latency_median, latency_std
                        )
                metrics = trajectory_metrics(episode, result)
                metric_rows.append(
                    {
                        **base,
                        "status": "ok",
                        "reason": "",
                        **metrics,
                        "inference_latency_ms": latency_mean,
                        "inference_latency_median_ms": latency_median,
                        "inference_latency_std_ms": latency_std,
                        "current_cycle": episode.current_cycle,
                        "num_context_points": len(episode.context_x),
                        "num_target_points": len(episode.target_x),
                        "reference_cycles": ",".join(map(str, episode.reference_cycles)),
                    }
                )
                prediction_frames.append(
                    prediction_frame(
                        episode,
                        result,
                        scalers,
                        model_name=model_name,
                        fold=fold,
                        seed=config.seed,
                    )
                )
                valid_plot_keys.append((cell_id, float(alpha), episode.current_cycle))

    per_cell = pd.DataFrame(metric_rows)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    aggregate = _aggregate(per_cell)
    per_cell.to_csv(destination / "per_cell_metrics.csv", index=False)
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    predictions.to_csv(destination / "trajectory_predictions.csv", index=False)
    valid_metrics = per_cell[per_cell["status"] == "ok"]
    if not valid_metrics.empty:
        plot_metric_vs_beta(valid_metrics, "future_rmse", "Future SOH RMSE", destination / "plots/rmse_vs_beta.png")
        plot_metric_vs_beta(valid_metrics, "current_soh_abs_error", "Current-cycle SOH absolute error", destination / "plots/current_error_vs_beta.png")
        plot_metric_vs_beta(valid_metrics, "interval_width_95", "95% interval width", destination / "plots/interval_width_vs_beta.png")
        plot_metric_vs_beta(valid_metrics, "coverage_95", "95% interval coverage", destination / "plots/coverage_vs_beta.png")
    if valid_plot_keys and not predictions.empty:
        requested = config.evaluation.trajectory_plot_cell
        eligible = [key for key in valid_plot_keys if requested is None or key[0] == requested]
        if not eligible:
            raise ValueError(f"trajectory_plot_cell is not a valid evaluated test cell: {requested}")
        cell_id, alpha, current_cycle = eligible[0]
        selected = predictions[(predictions["cell_id"] == cell_id) & (predictions["alpha"] == alpha)]
        plot_trajectory_betas(
            selected,
            destination / f"plots/trajectory_{cell_id}_alpha{alpha:g}.png",
            cell_id=cell_id,
            alpha=alpha,
            current_cycle=current_cycle,
        )
    checksum_after = parameter_checksum(model)
    if checksum_before != checksum_after:
        raise RuntimeError("evaluation changed model parameters")
    write_json(
        destination / "evaluation_manifest.json",
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(source),
            "dataset": "MATR",
            "model": model_name,
            "fold": fold,
            "seed": config.seed,
            "test_cells": split["test_cells"],
            "mc_samples": sample_count,
            "parameter_checksum_before": checksum_before,
            "parameter_checksum_after": checksum_after,
            "test_time_optimization": False,
            "num_valid_combinations": int((per_cell["status"] == "ok").sum()),
            "num_skipped_combinations": int((per_cell["status"] != "ok").sum()),
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a held-out MATR ANP fold")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--fold", type=int, help="optional assertion against checkpoint fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    data_root = resolve_data_root(config, args.data_root)
    if args.fold is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if int(payload["fold_split"]["fold"]) != args.fold:
            raise ValueError("--fold does not match the checkpoint fold")
    destination = evaluate_run(
        config,
        args.checkpoint,
        data_root,
        output_dir=args.output_dir,
        mc_samples=args.mc_samples,
    )
    print(f"Evaluation directory: {destination}")


if __name__ == "__main__":
    main()

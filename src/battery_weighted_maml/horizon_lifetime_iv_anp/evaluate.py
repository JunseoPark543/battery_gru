"""Held-out cell evaluation for lifetime prediction and derived RUL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from battery_weighted_maml.horizon_rul_anp.metrics import (
    interval_metrics,
    regression_metrics,
)
from battery_weighted_maml.matr_anp.runtime import (
    parameter_checksum,
    resolve_device,
    seed_everything,
    write_json,
)
from battery_weighted_maml.matr_anp.splits import FoldSplit

from .config import LifetimeIVConfig, load_config, resolve_data_root, save_config
from .data import (
    LabeledCell,
    LifetimeIVPrefixStore,
    LifetimeIVScalers,
    load_labeled_cells,
)
from .inference import LifetimePrediction, predict_batch
from .model import LifetimeIVANP, build_model
from .tasks import LifetimeTask, LifetimeTaskSampler, TaskUnavailable, collate_tasks
from .train import split_cells


@dataclass
class LoadedExperiment:
    config: LifetimeIVConfig
    model: LifetimeIVANP
    device: torch.device
    scalers: LifetimeIVScalers
    sampler: LifetimeTaskSampler
    train_cells: list[LabeledCell]
    validation_cells: list[LabeledCell]
    test_cells: list[LabeledCell]
    audit: pd.DataFrame
    payload: dict[str, Any]


def load_experiment(
    config: LifetimeIVConfig,
    checkpoint: str | Path,
    data_root: str | Path,
) -> LoadedExperiment:
    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"lifetime I-V checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "horizon_conditioned_lifetime_iv_anp":
        raise ValueError("checkpoint is not a lifetime I-V ANP")
    current, saved = config.to_dict(), payload["config"]
    for section in ("seed", "data", "q_grid", "split", "task", "model"):
        if current.get(section) != saved.get(section):
            raise ValueError(f"checkpoint config section differs: {section}")
    cells, audit = load_labeled_cells(data_root, config.data)  # type: ignore[arg-type]
    split = FoldSplit(**payload["fold_split"])
    split.validate([item.cell_id for item in cells])
    train, validation, test = split_cells(cells, split)
    scalers = LifetimeIVScalers.fit(train, max(config.task.horizons))
    if scalers.to_dict() != payload["scalers"]:
        raise ValueError("checkpoint train-only scalers differ")
    store = LifetimeIVPrefixStore(scalers, config.q_grid, max(config.task.horizons))
    sampler = LifetimeTaskSampler(config.task, scalers, store)
    model, spec = build_model(config.model, config.q_grid.num_points)
    if spec.to_dict() != payload["model_spec"]:
        raise ValueError("checkpoint model specification differs")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = resolve_device(config.device)
    model.to(device).eval()
    return LoadedExperiment(
        config, model, device, scalers, sampler,
        train, validation, test, audit, payload,
    )


def predict_task(
    experiment: LoadedExperiment,
    task: LifetimeTask,
    *,
    mc_samples: int,
    seed: int,
) -> LifetimePrediction:
    batch = collate_tasks([task]).to(experiment.device)
    return predict_batch(
        experiment.model, batch, experiment.scalers,
        mc_samples=mc_samples,
        interval_level=experiment.config.evaluation.interval_level,
        seed=seed,
    )


def _plot_metrics(frame: pd.DataFrame, destination: Path) -> None:
    valid = frame[frame["status"] == "ok"].sort_values("horizon")
    if valid.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, column, label in (
        (axes[0], "lifetime_rmse_cycles", "Lifetime RMSE (cycles)"),
        (axes[1], "rul_mae_cycles", "RUL MAE (cycles)"),
        (axes[2], "rul_coverage", "RUL interval coverage"),
    ):
        axis.plot(valid["horizon"], valid[column], marker="o", lw=2)
        axis.set(xlabel="Observation horizon k", ylabel=label, title=label)
        axis.grid(alpha=0.25)
    figure.suptitle("Lifetime I-V ANP: held-out MATR cells")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_cell(frame: pd.DataFrame, destination: Path, cell_id: str) -> None:
    selected = frame[frame["cell_id"] == cell_id].sort_values("horizon")
    if selected.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(selected["horizon"], selected["true_lifetime_cycles"], color="black", lw=2, label="True")
    axes[0].plot(selected["horizon"], selected["predicted_lifetime_mean_cycles"], marker="o", lw=2, label="Predicted")
    axes[0].fill_between(selected["horizon"], selected["lifetime_lower_cycles"], selected["lifetime_upper_cycles"], alpha=0.22)
    axes[0].set(xlabel="Observed cycle k", ylabel="Lifetime/EOL cycle", title="Lifetime")
    axes[1].plot(selected["horizon"], selected["true_rul_cycles"], color="black", lw=2, label="True")
    axes[1].plot(selected["horizon"], selected["predicted_rul_mean_cycles"], marker="o", lw=2, label="Predicted")
    axes[1].fill_between(selected["horizon"], selected["rul_lower_cycles"], selected["rul_upper_cycles"], alpha=0.22)
    axes[1].set(xlabel="Observed cycle k", ylabel="RUL (cycles)", title="Derived RUL")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Streaming prediction without parameter updates: {cell_id}")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def evaluate_checkpoint(
    config: LifetimeIVConfig,
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    horizons: Sequence[int] | None = None,
    mc_samples: int | None = None,
) -> Path:
    seed_everything(config.seed, config.training.deterministic)
    experiment = load_experiment(config, checkpoint, data_root)
    checksum_before = parameter_checksum(experiment.model)
    selected_horizons = [int(value) for value in (horizons or config.evaluation.horizons)]
    if not selected_horizons or selected_horizons != sorted(set(selected_horizons)):
        raise ValueError("evaluation horizons must be sorted and unique")
    if not set(selected_horizons).issubset(config.task.horizons):
        raise ValueError("evaluation horizons must be trained task horizons")
    sample_count = int(mc_samples or config.evaluation.mc_samples)
    source = Path(checkpoint).resolve()
    destination = Path(output_dir).resolve() if output_dir else source.parent.parent / "evaluation" / source.stem
    (destination / "plots").mkdir(parents=True, exist_ok=True)
    save_config(config, destination / "resolved_config.yaml")
    experiment.audit.to_csv(destination / "data_and_label_audit.csv", index=False)
    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for horizon in selected_horizons:
        try:
            task = experiment.sampler.evaluation(
                horizon, experiment.train_cells, experiment.test_cells,
                context_size=config.evaluation.context_size, seed=config.seed,
            )
        except TaskUnavailable as exc:
            aggregates.append({"horizon": horizon, "status": "skipped", "reason": str(exc)})
            continue
        prediction = predict_task(
            experiment, task, mc_samples=sample_count,
            seed=config.seed + horizon * 10_007,
        )
        true_lifetime = np.asarray([point.lifetime_cycles for point in task.query])
        true_rul = true_lifetime - horizon
        predicted_lifetime = prediction.lifetime_mean[0, :len(task.query)]
        predicted_rul = prediction.rul_mean[0, :len(task.query)]
        lifetime_metrics = regression_metrics(
            true_lifetime, predicted_lifetime,
            mape_epsilon_cycles=config.evaluation.mape_epsilon_cycles,
        )
        rul_metrics = regression_metrics(
            true_rul, predicted_rul,
            mape_epsilon_cycles=config.evaluation.mape_epsilon_cycles,
        )
        coverage = interval_metrics(
            true_rul,
            prediction.rul_lower[0, :len(task.query)],
            prediction.rul_upper[0, :len(task.query)],
        )
        aggregates.append({
            "horizon": horizon, "status": "ok", "reason": "",
            "num_context_cells": len(task.context), "num_test_cells": len(task.query),
            "lifetime_rmse_cycles": lifetime_metrics["rmse_cycles"],
            "lifetime_mae_cycles": lifetime_metrics["mae_cycles"],
            "rul_rmse_cycles": rul_metrics["rmse_cycles"],
            "rul_mae_cycles": rul_metrics["mae_cycles"],
            "rul_mape_percent": rul_metrics["mape_percent"],
            "rul_bias_cycles": rul_metrics["bias_cycles"],
            "rul_coverage": coverage["coverage"],
            "rul_mean_interval_width_cycles": coverage["mean_interval_width_cycles"],
        })
        context_ids = ",".join(point.cell_id for point in task.context)
        for index, point in enumerate(task.query):
            rows.append({
                "horizon": horizon, "cell_id": point.cell_id,
                "true_lifetime_cycles": point.lifetime_cycles,
                "predicted_lifetime_mean_cycles": float(prediction.lifetime_mean[0, index]),
                "predicted_lifetime_std_cycles": float(prediction.lifetime_std[0, index]),
                "lifetime_lower_cycles": float(prediction.lifetime_lower[0, index]),
                "lifetime_upper_cycles": float(prediction.lifetime_upper[0, index]),
                "true_rul_cycles": point.lifetime_cycles - horizon,
                "predicted_rul_mean_cycles": float(prediction.rul_mean[0, index]),
                "rul_lower_cycles": float(prediction.rul_lower[0, index]),
                "rul_upper_cycles": float(prediction.rul_upper[0, index]),
                "absolute_error_cycles": abs(float(prediction.lifetime_mean[0, index]) - point.lifetime_cycles),
                "context_cell_ids": context_ids,
                "query_label_used_as_input": False,
                "model_parameter_update": False,
                "interval_level": config.evaluation.interval_level,
            })
    per_cell = pd.DataFrame(rows)
    aggregate = pd.DataFrame(aggregates)
    per_cell.to_csv(destination / "per_cell_predictions.csv", index=False)
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    _plot_metrics(aggregate, destination / "plots/metrics_by_horizon.png")
    if not per_cell.empty:
        cell_id = config.evaluation.plot_cell or sorted(per_cell["cell_id"].unique())[0]
        _plot_cell(per_cell, destination / f"plots/streaming_lifetime_rul_{cell_id}.png", cell_id)
    checksum_after = parameter_checksum(experiment.model)
    if checksum_before != checksum_after:
        raise RuntimeError("evaluation changed model parameters")
    write_json(destination / "evaluation_manifest.json", {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(source),
        "horizons": selected_horizons,
        "prediction_target": "lifetime",
        "rul_formula": "predicted_lifetime - observation_horizon",
        "query_label_used_as_input": False,
        "model_parameter_update": False,
    })
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MATR lifetime I-V ANP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/matr_horizon_lifetime_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--mc-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    destination = evaluate_checkpoint(
        config, args.checkpoint, resolve_data_root(config, args.data_root),
        output_dir=args.output_dir, horizons=args.horizons,
        mc_samples=args.mc_samples,
    )
    print(f"Lifetime I-V evaluation: {destination}")


if __name__ == "__main__":
    main()

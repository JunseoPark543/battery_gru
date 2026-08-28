"""Held-out cell evaluation for the horizon-conditioned MATR RUL ANP."""

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

from battery_weighted_maml.matr_anp.runtime import (
    parameter_checksum,
    resolve_device,
    seed_everything,
    write_json,
)
from battery_weighted_maml.matr_anp.splits import FoldSplit

from .config import HorizonRULConfig, load_config, resolve_data_root, save_config
from .data import LabeledCell, RULScalers, load_labeled_cells
from .inference import RULPrediction, predict_batch
from .metrics import interval_metrics, regression_metrics
from .model import HorizonRULANP, build_model
from .tasks import HorizonTask, HorizonTaskSampler, TaskUnavailable, collate_tasks


@dataclass
class LoadedExperiment:
    config: HorizonRULConfig
    model: HorizonRULANP
    device: torch.device
    scalers: RULScalers
    sampler: HorizonTaskSampler
    train_cells: list[LabeledCell]
    validation_cells: list[LabeledCell]
    test_cells: list[LabeledCell]
    audit: pd.DataFrame
    payload: dict[str, Any]


def load_experiment(
    config: HorizonRULConfig,
    checkpoint: str | Path,
    data_root: str | Path,
) -> LoadedExperiment:
    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"horizon RUL checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "horizon_conditioned_rul_anp":
        raise ValueError("checkpoint is not a horizon-conditioned RUL ANP")
    if payload.get("dataset") != "MATR":
        raise ValueError("checkpoint dataset is not MATR")
    current = config.to_dict()
    saved = payload["config"]
    for section in ("seed", "data", "split", "task", "model"):
        if saved.get(section) != current.get(section):
            raise ValueError(f"checkpoint config section differs: {section}")
    cells, audit = load_labeled_cells(data_root, config.data)
    by_id = {item.cell_id: item for item in cells}
    split = FoldSplit(**payload["fold_split"])
    split.validate(list(by_id))
    train = [by_id[cell_id] for cell_id in split.train_cells]
    validation = [by_id[cell_id] for cell_id in split.validation_cells]
    test = [by_id[cell_id] for cell_id in split.test_cells]
    scalers = RULScalers.fit(train, config.task)
    if scalers.to_dict() != payload["scalers"]:
        raise ValueError("checkpoint train-only RUL scaler mismatch")
    model, spec = build_model(config.model)
    if spec.to_dict() != payload["model_spec"]:
        raise ValueError("checkpoint model specification mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = resolve_device(config.device)
    model.to(device).eval()
    return LoadedExperiment(
        config,
        model,
        device,
        scalers,
        HorizonTaskSampler(config.task, scalers),
        train,
        validation,
        test,
        audit,
        payload,
    )


def predict_task(
    experiment: LoadedExperiment,
    task: HorizonTask,
    *,
    mc_samples: int,
    seed: int,
) -> tuple[Any, RULPrediction]:
    batch = collate_tasks([task]).to(experiment.device)
    prediction = predict_batch(
        experiment.model,
        batch,
        experiment.scalers,
        mc_samples=mc_samples,
        interval_level=experiment.config.evaluation.interval_level,
        seed=seed,
    )
    return batch, prediction


def _plot_horizon_metrics(aggregate: pd.DataFrame, path: Path) -> None:
    valid = aggregate[aggregate["status"] == "ok"].sort_values("horizon")
    if valid.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, column, label in (
        (axes[0], "rmse_cycles", "RMSE (cycles)"),
        (axes[1], "mae_cycles", "MAE (cycles)"),
        (axes[2], "mape_percent", "MAPE (%)"),
    ):
        axis.plot(valid["horizon"], valid[column], marker="o", linewidth=2)
        axis.set(xlabel="Observation horizon k", ylabel=label, title=label)
        axis.grid(alpha=0.25)
    figure.suptitle("Horizon-conditioned direct RUL performance")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_streaming_cell(frame: pd.DataFrame, path: Path, cell_id: str) -> None:
    selected = frame[frame["cell_id"] == cell_id].sort_values("horizon")
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        selected["horizon"], selected["true_rul_cycles"],
        color="black", linewidth=2, label="True RUL",
    )
    axis.plot(
        selected["horizon"], selected["predicted_rul_mean_cycles"],
        color="#4C78A8", linewidth=2, marker="o", label="Predicted mean",
    )
    axis.fill_between(
        selected["horizon"],
        selected["lower_cycles"],
        selected["upper_cycles"],
        color="#4C78A8",
        alpha=0.22,
        label=f"{100 * float(selected['interval_level'].iloc[0]):g}% interval",
    )
    axis.set(
        xlabel="Observed cycle k",
        ylabel="Remaining useful life (cycles)",
        title=f"Streaming RUL without parameter updates: {cell_id}",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_checkpoint(
    config: HorizonRULConfig,
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
    selected_horizons = list(horizons or config.evaluation.horizons)
    if not selected_horizons or len(selected_horizons) != len(set(selected_horizons)):
        raise ValueError("evaluation horizons must be non-empty and unique")
    sample_count = int(mc_samples or config.evaluation.mc_samples)
    source = Path(checkpoint).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir
        else source.parent.parent / "evaluation" / source.stem
    )
    (destination / "plots").mkdir(parents=True, exist_ok=True)
    save_config(config, destination / "resolved_config.yaml")
    experiment.audit.to_csv(destination / "data_and_label_audit.csv", index=False)

    rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for horizon in selected_horizons:
        try:
            task = experiment.sampler.evaluation(
                int(horizon),
                experiment.train_cells,
                experiment.test_cells,
                context_size=config.evaluation.context_size,
                seed=config.seed,
            )
        except TaskUnavailable as exc:
            aggregate_rows.append(
                {
                    "horizon": int(horizon),
                    "status": "skipped",
                    "reason": str(exc),
                    "n_test_cells": 0,
                    "rmse_cycles": np.nan,
                    "mae_cycles": np.nan,
                    "mape_percent": np.nan,
                }
            )
            continue
        batch, prediction = predict_task(
            experiment,
            task,
            mc_samples=sample_count,
            seed=config.seed + 10_007 * int(horizon),
        )
        count = len(task.query)
        true = batch.query_rul_cycles[0, :count].cpu().numpy()
        mean = prediction.mean_cycles[0, :count]
        lower = prediction.lower_cycles[0, :count]
        upper = prediction.upper_cycles[0, :count]
        std = prediction.std_cycles[0, :count]
        context_ids = ",".join(point.cell_id for point in task.context)
        for index, point in enumerate(task.query):
            rows.append(
                {
                    "horizon": int(horizon),
                    "cell_id": point.cell_id,
                    "lifetime_cycle": point.lifetime,
                    "true_rul_cycles": float(true[index]),
                    "predicted_rul_mean_cycles": float(mean[index]),
                    "predicted_rul_std_cycles": float(std[index]),
                    "lower_cycles": float(lower[index]),
                    "upper_cycles": float(upper[index]),
                    "absolute_error_cycles": float(abs(mean[index] - true[index])),
                    "interval_level": config.evaluation.interval_level,
                    "context_cell_ids": context_ids,
                    "query_label_used_as_input": False,
                    "test_time_parameter_update": False,
                }
            )
        metrics = regression_metrics(
            true,
            mean,
            mape_epsilon_cycles=config.evaluation.mape_epsilon_cycles,
        )
        intervals = interval_metrics(true, lower, upper)
        aggregate_rows.append(
            {
                "horizon": int(horizon),
                "status": "ok",
                "reason": "",
                "n_test_cells": count,
                **metrics,
                **intervals,
            }
        )
    predictions = pd.DataFrame(rows)
    aggregate = pd.DataFrame(aggregate_rows)
    predictions.to_csv(destination / "per_cell_predictions.csv", index=False)
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    _plot_horizon_metrics(aggregate, destination / "plots/metrics_by_horizon.png")
    if not predictions.empty:
        plot_cell = config.evaluation.plot_cell
        if plot_cell is None or plot_cell not in set(predictions["cell_id"]):
            plot_cell = str(predictions["cell_id"].iloc[0])
        _plot_streaming_cell(
            predictions,
            destination / f"plots/streaming_rul_{plot_cell}.png",
            plot_cell,
        )
    checksum_after = parameter_checksum(experiment.model)
    if checksum_before != checksum_after:
        raise RuntimeError("evaluation changed model parameters")
    write_json(
        destination / "evaluation_manifest.json",
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(source),
            "fold": int(experiment.payload["fold_split"]["fold"]),
            "train_reference_cells": [item.cell_id for item in experiment.train_cells],
            "held_out_test_cells": [item.cell_id for item in experiment.test_cells],
            "horizons": selected_horizons,
            "mc_samples": sample_count,
            "query_label_used_as_input": False,
            "test_time_parameter_update": False,
            "parameter_checksum_before": checksum_before,
            "parameter_checksum_after": checksum_after,
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MATR horizon RUL ANP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/matr_horizon_rul_anp.yaml")
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
        config,
        args.checkpoint,
        resolve_data_root(config, args.data_root),
        output_dir=args.output_dir,
        horizons=args.horizons,
        mc_samples=args.mc_samples,
    )
    print(f"Horizon RUL evaluation: {destination}")


if __name__ == "__main__":
    main()

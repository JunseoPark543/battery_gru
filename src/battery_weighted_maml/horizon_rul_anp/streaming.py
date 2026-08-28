"""Streaming direct-RUL inference with prefix re-encoding and no updates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from battery_weighted_maml.matr_anp.runtime import parameter_checksum, write_json

from .config import load_config, resolve_data_root
from .data import LabeledCell
from .evaluate import LoadedExperiment, load_experiment, predict_task
from .tasks import TaskUnavailable


def predict_at_horizon(
    experiment: LoadedExperiment,
    target_cell: LabeledCell,
    horizon: int,
    *,
    mc_samples: int | None = None,
) -> dict[str, object]:
    """Re-encode cycles 1:k and predict RUL without a gradient update."""
    test_ids = {item.cell_id for item in experiment.test_cells}
    if target_cell.cell_id not in test_ids:
        raise ValueError("streaming target must be a held-out test cell")
    task = experiment.sampler.evaluation(
        int(horizon),
        experiment.train_cells,
        [target_cell],
        context_size=experiment.config.evaluation.context_size,
        seed=experiment.config.seed,
    )
    _, prediction = predict_task(
        experiment,
        task,
        mc_samples=int(mc_samples or experiment.config.evaluation.mc_samples),
        seed=experiment.config.seed + int(horizon) * 10_007,
    )
    point = task.query[0]
    mean = float(prediction.mean_cycles[0, 0])
    return {
        "cell_id": point.cell_id,
        "horizon": int(horizon),
        "lifetime_cycle": point.lifetime,
        "true_rul_cycles": point.rul_cycles,
        "predicted_rul_mean_cycles": mean,
        "predicted_rul_std_cycles": float(prediction.std_cycles[0, 0]),
        "lower_cycles": float(prediction.lower_cycles[0, 0]),
        "upper_cycles": float(prediction.upper_cycles[0, 0]),
        "absolute_error_cycles": abs(mean - point.rul_cycles),
        "num_reference_cells": len(task.context),
        "reference_cell_ids": ",".join(item.cell_id for item in task.context),
        "query_label_used_as_input": False,
        "model_parameter_update": False,
    }


def predict_streaming(
    experiment: LoadedExperiment,
    target_cell: LabeledCell,
    horizons: Sequence[int],
    *,
    mc_samples: int | None = None,
) -> pd.DataFrame:
    """Update the prediction as k grows, while keeping parameters unchanged."""
    values = [int(value) for value in horizons]
    if not values or values != sorted(set(values)):
        raise ValueError("streaming horizons must be non-empty, sorted, and unique")
    checksum = parameter_checksum(experiment.model)
    rows = []
    for horizon in values:
        try:
            rows.append(
                predict_at_horizon(
                    experiment,
                    target_cell,
                    horizon,
                    mc_samples=mc_samples,
                )
            )
        except TaskUnavailable as exc:
            rows.append(
                {
                    "cell_id": target_cell.cell_id,
                    "horizon": horizon,
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
    if checksum != parameter_checksum(experiment.model):
        raise RuntimeError("streaming inference changed model parameters")
    frame = pd.DataFrame(rows)
    if "status" not in frame:
        frame["status"] = "ok"
        frame["reason"] = ""
    else:
        frame["status"] = frame["status"].fillna("ok")
        frame["reason"] = frame["reason"].fillna("")
    return frame


def _plot(frame: pd.DataFrame, destination: Path, cell_id: str) -> None:
    valid = frame[frame["status"] == "ok"].sort_values("horizon")
    if valid.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(valid["horizon"], valid["true_rul_cycles"], color="black", lw=2, label="True RUL")
    axis.plot(
        valid["horizon"], valid["predicted_rul_mean_cycles"],
        color="#4C78A8", marker="o", lw=2, label="Predicted mean",
    )
    axis.fill_between(
        valid["horizon"], valid["lower_cycles"], valid["upper_cycles"],
        color="#4C78A8", alpha=0.22, label="Predictive interval",
    )
    axis.set(
        xlabel="Observed cycle k",
        ylabel="RUL (cycles)",
        title=f"Streaming horizon-conditioned RUL: {cell_id}",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream MATR horizon RUL predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/matr_horizon_rul_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--cell")
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    experiment = load_experiment(
        config,
        args.checkpoint,
        resolve_data_root(config, args.data_root),
    )
    by_id = {item.cell_id: item for item in experiment.test_cells}
    cell_id = args.cell or sorted(by_id)[0]
    if cell_id not in by_id:
        raise ValueError(
            f"--cell must be held out in this fold; available={sorted(by_id)}"
        )
    if args.horizons:
        horizons = args.horizons
    else:
        if args.step <= 0 or args.end < args.start:
            raise ValueError("streaming range requires step>0 and end>=start")
        horizons = list(range(args.start, args.end + 1, args.step))
    frame = predict_streaming(
        experiment,
        by_id[cell_id],
        horizons,
        mc_samples=args.mc_samples,
    )
    source = Path(args.checkpoint).resolve()
    destination = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else source.parent.parent / "streaming" / cell_id
    )
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "streaming_predictions.csv", index=False)
    _plot(frame, destination / "streaming_rul.png", cell_id)
    write_json(
        destination / "streaming_manifest.json",
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(source),
            "cell_id": cell_id,
            "horizons": horizons,
            "query_label_used_as_input": False,
            "model_parameter_update": False,
        },
    )
    print(f"Streaming RUL results: {destination}")


if __name__ == "__main__":
    main()

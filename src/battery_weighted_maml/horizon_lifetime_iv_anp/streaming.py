"""Update lifetime/RUL uncertainty as the observed horizon increases."""

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
    target: LabeledCell,
    horizon: int,
    *,
    mc_samples: int | None = None,
) -> dict[str, object]:
    if target.cell_id not in {item.cell_id for item in experiment.test_cells}:
        raise ValueError("streaming target must be a held-out test cell")
    task = experiment.sampler.evaluation(
        int(horizon), experiment.train_cells, [target],
        context_size=experiment.config.evaluation.context_size,
        seed=experiment.config.seed,
    )
    prediction = predict_task(
        experiment, task,
        mc_samples=int(mc_samples or experiment.config.evaluation.mc_samples),
        seed=experiment.config.seed + int(horizon) * 10_007,
    )
    point = task.query[0]
    predicted_lifetime = float(prediction.lifetime_mean[0, 0])
    return {
        "cell_id": point.cell_id,
        "horizon": int(horizon),
        "true_lifetime_cycles": point.lifetime_cycles,
        "predicted_lifetime_mean_cycles": predicted_lifetime,
        "predicted_lifetime_std_cycles": float(prediction.lifetime_std[0, 0]),
        "lifetime_lower_cycles": float(prediction.lifetime_lower[0, 0]),
        "lifetime_upper_cycles": float(prediction.lifetime_upper[0, 0]),
        "true_rul_cycles": point.lifetime_cycles - horizon,
        "predicted_rul_mean_cycles": float(prediction.rul_mean[0, 0]),
        "rul_lower_cycles": float(prediction.rul_lower[0, 0]),
        "rul_upper_cycles": float(prediction.rul_upper[0, 0]),
        "absolute_error_cycles": abs(predicted_lifetime - point.lifetime_cycles),
        "num_reference_cells": len(task.context),
        "reference_cell_ids": ",".join(item.cell_id for item in task.context),
        "query_label_used_as_input": False,
        "model_parameter_update": False,
    }


def predict_streaming(
    experiment: LoadedExperiment,
    target: LabeledCell,
    horizons: Sequence[int],
    *,
    mc_samples: int | None = None,
) -> pd.DataFrame:
    values = [int(value) for value in horizons]
    if not values or values != sorted(set(values)):
        raise ValueError("streaming horizons must be sorted and unique")
    if not set(values).issubset(experiment.config.task.horizons):
        raise ValueError("streaming horizons must be included in trained task horizons")
    checksum = parameter_checksum(experiment.model)
    rows: list[dict[str, object]] = []
    for horizon in values:
        try:
            rows.append(predict_at_horizon(
                experiment, target, horizon, mc_samples=mc_samples
            ))
        except TaskUnavailable as exc:
            rows.append({
                "cell_id": target.cell_id, "horizon": horizon,
                "status": "skipped", "reason": str(exc),
            })
    if checksum != parameter_checksum(experiment.model):
        raise RuntimeError("streaming inference changed model parameters")
    frame = pd.DataFrame(rows)
    if "status" not in frame:
        frame["status"], frame["reason"] = "ok", ""
    else:
        frame["status"] = frame["status"].fillna("ok")
        frame["reason"] = frame["reason"].fillna("")
    return frame


def _plot(frame: pd.DataFrame, destination: Path, cell_id: str) -> None:
    valid = frame[frame["status"] == "ok"].sort_values("horizon")
    if valid.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(valid["horizon"], valid["true_lifetime_cycles"], color="black", lw=2, label="True")
    axes[0].plot(valid["horizon"], valid["predicted_lifetime_mean_cycles"], marker="o", lw=2, label="Predicted")
    axes[0].fill_between(valid["horizon"], valid["lifetime_lower_cycles"], valid["lifetime_upper_cycles"], alpha=0.22)
    axes[0].set(xlabel="Observed cycle k", ylabel="Lifetime/EOL cycle", title="Lifetime")
    axes[1].plot(valid["horizon"], valid["true_rul_cycles"], color="black", lw=2, label="True")
    axes[1].plot(valid["horizon"], valid["predicted_rul_mean_cycles"], marker="o", lw=2, label="Predicted")
    axes[1].fill_between(valid["horizon"], valid["rul_lower_cycles"], valid["rul_upper_cycles"], alpha=0.22)
    axes[1].set(xlabel="Observed cycle k", ylabel="RUL (cycles)", title="Derived RUL")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Lifetime I-V streaming inference: {cell_id}")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream lifetime I-V ANP predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/matr_horizon_lifetime_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--cell")
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    experiment = load_experiment(
        config, args.checkpoint, resolve_data_root(config, args.data_root)
    )
    by_id = {item.cell_id: item for item in experiment.test_cells}
    cell_id = args.cell or sorted(by_id)[0]
    if cell_id not in by_id:
        raise ValueError(f"--cell must be held out; available={sorted(by_id)}")
    horizons = args.horizons or config.evaluation.horizons
    frame = predict_streaming(
        experiment, by_id[cell_id], horizons, mc_samples=args.mc_samples
    )
    source = Path(args.checkpoint).resolve()
    destination = Path(args.output_dir).resolve() if args.output_dir else source.parent.parent / "streaming" / cell_id
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "streaming_predictions.csv", index=False)
    _plot(frame, destination / "streaming_lifetime_rul.png", cell_id)
    write_json(destination / "streaming_manifest.json", {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(source),
        "cell_id": cell_id,
        "horizons": [int(value) for value in horizons],
        "prediction_target": "lifetime",
        "query_label_used_as_input": False,
        "model_parameter_update": False,
    })
    print(f"Lifetime I-V streaming results: {destination}")


if __name__ == "__main__":
    main()

"""Raw-cycle RUL metrics and diagnostic figures."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd


def regression_metrics(
    actual: Sequence[float], predicted: Sequence[float]
) -> dict[str, float | int | None]:
    truth = np.asarray(actual, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("metrics require non-empty equal-length 1-D arrays")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise FloatingPointError("metrics received non-finite values")
    error = estimate - truth
    absolute = np.abs(error)
    squared = np.square(error)
    denominator = float(np.sum(np.square(truth - truth.mean())))
    return {
        "count": int(truth.size),
        "mae_cycles": float(absolute.mean()),
        "median_absolute_error_cycles": float(np.median(absolute)),
        "rmse_cycles": float(np.sqrt(squared.mean())),
        "mape_percent": float(
            np.mean(absolute / np.maximum(np.abs(truth), 1.0)) * 100.0
        ),
        "mean_bias_cycles": float(error.mean()),
        "within_10_percent_accuracy": float(
            np.mean(absolute <= 0.10 * np.maximum(np.abs(truth), 1.0))
        ),
        "within_15_percent_accuracy": float(
            np.mean(absolute <= 0.15 * np.maximum(np.abs(truth), 1.0))
        ),
        "r2": None
        if truth.size < 2 or denominator <= 0
        else float(1.0 - squared.sum() / denominator),
        "max_absolute_error_cycles": float(absolute.max()),
    }


def save_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_prediction_figure(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = regression_metrics(frame.actual_rul_cycles, frame.predicted_rul_cycles)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    for protocol, group in frame.groupby("held_out_protocol", sort=True):
        axis.scatter(
            group.actual_rul_cycles,
            group.predicted_rul_cycles,
            s=45,
            alpha=0.8,
            label=protocol,
        )
    low = float(min(frame.actual_rul_cycles.min(), frame.predicted_rul_cycles.min()))
    high = float(max(frame.actual_rul_cycles.max(), frame.predicted_rul_cycles.max()))
    margin = max(50.0, 0.05 * (high - low))
    axis.plot([low - margin, high + margin], [low - margin, high + margin], "k--")
    r2 = "N/A" if metrics["r2"] is None else f"{metrics['r2']:.4f}"
    axis.set_title(
        f"{title}\nMAE={metrics['mae_cycles']:.2f} cycles | "
        f"RMSE={metrics['rmse_cycles']:.2f} | R²={r2}"
    )
    axis.set_xlabel("Actual raw RUL at cycle 100 (cycles)")
    axis.set_ylabel("Predicted raw RUL at cycle 100 (cycles)")
    axis.grid(alpha=0.25)
    if frame.held_out_protocol.nunique() > 1:
        axis.legend(title="Unseen protocol", fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_training_figure(history: list[dict[str, Any]], path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    frame = pd.DataFrame(history)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)
    for column, label in (
        ("joint_total", "joint total"),
        ("joint_task", "raw-RUL task"),
        ("meta_query", "BOIL query"),
    ):
        axes[0].plot(frame.iteration, frame[column], label=label, alpha=0.8)
    axes[0].set_yscale("symlog", linthresh=1.0e-4)
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    evaluated = frame.dropna(subset=["source_validation_mae_cycles"])
    if not evaluated.empty:
        axes[1].plot(
            evaluated.iteration,
            evaluated.source_validation_mae_cycles,
            marker="o",
            label="held-out source-cell validation MAE",
        )
        axes[1].legend()
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel("MAE (cycles)")
    axes[1].grid(alpha=0.25)
    figure.suptitle("HUST training diagnostics (target protocol entirely unused)")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _protocol_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(value))
    return (int(match.group(1)) if match else 10**9, str(value))


def save_key_results_dashboard(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    path: str | Path,
    title: str = "HUST Direct Raw-RUL: Key Results",
) -> None:
    """Create one self-contained figure for rapid model assessment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required_prediction_columns = {
        "held_out_protocol",
        "actual_rul_cycles",
        "predicted_rul_cycles",
    }
    required_metric_columns = {
        "held_out_protocol",
        "count",
        "mae_cycles",
        "source_mean_baseline_mae_cycles",
        "within_15_percent_accuracy",
    }
    if not required_prediction_columns.issubset(predictions):
        raise ValueError("prediction frame is missing dashboard columns")
    if not required_metric_columns.issubset(fold_metrics):
        raise ValueError("fold metric frame is missing dashboard columns")
    aggregate = regression_metrics(
        predictions.actual_rul_cycles, predictions.predicted_rul_cycles
    )
    protocol_order = sorted(
        predictions.held_out_protocol.unique().tolist(), key=_protocol_key
    )
    metric_group = (
        fold_metrics.groupby("held_out_protocol", as_index=True)
        .agg(
            model_mae=("mae_cycles", "mean"),
            baseline_mae=("source_mean_baseline_mae_cycles", "mean"),
            accuracy15=("within_15_percent_accuracy", "mean"),
            count=("count", "sum"),
        )
        .reindex(protocol_order)
    )
    bias = (
        predictions.assign(
            signed_error=(
                predictions.predicted_rul_cycles - predictions.actual_rul_cycles
            )
        )
        .groupby("held_out_protocol")
        .signed_error.mean()
        .reindex(protocol_order)
    )
    weighted_baseline_mae = float(
        np.average(metric_group.baseline_mae, weights=metric_group["count"])
    )
    improved_protocols = int(
        np.sum(metric_group.model_mae < metric_group.baseline_mae)
    )
    worst_protocol = str(metric_group.model_mae.idxmax())
    worst_mae = float(metric_group.model_mae.max())
    r2_text = "N/A" if aggregate["r2"] is None else f"{aggregate['r2']:.3f}"

    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    figure.subplots_adjust(top=0.80, hspace=0.32, wspace=0.25)
    figure.suptitle(title, fontsize=20, fontweight="bold", y=0.975)
    headline = (
        f"N={aggregate['count']}  |  MAE={aggregate['mae_cycles']:.1f} cycles  |  "
        f"RMSE={aggregate['rmse_cycles']:.1f}  |  MAPE={aggregate['mape_percent']:.1f}%  |  "
        f"R²={r2_text}  |  Within 15%={100 * aggregate['within_15_percent_accuracy']:.1f}%"
    )
    comparison = (
        f"Fold-weighted source-mean baseline MAE={weighted_baseline_mae:.1f} cycles  |  "
        f"Model beats baseline in {improved_protocols}/{len(protocol_order)} protocols  |  "
        f"Worst={worst_protocol} ({worst_mae:.1f} cycles)  |  Target adaptation: NONE"
    )
    figure.text(0.5, 0.925, headline, ha="center", va="center", fontsize=12)
    figure.text(
        0.5,
        0.888,
        comparison,
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f2f2f2", "alpha": 0.9},
    )

    scatter_axis = axes[0, 0]
    colors = plt.get_cmap("tab10")
    for index, protocol in enumerate(protocol_order):
        group = predictions[predictions.held_out_protocol == protocol]
        scatter_axis.scatter(
            group.actual_rul_cycles,
            group.predicted_rul_cycles,
            s=45,
            alpha=0.8,
            color=colors(index % 10),
            label=protocol,
        )
    low = float(
        min(predictions.actual_rul_cycles.min(), predictions.predicted_rul_cycles.min())
    )
    high = float(
        max(predictions.actual_rul_cycles.max(), predictions.predicted_rul_cycles.max())
    )
    margin = max(50.0, 0.05 * (high - low))
    scatter_axis.plot(
        [low - margin, high + margin],
        [low - margin, high + margin],
        "k--",
        linewidth=1,
        label="ideal",
    )
    scatter_axis.set_title("A. Actual vs predicted raw RUL")
    scatter_axis.set_xlabel("Actual RUL (cycles)")
    scatter_axis.set_ylabel("Predicted RUL (cycles)")
    scatter_axis.grid(alpha=0.25)
    scatter_axis.legend(fontsize=7, ncol=2)

    x = np.arange(len(protocol_order))
    width = 0.38
    mae_axis = axes[0, 1]
    mae_axis.bar(
        x - width / 2,
        metric_group.model_mae,
        width,
        label="Model",
        color="#2878b5",
    )
    mae_axis.bar(
        x + width / 2,
        metric_group.baseline_mae,
        width,
        label="Source-mean baseline",
        color="#f28e2b",
    )
    mae_axis.set_title("B. Protocol MAE: model vs baseline")
    mae_axis.set_ylabel("MAE (cycles, lower is better)")
    mae_axis.set_xticks(x, protocol_order, rotation=40, ha="right")
    mae_axis.grid(axis="y", alpha=0.25)
    mae_axis.legend()

    bias_axis = axes[1, 0]
    bias_colors = ["#d62728" if value > 0 else "#2ca02c" for value in bias]
    bias_axis.bar(x, bias, color=bias_colors)
    bias_axis.axhline(0.0, color="black", linewidth=1)
    bias_axis.set_title("C. Mean prediction bias")
    bias_axis.set_ylabel("Predicted - actual RUL (cycles)")
    bias_axis.set_xticks(x, protocol_order, rotation=40, ha="right")
    bias_axis.grid(axis="y", alpha=0.25)
    bias_axis.text(
        0.01,
        0.98,
        "Red: overprediction   Green: underprediction",
        transform=bias_axis.transAxes,
        va="top",
        fontsize=9,
    )

    accuracy_axis = axes[1, 1]
    accuracy_percent = 100.0 * metric_group.accuracy15
    bars = accuracy_axis.bar(x, accuracy_percent, color="#59a14f")
    accuracy_axis.axhline(80.0, color="#777777", linestyle="--", linewidth=1)
    accuracy_axis.set_ylim(0, 105)
    accuracy_axis.set_title("D. Predictions within ±15%")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_xticks(x, protocol_order, rotation=40, ha="right")
    accuracy_axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, accuracy_percent):
        accuracy_axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(102.0, value + 2.0),
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)

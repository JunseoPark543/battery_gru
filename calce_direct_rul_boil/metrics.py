"""Regression metrics and result figures for direct RUL experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float | None]:
    truth = np.asarray(actual, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("metrics require equally sized, non-empty 1-D arrays")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise FloatingPointError("metrics received non-finite values")
    error = estimate - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error.square())))
    mape = float(np.mean(np.abs(error) / np.maximum(np.abs(truth), 1.0)) * 100.0)
    denominator = float(np.sum((truth - truth.mean()).square()))
    r2 = None if truth.size < 2 or denominator <= 0 else float(1.0 - error.square().sum() / denominator)
    return {
        "count": int(truth.size),
        "mae_cycles": mae,
        "rmse_cycles": rmse,
        "mape_percent": mape,
        "r2": r2,
        "max_absolute_error_cycles": float(np.max(np.abs(error))),
    }


def save_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_prediction_figure(frame: pd.DataFrame, path: str | Path, title_prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = regression_metrics(frame["actual_rul_cycles"], frame["predicted_rul_cycles"])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    for domain, group in frame.groupby("held_out_domain", sort=True):
        axis.scatter(
            group["actual_rul_cycles"],
            group["predicted_rul_cycles"],
            s=55,
            alpha=0.85,
            label=domain,
        )
    low = float(min(frame["actual_rul_cycles"].min(), frame["predicted_rul_cycles"].min()))
    high = float(max(frame["actual_rul_cycles"].max(), frame["predicted_rul_cycles"].max()))
    margin = max(10.0, 0.05 * (high - low))
    axis.plot([low - margin, high + margin], [low - margin, high + margin], "k--", linewidth=1)
    r2_text = "N/A" if metrics["r2"] is None else f"{metrics['r2']:.4f}"
    axis.set_title(
        f"{title_prefix}\nMAE={metrics['mae_cycles']:.2f} cycles | "
        f"RMSE={metrics['rmse_cycles']:.2f} | R²={r2_text}"
    )
    axis.set_xlabel("Actual RUL at cycle 100 (cycles)")
    axis.set_ylabel("Predicted RUL at cycle 100 (cycles)")
    axis.grid(alpha=0.25)
    axis.legend(title="Unseen target domain")
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
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(frame["iteration"], frame["joint_total"], label="joint total", alpha=0.8)
    axes[0].plot(frame["iteration"], frame["joint_task"], label="joint task", alpha=0.8)
    axes[0].plot(frame["iteration"], frame["meta_query"], label="BOIL query", alpha=0.8)
    axes[0].set_yscale("symlog", linthresh=1.0e-4)
    axes[0].set_ylabel("Normalized loss")
    axes[0].legend()
    evaluation = frame.dropna(subset=["source_cv_meta_mae_cycles"])
    if not evaluation.empty:
        axes[1].plot(
            evaluation["iteration"],
            evaluation["source_cv_meta_mae_cycles"],
            marker="o",
            label="source-only meta-CV MAE",
        )
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel("MAE (cycles)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Training diagnostics (target-domain labels unused)")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)

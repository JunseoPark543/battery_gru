"""Compact visual summary for one HUST target experiment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_weighted_maml.evaluation.evaluator import EvaluationResult


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def save_key_results_figure(
    target_name: str,
    history_length: int,
    fast_results: Mapping[int, EvaluationResult],
    full_result: EvaluationResult,
    output_path: str | Path,
) -> None:
    """Plot trajectories and the main curve/RUL metrics in one PNG."""
    ordered = sorted(fast_results)
    methods = [f"fast {step}" for step in ordered] + ["full"]
    results = [fast_results[step] for step in ordered] + [full_result]
    comparison = pd.DataFrame(
        [
            {
                "method": method,
                "mae_percent": 100.0 * _finite(result.metrics.get("mae")),
                "rmse_percent": 100.0 * _finite(result.metrics.get("rmse")),
                "absolute_rul_error": _finite(
                    result.metrics.get("absolute_rul_error")
                ),
            }
            for method, result in zip(methods, results)
        ]
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.4),
        gridspec_kw={"width_ratios": [2.1, 1.0, 1.0]},
    )
    curve_axis, error_axis, rul_axis = axes
    full_frame = full_result.predictions
    observed = full_frame[full_frame["observed_soh"].notna()]
    curve_axis.plot(
        observed["cycle"], observed["observed_soh"], color="black", label="observed"
    )
    colors = plt.cm.viridis(np.linspace(0.12, 0.78, max(1, len(ordered))))
    for color, step in zip(colors, ordered):
        frame = fast_results[step].predictions
        future = frame[frame["split"] == "future"]
        curve_axis.plot(
            future["cycle"],
            future["predicted_soh"],
            color=color,
            alpha=0.72,
            linewidth=1.0,
            label=f"fast {step}",
        )
    full_future = full_frame[full_frame["split"] == "future"]
    curve_axis.plot(
        full_future["cycle"],
        full_future["predicted_soh"],
        color="tab:red",
        linewidth=1.8,
        label="full",
    )
    threshold = float(full_frame["eol_threshold"].iloc[0])
    curve_axis.axhline(threshold, color="tab:orange", linestyle="--", label="EOL")
    curve_axis.axvline(history_length, color="gray", linestyle=":", label="L=100")
    curve_axis.set(
        xlabel="Cycle",
        ylabel="SOH",
        title=f"{target_name}: observed vs recursive forecast",
    )
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(fontsize=7, ncol=2)

    x = np.arange(len(comparison))
    width = 0.38
    error_axis.bar(
        x - width / 2, comparison["mae_percent"], width, label="MAE (%)"
    )
    error_axis.bar(
        x + width / 2, comparison["rmse_percent"], width, label="RMSE (%)"
    )
    error_axis.set_xticks(x, comparison["method"], rotation=50, ha="right")
    error_axis.set(title="Curve error", ylabel="SOH error (%)")
    error_axis.grid(axis="y", alpha=0.25)
    error_axis.legend(fontsize=8)

    rul_axis.bar(x, comparison["absolute_rul_error"], color="tab:purple")
    rul_axis.set_xticks(x, comparison["method"], rotation=50, ha="right")
    rul_axis.set(title="RUL error", ylabel="Absolute error (cycles)")
    rul_axis.grid(axis="y", alpha=0.25)

    finite_mae = comparison["mae_percent"].replace([np.inf, -np.inf], np.nan)
    best_text = "N/A"
    if finite_mae.notna().any():
        best_index = finite_mae.idxmin()
        best_text = (
            f"{comparison.loc[best_index, 'method']} "
            f"(MAE {comparison.loc[best_index, 'mae_percent']:.3f}%)"
        )
    figure.suptitle(
        f"HUST weighted MAML | first {history_length} cycles | best: {best_text}",
        fontsize=13,
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def adaptation_comparison_frame(
    fast_results: Mapping[int, EvaluationResult], full_result: EvaluationResult
) -> pd.DataFrame:
    rows = [
        {"adaptation": f"fast_{step}", "fast_step": step, **result.metrics}
        for step, result in sorted(fast_results.items())
    ]
    rows.append({"adaptation": "full", "fast_step": np.nan, **full_result.metrics})
    return pd.DataFrame(rows)


"""Replot a completed weighted-MAML run without training or adaptation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAST_PATTERN = re.compile(r"target_fast_(\d+)_prediction\.csv$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_text(metrics: dict[str, Any]) -> str:
    def value(key: str, *, scale: float = 1.0, digits: int = 3) -> str:
        raw = metrics.get(key)
        if raw is None:
            return "N/A"
        number = float(raw) * scale
        return f"{number:.{digits}f}" if np.isfinite(number) else "N/A"

    return "\n".join(
        [
            f"MAE: {value('mae', scale=100.0)}%",
            f"RMSE: {value('rmse', scale=100.0)}%",
            f"R²: {value('r2')}",
            f"True / predicted RUL: {value('true_rul', digits=0)} / "
            f"{value('predicted_rul', digits=0)} cycles",
            f"Absolute RUL error: {value('absolute_rul_error', digits=0)} cycles",
        ]
    )


def _mode_label(mode: str) -> str:
    if mode == "full":
        return "Full adaptation"
    return f"Fast {mode.removeprefix('fast_')} steps"


def _load_results(run_dir: Path) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    predictions = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

    full_prediction = predictions / "target_full_prediction.csv"
    full_metrics = metrics_dir / "full_metrics.json"
    if full_prediction.exists() and full_metrics.exists():
        results["full"] = (pd.read_csv(full_prediction), _load_json(full_metrics))

    for prediction_path in sorted(predictions.glob("target_fast_*_prediction.csv")):
        match = FAST_PATTERN.match(prediction_path.name)
        if match is None:
            continue
        mode = f"fast_{int(match.group(1))}"
        metric_path = metrics_dir / f"{mode}_metrics.json"
        if metric_path.exists():
            results[mode] = (pd.read_csv(prediction_path), _load_json(metric_path))

    if not results:
        raise FileNotFoundError(f"no prediction/metric pairs found under {run_dir}")
    return results


def _run_description(
    run_dir: Path,
    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
) -> tuple[str, int, str]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    sample_frame = next(iter(results.values()))[0]
    target = str(manifest.get("target", "target")).removesuffix(".pkl")
    history_length = int(
        manifest.get("history_length", (sample_frame["split"] == "support").sum())
    )
    features = manifest.get("resolved_config", {}).get("model", {}).get("features", [])
    feature_text = "+".join(str(feature).upper() for feature in features) or "unknown input"
    return target, history_length, feature_text


def _plot_detailed(
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    destination: Path,
    title: str,
    history_length: int,
) -> None:
    observed = frame[frame["observed_soh"].notna()]
    future = frame[frame["split"] == "future"]
    threshold = float(frame["eol_threshold"].iloc[0])

    figure, axis = plt.subplots(figsize=(11, 6.4))
    axis.plot(
        observed["cycle"], observed["observed_soh"],
        color="tab:blue", linewidth=1.6, label="Observed SOH",
    )
    axis.plot(
        future["cycle"], future["predicted_soh"],
        color="tab:orange", linewidth=1.8, label="Recursive prediction",
    )
    axis.axhline(
        threshold, color="tab:red", linestyle="--", linewidth=1.3,
        label="EOL threshold",
    )
    axis.axvline(
        history_length, color="0.35", linestyle=":", linewidth=1.5,
        label=f"History L={history_length}",
    )

    true_eol = metrics.get("true_eol_cycle")
    predicted_eol = metrics.get("predicted_eol_cycle_interpolated")
    if true_eol is not None and np.isfinite(float(true_eol)):
        axis.axvline(
            float(true_eol), color="tab:green", linestyle="-.", linewidth=1.3,
            label=f"True EOL={float(true_eol):.0f}",
        )
    if predicted_eol is not None and np.isfinite(float(predicted_eol)):
        axis.axvline(
            float(predicted_eol), color="tab:purple", linestyle="-.", linewidth=1.3,
            label=f"Predicted EOL={float(predicted_eol):.1f}",
        )

    axis.text(
        0.985, 0.975, _metric_text(metrics), transform=axis.transAxes,
        ha="right", va="top", fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.55", "facecolor": "white",
            "edgecolor": "0.65", "alpha": 0.92,
        },
    )
    axis.set(xlabel="Cycle", ylabel="SOH", title=title)
    axis.grid(alpha=0.22)
    axis.legend(loc="lower left", fontsize=9)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_dashboard(
    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    destination: Path,
    target: str,
    history_length: int,
    feature_text: str,
) -> None:
    modes = sorted(
        results,
        key=lambda mode: (
            mode == "full",
            int(mode.removeprefix("fast_")) if mode != "full" else 10**9,
        ),
    )
    labels = [_mode_label(mode) for mode in modes]
    metric_rows = [results[mode][1] for mode in modes]
    reference = results["full"][0] if "full" in results else results[modes[0]][0]
    observed = reference[reference["observed_soh"].notna()]
    threshold = float(reference["eol_threshold"].iloc[0])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    trajectory_axis, error_axis, r2_axis, rul_axis = axes.flat

    trajectory_axis.plot(
        observed["cycle"], observed["observed_soh"],
        color="black", linewidth=1.8, label="Observed SOH", zorder=10,
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(modes)))
    for color, mode, label in zip(colors, modes, labels):
        frame = results[mode][0]
        future = frame[frame["split"] == "future"]
        trajectory_axis.plot(
            future["cycle"], future["predicted_soh"], color=color,
            linewidth=2.1 if mode == "full" else 1.15,
            alpha=1.0 if mode == "full" else 0.8, label=label,
        )
    trajectory_axis.axhline(
        threshold, color="tab:red", linestyle="--", linewidth=1.2,
        label="EOL threshold",
    )
    trajectory_axis.axvline(
        history_length, color="0.4", linestyle=":", linewidth=1.3,
        label=f"L={history_length}",
    )
    trajectory_axis.set(
        xlabel="Cycle", ylabel="SOH", title="Recursive trajectory comparison",
    )
    trajectory_axis.grid(alpha=0.2)
    trajectory_axis.legend(fontsize=7.5, ncol=2)

    positions = np.arange(len(modes))
    width = 0.38
    mae = [100.0 * float(metric["mae"]) for metric in metric_rows]
    rmse = [100.0 * float(metric["rmse"]) for metric in metric_rows]
    error_axis.bar(
        positions - width / 2, mae, width, label="MAE", color="tab:blue",
    )
    error_axis.bar(
        positions + width / 2, rmse, width, label="RMSE", color="tab:orange",
    )
    error_axis.set(ylabel="Error (%)", title="SOH prediction error")
    error_axis.legend()

    r2_values = [float(metric["r2"]) for metric in metric_rows]
    r2_colors = ["tab:green" if value >= 0 else "tab:red" for value in r2_values]
    r2_axis.bar(positions, r2_values, color=r2_colors)
    r2_axis.axhline(0.0, color="0.3", linewidth=0.9)
    r2_axis.set(ylabel="R²", title="Coefficient of determination")

    rul_values = [
        float(metric["absolute_rul_error"])
        if metric.get("absolute_rul_error") is not None else np.nan
        for metric in metric_rows
    ]
    rul_axis.bar(positions, rul_values, color="tab:purple")
    rul_axis.set(ylabel="Cycles", title="Absolute RUL error")

    for axis in (error_axis, r2_axis, rul_axis):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)

    figure.suptitle(
        f"{target} | L={history_length} | {feature_text}-only weighted MAML",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def replot(run_dir: Path) -> list[Path]:
    run_dir = run_dir.resolve()
    results = _load_results(run_dir)
    target, history_length, feature_text = _run_description(run_dir, results)
    figures = run_dir / "figures"
    written: list[Path] = []

    for mode, (frame, metrics) in results.items():
        destination = figures / f"target_soh_{mode}_metrics.png"
        _plot_detailed(
            frame, metrics, destination,
            f"{target} | L={history_length} | {feature_text}-only weighted MAML | "
            f"{_mode_label(mode)}",
            history_length,
        )
        written.append(destination)

    dashboard = figures / "adaptation_metrics_dashboard.png"
    _plot_dashboard(results, dashboard, target, history_length, feature_text)
    written.append(dashboard)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot stored weighted-MAML predictions with performance metrics."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output_path in replot(arguments.run_dir):
        print(output_path)

"""Headless local plots for runs and aggregate comparisons."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _formatted_metric(
    metrics: Mapping[str, Any],
    key: str,
    *,
    scale: float = 1.0,
    decimals: int = 3,
) -> str:
    value = metrics.get(key)
    if value is None:
        return "N/A"
    try:
        numeric = float(value) * scale
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.{decimals}f}" if np.isfinite(numeric) else "N/A"


def performance_title(title: str, metrics: Mapping[str, Any] | None) -> str:
    """Append consistently scaled evaluation metrics to a prediction title."""
    if metrics is None:
        return title
    mae = _formatted_metric(metrics, "mae", scale=100.0)
    rmse = _formatted_metric(metrics, "rmse", scale=100.0)
    r2 = _formatted_metric(metrics, "r2")
    rul_error = _formatted_metric(metrics, "absolute_rul_error", decimals=0)
    return (
        f"{title}\n"
        f"MAE={mae}% | RMSE={rmse}% | R²={r2} | "
        f"absolute RUL error={rul_error} cycles"
    )


def plot_target_prediction(
    frame: pd.DataFrame,
    path: str | Path,
    title: str,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    observed = frame[frame["observed_soh"].notna()]
    future = frame[frame["split"] == "future"]
    axis.plot(observed["cycle"], observed["observed_soh"], label="observed", linewidth=1.5)
    axis.plot(future["cycle"], future["predicted_soh"], label="recursive prediction", linewidth=1.3)
    axis.axhline(float(frame["eol_threshold"].iloc[0]), color="tab:red", linestyle="--", label="EOL")
    axis.axvline(int((frame["split"] == "support").sum()), color="gray", linestyle=":", label="L")
    axis.set(xlabel="Cycle", ylabel="SOH", title=performance_title(title, metrics))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transfer_zero_vs_full(
    run_dir: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Plot transfer 0-step and full fine-tuning forecasts side by side."""
    root = Path(run_dir)
    modes = ["transfer_0", "transfer_full"]
    labels = ["0 fine-tuning (zero-shot transfer)", "Full fine-tuning"]
    metrics_path = root / "metrics/transfer_metrics_summary.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"transfer metrics not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path).set_index("mode")
    frames: dict[str, pd.DataFrame] = {}
    for mode in modes:
        prediction_path = root / f"predictions/target_{mode}_prediction.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"transfer prediction not found: {prediction_path}")
        if mode not in metrics.index:
            raise ValueError(f"{mode} is missing from {metrics_path}")
        frames[mode] = pd.read_csv(prediction_path)

    manifest_path = root / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = str(manifest.get("target", "target")).removesuffix(".pkl")
    history_length = int(manifest.get("history_length", 0))
    full_step = manifest.get("full_fine_tuning_best_step")

    all_values = np.concatenate(
        [
            frame[["observed_soh", "predicted_soh"]].to_numpy().reshape(-1)
            for frame in frames.values()
        ]
    )
    finite = all_values[np.isfinite(all_values)]
    y_min = float(finite.min())
    y_max = float(finite.max())
    margin = max(0.02, 0.04 * (y_max - y_min))

    figure, axes = plt.subplots(1, 2, figsize=(17, 6.5), sharex=True, sharey=True)
    legend_handles = None
    for axis, mode, label in zip(axes, modes, labels):
        frame = frames[mode]
        row = metrics.loc[mode]
        available = frame["observed_soh"].notna()
        support = frame["split"] == "support"
        future = frame["split"] == "future"
        threshold = float(frame["eol_threshold"].dropna().iloc[0])
        observed_line = axis.plot(
            frame.loc[available, "cycle"],
            frame.loc[available, "observed_soh"],
            color="#303030",
            linewidth=1.35,
            label="Actual SOH",
            zorder=2,
        )[0]
        support_line = axis.plot(
            frame.loc[support, "cycle"],
            frame.loc[support, "observed_soh"],
            color="#2ca02c",
            linewidth=2.4,
            label=f"Observed input (cycles 1-{history_length})",
            zorder=3,
        )[0]
        prediction_line = axis.plot(
            frame.loc[future, "cycle"],
            frame.loc[future, "predicted_soh"],
            color="#d62728",
            linewidth=1.8,
            label="Recursive prediction",
            zorder=4,
        )[0]
        threshold_line = axis.axhline(
            threshold,
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label=f"EOL threshold ({threshold:g})",
            zorder=1,
        )
        axis.axvline(history_length, color="#2ca02c", linestyle=":", linewidth=1.0)
        if legend_handles is None:
            legend_handles = [
                observed_line,
                support_line,
                prediction_line,
                threshold_line,
            ]

        step_text = "0" if mode == "transfer_0" else str(full_step or "full")
        predicted_eol = row.get("predicted_eol_cycle_interpolated", np.nan)
        predicted_eol_text = (
            f"{float(predicted_eol):.1f}"
            if pd.notna(predicted_eol) and np.isfinite(float(predicted_eol))
            else "not reached"
        )
        rul_error = row.get("absolute_rul_error", np.nan)
        rul_error_text = (
            f"{float(rul_error):.0f}"
            if pd.notna(rul_error) and np.isfinite(float(rul_error))
            else "N/A"
        )
        axis.set_title(
            f"{label} | selected step={step_text}\n"
            f"MAE={100.0 * float(row['mae']):.2f}% | "
            f"RMSE={100.0 * float(row['rmse']):.2f}% | "
            f"R²={float(row['r2']):.3f}\n"
            f"predicted EOL={predicted_eol_text} | "
            f"absolute RUL error={rul_error_text} cycles",
            fontsize=11,
        )
        axis.set_xlabel("Cycle")
        axis.set_ylim(y_min - margin, y_max + margin)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("SOH")
    figure.suptitle(
        f"{target}: source-pretrained GRU transfer comparison (L={history_length})\n"
        "No meta-learning; SOH-only; same source-trained checkpoint",
        fontsize=15,
        y=0.99,
    )
    if legend_handles is not None:
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.865),
            frameon=False,
        )
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.1, top=0.72, wspace=0.06)
    output = (
        Path(destination)
        if destination is not None
        else root / "figures/target_soh_transfer_0_vs_full.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_training_outputs(run_dir: str | Path) -> None:
    root = Path(run_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    iterations = pd.read_csv(root / "training/iteration_history.csv")
    losses = pd.read_csv(root / "training/source_loss_history.csv")
    alphas = pd.read_csv(root / "weights/alpha_history.csv")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(iterations["iteration"], iterations["weighted_meta_loss"], label="meta loss")
    axis.plot(iterations["iteration"], iterations["ema_source_meta_loss"], label="EMA")
    axis.set(xlabel="Meta iteration", ylabel="Loss", title="Weighted source meta loss")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(figures / "training_loss.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    for source, group in alphas.groupby("source"):
        axis.plot(group["iteration"], group["alpha"], label=source)
    axis.set(xlabel="Meta iteration", ylabel="Alpha", title="Target-aware source weights")
    axis.grid(alpha=0.25); axis.legend(fontsize=7); fig.tight_layout()
    fig.savefig(figures / "alpha_trajectory.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    for source, group in losses.groupby("source"):
        axis.plot(group["iteration"], group["query_loss"], label=source)
    axis.set(xlabel="Meta iteration", ylabel="Query MSE", title="Adapted source query losses")
    axis.grid(alpha=0.25); axis.legend(fontsize=7); fig.tight_layout()
    fig.savefig(figures / "source_query_losses.png", dpi=150); plt.close(fig)
    path_columns = sorted(
        (column for column in losses.columns if column.startswith("query_loss_step_")),
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )
    if len(path_columns) > 1:
        fig, axis = plt.subplots(figsize=(9, 5))
        for column in path_columns:
            weighted = (
                losses.assign(weighted=losses[column] * losses["alpha"])
                .groupby("iteration", as_index=False)["weighted"]
                .sum()
            )
            step = column.rsplit("_", 1)[-1]
            axis.plot(weighted["iteration"], weighted["weighted"], label=f"step {step}")
        axis.set(
            xlabel="Meta iteration",
            ylabel="Alpha-weighted query MSE",
            title="Query loss along the source adaptation path",
        )
        axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
        fig.savefig(figures / "path_query_losses.png", dpi=150); plt.close(fig)
    component_columns = [
        "weighted_path_mean_query_loss",
        "weighted_path_worst_query_loss",
        "weighted_path_dispersion",
    ]
    if len(path_columns) > 1 and all(
        column in iterations.columns for column in component_columns
    ):
        fig, axis = plt.subplots(figsize=(9, 5))
        for column in component_columns:
            axis.plot(
                iterations["iteration"],
                iterations[column],
                label=column.removeprefix("weighted_path_").replace("_", " "),
            )
        axis.set(
            xlabel="Meta iteration",
            ylabel="Loss",
            title="Robust adaptation path components",
        )
        axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
        fig.savefig(figures / "path_loss_components.png", dpi=150); plt.close(fig)
    if not alphas.empty:
        pivot = alphas.pivot(index="source", columns="iteration", values="alpha")
        fig, axis = plt.subplots(figsize=(10, max(3, 0.45 * len(pivot))))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        axis.set_yticks(np.arange(len(pivot))); axis.set_yticklabels(pivot.index, fontsize=7)
        axis.set_xticks(np.arange(len(pivot.columns))); axis.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
        axis.set(xlabel="Meta iteration", title="Alpha heatmap")
        fig.colorbar(image, ax=axis, label="alpha"); fig.tight_layout()
        fig.savefig(root / "weights/alpha_heatmap.png", dpi=150); plt.close(fig)

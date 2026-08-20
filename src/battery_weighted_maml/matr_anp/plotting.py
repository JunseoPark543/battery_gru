"""Plots for MATR probabilistic trajectory evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectory_betas(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    cell_id: str,
    alpha: float,
    current_cycle: int,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6.5))
    first = predictions[predictions["beta"] == predictions["beta"].min()]
    context = first[first["split"] == "context"]
    target = first[first["split"] == "target"]
    axis.plot(context["cycle"], context["actual_soh"], "o-", ms=3, color="black", label="past SOH context")
    axis.plot(target["cycle"], target["actual_soh"], color="0.35", lw=1.5, label="actual future SOH")
    colors = plt.get_cmap("viridis")
    beta_values = sorted(predictions["beta"].unique())
    for index, beta in enumerate(beta_values):
        selected = predictions[(predictions["beta"] == beta) & (predictions["split"] == "target")]
        color = colors(index / max(1, len(beta_values) - 1))
        axis.plot(selected["cycle"], selected["predicted_mean"], color=color, label=f"beta={beta:g}")
        axis.fill_between(
            selected["cycle"], selected["lower_95"], selected["upper_95"],
            color=color, alpha=0.08,
        )
    axis.axvline(current_cycle, color="tab:red", linestyle="--", label="current cycle")
    axis.set(
        xlabel="Cycle number",
        ylabel="SOH",
        title=f"{cell_id}: probabilistic trajectory at alpha={alpha:g}",
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_metric_vs_beta(
    per_cell: pd.DataFrame,
    metric: str,
    ylabel: str,
    destination: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    grouped = per_cell.groupby(["alpha", "beta"])[metric].agg(["mean", "std"]).reset_index()
    for alpha, rows in grouped.groupby("alpha"):
        rows = rows.sort_values("beta")
        axis.errorbar(
            rows["beta"], rows["mean"], yerr=rows["std"].fillna(0.0),
            marker="o", capsize=3, label=f"alpha={alpha:g}",
        )
    axis.set(xlabel="Partial observation beta", ylabel=ylabel, title=f"{ylabel} versus partial I-V observation")
    axis.set_xticks(sorted(per_cell["beta"].unique()))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_model_comparison(
    per_cell: pd.DataFrame,
    metric: str,
    ylabel: str,
    destination: str | Path,
) -> None:
    """Compare model/alpha cell-averaged curves on one beta axis."""
    figure, axis = plt.subplots(figsize=(10, 6))
    grouped = per_cell.groupby(["model", "alpha", "beta"])[metric].agg(
        ["mean", "std"]
    ).reset_index()
    for (model, alpha), rows in grouped.groupby(["model", "alpha"]):
        rows = rows.sort_values("beta")
        axis.errorbar(
            rows["beta"],
            rows["mean"],
            yerr=rows["std"].fillna(0.0),
            marker="o",
            capsize=2,
            label=f"{model} / alpha={alpha:g}",
        )
    axis.set(
        xlabel="Partial observation beta",
        ylabel=ylabel,
        title=f"Model comparison: {ylabel}",
    )
    axis.set_xticks(sorted(per_cell["beta"].unique()))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_context_streaming_summary(
    aggregate: pd.DataFrame,
    destination: str | Path,
) -> None:
    """Plot accuracy and calibration while the SOH context moves forward."""
    valid = aggregate.sort_values(["cycle_step", "observed_through_cycle"])
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    panels = (
        ("future_rmse_mean", "Future SOH RMSE", None),
        ("coverage_95_mean", "95% interval coverage", 0.95),
        ("interval_width_95_mean", "Mean 95% interval width", None),
        ("num_cells", "Evaluated test cells", None),
    )
    for (metric, ylabel, reference), axis in zip(panels, axes.flat):
        for step, rows in valid.groupby("cycle_step", sort=False):
            rows = rows.sort_values("observed_through_cycle")
            first_cycle = int(rows["observed_through_cycle"].iloc[0])
            if int(step) == 1:
                axis.plot(
                    rows["observed_through_cycle"], rows[metric],
                    linewidth=1.2, alpha=0.8,
                    label=f"step1: {first_cycle}, {first_cycle + 1}, ...",
                )
            else:
                axis.plot(
                    rows["observed_through_cycle"], rows[metric],
                    marker="o", markersize=3, linewidth=1.0,
                    label=(
                        f"step{int(step)}: {first_cycle}, "
                        f"{first_cycle + int(step)}, ..."
                    ),
                )
        if reference is not None:
            axis.axhline(reference, color="0.35", linestyle="--", linewidth=1)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("SOH observed through cycle")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Held-out streaming context test (fixed checkpoint)")
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_context_trajectory_snapshots(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    cell_id: str,
    schedule: str,
) -> None:
    """Overlay the first three streaming forecasts for one test cell."""
    selected = predictions[
        (predictions["cell_id"] == cell_id)
        & (predictions["schedule"] == schedule)
    ]
    if selected.empty:
        return
    actual = (
        selected[["cycle", "actual_soh"]]
        .dropna()
        .drop_duplicates("cycle")
        .sort_values("cycle")
    )
    cutoffs = sorted(selected["observed_through_cycle"].unique())
    colors = plt.get_cmap("viridis")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(actual["cycle"], actual["actual_soh"], color="black", lw=1.7, label="actual SOH")
    for index, cutoff in enumerate(cutoffs):
        rows = selected[
            (selected["observed_through_cycle"] == cutoff)
            & (selected["split"] == "target")
        ].sort_values("cycle")
        color = colors(index / max(1, len(cutoffs) - 1))
        axis.plot(
            rows["cycle"], rows["predicted_mean"], color=color,
            label=f"observed through {int(cutoff)}",
        )
        axis.fill_between(
            rows["cycle"], rows["lower_95"], rows["upper_95"],
            color=color, alpha=0.07,
        )
        axis.axvline(int(cutoff), color=color, linestyle="--", linewidth=0.8, alpha=0.7)
    axis.set(
        xlabel="Cycle number",
        ylabel="SOH",
        title=f"{cell_id}: streaming forecasts ({schedule})",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

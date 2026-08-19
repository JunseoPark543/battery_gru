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

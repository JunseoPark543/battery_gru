"""Plots for MATR probabilistic trajectory evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .episodes import Episode
from .features import FoldScalers


def plot_hs_attention(
    episode: Episode,
    attention: dict[str, object],
    q_grid: np.ndarray,
    scalers: FoldScalers,
    destination: str | Path,
) -> tuple[Path, Path]:
    """Visualize hierarchical cycle attention alpha and signal attention beta.

    The plotted signals belong only to historical context cycles. Attention is
    model relevance, not a causal attribution score.
    """
    alpha = np.asarray(attention["cycle_attention"])[0]
    beta = np.asarray(attention["signal_attention"])[0]
    gate = np.asarray(attention["fusion_gate"])[0]
    context_count = len(episode.context_x)
    target_count = len(episode.target_x)
    alpha = alpha[:target_count, :context_count]
    beta = beta[:target_count, :context_count, : len(q_grid)]
    context_cycles = np.rint(
        episode.context_x[:, 0] * float(scalers.max_cycle_train)
    ).astype(int)
    target_cycles = episode.target_cycles.astype(int)

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    alpha_path = output / "hs_cycle_attention_alpha.png"
    beta_path = output / "hs_signal_attention_beta.png"

    figure, axis = plt.subplots(figsize=(11, 6.5))
    image_handle = axis.imshow(alpha, aspect="auto", origin="lower", cmap="viridis")
    x_positions = np.linspace(0, max(context_count - 1, 0), min(context_count, 10), dtype=int)
    y_positions = np.linspace(0, max(target_count - 1, 0), min(target_count, 10), dtype=int)
    axis.set_xticks(x_positions, context_cycles[x_positions])
    axis.set_yticks(y_positions, target_cycles[y_positions])
    axis.set(
        xlabel="Historical context cycle",
        ylabel="Future target cycle",
        title=f"{episode.cell_id}: cycle-level attention alpha",
    )
    figure.colorbar(image_handle, ax=axis, label="Attention weight")
    figure.tight_layout()
    figure.savefig(alpha_path, dpi=180)
    plt.close(figure)

    target_index = 0
    signal_present = episode.context_signal_mask.any(axis=-1)
    if signal_present.any():
        selectable_alpha = np.where(signal_present, alpha[target_index], -np.inf)
        context_index = int(np.argmax(selectable_alpha))
    else:
        context_index = int(np.argmax(alpha[target_index]))
    mask = episode.context_signal_mask[context_index]
    voltage = (
        episode.context_signal[context_index, :, 0] * scalers.voltage_std
        + scalers.voltage_mean
    )
    current = (
        episode.context_signal[context_index, :, 1] * scalers.current_std
        + scalers.current_mean
    )
    beta_values = beta[target_index, context_index]
    gate_mean = float(np.mean(gate[target_index])) if gate.size else 0.0
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(q_grid[mask], voltage[mask], color="tab:blue", label="Voltage [V]")
    twin = axes[0].twinx()
    twin.plot(
        q_grid[mask], current[mask], color="tab:orange", alpha=0.8,
        label="|Current| [A]",
    )
    axes[0].set_ylabel("Voltage [V]", color="tab:blue")
    twin.set_ylabel("|Current| [A]", color="tab:orange")
    axes[0].grid(alpha=0.2)
    axes[1].plot(q_grid[mask], beta_values[mask], color="tab:purple")
    axes[1].fill_between(q_grid[mask], 0.0, beta_values[mask], color="tab:purple", alpha=0.2)
    axes[1].set(xlabel="Normalized discharge capacity q", ylabel="Signal attention beta")
    axes[1].grid(alpha=0.2)
    figure.suptitle(
        f"{episode.cell_id}: target {target_cycles[target_index]} attends to "
        f"context {context_cycles[context_index]} "
        f"(alpha={alpha[target_index, context_index]:.3f}, mean gate={gate_mean:.3f})"
    )
    figure.tight_layout()
    figure.savefig(beta_path, dpi=180)
    plt.close(figure)
    return alpha_path, beta_path


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


def plot_trajectory_alpha_overlay(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    cell_id: str,
) -> None:
    """Overlay every requested context fraction for one held-out cell."""
    alpha_values = sorted(float(value) for value in predictions["alpha"].unique())
    if not alpha_values:
        return
    actual = (
        predictions[["cycle", "actual_soh"]]
        .dropna()
        .drop_duplicates("cycle")
        .sort_values("cycle")
    )
    colors = plt.get_cmap("viridis")
    figure, axis = plt.subplots(figsize=(13, 7.5))
    axis.plot(
        actual["cycle"], actual["actual_soh"],
        color="black", linewidth=2.0, label="actual SOH",
    )
    for index, alpha in enumerate(alpha_values):
        selected = predictions[predictions["alpha"] == alpha]
        beta = float(selected["beta"].min())
        selected = selected[selected["beta"] == beta]
        target = selected[selected["split"] == "target"].sort_values("cycle")
        color = colors(index / max(1, len(alpha_values) - 1))
        current_cycle = int(target["cycle"].iloc[0])
        axis.plot(
            target["cycle"], target["predicted_mean"],
            color=color,
            linewidth=1.5,
            label=f"alpha={alpha:g} (start={current_cycle})",
        )
        axis.fill_between(
            target["cycle"], target["lower_95"], target["upper_95"],
            color=color,
            alpha=0.045,
        )
        axis.axvline(
            current_cycle,
            color=color,
            linestyle=":",
            linewidth=0.9,
            alpha=0.75,
        )
    axis.set(
        xlabel="Cycle number",
        ylabel="SOH",
        title=f"{cell_id}: forecasts for all context fractions",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
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
    multiple_betas = "beta" in valid and valid["beta"].nunique() > 1
    group_columns = ["cycle_step", "beta"] if multiple_betas else ["cycle_step"]
    for (metric, ylabel, reference), axis in zip(panels, axes.flat):
        for keys, rows in valid.groupby(group_columns, sort=False):
            if multiple_betas:
                step, beta = keys
            else:
                step = keys[0] if isinstance(keys, tuple) else keys
                beta = None
            rows = rows.sort_values("observed_through_cycle")
            first_cycle = int(rows["observed_through_cycle"].iloc[0])
            beta_label = f", beta={float(beta):g}" if beta is not None else ""
            if int(step) == 1:
                axis.plot(
                    rows["observed_through_cycle"], rows[metric],
                    linewidth=1.2, alpha=0.8,
                    label=(
                        f"step1: {first_cycle}, {first_cycle + 1}, ..."
                        f"{beta_label}"
                    ),
                )
            else:
                axis.plot(
                    rows["observed_through_cycle"], rows[metric],
                    marker="o", markersize=3, linewidth=1.0,
                    label=(
                        f"step{int(step)}: {first_cycle}, "
                        f"{first_cycle + int(step)}, ...{beta_label}"
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
    beta: float | None = None,
) -> None:
    """Overlay selected early streaming forecasts for one test cell."""
    selected = predictions[
        (predictions["cell_id"] == cell_id)
        & (predictions["schedule"] == schedule)
    ]
    if beta is not None:
        selected = selected[selected["beta"] == beta]
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
    beta_title = f", beta={beta:g}" if beta is not None else ""
    axis.set(
        xlabel="Cycle number",
        ylabel="SOH",
        title=f"{cell_id}: streaming forecasts ({schedule}{beta_title})",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_within_cycle_beta_snapshot(
    predictions: pd.DataFrame,
    destination: str | Path,
    *,
    cell_id: str,
    schedule: str,
    observed_through_cycle: int,
) -> None:
    """Compare all partial I-V arrival levels at one SOH context cutoff."""
    selected = predictions[
        (predictions["cell_id"] == cell_id)
        & (predictions["schedule"] == schedule)
        & (predictions["observed_through_cycle"] == observed_through_cycle)
    ]
    if selected.empty:
        return
    beta_values = sorted(selected["beta"].unique())
    first = selected[selected["beta"] == beta_values[0]]
    context = first[first["split"] == "context"].sort_values("cycle")
    target = first[first["split"] == "target"].sort_values("cycle")
    colors = plt.get_cmap("viridis")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(
        context["cycle"], context["actual_soh"], "o-", ms=2.5,
        color="black", label="observed SOH context",
    )
    axis.plot(
        target["cycle"], target["actual_soh"], color="0.35",
        linewidth=1.6, label="actual future SOH",
    )
    for index, beta in enumerate(beta_values):
        rows = selected[
            (selected["beta"] == beta) & (selected["split"] == "target")
        ].sort_values("cycle")
        color = colors(index / max(1, len(beta_values) - 1))
        axis.plot(
            rows["cycle"], rows["predicted_mean"], color=color,
            label=f"beta={float(beta):g}",
        )
        axis.fill_between(
            rows["cycle"], rows["lower_95"], rows["upper_95"],
            color=color, alpha=0.07,
        )
    axis.axvline(
        observed_through_cycle, color="tab:red", linestyle="--",
        label="last observed SOH cycle",
    )
    axis.set(
        xlabel="Cycle number",
        ylabel="SOH",
        title=(
            f"{cell_id}: within-cycle partial I-V updates after SOH cycle "
            f"{observed_through_cycle}"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

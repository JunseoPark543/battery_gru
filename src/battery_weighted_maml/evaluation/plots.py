"""Headless local plots for runs and aggregate comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_target_prediction(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    observed = frame[frame["observed_soh"].notna()]
    future = frame[frame["split"] == "future"]
    axis.plot(observed["cycle"], observed["observed_soh"], label="observed", linewidth=1.5)
    axis.plot(future["cycle"], future["predicted_soh"], label="recursive prediction", linewidth=1.3)
    axis.axhline(float(frame["eol_threshold"].iloc[0]), color="tab:red", linestyle="--", label="EOL")
    axis.axvline(int((frame["split"] == "support").sum()), color="gray", linestyle=":", label="L")
    axis.set(xlabel="Cycle", ylabel="SOH", title=title)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    if not alphas.empty:
        pivot = alphas.pivot(index="source", columns="iteration", values="alpha")
        fig, axis = plt.subplots(figsize=(10, max(3, 0.45 * len(pivot))))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        axis.set_yticks(np.arange(len(pivot))); axis.set_yticklabels(pivot.index, fontsize=7)
        axis.set_xticks(np.arange(len(pivot.columns))); axis.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
        axis.set(xlabel="Meta iteration", title="Alpha heatmap")
        fig.colorbar(image, ax=axis, label="alpha"); fig.tight_layout()
        fig.savefig(root / "weights/alpha_heatmap.png", dpi=150); plt.close(fig)


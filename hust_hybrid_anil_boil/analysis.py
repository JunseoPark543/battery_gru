"""Plots and representation diagnostics for hybrid direct-RUL experiments."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def save_training_diagnostics(history: list[dict[str, Any]], path: str | Path) -> None:
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for column in ("total_loss", "query_loss", "L_GY", "L_G", "L_S", "L_R", "L_C", "L_O", "L_delta"):
        if column in frame:
            axes[0].plot(frame.iteration, frame[column], label=column, alpha=0.75)
    axes[0].set_yscale("symlog", linthresh=1.0e-5)
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8, ncol=3)
    evaluated = frame.dropna(subset=["source_validation_mae_cycles"])
    if not evaluated.empty:
        axes[1].plot(
            evaluated.iteration,
            evaluated.source_validation_mae_cycles,
            marker="o",
            label="source validation query MAE",
        )
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel("MAE (cycles)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Hybrid ANIL/BOIL training diagnostics")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_adaptation_curve(metrics: pd.DataFrame, path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = metrics.sort_values("adaptation_step")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(ordered.adaptation_step, ordered.mae_cycles, marker="o", label="MAE")
    axes[0].plot(ordered.adaptation_step, ordered.rmse_cycles, marker="s", label="RMSE")
    axes[0].set_xlabel("Adaptation steps")
    axes[0].set_ylabel("Error (cycles)")
    axes[0].set_title("Prediction error")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        ordered.adaptation_step,
        ordered.general_representation_cosine_distance,
        marker="o",
        label="General",
    )
    axes[1].plot(
        ordered.adaptation_step,
        ordered.specific_representation_cosine_distance,
        marker="s",
        label="Specific",
    )
    axes[1].set_xlabel("Adaptation steps")
    axes[1].set_ylabel("Mean cosine distance")
    axes[1].set_title("Representation change")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _tsne(features: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    if len(features) < 3:
        raise ValueError("t-SNE requires at least three samples")
    kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": min(30.0, max(2.0, (len(features) - 1) / 3.0)),
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    return TSNE(**kwargs).fit_transform(features)


def save_feature_visualization(
    general_features: np.ndarray,
    specific_features: np.ndarray,
    domains: Sequence[str],
    normalized_rul: Sequence[float],
    path: str | Path,
    *,
    seed: int,
) -> None:
    """Save domain- and degradation-colored t-SNE plots for both branches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    general_2d = _tsne(np.asarray(general_features, dtype=np.float64), seed)
    specific_2d = _tsne(np.asarray(specific_features, dtype=np.float64), seed)
    domain_values = np.asarray(list(domains), dtype=object)
    rul_values = np.asarray(normalized_rul, dtype=np.float64)
    unique_domains = sorted(np.unique(domain_values))
    color_map = {domain: index for index, domain in enumerate(unique_domains)}
    domain_color = np.asarray([color_map[value] for value in domain_values])
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    for axis, points, name in (
        (axes[0, 0], general_2d, "General"),
        (axes[0, 1], specific_2d, "Specific"),
    ):
        scatter = axis.scatter(points[:, 0], points[:, 1], c=domain_color, cmap="tab10", s=35)
        axis.set_title(f"{name} representation by protocol")
        handles, _ = scatter.legend_elements()
        axis.legend(handles, unique_domains, fontsize=7, ncol=2)
    for axis, points, name in (
        (axes[1, 0], general_2d, "General"),
        (axes[1, 1], specific_2d, "Specific"),
    ):
        scatter = axis.scatter(points[:, 0], points[:, 1], c=rul_values, cmap="viridis", s=35)
        axis.set_title(f"{name} representation by normalized RUL")
        figure.colorbar(scatter, ax=axis, label="RUL / 500")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("HUST held-out query representation diagnostics")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_method_comparison_figure(table: pd.DataFrame, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.empty:
        return
    ordered = table.sort_values("mae_cycles")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(ordered))
    width = 0.38
    axes[0].bar(x - width / 2, ordered.mae_cycles, width, label="MAE")
    axes[0].bar(x + width / 2, ordered.rmse_cycles, width, label="RMSE")
    axes[0].set_xticks(x, ordered.method, rotation=30, ha="right")
    axes[0].set_ylabel("Cycles (lower is better)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, ordered.r2.fillna(0.0), color="#59a14f")
    axes[1].set_xticks(x, ordered.method, rotation=30, ha="right")
    axes[1].set_ylabel("R² (higher is better)")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Architecture-matched meta-learning comparison")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


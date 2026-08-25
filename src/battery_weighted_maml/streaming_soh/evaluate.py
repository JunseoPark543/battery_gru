"""Held-out-cell evaluation and plots for streaming SOH forecasting."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from battery_weighted_maml.matr_anp.data import CellData, load_dataset
from battery_weighted_maml.matr_anp.runtime import resolve_device, seed_everything, write_json
from battery_weighted_maml.matr_anp.splits import FoldSplit

from .config import config_from_dict, resolve_data_root
from .episodes import EpisodeSampler, StreamingEpisode, collate_episodes
from .features import CycleGridProcessor, EpisodeUnavailable, SignalScaler
from .losses import soh_metrics
from .model import build_model
from .train import model_forward


def _entries(
    cells: list[CellData], sampler: EpisodeSampler, alphas: list[float], betas: list[float]
) -> list[tuple[float, float, StreamingEpisode]]:
    entries: list[tuple[float, float, StreamingEpisode]] = []
    for cell in cells:
        for alpha in alphas:
            for beta in betas:
                try:
                    entries.append((alpha, beta, sampler.evaluation(cell, alpha, beta)))
                except EpisodeUnavailable:
                    continue
    if not entries:
        raise EpisodeUnavailable("test split produced no streaming SOH episodes")
    return entries


def _predict(
    model: torch.nn.Module,
    entries: list[tuple[float, float, StreamingEpisode]],
    device: torch.device,
    cycle_scale: float,
    interval_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []
    point_rows: list[dict] = []
    z_value = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(entries), 8):
            chunk = entries[start : start + 8]
            batch = collate_episodes([item[2] for item in chunk]).to(device)
            output = model_forward(model, batch)
            mean = output["soh_mean"].float().cpu().numpy()
            std = output["soh_std"].float().cpu().numpy()
            target = batch.target_soh.float().cpu().numpy()
            mask = batch.query_mask.cpu().numpy()
            for row, (alpha, beta, episode) in enumerate(chunk):
                valid = mask[row]
                metrics = soh_metrics(target[row, valid], mean[row, valid])
                lower = mean[row, valid] - z_value * std[row, valid]
                upper = mean[row, valid] + z_value * std[row, valid]
                coverage = np.mean(
                    (target[row, valid] >= lower) & (target[row, valid] <= upper)
                )
                episode_rows.append(
                    {
                        "cell_id": episode.cell_id,
                        "current_cycle": episode.current_cycle,
                        "cycle_alpha": alpha,
                        "beta": beta,
                        "q_cut": episode.q_cut,
                        "q_end": episode.q_end,
                        "forecast_points": int(np.count_nonzero(valid)),
                        "current_soh_absolute_error": abs(
                            float(mean[row, 0]) - float(target[row, 0])
                        ),
                        "interval_coverage": float(coverage),
                        **metrics,
                    }
                )
                cycles = np.rint(
                    episode.query_cycle_scaled * float(cycle_scale)
                ).astype(np.int64)
                for index, cycle in enumerate(cycles):
                    point_rows.append(
                        {
                            "cell_id": episode.cell_id,
                            "current_cycle": episode.current_cycle,
                            "cycle_alpha": alpha,
                            "beta": beta,
                            "forecast_cycle": int(cycle),
                            "actual_soh": float(episode.target_soh[index]),
                            "predicted_soh": float(mean[row, index]),
                            "predicted_std": float(std[row, index]),
                            "lower_interval": float(mean[row, index] - z_value * std[row, index]),
                            "upper_interval": float(mean[row, index] + z_value * std[row, index]),
                        }
                    )
    return pd.DataFrame(episode_rows), pd.DataFrame(point_rows)


def _plot_heatmap(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    grouped = metrics.groupby(["cycle_alpha", "beta"], as_index=False).agg(
        soh_rmse=("soh_rmse", "mean"),
        current_soh_mae=("current_soh_absolute_error", "mean"),
    )
    alphas = sorted(grouped["cycle_alpha"].unique())
    betas = sorted(grouped["beta"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, column, title in (
        (axes[0], "soh_rmse", "Future SOH RMSE"),
        (axes[1], "current_soh_mae", "Current SOH MAE"),
    ):
        matrix = grouped.pivot(index="cycle_alpha", columns="beta", values=column).reindex(
            index=alphas, columns=betas
        )
        image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(betas)), [f"{value:g}" for value in betas])
        axis.set_yticks(range(len(alphas)), [f"{value:g}" for value in alphas])
        axis.set_xlabel("Observed current-cycle fraction beta")
        axis.set_ylabel("Current-cycle position alpha")
        axis.set_title(title)
        for row in range(len(alphas)):
            for column_index in range(len(betas)):
                value = matrix.iloc[row, column_index]
                axis.text(column_index, row, f"{value:.4f}", ha="center", va="center", color="white")
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.suptitle("Held-out-cell streaming SOH trajectory evaluation")
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _plot_examples(
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    path: Path,
    plot_cells: int,
    dpi: int,
) -> None:
    cells = sorted(metrics["cell_id"].unique())[:plot_cells]
    if not cells:
        return
    beta = sorted(metrics["beta"].unique())[len(metrics["beta"].unique()) // 2]
    alpha = sorted(metrics["cycle_alpha"].unique())[len(metrics["cycle_alpha"].unique()) // 2]
    figure, axes = plt.subplots(
        len(cells), 1, figsize=(9, 3.2 * len(cells)), squeeze=False, constrained_layout=True
    )
    for row, cell_id in enumerate(cells):
        axis = axes[row, 0]
        selected = points[
            (points["cell_id"] == cell_id)
            & np.isclose(points["cycle_alpha"], alpha)
            & np.isclose(points["beta"], beta)
        ].sort_values("forecast_cycle")
        if selected.empty:
            axis.axis("off")
            continue
        record = metrics[
            (metrics["cell_id"] == cell_id)
            & np.isclose(metrics["cycle_alpha"], alpha)
            & np.isclose(metrics["beta"], beta)
        ].iloc[0]
        axis.plot(selected["forecast_cycle"], selected["actual_soh"], color="black", label="actual")
        axis.plot(
            selected["forecast_cycle"], selected["predicted_soh"], color="#d62728", label="prediction"
        )
        axis.fill_between(
            selected["forecast_cycle"],
            selected["lower_interval"],
            selected["upper_interval"],
            color="#d62728",
            alpha=0.18,
            label="prediction interval",
        )
        axis.axvline(record["current_cycle"], color="#1f77b4", ls="--", lw=1)
        axis.set_title(
            f"{cell_id}, current={int(record['current_cycle'])}, beta={beta:g}, "
            f"RMSE={record['soh_rmse']:.4f}"
        )
        axis.set_xlabel("Cycle")
        axis.set_ylabel("SOH")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    data_root: str | Path | None = None,
    device_name: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "streaming_soh":
        raise ValueError("checkpoint belongs to another algorithm")
    config = config_from_dict(payload["config"])
    if device_name:
        config.device = device_name
    seed_everything(config.seed, config.training.deterministic)
    device = resolve_device(config.device)
    root = resolve_data_root(config, str(data_root) if data_root is not None else None)
    cells, _ = load_dataset(root, config.data, tolerate_invalid_cells=True)
    split_keys = {field.name for field in fields(FoldSplit)}
    split = FoldSplit(
        **{key: value for key, value in payload["fold_split"].items() if key in split_keys}
    )
    by_id = {cell.cell_id: cell for cell in cells}
    missing = set(split.test_cells) - set(by_id)
    if missing:
        raise ValueError(f"checkpoint test cells are missing: {sorted(missing)}")
    test_cells = [by_id[cell_id] for cell_id in split.test_cells]
    scaler = SignalScaler.from_dict(payload["signal_scaler"])
    processor = CycleGridProcessor(
        config.q_grid,
        config.episode.minimum_observed_q_points,
        config.episode.minimum_future_q_points,
    )
    sampler = EpisodeSampler(config.episode, processor, scaler)
    model = build_model(config.model).to(device)
    model.load_state_dict(payload["model_state_dict"])
    entries = _entries(
        test_cells,
        sampler,
        config.episode.evaluation_cycle_alphas,
        config.episode.evaluation_betas,
    )
    metrics, points = _predict(
        model,
        entries,
        device,
        config.episode.cycle_scale,
        config.evaluation.interval_level,
    )
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else checkpoint_path.parent.parent / "evaluation" / checkpoint_path.stem
    )
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "episode_metrics.csv", index=False)
    points.to_csv(destination / "per_cycle_predictions.csv", index=False)
    aggregate = metrics.groupby(["cycle_alpha", "beta"], as_index=False).agg(
        cell_count=("cell_id", "nunique"),
        episode_count=("cell_id", "size"),
        soh_mae=("soh_mae", "mean"),
        soh_rmse=("soh_rmse", "mean"),
        current_soh_mae=("current_soh_absolute_error", "mean"),
        interval_coverage=("interval_coverage", "mean"),
    )
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    write_json(
        destination / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "test_cells": split.test_cells,
            "episodes": len(metrics),
            "soh_mae": float(metrics["soh_mae"].mean()),
            "soh_rmse": float(metrics["soh_rmse"].mean()),
            "current_soh_mae": float(metrics["current_soh_absolute_error"].mean()),
            "interval_coverage": float(metrics["interval_coverage"].mean()),
        },
    )
    _plot_heatmap(metrics, destination / "metrics_heatmap.png", config.evaluation.dpi)
    _plot_examples(
        metrics,
        points,
        destination / "trajectory_examples.png",
        config.evaluation.plot_cells,
        config.evaluation.dpi,
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate streaming SOH checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = evaluate_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    print(f"Evaluation directory: {destination}")


if __name__ == "__main__":
    main()

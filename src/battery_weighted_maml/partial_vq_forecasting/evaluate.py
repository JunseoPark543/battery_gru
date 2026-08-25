"""Evaluate and plot a trained partial V-Q forecaster on held-out cells."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from battery_weighted_maml.matr_anp.data import CellData, load_dataset
from battery_weighted_maml.matr_anp.runtime import resolve_device, seed_everything, write_json
from battery_weighted_maml.matr_anp.splits import FoldSplit

from .config import config_from_dict, resolve_data_root
from .episodes import EpisodeSampler, VQEpisode, collate_episodes
from .features import EpisodeUnavailable, PartialVQProcessor, VoltageScaler
from .losses import voltage_metrics
from .model import build_model


def _episodes(
    cells: list[CellData], sampler: EpisodeSampler, alphas: list[float], betas: list[float]
) -> list[tuple[float, float, VQEpisode]]:
    output: list[tuple[float, float, VQEpisode]] = []
    for cell in cells:
        for alpha in alphas:
            for beta in betas:
                try:
                    output.append((alpha, beta, sampler.evaluation(cell, alpha, beta)))
                except EpisodeUnavailable:
                    continue
    if not output:
        raise EpisodeUnavailable("test split produced no valid episodes")
    return output


def _predict(
    model: torch.nn.Module,
    entries: list[tuple[float, float, VQEpisode]],
    sampler: EpisodeSampler,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []
    point_rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(entries), 64):
            chunk = entries[start : start + 64]
            episodes = [item[2] for item in chunk]
            batch = collate_episodes(episodes).to(device)
            output = model(batch.input_feature, batch.q_coordinate)
            predicted_voltage = sampler.scaler.inverse(
                output["voltage"].float().cpu().numpy()
            )
            predicted_q_end = (
                output["endpoint_fraction"].float().cpu().numpy() * sampler.processor.q_max
            )
            for index, (alpha, beta, episode) in enumerate(chunk):
                target_voltage = sampler.scaler.inverse(episode.target_voltage)
                metrics = voltage_metrics(
                    target_voltage[episode.future_mask],
                    predicted_voltage[index, episode.future_mask],
                )
                episode_rows.append(
                    {
                        "cell_id": episode.cell_id,
                        "cycle_number": episode.cycle_number,
                        "cycle_alpha": alpha,
                        "beta": beta,
                        "q_cut": episode.q_cut,
                        "actual_q_end": episode.q_end,
                        "predicted_q_end": float(predicted_q_end[index]),
                        "endpoint_absolute_error_q": abs(
                            float(predicted_q_end[index]) - episode.q_end
                        ),
                        "observed_points": episode.observed_points,
                        "future_points": episode.future_points,
                        **metrics,
                    }
                )
                for q_index in np.flatnonzero(episode.valid_mask):
                    point_rows.append(
                        {
                            "cell_id": episode.cell_id,
                            "cycle_number": episode.cycle_number,
                            "cycle_alpha": alpha,
                            "beta": beta,
                            "q": float(sampler.processor.grid[q_index]),
                            "actual_voltage_v": float(target_voltage[q_index]),
                            "predicted_voltage_v": float(predicted_voltage[index, q_index]),
                            "observed": bool(episode.observed_mask[q_index]),
                            "future": bool(episode.future_mask[q_index]),
                            "actual_q_end": episode.q_end,
                            "predicted_q_end": float(predicted_q_end[index]),
                        }
                    )
    return pd.DataFrame(episode_rows), pd.DataFrame(point_rows)


def _plot_metric_heatmaps(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    grouped = metrics.groupby(["cycle_alpha", "beta"], as_index=False).agg(
        voltage_rmse_v=("voltage_rmse_v", "mean"),
        endpoint_mae_q=("endpoint_absolute_error_q", "mean"),
    )
    alphas = sorted(grouped["cycle_alpha"].unique())
    betas = sorted(grouped["beta"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, column, title, fmt in (
        (axes[0], "voltage_rmse_v", "Future voltage RMSE (V)", ".4f"),
        (axes[1], "endpoint_mae_q", "Discharge endpoint MAE (q)", ".4f"),
    ):
        matrix = grouped.pivot(index="cycle_alpha", columns="beta", values=column).reindex(
            index=alphas, columns=betas
        )
        image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(betas)), [f"{value:g}" for value in betas])
        axis.set_yticks(range(len(alphas)), [f"{value:g}" for value in alphas])
        axis.set_xlabel("Observed within-cycle fraction beta")
        axis.set_ylabel("Cycle position alpha")
        axis.set_title(title)
        for row in range(len(alphas)):
            for column_index in range(len(betas)):
                value = matrix.iloc[row, column_index]
                axis.text(
                    column_index,
                    row,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color="white" if np.isfinite(value) else "black",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.suptitle("Held-out cell partial V-Q forecasting")
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _plot_trajectories(
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    path: Path,
    number_of_cells: int,
    dpi: int,
) -> None:
    cells = sorted(metrics["cell_id"].unique())[:number_of_cells]
    betas = sorted(metrics["beta"].unique())
    if not cells or not betas:
        return
    middle_alpha = sorted(metrics["cycle_alpha"].unique())[
        len(metrics["cycle_alpha"].unique()) // 2
    ]
    figure, axes = plt.subplots(
        len(cells), len(betas),
        figsize=(4.1 * len(betas), 3.0 * len(cells)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, cell_id in enumerate(cells):
        for column, beta in enumerate(betas):
            axis = axes[row, column]
            selected_metrics = metrics[
                (metrics["cell_id"] == cell_id)
                & np.isclose(metrics["cycle_alpha"], middle_alpha)
                & np.isclose(metrics["beta"], beta)
            ]
            if selected_metrics.empty:
                axis.axis("off")
                continue
            record = selected_metrics.iloc[0]
            selected = points[
                (points["cell_id"] == cell_id)
                & (points["cycle_number"] == record["cycle_number"])
                & np.isclose(points["cycle_alpha"], middle_alpha)
                & np.isclose(points["beta"], beta)
            ].sort_values("q")
            observed = selected["observed"]
            future = selected["future"]
            axis.plot(selected["q"], selected["actual_voltage_v"], color="black", lw=1.5, label="actual full")
            axis.plot(
                selected.loc[observed, "q"], selected.loc[observed, "actual_voltage_v"],
                color="#1f77b4", lw=2.4, label="observed prefix",
            )
            predicted_visible = future & (selected["q"] <= record["predicted_q_end"])
            axis.plot(
                selected.loc[predicted_visible, "q"],
                selected.loc[predicted_visible, "predicted_voltage_v"],
                color="#d62728", lw=2.0, ls="--", label="predicted future",
            )
            axis.axvline(record["q_cut"], color="#1f77b4", ls=":", lw=1)
            axis.axvline(record["actual_q_end"], color="black", ls=":", lw=1)
            axis.axvline(record["predicted_q_end"], color="#d62728", ls=":", lw=1)
            axis.set_title(
                f"{cell_id} c{int(record['cycle_number'])} beta={beta:g}\n"
                f"V-RMSE={record['voltage_rmse_v']:.4f} V, "
                f"qend err={record['endpoint_absolute_error_q']:.4f}"
            )
            axis.set_xlabel("q = discharged capacity / nominal capacity")
            axis.set_ylabel("Voltage (V)")
            axis.grid(alpha=0.25)
            if row == 0 and column == 0:
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
    if payload.get("algorithm") != "partial_vq_forecasting":
        raise ValueError("checkpoint belongs to another algorithm")
    config = config_from_dict(payload["config"])
    if device_name:
        config.device = device_name
    seed_everything(config.seed, config.training.deterministic)
    device = resolve_device(config.device)
    root = resolve_data_root(config, str(data_root) if data_root is not None else None)
    cells, _ = load_dataset(root, config.data, tolerate_invalid_cells=True)
    split_keys = {field.name for field in fields(FoldSplit)}
    split = FoldSplit(**{key: value for key, value in payload["fold_split"].items() if key in split_keys})
    by_id = {cell.cell_id: cell for cell in cells}
    missing = set(split.test_cells) - set(by_id)
    if missing:
        raise ValueError(f"checkpoint test cells are missing from data root: {sorted(missing)}")
    test_cells = [by_id[cell_id] for cell_id in split.test_cells]
    scaler = VoltageScaler.from_dict(payload["voltage_scaler"])
    processor = PartialVQProcessor(
        config.q_grid,
        config.episode.minimum_observed_points,
        config.episode.minimum_future_points,
    )
    sampler = EpisodeSampler(config.episode, processor, scaler)
    model = build_model(config.model).to(device)
    model.load_state_dict(payload["model_state_dict"])
    entries = _episodes(
        test_cells,
        sampler,
        config.episode.evaluation_cycle_alphas,
        config.episode.evaluation_betas,
    )
    metrics, points = _predict(model, entries, sampler, device)
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else checkpoint_path.parent.parent / "evaluation" / checkpoint_path.stem
    )
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "episode_metrics.csv", index=False)
    points.to_csv(destination / "per_q_predictions.csv", index=False)
    aggregate = metrics.groupby(["cycle_alpha", "beta"], as_index=False).agg(
        cell_count=("cell_id", "nunique"),
        episode_count=("cell_id", "size"),
        voltage_mae_v=("voltage_mae_v", "mean"),
        voltage_rmse_v=("voltage_rmse_v", "mean"),
        endpoint_mae_q=("endpoint_absolute_error_q", "mean"),
    )
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "test_cells": split.test_cells,
        "episodes": len(metrics),
        "future_voltage_mae_v": float(metrics["voltage_mae_v"].mean()),
        "future_voltage_rmse_v": float(metrics["voltage_rmse_v"].mean()),
        "endpoint_mae_q": float(metrics["endpoint_absolute_error_q"].mean()),
    }
    write_json(destination / "summary.json", summary)
    _plot_metric_heatmaps(
        metrics, destination / "metrics_heatmap.png", config.evaluation.dpi
    )
    _plot_trajectories(
        metrics,
        points,
        destination / "trajectory_examples.png",
        config.evaluation.plot_cells,
        config.evaluation.dpi,
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate partial V-Q forecaster")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = evaluate_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    print(f"Evaluation directory: {output}")


if __name__ == "__main__":
    main()

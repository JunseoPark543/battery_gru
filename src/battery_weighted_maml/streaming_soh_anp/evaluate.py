"""Held-out evaluation, calibration, and plots for streaming latent ANP."""

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
from battery_weighted_maml.streaming_soh.episodes import (
    EpisodeSampler,
    StreamingEpisode,
    collate_episodes,
)
from battery_weighted_maml.streaming_soh.features import (
    CycleGridProcessor,
    EpisodeUnavailable,
    SignalScaler,
)

from .config import config_from_dict, resolve_data_root
from .losses import regression_metrics
from .model import build_model
from .train import model_forward


CALIBRATION_LEVELS = (0.5, 0.8, 0.9, 0.95)


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
        raise EpisodeUnavailable("test split produced no latent ANP episodes")
    return entries


def _coverage(target: np.ndarray, mean: np.ndarray, std: np.ndarray, level: float) -> float:
    z_value = NormalDist().inv_cdf(0.5 + level / 2.0)
    return float(np.mean(np.abs(target - mean) <= z_value * std))


def _predict(
    model: torch.nn.Module,
    entries: list[tuple[float, float, StreamingEpisode]],
    device: torch.device,
    *,
    cycle_scale: float,
    latent_samples: int,
    interval_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []
    point_rows: list[dict] = []
    z_value = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(entries), 4):
            chunk = entries[start : start + 4]
            batch = collate_episodes([item[2] for item in chunk]).to(device)
            output = model_forward(
                model,
                batch,
                use_posterior=False,
                num_latent_samples=latent_samples,
            )
            mean = output["soh_mean"].float().cpu().numpy()
            total_std = output["soh_std"].float().cpu().numpy()
            epistemic_std = output["soh_epistemic_std"].float().cpu().numpy()
            aleatoric_std = output["soh_aleatoric_std"].float().cpu().numpy()
            target = batch.target_soh.float().cpu().numpy()
            mask = batch.query_mask.cpu().numpy()
            prior_std = output["prior_std"].float().cpu().numpy()
            for row, (alpha, beta, episode) in enumerate(chunk):
                valid = mask[row]
                metrics = regression_metrics(
                    target[row, valid], mean[row, valid], total_std[row, valid]
                )
                coverage = {
                    f"coverage_{int(level * 100)}": _coverage(
                        target[row, valid], mean[row, valid], total_std[row, valid], level
                    )
                    for level in CALIBRATION_LEVELS
                }
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
                        "mean_predictive_std": float(np.mean(total_std[row, valid])),
                        "mean_epistemic_std": float(np.mean(epistemic_std[row, valid])),
                        "mean_aleatoric_std": float(np.mean(aleatoric_std[row, valid])),
                        "mean_prior_latent_std": float(np.mean(prior_std[row])),
                        "interval_width": float(
                            np.mean(2.0 * z_value * total_std[row, valid])
                        ),
                        **coverage,
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
                            "predictive_std": float(total_std[row, index]),
                            "epistemic_std": float(epistemic_std[row, index]),
                            "aleatoric_std": float(aleatoric_std[row, index]),
                            "lower_interval": float(
                                mean[row, index] - z_value * total_std[row, index]
                            ),
                            "upper_interval": float(
                                mean[row, index] + z_value * total_std[row, index]
                            ),
                        }
                    )
    return pd.DataFrame(episode_rows), pd.DataFrame(point_rows)


def _plot_heatmaps(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    grouped = metrics.groupby(["cycle_alpha", "beta"], as_index=False).agg(
        soh_rmse=("soh_rmse", "mean"),
        soh_crps=("soh_crps", "mean"),
        coverage_95=("coverage_95", "mean"),
    )
    alphas = sorted(grouped["cycle_alpha"].unique())
    betas = sorted(grouped["beta"].unique())
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, column, title in (
        (axes[0], "soh_rmse", "SOH RMSE"),
        (axes[1], "soh_crps", "SOH CRPS"),
        (axes[2], "coverage_95", "95% interval coverage"),
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
                axis.text(
                    column_index,
                    row,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.suptitle("Held-out latent ANP SOH accuracy and calibration")
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _plot_calibration(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    nominal = np.asarray(CALIBRATION_LEVELS, dtype=np.float64)
    empirical = np.asarray(
        [metrics[f"coverage_{int(level * 100)}"].mean() for level in nominal]
    )
    figure, axis = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="black", ls="--", label="ideal")
    axis.plot(nominal, empirical, marker="o", color="#d62728", label="latent ANP")
    axis.set_xlim(0.45, 1.0)
    axis.set_ylim(0.45, 1.0)
    axis.set_xlabel("Nominal interval coverage")
    axis.set_ylabel("Empirical held-out coverage")
    axis.set_title("SOH predictive uncertainty calibration")
    axis.grid(alpha=0.25)
    axis.legend()
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
        selected_metrics = metrics[
            (metrics["cell_id"] == cell_id)
            & np.isclose(metrics["cycle_alpha"], alpha)
            & np.isclose(metrics["beta"], beta)
        ]
        if selected.empty or selected_metrics.empty:
            axis.axis("off")
            continue
        record = selected_metrics.iloc[0]
        epistemic_z = 1.96
        axis.fill_between(
            selected["forecast_cycle"],
            selected["lower_interval"],
            selected["upper_interval"],
            color="#d62728",
            alpha=0.13,
            label="total 95% interval",
        )
        axis.fill_between(
            selected["forecast_cycle"],
            selected["predicted_soh"] - epistemic_z * selected["epistemic_std"],
            selected["predicted_soh"] + epistemic_z * selected["epistemic_std"],
            color="#ff7f0e",
            alpha=0.22,
            label="epistemic component",
        )
        axis.plot(selected["forecast_cycle"], selected["actual_soh"], color="black", label="actual")
        axis.plot(
            selected["forecast_cycle"],
            selected["predicted_soh"],
            color="#d62728",
            label="predictive mean",
        )
        axis.axvline(record["current_cycle"], color="#1f77b4", ls="--", lw=1)
        axis.set_title(
            f"{cell_id}, current={int(record['current_cycle'])}, beta={beta:g}, "
            f"RMSE={record['soh_rmse']:.4f}, CRPS={record['soh_crps']:.4f}"
        )
        axis.set_xlabel("Cycle")
        axis.set_ylabel("SOH")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
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
    if payload.get("algorithm") != "streaming_soh_latent_anp":
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
        cycle_scale=config.episode.cycle_scale,
        latent_samples=config.evaluation.latent_samples,
        interval_level=config.evaluation.interval_level,
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
        soh_nll=("soh_nll", "mean"),
        soh_crps=("soh_crps", "mean"),
        coverage_90=("coverage_90", "mean"),
        coverage_95=("coverage_95", "mean"),
        interval_width=("interval_width", "mean"),
        epistemic_std=("mean_epistemic_std", "mean"),
        aleatoric_std=("mean_aleatoric_std", "mean"),
    )
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    calibration = pd.DataFrame(
        {
            "nominal_coverage": CALIBRATION_LEVELS,
            "empirical_coverage": [
                float(metrics[f"coverage_{int(level * 100)}"].mean())
                for level in CALIBRATION_LEVELS
            ],
        }
    )
    calibration.to_csv(destination / "calibration.csv", index=False)
    write_json(
        destination / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "test_cells": split.test_cells,
            "episodes": len(metrics),
            "latent_samples": config.evaluation.latent_samples,
            "soh_mae": float(metrics["soh_mae"].mean()),
            "soh_rmse": float(metrics["soh_rmse"].mean()),
            "soh_nll": float(metrics["soh_nll"].mean()),
            "soh_crps": float(metrics["soh_crps"].mean()),
            "coverage_90": float(metrics["coverage_90"].mean()),
            "coverage_95": float(metrics["coverage_95"].mean()),
            "mean_epistemic_std": float(metrics["mean_epistemic_std"].mean()),
            "mean_aleatoric_std": float(metrics["mean_aleatoric_std"].mean()),
        },
    )
    _plot_heatmaps(metrics, destination / "metrics_heatmap.png", config.evaluation.dpi)
    _plot_calibration(metrics, destination / "calibration_plot.png", config.evaluation.dpi)
    _plot_examples(
        metrics,
        points,
        destination / "trajectory_uncertainty_examples.png",
        config.evaluation.plot_cells,
        config.evaluation.dpi,
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate streaming latent ANP checkpoint")
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

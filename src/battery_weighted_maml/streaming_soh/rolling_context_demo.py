"""Compare one held-out cell as completed-cycle context grows over time."""

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


def _validate_cycles(cycles: list[int]) -> list[int]:
    values = sorted(set(int(value) for value in cycles))
    if not values or any(value <= 0 for value in values):
        raise ValueError("cycles must contain positive cycle numbers")
    return values


def _select_cell(
    cells: list[CellData],
    sampler: EpisodeSampler,
    cycles: list[int],
    requested: str | None,
) -> CellData:
    candidates: list[CellData] = []
    required = set(cycles)
    for cell in cells:
        eligible = {cell.cycles[index].cycle_number for index in sampler.candidate_indices(cell)}
        if required.issubset(eligible):
            candidates.append(cell)
    if requested:
        selected = [cell for cell in candidates if cell.cell_id == requested]
        if not selected:
            raise EpisodeUnavailable(
                f"{requested}: requested cycles {cycles} are not all eligible held-out cuts"
            )
        return selected[0]
    if not candidates:
        raise EpisodeUnavailable(
            f"no held-out cell supports every requested current cycle: {cycles}"
        )
    return sorted(candidates, key=lambda item: item.cell_id)[0]


def _predict_rolling_context(
    model: torch.nn.Module,
    episodes: list[StreamingEpisode],
    device: torch.device,
    cycle_scale: float,
    interval_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch = collate_episodes(episodes).to(device)
    model.eval()
    with torch.no_grad():
        output = model_forward(model, batch)
    mean = output["soh_mean"].float().cpu().numpy()
    std = output["soh_std"].float().cpu().numpy()
    mask = batch.query_mask.cpu().numpy()
    completed_state = output["completed_state"].float().cpu().numpy()
    candidate_state = output["candidate_state"].float().cpu().numpy()
    z_value = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    point_rows: list[dict] = []
    for row, episode in enumerate(episodes):
        cycles = np.rint(episode.query_cycle_scaled * cycle_scale).astype(np.int64)
        for index, cycle in enumerate(cycles):
            point_rows.append(
                {
                    "cell_id": episode.cell_id,
                    "current_cycle": episode.current_cycle,
                    "beta": episode.beta,
                    "history_cycles": len(episode.history_soh),
                    "q_cut": episode.q_cut,
                    "forecast_cycle": int(cycle),
                    "actual_soh": float(episode.target_soh[index]),
                    "predicted_soh": float(mean[row, index]),
                    "predicted_std": float(std[row, index]),
                    "lower_interval": float(mean[row, index] - z_value * std[row, index]),
                    "upper_interval": float(mean[row, index] + z_value * std[row, index]),
                    "completed_state_l2": float(np.linalg.norm(completed_state[row])),
                    "candidate_state_change_l2": float(
                        np.linalg.norm(candidate_state[row] - completed_state[row])
                    ),
                }
            )
    points = pd.DataFrame(point_rows)
    common_start = max(episode.current_cycle for episode in episodes)
    metric_rows: list[dict] = []
    for row, episode in enumerate(episodes):
        valid = mask[row]
        remaining_metrics = soh_metrics(
            batch.target_soh[row, valid].cpu().numpy(), mean[row, valid]
        )
        selected = points[
            (points["current_cycle"] == episode.current_cycle)
            & (points["forecast_cycle"] >= common_start)
        ]
        common_metrics = soh_metrics(
            selected["actual_soh"].to_numpy(), selected["predicted_soh"].to_numpy()
        )
        metric_rows.append(
            {
                "cell_id": episode.cell_id,
                "current_cycle": episode.current_cycle,
                "beta": episode.beta,
                "completed_through_cycle": episode.current_cycle - 1,
                "history_cycles": len(episode.history_soh),
                "q_cut": episode.q_cut,
                "forecast_points": int(np.count_nonzero(valid)),
                "remaining_soh_mae": remaining_metrics["soh_mae"],
                "remaining_soh_rmse": remaining_metrics["soh_rmse"],
                "common_horizon_start": common_start,
                "common_horizon_soh_mae": common_metrics["soh_mae"],
                "common_horizon_soh_rmse": common_metrics["soh_rmse"],
                "current_soh_absolute_error": abs(
                    float(mean[row, 0]) - float(batch.target_soh[row, 0].cpu())
                ),
                "mean_predicted_std": float(np.mean(std[row, valid])),
            }
        )
    return pd.DataFrame(metric_rows), points


def _plot_panels(
    cell: CellData,
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    cycles = sorted(metrics["current_cycle"].unique())
    x_min = max(1, min(cycles) - 40)
    x_max = int(cell.cycle_numbers[-1])
    relevant = cell.soh[cell.cycle_numbers >= x_min]
    y_margin = 0.03
    y_min = float(np.min(relevant) - y_margin)
    y_max = float(np.max(relevant) + y_margin)
    figure, axes = plt.subplots(
        len(cycles),
        1,
        figsize=(11, 3.6 * len(cycles)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, current_cycle in enumerate(cycles):
        axis = axes[row, 0]
        record = metrics[metrics["current_cycle"] == current_cycle].iloc[0]
        selected = points[points["current_cycle"] == current_cycle].sort_values(
            "forecast_cycle"
        )
        observed = cell.cycle_numbers < current_cycle
        axis.plot(
            cell.cycle_numbers[observed],
            cell.soh[observed],
            color="#1f77b4",
            lw=1.7,
            label=f"completed SOH through {current_cycle - 1}",
        )
        axis.plot(
            selected["forecast_cycle"],
            selected["actual_soh"],
            color="black",
            lw=1.9,
            label="actual current/future",
        )
        axis.fill_between(
            selected["forecast_cycle"],
            selected["lower_interval"],
            selected["upper_interval"],
            color="#d62728",
            alpha=0.15,
            label="prediction interval",
        )
        axis.plot(
            selected["forecast_cycle"],
            selected["predicted_soh"],
            color="#d62728",
            lw=1.9,
            label="predicted SOH",
        )
        axis.axvline(current_cycle, color="#555555", ls="--", lw=1)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_ylabel("SOH")
        axis.set_title(
            f"Current cycle {current_cycle}, beta={record['beta']:g}, "
            f"history={int(record['history_cycles'])} | "
            f"remaining RMSE={record['remaining_soh_rmse']:.4f}, "
            f"common-horizon RMSE={record['common_horizon_soh_rmse']:.4f}"
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("Cycle")
    figure.suptitle(
        f"{cell.cell_id}: non-NP forecast as completed-cycle context grows",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _plot_overlay(
    cell: CellData,
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(cell.cycle_numbers, cell.soh, color="black", lw=2, label="actual SOH")
    cycles = sorted(metrics["current_cycle"].unique())
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(cycles)))
    for color, current_cycle in zip(colors, cycles):
        selected = points[points["current_cycle"] == current_cycle].sort_values(
            "forecast_cycle"
        )
        record = metrics[metrics["current_cycle"] == current_cycle].iloc[0]
        axis.plot(
            selected["forecast_cycle"],
            selected["predicted_soh"],
            color=color,
            lw=1.8,
            label=(
                f"current {current_cycle}, context through {current_cycle - 1}, "
                f"common RMSE={record['common_horizon_soh_rmse']:.4f}"
            ),
        )
        axis.scatter([current_cycle], [selected.iloc[0]["predicted_soh"]], color=color, s=28)
    axis.set_xlim(max(1, min(cycles) - 40), int(cell.cycle_numbers[-1]))
    axis.set_xlabel("Cycle")
    axis.set_ylabel("SOH")
    axis.set_title(f"{cell.cell_id}: forecasts at cycles {', '.join(map(str, cycles))}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def run_rolling_context_demo(
    checkpoint: str | Path,
    *,
    cycles: list[int],
    beta: float,
    cell_id: str | None = None,
    data_root: str | Path | None = None,
    device_name: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0,1)")
    requested_cycles = _validate_cycles(cycles)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "streaming_soh":
        raise ValueError("checkpoint belongs to another algorithm; use the non-NP streaming_soh checkpoint")
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
    test_cells = [by_id[name] for name in split.test_cells if name in by_id]
    scaler = SignalScaler.from_dict(payload["signal_scaler"])
    processor = CycleGridProcessor(
        config.q_grid,
        config.episode.minimum_observed_q_points,
        config.episode.minimum_future_q_points,
    )
    sampler = EpisodeSampler(config.episode, processor, scaler)
    cell = _select_cell(test_cells, sampler, requested_cycles, cell_id)
    episodes = [
        sampler.evaluation(cell, 0.5, beta, current_cycle=current_cycle)
        for current_cycle in requested_cycles
    ]
    model = build_model(config.model).to(device)
    model.load_state_dict(payload["model_state_dict"])
    metrics, points = _predict_rolling_context(
        model,
        episodes,
        device,
        config.episode.cycle_scale,
        config.evaluation.interval_level,
    )
    cycle_tag = "-".join(map(str, requested_cycles))
    beta_tag = f"{beta:g}".replace(".", "p")
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else checkpoint_path.parent.parent
        / "rolling_context"
        / f"{cell.cell_id}_c{cycle_tag}_b{beta_tag}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "rolling_context_metrics.csv", index=False)
    points.to_csv(destination / "rolling_context_predictions.csv", index=False)
    _plot_panels(
        cell,
        metrics,
        points,
        destination / "rolling_context_panels.png",
        config.evaluation.dpi,
    )
    _plot_overlay(
        cell,
        metrics,
        points,
        destination / "rolling_context_overlay.png",
        config.evaluation.dpi,
    )
    write_json(
        destination / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "cell_id": cell.cell_id,
            "current_cycles": requested_cycles,
            "current_cycle_prefix_beta": beta,
            "semantics": (
                "at current cycle k, SOH/V-I from cycles < k are completed context and "
                "only the beta prefix of cycle k V/I is observed"
            ),
            "model_weights_updated": False,
            "common_horizon_start": max(requested_cycles),
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare non-NP forecasts as completed-cycle context grows"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cycles", type=int, nargs="+", default=[130, 135, 140])
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--cell-id")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = run_rolling_context_demo(
        args.checkpoint,
        cycles=args.cycles,
        beta=args.beta,
        cell_id=args.cell_id,
        data_root=args.data_root,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    print(f"Rolling-context output: {destination}")


if __name__ == "__main__":
    main()

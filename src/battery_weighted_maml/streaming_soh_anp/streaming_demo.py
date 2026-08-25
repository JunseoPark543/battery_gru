"""Replay a held-out cycle and update the latent context prior online."""

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
from battery_weighted_maml.streaming_soh.episodes import EpisodeSampler
from battery_weighted_maml.streaming_soh.features import (
    CycleGridProcessor,
    EpisodeUnavailable,
    SignalScaler,
)

from .config import config_from_dict, resolve_data_root
from .model import build_model
from .online import OnlineLatentANPSession


def _select_cell(
    cells: list[CellData], sampler: EpisodeSampler, cycle: int, requested: str | None
) -> CellData:
    candidates = [
        cell
        for cell in cells
        if any(cell.cycles[index].cycle_number == cycle for index in sampler.candidate_indices(cell))
    ]
    if requested:
        selected = [cell for cell in candidates if cell.cell_id == requested]
        if not selected:
            raise EpisodeUnavailable(f"{requested}: cycle {cycle} is not an eligible held-out cut")
        return selected[0]
    if not candidates:
        raise EpisodeUnavailable(f"no held-out cell has eligible cycle {cycle}")
    return sorted(candidates, key=lambda item: item.cell_id)[0]


def run_streaming_demo(
    checkpoint: str | Path,
    *,
    cycle: int,
    betas: list[float],
    cell_id: str | None = None,
    data_root: str | Path | None = None,
    device_name: str | None = None,
    output_dir: str | Path | None = None,
    latent_samples: int | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "streaming_soh_latent_anp":
        raise ValueError("checkpoint belongs to another algorithm")
    if not betas or any(not 0.0 < beta < 1.0 for beta in betas):
        raise ValueError("betas must contain values in (0,1)")
    config = config_from_dict(payload["config"])
    if device_name:
        config.device = device_name
    sample_count = int(latent_samples or config.evaluation.latent_samples)
    if sample_count <= 1:
        raise ValueError("latent_samples must be greater than one")
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
    cell = _select_cell(test_cells, sampler, cycle, cell_id)
    current = cell.cycle_by_number(cycle)
    if current.discharge is None:
        raise EpisodeUnavailable(f"{cell.cell_id} cycle {cycle}: no discharge curve")
    forecast_cycles = cell.cycle_numbers[cell.cycle_numbers >= cycle]
    actual_soh = cell.soh[cell.cycle_numbers >= cycle]
    model = build_model(config.model).to(device)
    model.load_state_dict(payload["model_state_dict"])
    session = OnlineLatentANPSession(
        model,
        processor,
        scaler,
        cell,
        current_cycle=cycle,
        forecast_cycles=forecast_cycles,
        maximum_history_cycles=config.episode.maximum_history_cycles,
        cycle_scale=config.episode.cycle_scale,
        latent_samples=sample_count,
        device=device,
    )
    z_value = NormalDist().inv_cdf(0.5 + config.evaluation.interval_level / 2.0)
    prediction_rows: list[dict] = []
    prior_rows: list[dict] = []
    sorted_betas = sorted(set(float(value) for value in betas))
    forecasts = []
    for beta in sorted_betas:
        cut = int(round(beta * (len(current.discharge.q) - 1))) + 1
        cut = min(max(cut, 2), len(current.discharge.q) - 1)
        forecast = session.observe(
            current.discharge.q[:cut],
            current.discharge.voltage_v[:cut],
            current.discharge.current_a_magnitude[:cut],
        )
        forecasts.append(forecast)
        for index, forecast_cycle in enumerate(forecast.cycles):
            prediction_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "current_cycle": cycle,
                    "beta": beta,
                    "q_prefix_fraction": forecast.prefix_fraction_q_grid,
                    "forecast_cycle": int(forecast_cycle),
                    "actual_soh": float(actual_soh[index]),
                    "predicted_soh": float(forecast.soh_mean[index]),
                    "predictive_std": float(forecast.predictive_std[index]),
                    "epistemic_std": float(forecast.epistemic_std[index]),
                    "aleatoric_std": float(forecast.aleatoric_std[index]),
                    "lower_interval": float(
                        forecast.soh_mean[index] - z_value * forecast.predictive_std[index]
                    ),
                    "upper_interval": float(
                        forecast.soh_mean[index] + z_value * forecast.predictive_std[index]
                    ),
                    "candidate_state_change_l2": forecast.candidate_state_change_l2,
                }
            )
        for dimension, (mean, std) in enumerate(zip(forecast.prior_mean, forecast.prior_std)):
            prior_rows.append(
                {
                    "beta": beta,
                    "q_prefix_fraction": forecast.prefix_fraction_q_grid,
                    "latent_dimension": dimension,
                    "prior_mean": float(mean),
                    "prior_std": float(std),
                }
            )
    frame = pd.DataFrame(prediction_rows)
    prior_frame = pd.DataFrame(prior_rows)
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else checkpoint_path.parent.parent / "streaming" / f"{cell.cell_id}_cycle{cycle}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "streaming_predictions.csv", index=False)
    prior_frame.to_csv(destination / "latent_prior_by_prefix.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    history_mask = cell.cycle_numbers < cycle
    axes[0].plot(
        cell.cycle_numbers[history_mask],
        cell.soh[history_mask],
        color="#1f77b4",
        lw=1.7,
        label="completed observed SOH",
    )
    axes[0].plot(forecast_cycles, actual_soh, color="black", lw=2.0, label="actual future")
    colors = plt.cm.plasma(np.linspace(0.12, 0.9, len(sorted_betas)))
    for color, beta, forecast in zip(colors, sorted_betas, forecasts):
        axes[0].plot(
            forecast.cycles,
            forecast.soh_mean,
            color=color,
            lw=1.6,
            label=f"beta={beta:g}, q={forecast.prefix_fraction_q_grid:.3f}",
        )
    latest = forecasts[-1]
    axes[0].fill_between(
        latest.cycles,
        latest.soh_mean - z_value * latest.predictive_std,
        latest.soh_mean + z_value * latest.predictive_std,
        color=colors[-1],
        alpha=0.15,
        label=f"latest {config.evaluation.interval_level:.0%} interval",
    )
    axes[0].axvline(cycle, color="#555555", ls="--", lw=1)
    axes[0].set_xlabel("Cycle")
    axes[0].set_ylabel("SOH")
    axes[0].set_title(f"{cell.cell_id}: latent ANP forecast updates at cycle {cycle}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    mean_total = [float(np.mean(item.predictive_std)) for item in forecasts]
    mean_epistemic = [float(np.mean(item.epistemic_std)) for item in forecasts]
    mean_aleatoric = [float(np.mean(item.aleatoric_std)) for item in forecasts]
    axes[1].plot(sorted_betas, mean_total, marker="o", label="total predictive std")
    axes[1].plot(sorted_betas, mean_epistemic, marker="o", label="epistemic std")
    axes[1].plot(sorted_betas, mean_aleatoric, marker="o", label="aleatoric std")
    axes[1].set_xlabel("Observed within-cycle fraction beta (offline replay index)")
    axes[1].set_ylabel("Mean SOH uncertainty over forecast horizon")
    axes[1].set_title("Uncertainty update as more V/I samples arrive")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(destination / "streaming_anp_updates.png", dpi=config.evaluation.dpi)
    plt.close(figure)
    write_json(
        destination / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "cell_id": cell.cell_id,
            "current_cycle": cycle,
            "betas": sorted_betas,
            "latent_samples": sample_count,
            "model_weights_updated": False,
            "future_soh_used_by_model": False,
            "update": "context prior and candidate state recomputed from observed prefix",
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay online latent ANP SOH updates")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cycle", type=int, default=130)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--latent-samples", type=int)
    parser.add_argument("--cell-id")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = run_streaming_demo(
        args.checkpoint,
        cycle=args.cycle,
        betas=args.betas,
        cell_id=args.cell_id,
        data_root=args.data_root,
        device_name=args.device,
        output_dir=args.output_dir,
        latent_samples=args.latent_samples,
    )
    print(f"Streaming output: {destination}")


if __name__ == "__main__":
    main()

"""Held-out-cell evaluation and visual summaries for future V-Q surfaces."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from battery_weighted_maml.matr_anp.data import load_dataset
from battery_weighted_maml.matr_anp.runtime import resolve_device, seed_everything, write_json
from battery_weighted_maml.matr_anp.splits import FoldSplit

from .config import config_from_dict, resolve_data_root
from .episodes import EpisodeSampler, FutureVQEpisode, collate_episodes
from .features import CurveGridProcessor, EpisodeUnavailable, VoltageScaler
from .losses import gaussian_metrics
from .model import FutureVQLatentANP, build_model
from .train import ALGORITHM


@torch.no_grad()
def predict_episode(
    model: FutureVQLatentANP,
    episode: FutureVQEpisode,
    *,
    scaler: VoltageScaler,
    q_min: float,
    q_max: float,
    device: torch.device,
    latent_samples: int,
    query_chunk_size: int,
) -> dict[str, np.ndarray]:
    """Predict all future cycles while reusing one coherent latent sample set."""
    batch = collate_episodes([episode]).to(device)
    encoded = model.encode_history(
        batch.history_curve,
        batch.history_endpoint_fraction,
        batch.history_cycle_scaled,
        batch.history_gap_scaled,
        batch.history_mask,
        batch.q_coordinate,
    )
    latent = model.sample_latent(
        encoded["prior_mean"], encoded["prior_std"], latent_samples
    )
    outputs: dict[str, list[torch.Tensor]] = {
        "voltage_mean": [], "voltage_std": [], "voltage_epistemic_std": [],
        "voltage_aleatoric_std": [], "endpoint_mean": [], "endpoint_std": [],
        "endpoint_epistemic_std": [], "endpoint_aleatoric_std": [],
    }
    future_count = len(episode.query_cycle_scaled)
    for start in range(0, future_count, query_chunk_size):
        stop = min(start + query_chunk_size, future_count)
        decoded = model.summarize_samples(
            model.decode_queries(
                encoded,
                batch.query_cycle_scaled[:, start:stop],
                batch.q_coordinate,
                latent,
            )
        )
        for key in outputs:
            outputs[key].append(decoded[key][0].cpu())
    combined = {key: torch.cat(value, dim=0).numpy() for key, value in outputs.items()}
    q = q_min + episode.q_coordinate * (q_max - q_min)
    return {
        "cycles": episode.query_cycle_numbers.copy(),
        "q": q,
        "target_voltage_v": scaler.inverse(episode.target_voltage),
        "target_q_mask": episode.target_q_mask.copy(),
        "target_endpoint_q": episode.target_endpoint_fraction * q_max,
        "voltage_mean_v": scaler.inverse(combined["voltage_mean"]),
        "voltage_std_v": combined["voltage_std"] * scaler.std,
        "voltage_epistemic_std_v": combined["voltage_epistemic_std"] * scaler.std,
        "voltage_aleatoric_std_v": combined["voltage_aleatoric_std"] * scaler.std,
        "endpoint_mean_q": combined["endpoint_mean"] * q_max,
        "endpoint_std_q": combined["endpoint_std"] * q_max,
        "endpoint_epistemic_std_q": combined["endpoint_epistemic_std"] * q_max,
        "endpoint_aleatoric_std_q": combined["endpoint_aleatoric_std"] * q_max,
    }


def _episode_metrics(
    cell_id: str,
    cut_cycle: int,
    prediction: dict[str, np.ndarray],
    interval_level: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    mask = prediction["target_q_mask"]
    target = prediction["target_voltage_v"][mask]
    mean = prediction["voltage_mean_v"][mask]
    std = prediction["voltage_std_v"][mask]
    metrics: dict[str, Any] = {
        "cell_id": cell_id,
        "cut_cycle": cut_cycle,
        "future_cycles": len(prediction["cycles"]),
        **gaussian_metrics(target, mean, std, prefix="voltage"),
    }
    endpoint = gaussian_metrics(
        prediction["target_endpoint_q"],
        prediction["endpoint_mean_q"],
        prediction["endpoint_std_q"],
        prefix="endpoint_q",
    )
    metrics.update(endpoint)
    z = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    metrics[f"voltage_coverage_{int(round(100 * interval_level))}"] = float(
        np.mean(np.abs(target - mean) <= z * std)
    )
    metrics["mean_voltage_epistemic_std_v"] = float(
        np.mean(prediction["voltage_epistemic_std_v"][mask])
    )
    metrics["mean_voltage_aleatoric_std_v"] = float(
        np.mean(prediction["voltage_aleatoric_std_v"][mask])
    )
    per_curve: list[dict[str, Any]] = []
    for row, cycle in enumerate(prediction["cycles"]):
        curve_mask = mask[row]
        curve_metrics = gaussian_metrics(
            prediction["target_voltage_v"][row, curve_mask],
            prediction["voltage_mean_v"][row, curve_mask],
            prediction["voltage_std_v"][row, curve_mask],
            prefix="voltage",
        )
        per_curve.append(
            {
                "cell_id": cell_id,
                "cut_cycle": cut_cycle,
                "future_cycle": int(cycle),
                "horizon": int(cycle - cut_cycle),
                **curve_metrics,
                "target_endpoint_q": float(prediction["target_endpoint_q"][row]),
                "predicted_endpoint_q": float(prediction["endpoint_mean_q"][row]),
                "endpoint_std_q": float(prediction["endpoint_std_q"][row]),
            }
        )
    standardized = (target - mean) / np.maximum(std, 1.0e-8)
    return metrics, per_curve, standardized


def _plot_surface(
    cell_id: str,
    cut_cycle: int,
    prediction: dict[str, np.ndarray],
    path: Path,
    dpi: int,
) -> None:
    target = prediction["target_voltage_v"].copy()
    target[~prediction["target_q_mask"]] = np.nan
    predicted = prediction["voltage_mean_v"].copy()
    predicted[~prediction["target_q_mask"]] = np.nan
    error = np.abs(predicted - target)
    extent = [prediction["q"][0], prediction["q"][-1],
              prediction["cycles"][-1], prediction["cycles"][0]]
    voltage_values = target[np.isfinite(target)]
    vmin, vmax = np.quantile(voltage_values, [0.01, 0.99])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, values, title, limits, cmap in (
        (axes[0], target, "Actual future V-Q surface", (vmin, vmax), "viridis"),
        (axes[1], predicted, "Predicted mean", (vmin, vmax), "viridis"),
        (axes[2], error, "Absolute error", (0.0, np.nanquantile(error, 0.99)), "magma"),
    ):
        image = axis.imshow(
            values, aspect="auto", interpolation="nearest", extent=extent,
            vmin=limits[0], vmax=max(limits[1], limits[0] + 1.0e-8), cmap=cmap,
        )
        axis.set_xlabel("Q / nominal capacity")
        axis.set_ylabel("Cycle")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Voltage (V)" if title != "Absolute error" else "|error| (V)")
    figure.suptitle(f"{cell_id} | observed through cycle {cut_cycle}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _plot_curves(
    cell_id: str,
    cut_cycle: int,
    prediction: dict[str, np.ndarray],
    path: Path,
    dpi: int,
) -> None:
    count = len(prediction["cycles"])
    rows = np.unique(np.linspace(0, count - 1, min(6, count)).round().astype(int))
    columns = 3
    figure, axes = plt.subplots(
        int(np.ceil(len(rows) / columns)), columns, figsize=(13, 3.6 * np.ceil(len(rows) / columns)),
        squeeze=False, constrained_layout=True,
    )
    for axis, row in zip(axes.flat, rows):
        mask = prediction["target_q_mask"][row]
        q = prediction["q"][mask]
        target = prediction["target_voltage_v"][row, mask]
        mean = prediction["voltage_mean_v"][row, mask]
        std = prediction["voltage_std_v"][row, mask]
        axis.plot(q, target, color="black", linewidth=1.4, label="actual")
        axis.plot(q, mean, color="tab:blue", linewidth=1.4, label="predicted")
        axis.fill_between(q, mean - 1.96 * std, mean + 1.96 * std,
                          color="tab:blue", alpha=0.2, label="95% interval")
        axis.axvline(prediction["endpoint_mean_q"][row], color="tab:orange", linestyle="--")
        axis.set_title(
            f"cycle {prediction['cycles'][row]} (h={prediction['cycles'][row] - cut_cycle})"
        )
        axis.set_xlabel("Q / nominal capacity")
        axis.set_ylabel("Voltage (V)")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(rows):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(f"{cell_id} | future V-Q curves after cycle {cut_cycle}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _plot_calibration(standardized: np.ndarray, path: Path, dpi: int) -> None:
    nominal = np.asarray([0.5, 0.68, 0.8, 0.9, 0.95, 0.99])
    observed = []
    for level in nominal:
        z = NormalDist().inv_cdf(0.5 + float(level) / 2.0)
        observed.append(np.mean(np.abs(standardized) <= z))
    figure, axis = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    axis.plot(nominal, observed, "o-", color="tab:blue", label="model")
    axis.set(xlabel="Nominal coverage", ylabel="Observed coverage", xlim=(0.45, 1.0), ylim=(0.45, 1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Future V-Q uncertainty calibration")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    data_root: str | Path | None = None,
    device_name: str | None = None,
    cut_cycles: list[int] | None = None,
) -> Path:
    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != ALGORITHM:
        raise ValueError("checkpoint belongs to another algorithm")
    config = config_from_dict(payload["config"])
    if device_name:
        config.device = device_name
    root = Path(data_root).resolve() if data_root else resolve_data_root(config, None)
    device = resolve_device(config.device)
    seed_everything(config.seed, config.training.deterministic)
    cells, _ = load_dataset(root, config.data, tolerate_invalid_cells=True)
    split = FoldSplit(**payload["fold_split"])
    split.validate([cell.cell_id for cell in cells])
    scaler = VoltageScaler.from_dict(payload["voltage_scaler"])
    if set(scaler.fit_cell_ids) != set(split.train_cells):
        raise RuntimeError("checkpoint scaler provenance does not match training split")
    processor = CurveGridProcessor(config.q_grid, config.episode.minimum_q_points)
    sampler = EpisodeSampler(config.episode, processor, scaler)
    model = build_model(config.model).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    by_id = {cell.cell_id: cell for cell in cells}
    test_cells = [by_id[cell_id] for cell_id in split.test_cells]
    requested_cuts = cut_cycles or config.episode.evaluation_cut_cycles
    evaluation_dir = source.parent.parent / "evaluation" / source.stem
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    standardized: list[np.ndarray] = []
    plot_count = 0
    skipped: list[dict[str, Any]] = []
    for cell in test_cells:
        for cut in requested_cuts:
            try:
                episode = sampler.evaluation(cell, cut)
            except EpisodeUnavailable as exc:
                skipped.append({"cell_id": cell.cell_id, "cut_cycle": cut, "reason": str(exc)})
                continue
            prediction = predict_episode(
                model,
                episode,
                scaler=scaler,
                q_min=config.q_grid.minimum,
                q_max=config.q_grid.maximum,
                device=device,
                latent_samples=config.evaluation.latent_samples,
                query_chunk_size=config.evaluation.query_chunk_size,
            )
            metrics, per_curve, z_values = _episode_metrics(
                cell.cell_id, cut, prediction, config.evaluation.interval_level
            )
            episode_rows.append(metrics)
            curve_rows.extend(per_curve)
            standardized.append(z_values)
            if plot_count < config.evaluation.plot_cells:
                stem = f"{cell.cell_id}_cut{cut}"
                _plot_surface(
                    cell.cell_id, cut, prediction,
                    evaluation_dir / "plots" / f"{stem}_surface.png",
                    config.evaluation.dpi,
                )
                _plot_curves(
                    cell.cell_id, cut, prediction,
                    evaluation_dir / "plots" / f"{stem}_curves.png",
                    config.evaluation.dpi,
                )
                plot_count += 1
    if not episode_rows:
        raise EpisodeUnavailable("none of the held-out cells support the requested cuts")
    episode_frame = pd.DataFrame(episode_rows)
    curve_frame = pd.DataFrame(curve_rows)
    numeric_columns = [
        column for column in episode_frame.select_dtypes(include=[np.number]).columns
        if column not in {"cut_cycle", "future_cycles"}
    ]
    aggregate = episode_frame.groupby("cut_cycle", as_index=False)[numeric_columns].mean()
    aggregate.insert(1, "episodes", episode_frame.groupby("cut_cycle").size().values)
    overall = {"cut_cycle": "all", "episodes": len(episode_frame)}
    overall.update({column: float(episode_frame[column].mean()) for column in numeric_columns})
    aggregate = pd.concat([aggregate, pd.DataFrame([overall])], ignore_index=True)
    episode_frame.to_csv(evaluation_dir / "episode_metrics.csv", index=False)
    curve_frame.to_csv(evaluation_dir / "curve_metrics.csv", index=False)
    aggregate.to_csv(evaluation_dir / "aggregate_metrics.csv", index=False)
    pd.DataFrame(skipped).to_csv(evaluation_dir / "skipped_episodes.csv", index=False)
    _plot_calibration(
        np.concatenate(standardized), evaluation_dir / "plots" / "calibration.png",
        config.evaluation.dpi,
    )
    write_json(
        evaluation_dir / "summary.json",
        {
            "algorithm": ALGORITHM,
            "checkpoint": str(source),
            "checkpoint_step": int(payload["step"]),
            "best_step": int(payload["best_step"]),
            "test_cells": split.test_cells,
            "requested_cut_cycles": requested_cuts,
            "evaluated_episodes": len(episode_rows),
            "skipped_episodes": len(skipped),
            "latent_samples": config.evaluation.latent_samples,
            "query_chunk_size": config.evaluation.query_chunk_size,
            "target_leakage": "none: inference uses p(z|completed history) only",
            "recursive_feedback": False,
            "aggregate": json.loads(aggregate.to_json(orient="records")),
        },
    )
    return evaluation_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate future V-Q latent ANP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--cut-cycles", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = evaluate_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        device_name=args.device,
        cut_cycles=args.cut_cycles,
    )
    print(f"Evaluation directory: {output}")


if __name__ == "__main__":
    main()

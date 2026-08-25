"""Replay one held-out cycle as an online V/I stream without gradient updates."""

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
from .episodes import EpisodeSampler, collate_episodes
from .features import CycleGridProcessor, EpisodeUnavailable, SignalScaler
from .model import build_model
from .train import model_forward


def _select_cell(
    cells: list[CellData], sampler: EpisodeSampler, cycle: int, requested: str | None
) -> CellData:
    candidates = [cell for cell in cells if any(item.cycle_number == cycle for item in cell.cycles)]
    candidates = [
        cell
        for cell in candidates
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
) -> Path:
    checkpoint_path = Path(checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "streaming_soh":
        raise ValueError("checkpoint belongs to another algorithm")
    if not betas or any(not 0.0 < beta < 1.0 for beta in betas):
        raise ValueError("betas must contain values in (0,1)")
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
    cell = _select_cell(test_cells, sampler, cycle, cell_id)
    episodes = [
        sampler.evaluation(cell, 0.5, beta, current_cycle=cycle) for beta in sorted(set(betas))
    ]
    model = build_model(config.model).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    batch = collate_episodes(episodes).to(device)
    with torch.no_grad():
        output = model_forward(model, batch)
    prediction = output["soh_mean"].float().cpu().numpy()
    uncertainty = output["soh_std"].float().cpu().numpy()
    completed_state = output["completed_state"].float().cpu().numpy()
    candidate_state = output["candidate_state"].float().cpu().numpy()
    if not np.allclose(completed_state, completed_state[:1], atol=1.0e-6):
        raise RuntimeError("completed history state changed across prefixes of the same cycle")
    z_value = NormalDist().inv_cdf(0.5 + config.evaluation.interval_level / 2.0)
    rows: list[dict] = []
    for row, episode in enumerate(episodes):
        cycles = np.rint(
            episode.query_cycle_scaled * config.episode.cycle_scale
        ).astype(np.int64)
        state_change = float(np.linalg.norm(candidate_state[row] - completed_state[row]))
        for index, forecast_cycle in enumerate(cycles):
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "current_cycle": cycle,
                    "beta": episode.beta,
                    "q_cut": episode.q_cut,
                    "forecast_cycle": int(forecast_cycle),
                    "actual_soh": float(episode.target_soh[index]),
                    "predicted_soh": float(prediction[row, index]),
                    "predicted_std": float(uncertainty[row, index]),
                    "lower_interval": float(prediction[row, index] - z_value * uncertainty[row, index]),
                    "upper_interval": float(prediction[row, index] + z_value * uncertainty[row, index]),
                    "candidate_state_change_l2": state_change,
                }
            )
    frame = pd.DataFrame(rows)
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else checkpoint_path.parent.parent / "streaming" / f"{cell.cell_id}_cycle{cycle}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "streaming_predictions.csv", index=False)

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    history_cycles = cell.cycle_numbers[cell.cycle_numbers < cycle]
    history_soh = cell.soh[cell.cycle_numbers < cycle]
    axis.plot(history_cycles, history_soh, color="#1f77b4", lw=1.8, label="completed observed SOH")
    first = episodes[0]
    actual_cycles = np.rint(
        first.query_cycle_scaled * config.episode.cycle_scale
    ).astype(np.int64)
    axis.plot(actual_cycles, first.target_soh, color="black", lw=2.0, label="actual current/future SOH")
    colors = plt.cm.plasma(np.linspace(0.12, 0.9, len(episodes)))
    for color, episode in zip(colors, episodes):
        selected = frame[np.isclose(frame["beta"], episode.beta)]
        axis.plot(
            selected["forecast_cycle"],
            selected["predicted_soh"],
            color=color,
            lw=1.7,
            label=f"prefix beta={episode.beta:g} (q_cut={episode.q_cut:.3f})",
        )
    axis.axvline(cycle, color="#555555", ls="--", lw=1.2, label="streaming current cycle")
    axis.set_xlabel("Cycle")
    axis.set_ylabel("SOH")
    axis.set_title(
        f"{cell.cell_id}: SOH forecast updates as cycle {cycle} V/I samples arrive\n"
        "Model weights fixed; current-cycle candidate state recomputed from completed history"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(destination / "streaming_trajectory_updates.png", dpi=config.evaluation.dpi)
    plt.close(figure)
    write_json(
        destination / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "cell_id": cell.cell_id,
            "current_cycle": cycle,
            "betas": [episode.beta for episode in episodes],
            "model_weights_updated": False,
            "state_rule": "recompute candidate from immutable completed-cycle state for each prefix",
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay streaming SOH updates on one held-out cell")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cycle", type=int, default=130)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
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
    )
    print(f"Streaming output: {destination}")


if __name__ == "__main__":
    main()

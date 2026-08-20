"""Held-out streaming tests with progressively later SOH context cutoffs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import ExperimentConfig, load_config, resolve_data_root, save_config
from .data import CellData, load_matr_dataset
from .episodes import EpisodeSampler
from .evaluate import _validate_checkpoint_config
from .features import EpisodeUnavailable, FoldScalers, PartialIVProcessor
from .inference import predict_episode, prediction_frame, trajectory_metrics
from .model import build_model
from .plotting import plot_context_streaming_summary, plot_context_trajectory_snapshots
from .runtime import (
    configure_logger,
    parameter_checksum,
    resolve_device,
    seed_everything,
    write_json,
)


def cycle_schedule(start_cycle: int, end_cycle: int, step: int) -> list[int]:
    """Return exact requested cycle cutoffs for one streaming schedule."""
    if start_cycle <= 0:
        raise ValueError("start_cycle must be positive")
    if end_cycle < start_cycle:
        return []
    if step <= 0:
        raise ValueError("cycle steps must be positive")
    return list(range(start_cycle, end_cycle + 1, step))


def _unique_positive_steps(steps: Iterable[int]) -> list[int]:
    output: list[int] = []
    for raw in steps:
        step = int(raw)
        if step <= 0:
            raise ValueError("cycle steps must be positive")
        if step not in output:
            output.append(step)
    if not output:
        raise ValueError("at least one cycle step is required")
    return output


def _aggregate(per_cutoff: pd.DataFrame) -> pd.DataFrame:
    valid = per_cutoff[per_cutoff["status"] == "ok"]
    rows: list[dict[str, Any]] = []
    for (schedule, step, cutoff), group in valid.groupby(
        ["schedule", "cycle_step", "requested_observed_cycle"], sort=False
    ):
        rows.append(
            {
                "schedule": schedule,
                "cycle_step": int(step),
                "observed_through_cycle": int(cutoff),
                "num_cells": int(group["cell_id"].nunique()),
                "future_rmse_mean": float(group["future_rmse"].mean()),
                "future_rmse_std": float(group["future_rmse"].std(ddof=1))
                if len(group) > 1 else 0.0,
                "future_rmse_median": float(group["future_rmse"].median()),
                "current_soh_abs_error_mean": float(group["current_soh_abs_error"].mean()),
                "nll_mean": float(group["nll"].mean()),
                "coverage_95_mean": float(group["coverage_95"].mean()),
                "interval_width_95_mean": float(group["interval_width_95"].mean()),
                "num_context_points_mean": float(group["num_context_points"].mean()),
                "num_available_context_points_mean": float(
                    group["num_available_context_points"].mean()
                ),
                "num_target_points_mean": float(group["num_target_points"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _cell_end_cycle(cell: CellData, requested_end: int | None) -> int:
    # A cutoff at or after the final recorded cycle has no future target.
    maximum = int(cell.cycle_numbers[-1]) - 1
    return maximum if requested_end is None else min(int(requested_end), maximum)


def context_streaming_run(
    config: ExperimentConfig,
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    start_cycle: int = 100,
    cycle_steps: Iterable[int] = (10, 1),
    end_cycle: int | None = None,
    cell_ids: Iterable[str] | None = None,
    beta: float = 0.0,
    plot_cell: str | None = None,
    output_dir: str | Path | None = None,
    mc_samples: int | None = None,
    log_interval: int = 25,
) -> Path:
    """Evaluate immutable predictions as observed SOH cycles arrive.

    The union of all requested schedules is inferred only once.  Consequently,
    shared cutoffs (for example cycle 100 in step1 and step10) are exactly
    identical and are merely labelled into both schedules in the output.
    """
    steps = _unique_positive_steps(cycle_steps)
    if start_cycle <= 0:
        raise ValueError("start_cycle must be positive")
    if end_cycle is not None and end_cycle < start_cycle:
        raise ValueError("end_cycle must be at least start_cycle")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")

    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MATR ANP checkpoint not found: {source}")
    payload: dict[str, Any] = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("dataset") != "MATR" or payload.get("algorithm") != "attentive_neural_process":
        raise ValueError("checkpoint is not a MATR Attentive Neural Process checkpoint")
    _validate_checkpoint_config(config, payload)

    sample_count = mc_samples or config.evaluation.mc_samples
    if sample_count <= 0:
        raise ValueError("mc_samples must be positive")
    seed_everything(config.seed, config.training.deterministic)
    device = resolve_device(config.device)
    spec = payload["model_spec"]
    model_name = str(spec["model_name"])
    model, rebuilt = build_model(
        model_name, config.model, resolved_hidden_dim=int(spec["hidden_dim"])
    )
    if rebuilt.parameter_count != int(spec["parameter_count"]):
        raise ValueError("checkpoint/model parameter count mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    state_before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    checksum_before = parameter_checksum(model)

    cells, audit = load_matr_dataset(data_root, config.data, tolerate_invalid_cells=True)
    by_id = {cell.cell_id: cell for cell in cells}
    split = payload["fold_split"]
    test_cells = list(split["test_cells"])
    requested_ids = list(cell_ids) if cell_ids is not None else test_cells
    if not requested_ids:
        raise ValueError("at least one held-out test cell is required")
    invalid = sorted(set(requested_ids) - set(test_cells))
    if invalid:
        raise ValueError(f"streaming cells must belong to the checkpoint test fold: {invalid}")
    missing = sorted(set(requested_ids) - set(by_id))
    if missing:
        raise ValueError(f"streaming test cells are missing from MATR data: {missing}")
    selected_plot_cell = plot_cell or requested_ids[0]
    if selected_plot_cell not in requested_ids:
        raise ValueError("plot_cell must be one of the selected held-out cells")

    scalers = FoldScalers.from_dict(payload["scalers"])
    if set(scalers.fit_cell_ids) & set(test_cells):
        raise ValueError("test-cell leakage detected in checkpoint scalers")
    sampler = EpisodeSampler(
        config.episode,
        PartialIVProcessor(config.q_grid, config.data),
        scalers,
    )

    step_label = "-".join(map(str, steps))
    destination = (
        Path(output_dir).resolve()
        if output_dir
        else source.parent.parent / "streaming_context" / f"c{start_cycle}_steps{step_label}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "plots").mkdir(exist_ok=True)
    save_config(config, destination / "resolved_config.yaml")
    audit.to_csv(destination / "data_audit.csv", index=False)
    logger = configure_logger(destination / "streaming_context.log")
    logger.info(
        "Starting context streaming model=%s cells=%d start=%d end=%s steps=%s beta=%g "
        "mc_samples=%d max_context=%d device=%s",
        model_name, len(requested_ids), start_cycle, end_cycle, steps, beta,
        sample_count, config.episode.max_context_points, device,
    )

    schedules_by_cell: dict[str, dict[int, list[int]]] = {}
    total_unique = 0
    for cell_id in requested_ids:
        effective_end = _cell_end_cycle(by_id[cell_id], end_cycle)
        schedules = {
            step: cycle_schedule(start_cycle, effective_end, step) for step in steps
        }
        schedules_by_cell[cell_id] = schedules
        total_unique += len(set().union(*(set(values) for values in schedules.values())))
    if total_unique == 0:
        raise ValueError("no requested cell has a future target after start_cycle")

    metric_rows: list[dict[str, Any]] = []
    snapshot_frames: list[pd.DataFrame] = []
    fold = int(split["fold"])
    completed = 0
    progress = tqdm(total=total_unique, desc="context-streaming", unit="cutoff")
    for cell_index, cell_id in enumerate(requested_ids):
        cell = by_id[cell_id]
        schedules = schedules_by_cell[cell_id]
        schedule_sets = {step: set(values) for step, values in schedules.items()}
        snapshot_sets = {step: set(values[:3]) for step, values in schedules.items()}
        requested_cutoffs = sorted(set().union(*schedule_sets.values()))
        for requested_cutoff in requested_cutoffs:
            available = int(np.searchsorted(
                cell.cycle_numbers, requested_cutoff, side="right"
            ))
            actual_observed = (
                int(cell.cycle_numbers[available - 1]) if available > 0 else None
            )
            base: dict[str, Any] = {
                "fold": fold,
                "seed": config.seed,
                "model": model_name,
                "cell_id": cell_id,
                "beta": beta,
                "requested_observed_cycle": requested_cutoff,
                "observed_through_cycle": actual_observed,
                "num_available_context_points": available,
            }
            episode = None
            result = None
            try:
                episode = sampler.evaluation_after_cycle(cell, requested_cutoff, beta)
                result = predict_episode(
                    model,
                    episode,
                    scalers,
                    device,
                    mc_samples=sample_count,
                    interval_level=config.evaluation.interval_level,
                    # Keep MC draws paired as context changes for the same cell.
                    seed=config.seed + fold * 1_000_003 + cell_index * 10_007,
                )
                base.update(
                    {
                        "status": "ok",
                        "reason": "",
                        **trajectory_metrics(episode, result),
                        "forecast_start_cycle": episode.current_cycle,
                        "num_context_points": len(episode.context_x),
                        "num_target_points": len(episode.target_x),
                    }
                )
            except EpisodeUnavailable as exc:
                base.update(
                    {
                        "status": "skipped",
                        "reason": str(exc),
                        "future_rmse": np.nan,
                        "current_soh_abs_error": np.nan,
                        "nll": np.nan,
                        "coverage_95": np.nan,
                        "interval_width_95": np.nan,
                        "forecast_start_cycle": np.nan,
                        "num_context_points": np.nan,
                        "num_target_points": np.nan,
                    }
                )

            for step in steps:
                if requested_cutoff not in schedule_sets[step]:
                    continue
                schedule = f"step{step}"
                metric_rows.append({**base, "schedule": schedule, "cycle_step": step})
                if (
                    cell_id == selected_plot_cell
                    and requested_cutoff in snapshot_sets[step]
                    and episode is not None
                    and result is not None
                ):
                    frame = prediction_frame(
                        episode,
                        result,
                        scalers,
                        model_name=model_name,
                        fold=fold,
                        seed=config.seed,
                    )
                    frame["schedule"] = schedule
                    frame["cycle_step"] = step
                    frame["requested_observed_cycle"] = requested_cutoff
                    frame["observed_through_cycle"] = actual_observed
                    snapshot_frames.append(frame)

            completed += 1
            progress.update(1)
            progress.set_postfix(cell=cell_id, cycle=requested_cutoff)
            if completed == 1 or completed % log_interval == 0 or completed == total_unique:
                logger.info(
                    "progress=%d/%d cell=%s requested_cycle=%d actual_observed=%s "
                    "context=%s/%d target=%s rmse=%s status=%s",
                    completed, total_unique, cell_id, requested_cutoff, actual_observed,
                    base["num_context_points"], available, base["num_target_points"],
                    base["future_rmse"], base["status"],
                )
    progress.close()

    per_cutoff = pd.DataFrame(metric_rows)
    aggregate = _aggregate(per_cutoff)
    snapshots = (
        pd.concat(snapshot_frames, ignore_index=True)
        if snapshot_frames else pd.DataFrame()
    )
    per_cutoff.to_csv(destination / "per_cutoff_metrics.csv", index=False)
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    snapshots.to_csv(destination / "trajectory_snapshots.csv", index=False)
    if not aggregate.empty:
        plot_context_streaming_summary(
            aggregate, destination / "plots" / "streaming_context_summary.png"
        )
    if not snapshots.empty:
        for step in steps:
            plot_context_trajectory_snapshots(
                snapshots,
                destination / "plots" / f"trajectory_{selected_plot_cell}_step{step}.png",
                cell_id=selected_plot_cell,
                schedule=f"step{step}",
            )

    checksum_after = parameter_checksum(model)
    exactly_equal = all(
        torch.equal(state_before[name], value.detach().cpu())
        for name, value in model.state_dict().items()
    )
    if checksum_before != checksum_after or not exactly_equal:
        raise RuntimeError("streaming context inference changed model parameters or buffers")
    manifest = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(source),
        "dataset": "MATR",
        "model": model_name,
        "fold": fold,
        "seed": config.seed,
        "test_cells": requested_ids,
        "plot_cell": selected_plot_cell,
        "start_cycle": start_cycle,
        "end_cycle": end_cycle,
        "cycle_steps": steps,
        "beta": beta,
        "mc_samples": sample_count,
        "context_policy": "all available through cutoff, uniformly capped at training max_context_points",
        "max_context_points": config.episode.max_context_points,
        "test_time_optimizer": False,
        "backward_called": False,
        "num_unique_inferences": total_unique,
        "num_metric_rows": len(per_cutoff),
        "num_skipped_rows": int((per_cutoff["status"] != "ok").sum()),
        "parameter_checksum_before": checksum_before,
        "parameter_checksum_after": checksum_after,
        "parameters_exactly_equal": exactly_equal,
    }
    write_json(destination / "streaming_context_manifest.json", manifest)
    logger.info("Completed context streaming: %s", destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a fixed MATR ANP checkpoint at expanding SOH cycle cutoffs"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--start-cycle", type=int, default=100)
    parser.add_argument("--end-cycle", type=int)
    parser.add_argument("--cycle-steps", nargs="+", type=int, default=[10, 1])
    parser.add_argument("--cell-id", dest="cell_ids", nargs="+")
    parser.add_argument("--plot-cell")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    destination = context_streaming_run(
        config,
        args.checkpoint,
        resolve_data_root(config, args.data_root),
        start_cycle=args.start_cycle,
        end_cycle=args.end_cycle,
        cycle_steps=args.cycle_steps,
        cell_ids=args.cell_ids,
        beta=args.beta,
        plot_cell=args.plot_cell,
        output_dir=args.output_dir,
        mc_samples=args.mc_samples,
        log_interval=args.log_interval,
    )
    print(f"Streaming context directory: {destination}")


if __name__ == "__main__":
    main()

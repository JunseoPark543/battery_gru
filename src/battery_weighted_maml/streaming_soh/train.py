"""Train the hierarchical streaming SOH trajectory model."""

from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from battery_weighted_maml.matr_anp.data import CellData, load_dataset
from battery_weighted_maml.matr_anp.runtime import (
    configure_logger,
    git_commit,
    resolve_device,
    seed_everything,
    write_json,
)
from battery_weighted_maml.matr_anp.splits import FoldSplit, make_splits, save_splits

from .config import ExperimentConfig, load_config, resolve_data_root, save_config
from .episodes import EpisodeSampler, StreamingBatch, StreamingEpisode, collate_episodes
from .features import CycleGridProcessor, EpisodeUnavailable, SignalScaler, fit_signal_scaler
from .losses import streaming_soh_loss
from .model import StreamingSOHForecaster, build_model


def model_forward(
    model: StreamingSOHForecaster, batch: StreamingBatch
) -> dict[str, torch.Tensor]:
    return model(
        batch.history_curve,
        batch.history_soh,
        batch.history_gap,
        batch.history_cycle_scaled,
        batch.history_mask,
        batch.current_curve,
        batch.q_coordinate,
        batch.current_gap,
        batch.current_cycle_scaled,
        batch.prefix_fraction,
        batch.query_cycle_scaled,
    )


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _capture_rng(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(rng.bit_generator.state),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(raw: dict[str, Any], rng: np.random.Generator) -> None:
    random.setstate(raw["python"])
    rng.bit_generator.state = raw["numpy"]
    torch.set_rng_state(raw["torch"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and raw.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu().to(torch.uint8) for item in raw["cuda"]])


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_step: int,
    best_rmse: float,
    stale: int,
    config: ExperimentConfig,
    split: FoldSplit,
    scaler: SignalScaler,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "algorithm": "streaming_soh",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_step": best_step,
        "best_validation_soh_rmse": best_rmse,
        "stale_validations": stale,
        "config": config.to_dict(),
        "fold_split": asdict(split),
        "signal_scaler": scaler.to_dict(),
        "rng_states": _capture_rng(rng),
        "git_commit": git_commit(),
    }


def _sample_batch(
    cells: Sequence[CellData],
    sampler: EpisodeSampler,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list[StreamingEpisode], int]:
    episodes: list[StreamingEpisode] = []
    attempts = 0
    while len(episodes) < batch_size and attempts < batch_size * 30:
        attempts += 1
        cell = cells[int(rng.integers(0, len(cells)))]
        try:
            episodes.append(sampler.training(cell, rng))
        except EpisodeUnavailable:
            continue
    if len(episodes) != batch_size:
        raise EpisodeUnavailable(
            f"sampled only {len(episodes)}/{batch_size} episodes in {attempts} attempts"
        )
    return episodes, attempts


def _validation_entries(
    cells: Sequence[CellData], sampler: EpisodeSampler, config: ExperimentConfig
) -> list[tuple[str, StreamingEpisode]]:
    entries: list[tuple[str, StreamingEpisode]] = []
    for cell in cells:
        for alpha in config.episode.evaluation_cycle_alphas:
            for beta in config.episode.evaluation_betas:
                try:
                    entries.append((cell.cell_id, sampler.evaluation(cell, alpha, beta)))
                except EpisodeUnavailable:
                    continue
    if not entries:
        raise EpisodeUnavailable("validation produced no streaming episodes")
    return entries


def validation_metrics(
    model: StreamingSOHForecaster,
    cells: Sequence[CellData],
    sampler: EpisodeSampler,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    accumulators: dict[str, dict[str, float]] = {}
    entries = _validation_entries(cells, sampler, config)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(entries), 8):
            chunk = entries[start : start + 8]
            batch = collate_episodes([item[1] for item in chunk]).to(device)
            output = model_forward(model, batch)
            prediction = output["soh_mean"].float().cpu().numpy()
            target = batch.target_soh.float().cpu().numpy()
            mask = batch.query_mask.cpu().numpy()
            for row, (cell_id, _) in enumerate(chunk):
                error = prediction[row, mask[row]] - target[row, mask[row]]
                state = accumulators.setdefault(cell_id, {"squared": 0.0, "count": 0.0})
                state["squared"] += float(np.sum(np.square(error)))
                state["count"] += float(error.size)
    model.train()
    per_cell = {
        cell_id: float(np.sqrt(values["squared"] / values["count"]))
        for cell_id, values in accumulators.items()
        if values["count"] > 0
    }
    if not per_cell:
        raise EpisodeUnavailable("validation SOH masks were empty")
    return float(np.mean(list(per_cell.values()))), per_cell


def train_run(
    config: ExperimentConfig,
    fold: int,
    data_root: str | Path,
    *,
    resume: str | Path | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
) -> Path:
    resolved = copy.deepcopy(config)
    if max_steps is not None:
        resolved.training.max_steps = int(max_steps)
    if batch_size is not None:
        resolved.training.batch_size = int(batch_size)
    resolved.validate()
    seed_everything(resolved.seed, resolved.training.deterministic)
    device = resolve_device(resolved.device)
    if device.type == "cuda" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    cells, audit = load_dataset(data_root, resolved.data, tolerate_invalid_cells=True)
    splits = make_splits([cell.cell_id for cell in cells], resolved.split)
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must lie in [0,{len(splits) - 1}]")
    split = splits[fold]
    by_id = {cell.cell_id: cell for cell in cells}
    train_cells = [by_id[cell_id] for cell_id in split.train_cells]
    validation_cells = [by_id[cell_id] for cell_id in split.validation_cells]
    scaler = fit_signal_scaler(train_cells)
    if set(scaler.fit_cell_ids) != set(split.train_cells):
        raise RuntimeError("signal scaler was not fit on exactly the training cells")
    processor = CycleGridProcessor(
        resolved.q_grid,
        resolved.episode.minimum_observed_q_points,
        resolved.episode.minimum_future_q_points,
    )
    sampler = EpisodeSampler(resolved.episode, processor, scaler)
    run_name = f"{resolved.data.dataset.lower()}_stream_soh_f{fold}_s{resolved.seed}"
    if resume:
        checkpoint_path = Path(resume).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        run_dir = checkpoint_path.parent.parent
    else:
        run_dir = Path(resolved.paths.output_root).resolve() / run_name
        if (run_dir / "checkpoints").exists() and list((run_dir / "checkpoints").glob("*.pt")):
            raise FileExistsError(f"run already has checkpoints: {run_dir}; use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "train.log")
    audit.to_csv(run_dir / "data_audit.csv", index=False)
    save_splits(splits, run_dir / "splits.json", resolved.split)
    save_config(resolved, run_dir / "config_resolved.yaml")

    model = build_model(resolved.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved.training.learning_rate,
        weight_decay=resolved.training.weight_decay,
    )
    rng = np.random.default_rng(resolved.seed + 1009 * (fold + 1))
    start_step, best_step, stale = 1, 0, 0
    best_rmse = float("inf")
    if resume:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != "streaming_soh":
            raise ValueError("resume checkpoint belongs to another algorithm")
        if payload["fold_split"] != asdict(split):
            raise ValueError("resume checkpoint uses a different cell split")
        if payload["signal_scaler"] != scaler.to_dict():
            raise ValueError("resume data produce a different train-only signal scaler")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload["step"]) + 1
        best_step = int(payload["best_step"])
        best_rmse = float(payload["best_validation_soh_rmse"])
        stale = int(payload["stale_validations"])
        _restore_rng(payload["rng_states"], rng)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "start dataset=%s fold=%d device=%s train_cells=%d validation_cells=%d test_cells=%d "
        "parameters=%d batch=%d max_steps=%d q_points=%d history_max=%d "
        "input=completed_soh_vi_history|streaming_current_vi_prefix output=current_and_future_soh",
        resolved.data.dataset.upper(), fold, device, len(train_cells), len(validation_cells),
        len(split.test_cells), parameters, resolved.training.batch_size,
        resolved.training.max_steps, resolved.q_grid.num_points,
        resolved.episode.maximum_history_cycles,
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "status": "running",
            "algorithm": "streaming_soh",
            "run_name": run_name,
            "fold": fold,
            "causal_rule": "each current-cycle prefix is recomputed from immutable completed history",
            "inputs": ["completed_cycle_soh", "completed_cycle_v_i_q", "current_v_i_q_prefix"],
            "outputs": ["current_and_future_soh_mean_std", "current_curve_completion", "q_end"],
        },
    )
    history_path = run_dir / "training_history.csv"
    records = pd.read_csv(history_path).to_dict("records") if resume and history_path.is_file() else []
    started = time.perf_counter()
    progress = tqdm(
        range(start_step, resolved.training.max_steps + 1), desc=run_name, unit="step"
    )
    last_step = start_step - 1
    for step in progress:
        sample_started = time.perf_counter()
        episodes, attempts = _sample_batch(
            train_cells, sampler, resolved.training.batch_size, rng
        )
        sample_seconds = time.perf_counter() - sample_started
        batch = collate_episodes(episodes).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model_forward(model, batch)
        loss, components = streaming_soh_loss(
            output,
            batch,
            soh_huber_delta=resolved.training.soh_huber_delta,
            voltage_huber_delta=resolved.training.voltage_huber_delta,
            endpoint_huber_delta=resolved.training.endpoint_huber_delta,
            uncertainty_weight=resolved.training.uncertainty_weight,
            voltage_completion_weight=resolved.training.voltage_completion_weight,
            endpoint_weight=resolved.training.endpoint_weight,
            monotonic_weight=resolved.training.monotonic_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), resolved.training.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()

        validate_now = (
            step == 1
            or step % resolved.training.validation_interval == 0
            or step == resolved.training.max_steps
        )
        validation_rmse = float("nan")
        per_cell: dict[str, float] = {}
        improved = False
        if validate_now:
            validation_rmse, per_cell = validation_metrics(
                model, validation_cells, sampler, resolved, device
            )
            improved = validation_rmse < best_rmse
            if improved:
                best_rmse, best_step, stale = validation_rmse, step, 0
            else:
                stale += 1
        elapsed = time.perf_counter() - started
        record = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "soh_huber": float(components["soh_huber"].detach().cpu()),
            "uncertainty_nll": float(components["uncertainty_nll"].detach().cpu()),
            "future_voltage": float(components["future_voltage"].detach().cpu()),
            "endpoint": float(components["endpoint"].detach().cpu()),
            "monotonic": float(components["monotonic"].detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "validation_soh_rmse": validation_rmse,
            "best_validation_soh_rmse": best_rmse,
            "best_step": best_step,
            "stale_validations": stale,
            "elapsed_seconds": elapsed,
            "sampling_seconds": sample_seconds,
            "cells": "|".join(item.cell_id for item in episodes),
            "current_cycles": "|".join(str(item.current_cycle) for item in episodes),
            "betas": "|".join(f"{item.beta:.4f}" for item in episodes),
            "history_counts": "|".join(str(len(item.history_soh)) for item in episodes),
            "forecast_counts": "|".join(str(len(item.target_soh)) for item in episodes),
        }
        records.append(record)
        checkpoint = _checkpoint_payload(
            model, optimizer, step, best_step, best_rmse, stale, resolved, split, scaler, rng
        )
        if improved:
            _atomic_save(checkpoint, run_dir / "checkpoints" / "best.pt")
        if (
            validate_now
            or step % resolved.training.checkpoint_interval == 0
            or step == resolved.training.max_steps
        ):
            _atomic_save(checkpoint, run_dir / "checkpoints" / "last.pt")
            pd.DataFrame(records).to_csv(history_path, index=False)
        progress.set_postfix(
            loss=f"{record['loss']:.4g}",
            val=f"{validation_rmse:.4g}" if validate_now else "-",
            refresh=False,
        )
        if step % resolved.training.log_interval == 0 or validate_now:
            eta = elapsed / max(step - start_step + 1, 1) * (resolved.training.max_steps - step)
            logger.info(
                "step=%d/%d loss=%.7g soh=%.7g nll=%.7g future_v=%.7g qend=%.7g "
                "mono=%.7g grad=%.6g val_soh_rmse=%s best=%.7g@%d stale=%d "
                "elapsed=%.1fs eta=%.1fs sample_ms=%.2f attempts=%d cells=%s "
                "current_cycles=%s betas=%s history=%s forecast=%s",
                step, resolved.training.max_steps, record["loss"], record["soh_huber"],
                record["uncertainty_nll"], record["future_voltage"], record["endpoint"],
                record["monotonic"], record["gradient_norm"],
                f"{validation_rmse:.7g}" if validate_now else "-", best_rmse, best_step,
                stale, elapsed, eta, 1000.0 * sample_seconds, attempts, record["cells"],
                record["current_cycles"], record["betas"], record["history_counts"],
                record["forecast_counts"],
            )
            if validate_now:
                logger.info(
                    "validation_per_cell step=%d values=%s",
                    step,
                    "|".join(f"{cell}:{value:.7g}" for cell, value in per_cell.items()),
                )
        last_step = step
        if validate_now and stale >= resolved.training.early_stopping_patience:
            logger.info(
                "early stopping step=%d best_soh_rmse=%.7g@%d", step, best_rmse, best_step
            )
            break
    pd.DataFrame(records).to_csv(history_path, index=False)
    write_json(
        run_dir / "run_manifest.json",
        {
            "status": "completed",
            "algorithm": "streaming_soh",
            "run_name": run_name,
            "last_step": last_step,
            "best_step": best_step,
            "best_validation_soh_rmse": best_rmse,
        },
    )
    logger.info("completed run=%s best_soh_rmse=%.7g@%d", run_dir, best_rmse, best_step)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train streaming V/I-conditioned SOH model")
    parser.add_argument("--config", default="configs/matr_streaming_soh.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    root = resolve_data_root(config, args.data_root)
    run_dir = train_run(
        config,
        args.fold,
        root,
        resume=args.resume,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

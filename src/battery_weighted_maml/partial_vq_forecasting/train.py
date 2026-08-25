"""Train a cell-split partial V-Q curve completion model."""

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
from .episodes import EpisodeSampler, VQEpisode, collate_episodes
from .features import EpisodeUnavailable, PartialVQProcessor, VoltageScaler, fit_voltage_scaler
from .losses import forecasting_loss
from .model import build_model


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
    amp_scaler: Any,
    step: int,
    best_step: int,
    best_rmse: float,
    stale: int,
    config: ExperimentConfig,
    split: FoldSplit,
    voltage_scaler: VoltageScaler,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "algorithm": "partial_vq_forecasting",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": amp_scaler.state_dict(),
        "step": step,
        "best_step": best_step,
        "best_validation_voltage_rmse_v": best_rmse,
        "stale_validations": stale,
        "config": config.to_dict(),
        "fold_split": asdict(split),
        "voltage_scaler": voltage_scaler.to_dict(),
        "rng_states": _capture_rng(rng),
        "git_commit": git_commit(),
    }


def _sample_batch(
    cells: Sequence[CellData],
    sampler: EpisodeSampler,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list[VQEpisode], int]:
    episodes: list[VQEpisode] = []
    attempts = 0
    while len(episodes) < batch_size and attempts < batch_size * 20:
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


def validation_metrics(
    model: nn.Module,
    cells: Sequence[CellData],
    sampler: EpisodeSampler,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    entries: list[tuple[str, VQEpisode]] = []
    for cell in cells:
        for alpha in config.episode.evaluation_cycle_alphas:
            for beta in config.episode.evaluation_betas:
                try:
                    entries.append((cell.cell_id, sampler.evaluation(cell, alpha, beta)))
                except EpisodeUnavailable:
                    continue
    if not entries:
        raise EpisodeUnavailable("validation produced no valid episodes")
    accumulators: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(entries), 64):
            chunk = entries[start : start + 64]
            episodes = [item[1] for item in chunk]
            batch = collate_episodes(episodes).to(device)
            output = model(batch.input_feature, batch.q_coordinate)
            predicted = sampler.scaler.inverse(output["voltage"].float().cpu().numpy())
            predicted_endpoint = (
                output["endpoint_fraction"].float().cpu().numpy() * sampler.processor.q_max
            )
            for index, (cell_id, episode) in enumerate(chunk):
                state = accumulators.setdefault(
                    cell_id, {"squared": 0.0, "count": 0.0, "endpoint": 0.0, "episodes": 0.0}
                )
                target = sampler.scaler.inverse(episode.target_voltage)
                error = predicted[index, episode.future_mask] - target[episode.future_mask]
                state["squared"] += float(np.sum(np.square(error)))
                state["count"] += float(error.size)
                state["endpoint"] += abs(float(predicted_endpoint[index]) - episode.q_end)
                state["episodes"] += 1.0
    model.train()
    per_cell = {
        cell_id: (
            float(np.sqrt(values["squared"] / values["count"])),
            values["endpoint"] / values["episodes"],
        )
        for cell_id, values in accumulators.items()
        if values["count"] > 0 and values["episodes"] > 0
    }
    if not per_cell:
        raise EpisodeUnavailable("validation future masks were empty")
    voltage_rmse = float(np.mean([value[0] for value in per_cell.values()]))
    endpoint_mae = float(np.mean([value[1] for value in per_cell.values()]))
    return voltage_rmse, endpoint_mae, per_cell


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
    cells, audit = load_dataset(data_root, resolved.data, tolerate_invalid_cells=True)
    splits = make_splits([cell.cell_id for cell in cells], resolved.split)
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must lie in [0,{len(splits) - 1}]")
    split = splits[fold]
    by_id = {cell.cell_id: cell for cell in cells}
    train_cells = [by_id[cell_id] for cell_id in split.train_cells]
    validation_cells = [by_id[cell_id] for cell_id in split.validation_cells]
    voltage_scaler = fit_voltage_scaler(
        train_cells, resolved.episode.minimum_cycle_position
    )
    if set(voltage_scaler.fit_cell_ids) != set(split.train_cells):
        raise RuntimeError("voltage scaler was not fit on exactly the training cells")
    processor = PartialVQProcessor(
        resolved.q_grid,
        resolved.episode.minimum_observed_points,
        resolved.episode.minimum_future_points,
    )
    sampler = EpisodeSampler(resolved.episode, processor, voltage_scaler)
    architecture = "attn" if resolved.model.use_attention else "cnn"
    run_name = f"{resolved.data.dataset.lower()}_partial_vq_{architecture}_f{fold}_s{resolved.seed}"
    if resume:
        checkpoint_path = Path(resume).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        run_dir = checkpoint_path.parent.parent
    else:
        run_dir = Path(resolved.paths.output_root).resolve() / run_name
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"run directory already exists: {run_dir}; use --resume or change output_root"
            )
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
    amp_enabled = resolved.training.use_amp and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    rng = np.random.default_rng(resolved.seed + 1009 * (fold + 1))
    start_step, best_step, stale = 1, 0, 0
    best_rmse = float("inf")
    if resume:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != "partial_vq_forecasting":
            raise ValueError("resume checkpoint belongs to another algorithm")
        if payload["fold_split"] != asdict(split):
            raise ValueError("resume checkpoint uses a different cell split")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        amp_scaler.load_state_dict(payload.get("amp_scaler_state_dict", {}))
        start_step = int(payload["step"]) + 1
        best_step = int(payload["best_step"])
        best_rmse = float(payload["best_validation_voltage_rmse_v"])
        stale = int(payload["stale_validations"])
        _restore_rng(payload["rng_states"], rng)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "start dataset=%s fold=%d device=%s train_cells=%d validation_cells=%d test_cells=%d "
        "q=[%.4g,%.4g]x%d attention=%s parameters=%d batch=%d max_steps=%d "
        "input=observed_current_cycle_q_voltage_prefix output=future_voltage_q|q_end",
        resolved.data.dataset.upper(), fold, device, len(train_cells), len(validation_cells),
        len(split.test_cells), processor.q_min, processor.q_max, len(processor.grid),
        resolved.model.use_attention, parameters, resolved.training.batch_size,
        resolved.training.max_steps,
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "status": "running",
            "algorithm": "partial_vq_forecasting",
            "run_name": run_name,
            "fold": fold,
            "inputs": ["observed_q", "observed_voltage", "observation_mask"],
            "outputs": ["future_voltage_at_q", "q_end"],
            "excluded": ["soh", "cycle_number", "future_voltage", "future_q_end"],
        },
    )
    history_path = run_dir / "training_history.csv"
    records = (
        pd.read_csv(history_path).to_dict("records")
        if resume and history_path.is_file()
        else []
    )
    started = time.perf_counter()
    progress = tqdm(
        range(start_step, resolved.training.max_steps + 1), desc=run_name, unit="step"
    )
    last_step = start_step - 1
    for step in progress:
        sampling_started = time.perf_counter()
        episodes, attempts = _sample_batch(
            train_cells, sampler, resolved.training.batch_size, rng
        )
        sampling_seconds = time.perf_counter() - sampling_started
        batch = collate_episodes(episodes).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(batch.input_feature, batch.q_coordinate)
            loss, components = forecasting_loss(
                output,
                batch.target_voltage,
                batch.observed_mask,
                batch.future_mask,
                batch.valid_mask,
                batch.endpoint_fraction,
                voltage_huber_delta=resolved.training.voltage_huber_delta,
                endpoint_huber_delta=resolved.training.endpoint_huber_delta,
                endpoint_weight=resolved.training.endpoint_weight,
                observed_reconstruction_weight=resolved.training.observed_reconstruction_weight,
                monotonic_weight=resolved.training.monotonic_weight,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), resolved.training.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        amp_scaler.step(optimizer)
        amp_scaler.update()

        validate_now = (
            step == 1
            or step % resolved.training.validation_interval == 0
            or step == resolved.training.max_steps
        )
        validation_rmse = float("nan")
        endpoint_mae = float("nan")
        per_cell: dict[str, tuple[float, float]] = {}
        improved = False
        if validate_now:
            validation_rmse, endpoint_mae, per_cell = validation_metrics(
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
            "future_voltage_loss": float(components["future_voltage"].detach().cpu()),
            "observed_reconstruction_loss": float(
                components["observed_reconstruction"].detach().cpu()
            ),
            "endpoint_loss": float(components["endpoint"].detach().cpu()),
            "monotonic_loss": float(components["monotonic"].detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "validation_voltage_rmse_v": validation_rmse,
            "validation_endpoint_mae_q": endpoint_mae,
            "best_validation_voltage_rmse_v": best_rmse,
            "best_step": best_step,
            "stale_validations": stale,
            "elapsed_seconds": elapsed,
            "sampling_seconds": sampling_seconds,
            "cells": "|".join(item.cell_id for item in episodes),
            "cycles": "|".join(str(item.cycle_number) for item in episodes),
            "betas": "|".join(f"{item.beta:.4f}" for item in episodes),
            "q_cuts": "|".join(f"{item.q_cut:.5g}" for item in episodes),
            "q_ends": "|".join(f"{item.q_end:.5g}" for item in episodes),
        }
        records.append(record)
        checkpoint = _checkpoint_payload(
            model, optimizer, amp_scaler, step, best_step, best_rmse, stale,
            resolved, split, voltage_scaler, rng,
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
                "step=%d/%d loss=%.7g future_v=%.7g observed_recon=%.7g endpoint=%.7g "
                "monotonic=%.7g grad=%.6g val_v_rmse=%s val_qend_mae=%s best=%.7g@%d "
                "stale=%d elapsed=%.1fs eta=%.1fs sample_ms=%.2f attempts=%d "
                "cells=%s cycles=%s betas=%s q_cut=%s q_end=%s",
                step, resolved.training.max_steps, record["loss"],
                record["future_voltage_loss"], record["observed_reconstruction_loss"],
                record["endpoint_loss"], record["monotonic_loss"],
                record["gradient_norm"],
                f"{validation_rmse:.7g}" if validate_now else "-",
                f"{endpoint_mae:.7g}" if validate_now else "-",
                best_rmse, best_step, stale, elapsed, eta, 1000.0 * sampling_seconds,
                attempts, record["cells"], record["cycles"], record["betas"],
                record["q_cuts"], record["q_ends"],
            )
            if validate_now:
                logger.info(
                    "validation_per_cell step=%d values=%s",
                    step,
                    "|".join(
                        f"{cell}:v_rmse={values[0]:.7g},qend_mae={values[1]:.7g}"
                        for cell, values in per_cell.items()
                    ),
                )
        last_step = step
        if validate_now and stale >= resolved.training.early_stopping_patience:
            logger.info("early stopping step=%d best_v_rmse=%.7g@%d", step, best_rmse, best_step)
            break
    pd.DataFrame(records).to_csv(history_path, index=False)
    write_json(
        run_dir / "run_manifest.json",
        {
            "status": "completed",
            "algorithm": "partial_vq_forecasting",
            "run_name": run_name,
            "last_step": last_step,
            "best_step": best_step,
            "best_validation_voltage_rmse_v": best_rmse,
        },
    )
    logger.info("completed run=%s best_v_rmse=%.7g@%d", run_dir, best_rmse, best_step)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train within-cycle partial V-Q forecaster")
    parser.add_argument("--config", default="configs/matr_partial_vq_forecasting.yaml")
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
    data_root = resolve_data_root(config, args.data_root)
    run_dir = train_run(
        config,
        args.fold,
        data_root,
        resume=args.resume,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

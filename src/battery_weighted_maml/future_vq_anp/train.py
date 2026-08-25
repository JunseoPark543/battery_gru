"""Training and resumable checkpointing for future V-Q latent ANP."""

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
from tqdm import tqdm

from battery_weighted_maml.matr_anp.data import CellData, load_dataset
from battery_weighted_maml.matr_anp.runtime import (
    configure_logger,
    git_commit,
    parameter_checksum,
    resolve_device,
    seed_everything,
    write_json,
)
from battery_weighted_maml.matr_anp.splits import FoldSplit, make_splits, save_splits

from .config import ExperimentConfig, load_config, resolve_data_root, save_config
from .episodes import EpisodeSampler, FutureVQBatch, collate_episodes
from .features import CurveGridProcessor, EpisodeUnavailable, VoltageScaler, fit_voltage_scaler
from .losses import future_vq_loss, gaussian_metrics
from .model import FutureVQLatentANP, build_model


ALGORITHM = "future_vq_latent_anp"


def model_forward(
    model: FutureVQLatentANP,
    batch: FutureVQBatch,
    *,
    use_posterior: bool,
    num_latent_samples: int,
) -> dict[str, torch.Tensor]:
    return model(
        history_curve=batch.history_curve,
        history_endpoint_fraction=batch.history_endpoint_fraction,
        history_cycle_scaled=batch.history_cycle_scaled,
        history_gap_scaled=batch.history_gap_scaled,
        history_mask=batch.history_mask,
        q_coordinate=batch.q_coordinate,
        query_cycle_scaled=batch.query_cycle_scaled,
        query_mask=batch.query_mask,
        target_voltage=batch.target_voltage if use_posterior else None,
        target_q_mask=batch.target_q_mask if use_posterior else None,
        target_endpoint_fraction=(batch.target_endpoint_fraction if use_posterior else None),
        use_posterior=use_posterior,
        num_latent_samples=num_latent_samples,
    )


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_generator": copy.deepcopy(rng.bit_generator.state),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any], rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
    torch.set_rng_state(torch_state.cpu().to(torch.uint8))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = [
            item if isinstance(item, torch.Tensor) else torch.as_tensor(item, dtype=torch.uint8)
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
    rng.bit_generator.state = state["numpy_generator"]


def _checkpoint_payload(
    model: FutureVQLatentANP,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_step: int,
    best_score: float,
    stale: int,
    config: ExperimentConfig,
    split: FoldSplit,
    scaler: VoltageScaler,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "algorithm": ALGORITHM,
        "step": step,
        "best_step": best_step,
        "best_validation_score": best_score,
        "stale_validations": stale,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "fold_split": asdict(split),
        "voltage_scaler": scaler.to_dict(),
        "rng_states": _rng_state(rng),
        "parameter_checksum": parameter_checksum(model),
        "git_commit": git_commit(),
    }


def _sample_batch(
    cells: Sequence[CellData],
    sampler: EpisodeSampler,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list, int]:
    episodes = []
    attempts = 0
    maximum_attempts = max(50, batch_size * 20)
    while len(episodes) < batch_size and attempts < maximum_attempts:
        cell = cells[int(rng.integers(0, len(cells)))]
        attempts += 1
        try:
            episodes.append(sampler.training(cell, rng))
        except EpisodeUnavailable:
            continue
    if len(episodes) != batch_size:
        raise EpisodeUnavailable(
            f"could sample only {len(episodes)}/{batch_size} episodes in {attempts} attempts"
        )
    return episodes, attempts


@torch.no_grad()
def validation_metrics(
    model: FutureVQLatentANP,
    cells: Sequence[CellData],
    sampler: EpisodeSampler,
    config: ExperimentConfig,
    scaler: VoltageScaler,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    per_cell: dict[str, list[float]] = {}
    all_target: list[np.ndarray] = []
    all_mean: list[np.ndarray] = []
    all_std: list[np.ndarray] = []
    for cell in cells:
        for cut in config.episode.evaluation_cut_cycles:
            try:
                episode = sampler.evaluation(cell, cut)
            except EpisodeUnavailable:
                continue
            # Validation samples the same maximum number of horizons as training.
            if len(episode.query_cycle_scaled) > config.episode.maximum_training_future_cycles:
                indices = EpisodeSampler._even_subsample(
                    np.arange(len(episode.query_cycle_scaled)),
                    config.episode.maximum_training_future_cycles,
                )
                episode = type(episode)(
                    **{
                        **vars(episode),
                        "query_cycle_numbers": episode.query_cycle_numbers[indices],
                        "query_cycle_scaled": episode.query_cycle_scaled[indices],
                        "target_voltage": episode.target_voltage[indices],
                        "target_q_mask": episode.target_q_mask[indices],
                        "target_endpoint_fraction": episode.target_endpoint_fraction[indices],
                    }
                )
            batch = collate_episodes([episode]).to(device)
            output = model_forward(
                model,
                batch,
                use_posterior=False,
                num_latent_samples=config.training.validation_latent_samples,
            )
            mask = batch.target_q_mask[0].cpu().numpy()
            target = scaler.inverse(batch.target_voltage[0].cpu().numpy()[mask])
            mean = scaler.inverse(output["voltage_mean"][0].cpu().numpy()[mask])
            std = output["voltage_std"][0].cpu().numpy()[mask] * scaler.std
            metrics = gaussian_metrics(target, mean, std, prefix="voltage")
            per_cell.setdefault(cell.cell_id, []).append(metrics["voltage_rmse"])
            all_target.append(target)
            all_mean.append(mean)
            all_std.append(std)
    model.train()
    if not all_target:
        raise EpisodeUnavailable("no validation cell supports configured cut cycles")
    aggregate = gaussian_metrics(
        np.concatenate(all_target), np.concatenate(all_mean), np.concatenate(all_std),
        prefix="voltage",
    )
    cell_rmse = {key: float(np.mean(value)) for key, value in per_cell.items()}
    return aggregate["voltage_rmse"], aggregate["voltage_crps"], cell_rmse


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
    scaler = fit_voltage_scaler(train_cells)
    if set(scaler.fit_cell_ids) != set(split.train_cells):
        raise RuntimeError("voltage scaler was not fit on exactly the training cells")
    processor = CurveGridProcessor(resolved.q_grid, resolved.episode.minimum_q_points)
    sampler = EpisodeSampler(resolved.episode, processor, scaler)
    run_name = f"{resolved.data.dataset.lower()}_future_vq_anp_f{fold}_s{resolved.seed}"
    checkpoint_path: Path | None = None
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
    best_score = float("inf")
    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != ALGORITHM:
            raise ValueError("resume checkpoint belongs to another algorithm")
        if payload["fold_split"] != asdict(split):
            raise ValueError("resume checkpoint uses a different cell split")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload["step"]) + 1
        best_step = int(payload["best_step"])
        best_score = float(payload["best_validation_score"])
        stale = int(payload["stale_validations"])
        _restore_rng(payload["rng_states"], rng)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "start algorithm=%s dataset=%s fold=%d device=%s train=%d validation=%d "
        "test=%d parameters=%d history=%d max_future_train=%d q_points=%d batch=%d "
        "max_steps=%d latent=%d",
        ALGORITHM, resolved.data.dataset.upper(), fold, device, len(train_cells),
        len(validation_cells), len(split.test_cells), parameters,
        resolved.episode.history_cycles,
        resolved.episode.maximum_training_future_cycles,
        resolved.q_grid.num_points, resolved.training.batch_size,
        resolved.training.max_steps, resolved.model.latent_dim,
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "status": "running",
            "algorithm": ALGORITHM,
            "run_name": run_name,
            "input": "most recent completed full V-Q curves only",
            "output": "all future V(cycle,Q) curves and q_end",
            "inference": "context prior only; no future target or recursive prediction input",
        },
    )
    history_path = run_dir / "training_history.csv"
    records = pd.read_csv(history_path).to_dict("records") if resume and history_path.is_file() else []
    started = time.perf_counter()
    last_step = start_step - 1
    progress = tqdm(
        range(start_step, resolved.training.max_steps + 1), desc=run_name, unit="step"
    )
    for step in progress:
        sample_started = time.perf_counter()
        episodes, attempts = _sample_batch(
            train_cells, sampler, resolved.training.batch_size, rng
        )
        sample_seconds = time.perf_counter() - sample_started
        batch = collate_episodes(episodes).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model_forward(model, batch, use_posterior=True, num_latent_samples=1)
        kl_coefficient = resolved.training.kl_weight * min(
            1.0, step / float(resolved.training.kl_warmup_steps)
        )
        loss, components = future_vq_loss(
            output,
            batch,
            voltage_huber_delta=resolved.training.voltage_huber_delta,
            voltage_huber_weight=resolved.training.voltage_huber_weight,
            endpoint_huber_delta=resolved.training.endpoint_huber_delta,
            endpoint_weight=resolved.training.endpoint_weight,
            kl_coefficient=kl_coefficient,
            kl_free_bits=resolved.training.kl_free_bits,
            q_monotonic_weight=resolved.training.q_monotonic_weight,
            endpoint_monotonic_weight=resolved.training.endpoint_monotonic_weight,
            temporal_smoothness_weight=resolved.training.temporal_smoothness_weight,
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
        validation_rmse = validation_crps = float("nan")
        per_cell: dict[str, float] = {}
        improved = False
        if validate_now:
            validation_rmse, validation_crps, per_cell = validation_metrics(
                model, validation_cells, sampler, resolved, scaler, device
            )
            score = (
                validation_crps
                if resolved.training.selection_metric == "crps"
                else validation_rmse
            )
            improved = score < best_score
            if improved:
                best_score, best_step, stale = score, step, 0
            else:
                stale += 1
        elapsed = time.perf_counter() - started
        record: dict[str, Any] = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            **{key: float(value.detach().cpu()) for key, value in components.items()},
            "kl_coefficient": kl_coefficient,
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "prior_std": float(output["prior_std"].mean().detach().cpu()),
            "posterior_std": float(output["posterior_std"].mean().detach().cpu()),
            "validation_voltage_rmse_v": validation_rmse,
            "validation_voltage_crps_v": validation_crps,
            "best_validation_score": best_score,
            "best_step": best_step,
            "stale_validations": stale,
            "elapsed_seconds": elapsed,
            "sampling_seconds": sample_seconds,
            "cells": "|".join(item.cell_id for item in episodes),
            "cut_cycles": "|".join(str(item.cut_cycle) for item in episodes),
            "future_counts": "|".join(str(len(item.query_cycle_scaled)) for item in episodes),
        }
        records.append(record)
        checkpoint = _checkpoint_payload(
            model, optimizer, step=step, best_step=best_step, best_score=best_score,
            stale=stale, config=resolved, split=split, scaler=scaler, rng=rng,
        )
        if improved:
            _atomic_save(checkpoint, run_dir / "checkpoints" / "best.pt")
        if validate_now or step % resolved.training.checkpoint_interval == 0:
            _atomic_save(checkpoint, run_dir / "checkpoints" / "last.pt")
            pd.DataFrame(records).to_csv(history_path, index=False)
        progress.set_postfix(
            loss=f"{record['loss']:.4g}",
            val=f"{best_score:.4g}" if validate_now else "-",
            refresh=False,
        )
        if step % resolved.training.log_interval == 0 or validate_now:
            eta = elapsed / max(step - start_step + 1, 1) * (resolved.training.max_steps - step)
            logger.info(
                "step=%d/%d loss=%.7g v_nll=%.7g v_huber=%.7g endpoint_nll=%.7g "
                "kl=%.7g kl_coef=%.6g q_mono=%.7g endpoint_mono=%.7g smooth=%.7g "
                "grad=%.6g prior_std=%.6g posterior_std=%.6g val_rmse_v=%s "
                "val_crps_v=%s best=%.7g@%d stale=%d elapsed=%.1fs eta=%.1fs "
                "sample_ms=%.2f attempts=%d cells=%s cuts=%s future=%s",
                step, resolved.training.max_steps, record["loss"], record["voltage_nll"],
                record["voltage_huber"], record["endpoint_nll"], record["kl"],
                kl_coefficient, record["q_monotonic"], record["endpoint_monotonic"],
                record["temporal_smoothness"], record["gradient_norm"],
                record["prior_std"], record["posterior_std"],
                f"{validation_rmse:.7g}" if validate_now else "-",
                f"{validation_crps:.7g}" if validate_now else "-",
                best_score, best_step, stale, elapsed, eta, 1000.0 * sample_seconds,
                attempts, record["cells"], record["cut_cycles"], record["future_counts"],
            )
            if validate_now:
                logger.info(
                    "validation_per_cell step=%d rmse_v=%s",
                    step,
                    "|".join(f"{cell}:{value:.7g}" for cell, value in per_cell.items()),
                )
        last_step = step
        if validate_now and stale >= resolved.training.early_stopping_patience:
            logger.info("early stopping step=%d best=%.7g@%d", step, best_score, best_step)
            break
    pd.DataFrame(records).to_csv(history_path, index=False)
    write_json(
        run_dir / "run_manifest.json",
        {
            "status": "completed",
            "algorithm": ALGORITHM,
            "run_name": run_name,
            "last_step": last_step,
            "best_step": best_step,
            "selection_metric": resolved.training.selection_metric,
            "best_validation_score": best_score,
        },
    )
    logger.info("completed run=%s best=%.7g@%d", run_dir, best_score, best_step)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train future V-Q latent ANP")
    parser.add_argument("--config", default="configs/matr_future_vq_anp.yaml")
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

"""Training CLI for leakage-safe cell-level MATR ANP folds."""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .config import ExperimentConfig, load_config, resolve_data_root, save_config
from .data import CellData, load_matr_dataset
from .episodes import EpisodeSampler, collate_episodes
from .features import EpisodeUnavailable, FoldScalers, PartialIVProcessor
from .losses import anp_elbo_loss
from .model import MODEL_NAMES, ModelSpec, build_model
from .runtime import configure_logger, git_commit, resolve_device, seed_everything, write_json
from .splits import FoldSplit, make_splits, save_splits


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _capture_rng(rng: np.random.Generator) -> dict[str, Any]:
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
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [item.cpu().to(torch.uint8) for item in state["cuda"]]
        )


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    best_step: int,
    best_validation_rmse: float,
    stale_validations: int,
    config: ExperimentConfig,
    model_spec: ModelSpec,
    fold_split: FoldSplit,
    scalers: FoldScalers,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "algorithm": "attentive_neural_process",
        "dataset": "MATR",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "step": step,
        "best_step": best_step,
        "best_validation_rmse": best_validation_rmse,
        "stale_validations": stale_validations,
        "config": config.to_dict(),
        "model_spec": model_spec.to_dict(),
        "fold_split": asdict(fold_split),
        "scalers": scalers.to_dict(),
        "rng_states": _capture_rng(rng),
        "git_commit": git_commit(),
    }


def _validation_rmse(
    model: nn.Module,
    validation_cells: Sequence[CellData],
    sampler: EpisodeSampler,
    model_name: str,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    model.eval()
    cell_values: list[float] = []
    alphas = config.episode.evaluation_alphas
    requested = config.training.validation_episodes_per_cell
    selected_alphas = [alphas[index % len(alphas)] for index in range(requested)]
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.no_grad(), torch.random.fork_rng(
        devices=fork_devices
    ):
        torch.manual_seed(config.seed + 700_001)
        for cell in validation_cells:
            episode_errors: list[float] = []
            for alpha in selected_alphas:
                beta = 0.5 if model_name == "partial_iv_anp" else 0.0
                try:
                    episode = sampler.evaluation(cell, alpha, beta)
                except EpisodeUnavailable:
                    continue
                batch = collate_episodes([episode]).to(device)
                output = model(
                    batch.context_x,
                    batch.context_y,
                    batch.context_mask,
                    batch.target_x,
                    iv_feature=batch.iv_feature,
                    sample_latent=False,
                )
                predicted = sampler.scalers.inverse_soh(
                    output["mean"][0, : len(episode.target_y), 0].cpu().numpy()
                )
                episode_errors.append(
                    float(np.sqrt(np.mean(np.square(predicted - episode.target_soh_raw))))
                )
            if episode_errors:
                cell_values.append(float(np.mean(episode_errors)))
    model.train()
    if not cell_values:
        raise ValueError("validation produced no valid cell episodes")
    return float(np.mean(cell_values))


def train_run(
    config: ExperimentConfig,
    model_name: str,
    fold: int,
    data_root: str | Path,
    *,
    resume: str | Path | None = None,
    max_steps: int | None = None,
    output_root: str | Path | None = None,
) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    resolved = copy.deepcopy(config)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps override must be positive")
        resolved.training.max_steps = max_steps
    resolved.validate()
    seed_everything(resolved.seed, resolved.training.deterministic)
    cells, audit = load_matr_dataset(
        data_root, resolved.data, tolerate_invalid_cells=True
    )
    splits = make_splits([cell.cell_id for cell in cells], resolved.split)
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in [0,{len(splits) - 1}]")
    split = splits[fold]
    by_id = {cell.cell_id: cell for cell in cells}
    train_cells = [by_id[cell_id] for cell_id in split.train_cells]
    validation_cells = [by_id[cell_id] for cell_id in split.validation_cells]
    processor = PartialIVProcessor(resolved.q_grid, resolved.data)
    scalers = FoldScalers.fit(
        train_cells, processor, resolved.episode.minimum_current_cycle_position - 1
    )
    if set(scalers.fit_cell_ids) != set(split.train_cells):
        raise RuntimeError("scaler fit cells do not exactly equal the training split")
    sampler = EpisodeSampler(resolved.episode, processor, scalers)

    if resume is not None:
        checkpoint_path = Path(resume).resolve()
        run_dir = checkpoint_path.parent.parent
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload["dataset"] != "MATR" or payload["model_spec"]["model_name"] != model_name:
            raise ValueError("resume checkpoint dataset/model mismatch")
        saved_config = payload["config"]
        current_config = resolved.to_dict()
        for section in ("seed", "data", "q_grid", "split", "episode", "model"):
            if saved_config.get(section) != current_config.get(section):
                raise ValueError(f"resume config section '{section}' differs from checkpoint")
        saved_training = dict(saved_config["training"])
        current_training = dict(current_config["training"])
        # Extending max_steps is the supported reason to change training config on resume.
        saved_training.pop("max_steps", None)
        current_training.pop("max_steps", None)
        if saved_training != current_training:
            raise ValueError("resume training config differs from checkpoint (except max_steps)")
        if payload["fold_split"] != asdict(split):
            raise ValueError("resume checkpoint cell split mismatch")
        if payload["scalers"] != scalers.to_dict():
            raise ValueError("resume checkpoint scaler values or training cells mismatch")
        resolved_hidden = int(payload["model_spec"]["hidden_dim"])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_output = Path(output_root or resolved.paths.output_root).resolve()
        run_dir = base_output / f"{timestamp}_fold{fold}_{model_name}_s{resolved.seed}"
        payload = None
        resolved_hidden = None
    for directory in (
        "checkpoints", "training", "logs", "scalers", "audit", "evaluation", "plots"
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "logs/train.log")
    save_config(resolved, run_dir / "resolved_config.yaml")
    save_splits(splits, run_dir / "splits.json", resolved.split)
    scalers.save(run_dir / "scalers/fold_scalers.json")
    audit.to_csv(run_dir / "audit/data_audit.csv", index=False)

    device = resolve_device(resolved.device)
    model, model_spec = build_model(
        model_name, resolved.model, resolved_hidden_dim=resolved_hidden
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved.training.learning_rate)
    amp_enabled = bool(resolved.training.use_amp and device.type == "cuda")
    try:
        amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except AttributeError:  # PyTorch < 2.3 compatibility
        amp_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    rng = np.random.default_rng(resolved.seed + fold * 100_003)
    start_step = 1
    best_step = 0
    best_validation_rmse = float("inf")
    stale_validations = 0
    history_path = run_dir / "training/history.csv"
    records: list[dict[str, Any]] = []
    if payload is not None:
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        amp_scaler.load_state_dict(payload["amp_scaler_state_dict"])
        _restore_rng(payload["rng_states"], rng)
        start_step = int(payload["step"]) + 1
        best_step = int(payload["best_step"])
        best_validation_rmse = float(payload["best_validation_rmse"])
        stale_validations = int(payload["stale_validations"])
        if history_path.is_file():
            records = pd.read_csv(history_path).to_dict("records")
        logger.info("Resuming at step %d", start_step)

    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MATR",
        "algorithm": "attentive_neural_process",
        "model": model_name,
        "model_spec": model_spec.to_dict(),
        "fold": fold,
        "split": asdict(split),
        "scaler_fit_cells": scalers.fit_cell_ids,
        "device": str(device),
        "seed": resolved.seed,
        "git_commit": git_commit(),
        "data_root": str(Path(data_root).resolve()),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    logger.info(
        "MATR fold=%d model=%s parameters=%d hidden=%d train=%s validation=%s test=%s",
        fold, model_name, model_spec.parameter_count, model_spec.hidden_dim,
        split.train_cells, split.validation_cells, split.test_cells,
    )

    began = time.perf_counter()
    progress = tqdm(
        range(start_step, resolved.training.max_steps + 1),
        desc=f"MATR-{model_name}-fold{fold}",
        unit="step",
    )
    last_step = start_step - 1
    for step in progress:
        episodes = []
        attempts = 0
        while len(episodes) < resolved.training.batch_size:
            attempts += 1
            if attempts > resolved.training.batch_size * 50:
                raise RuntimeError("could not sample enough valid MATR training episodes")
            cell = train_cells[int(rng.integers(0, len(train_cells)))]
            try:
                episodes.append(sampler.sample_training(cell, rng))
            except EpisodeUnavailable:
                continue
        batch = collate_episodes(episodes).to(device)
        kl_weight = (
            1.0
            if resolved.training.kl_warmup_steps == 0
            else min(1.0, step / resolved.training.kl_warmup_steps)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(
                batch.context_x,
                batch.context_y,
                batch.context_mask,
                batch.target_x,
                target_y=batch.target_y,
                target_mask=batch.target_mask,
                iv_feature=batch.iv_feature,
            )
            losses = anp_elbo_loss(output, batch.target_y, batch.target_mask, kl_weight)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(
                f"non-finite ANP loss at step {step}: "
                f"loss={float(losses['loss'].detach().cpu())}, "
                f"nll={float(losses['nll'].detach().cpu())}, "
                f"kl={float(losses['kl'].detach().cpu())}"
            )
        amp_scaler.scale(losses["loss"]).backward()
        amp_scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), resolved.training.gradient_clip_norm
            ).detach().cpu()
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        amp_scaler.step(optimizer)
        amp_scaler.update()
        record: dict[str, Any] = {
            "step": step,
            "loss": float(losses["loss"].detach().cpu()),
            "nll": float(losses["nll"].detach().cpu()),
            "kl": float(losses["kl"].detach().cpu()),
            "kl_weight": kl_weight,
            "gradient_norm": gradient_norm,
            "validation_rmse": np.nan,
            "elapsed_seconds": time.perf_counter() - began,
        }

        validate_now = (
            step % resolved.training.validation_interval == 0
            or step == resolved.training.max_steps
        )
        improved = False
        if validate_now:
            validation_rmse = _validation_rmse(
                model, validation_cells, sampler, model_name, resolved, device
            )
            record["validation_rmse"] = validation_rmse
            improved = validation_rmse < best_validation_rmse - 1.0e-10
            if improved:
                best_validation_rmse = validation_rmse
                best_step = step
                stale_validations = 0
            else:
                stale_validations += 1
        records.append(record)
        checkpoint = _checkpoint_payload(
            model, optimizer, amp_scaler, step, best_step, best_validation_rmse,
            stale_validations, resolved, model_spec, split, scalers, rng,
        )
        if improved:
            _atomic_save(checkpoint, run_dir / "checkpoints/best.pt")
        if (
            step % resolved.training.checkpoint_interval == 0
            or validate_now
            or step == resolved.training.max_steps
        ):
            _atomic_save(checkpoint, run_dir / "checkpoints/last.pt")
            pd.DataFrame(records).to_csv(history_path, index=False)
        progress.set_postfix(
            loss=f"{record['loss']:.4g}",
            val=(f"{record['validation_rmse']:.4g}" if validate_now else "-"),
        )
        if step % resolved.training.log_interval == 0 or validate_now or step == start_step:
            logger.info(
                "step=%d/%d loss=%.7g nll=%.7g kl=%.7g kl_weight=%.4f "
                "grad=%.6g validation_rmse=%s best=%.7g@%d stale=%d",
                step, resolved.training.max_steps, record["loss"], record["nll"],
                record["kl"], kl_weight, gradient_norm,
                f"{record['validation_rmse']:.7g}" if validate_now else "-",
                best_validation_rmse, best_step, stale_validations,
            )
        last_step = step
        if validate_now and stale_validations >= resolved.training.early_stopping_patience:
            logger.info("Early stopping at step %d; best step=%d", step, best_step)
            break
    if not (run_dir / "checkpoints/best.pt").is_file():
        raise RuntimeError("training produced no best validation checkpoint")
    pd.DataFrame(records).to_csv(history_path, index=False)
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_step": last_step,
            "best_step": best_step,
            "best_validation_rmse": best_validation_rmse,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    logger.info("Completed MATR ANP run: %s", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one cell-level MATR ANP fold")
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    data_root = resolve_data_root(config, args.data_root)
    run_dir = train_run(
        config, args.model, args.fold, data_root,
        resume=args.resume, max_steps=args.max_steps,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

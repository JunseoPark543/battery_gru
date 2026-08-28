"""Training CLI for leakage-safe cell-level MATR/CALCE ANP folds."""

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

from .config import (
    ExperimentConfig,
    complete_model_config_dict,
    load_config,
    resolve_data_root,
    save_config,
)
from .data import CellData, load_dataset
from .episodes import EpisodeSampler, collate_episodes
from .features import EpisodeUnavailable, FoldScalers, PartialIVProcessor
from .losses import anp_elbo_loss
from .model import HS_MODEL_NAMES, MODEL_NAMES, ModelSpec, build_model
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


def _masked_range(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    selected = values.masked_select(mask.unsqueeze(-1))
    if selected.numel() == 0:
        return float("nan"), float("nan")
    return float(selected.min().detach().cpu()), float(selected.max().detach().cpu())


def _nonfinite_gradient_names(model: nn.Module, limit: int = 8) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _gpu_memory_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated(device) / (1024.0**2),
        torch.cuda.memory_reserved(device) / (1024.0**2),
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
        "dataset": config.data.dataset.upper(),
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
) -> tuple[float, dict[str, float]]:
    model.eval()
    cell_values: list[float] = []
    by_cell: dict[str, float] = {}
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
                    context_signal=batch.context_signal,
                    context_signal_mask=batch.context_signal_mask,
                    sample_latent=False,
                )
                predicted = sampler.scalers.inverse_soh(
                    output["mean"][0, : len(episode.target_y), 0].cpu().numpy()
                )
                episode_errors.append(
                    float(np.sqrt(np.mean(np.square(predicted - episode.target_soh_raw))))
                )
            if episode_errors:
                value = float(np.mean(episode_errors))
                cell_values.append(value)
                by_cell[cell.cell_id] = value
    model.train()
    if not cell_values:
        raise ValueError("validation produced no valid cell episodes")
    return float(np.mean(cell_values)), by_cell


def train_run(
    config: ExperimentConfig,
    model_name: str,
    fold: int,
    data_root: str | Path,
    *,
    resume: str | Path | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
    output_root: str | Path | None = None,
) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    resolved = copy.deepcopy(config)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps override must be positive")
        resolved.training.max_steps = max_steps
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size override must be positive")
        resolved.training.batch_size = batch_size
    resolved.validate()
    seed_everything(resolved.seed, resolved.training.deterministic)
    dataset = resolved.data.dataset.upper()
    cells, audit = load_dataset(
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
    sampler = EpisodeSampler(
        resolved.episode,
        processor,
        scalers,
        # Preserve the pre-existing baseline episode construction exactly.
        # Only HS variants disable current/target-cycle I-V access.
        include_current_iv=model_name not in HS_MODEL_NAMES,
        include_context_signal=model_name in HS_MODEL_NAMES,
    )

    if resume is not None:
        checkpoint_path = Path(resume).resolve()
        run_dir = checkpoint_path.parent.parent
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload["dataset"] != dataset or payload["model_spec"]["model_name"] != model_name:
            raise ValueError("resume checkpoint dataset/model mismatch")
        saved_config = payload["config"]
        current_config = resolved.to_dict()
        for section in ("seed", "data", "q_grid", "split", "episode", "model"):
            saved_section = saved_config.get(section)
            current_section = current_config.get(section)
            if section == "model" and isinstance(saved_section, dict):
                saved_section = complete_model_config_dict(saved_section)
            if saved_section != current_section:
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
        saved_scalers = dict(payload["scalers"])
        current_scalers = scalers.to_dict()
        for field in ("voltage_mean", "voltage_std"):
            if field not in saved_scalers:
                current_scalers.pop(field, None)
        if saved_scalers != current_scalers:
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
        amp_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
            init_scale=resolved.training.amp_initial_scale,
        )
    except (AttributeError, TypeError):  # PyTorch < 2.3 compatibility
        amp_scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled,
            init_scale=resolved.training.amp_initial_scale,
        )
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
        "dataset": dataset,
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
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "feature_cache": processor.cache_info(),
    }
    if device.type == "cuda":
        manifest.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    write_json(run_dir / "run_manifest.json", manifest)
    train_soh = np.concatenate([cell.soh for cell in train_cells])
    logger.info(
        "%s fold=%d model=%s parameters=%d hidden=%d train_cells=%d "
        "validation_cells=%d test_cells=%d",
        dataset,
        fold,
        model_name,
        model_spec.parameter_count,
        model_spec.hidden_dim,
        len(split.train_cells),
        len(split.validation_cells),
        len(split.test_cells),
    )
    logger.info("held_out_test_cells=%s", split.test_cells)
    logger.info(
        "train_data soh_min=%.7g soh_median=%.7g soh_max=%.7g cycles_max=%d "
        "valid_cells=%d invalid_files=%d",
        float(np.min(train_soh)),
        float(np.median(train_soh)),
        float(np.max(train_soh)),
        scalers.max_cycle_train,
        len(cells),
        int((audit["status"] != "valid").sum()),
    )
    logger.info(
        "scalers soh_mean=%.7g soh_std=%.7g delta_v_mean=%.7g delta_v_std=%.7g "
        "voltage_mean=%.7g voltage_std=%.7g current_mean=%.7g current_std=%.7g "
        "fit_cells=%d",
        scalers.soh_mean,
        scalers.soh_std,
        scalers.delta_voltage_mean,
        scalers.delta_voltage_std,
        scalers.voltage_mean,
        scalers.voltage_std,
        scalers.current_mean,
        scalers.current_std,
        len(scalers.fit_cell_ids),
    )
    logger.info(
        "runtime device=%s amp=%s amp_initial_scale=%.7g deterministic=%s "
        "batch_size=%d lr=%.7g grad_clip=%.7g log_interval=%d",
        device,
        amp_enabled,
        float(amp_scaler.get_scale()),
        resolved.training.deterministic,
        resolved.training.batch_size,
        resolved.training.learning_rate,
        resolved.training.gradient_clip_norm,
        resolved.training.log_interval,
    )
    if device.type == "cuda":
        logger.info(
            "cuda device_name=%s capability=%s torch_cuda=%s current_device=%d",
            torch.cuda.get_device_name(device),
            torch.cuda.get_device_capability(device),
            torch.version.cuda,
            torch.cuda.current_device(),
        )
    cache = processor.cache_info()
    logger.info(
        "feature_cache grid_curves=%d references=%d raw_features=%d context_signals=%d",
        cache["grid_curves"],
        cache["references"],
        cache["raw_features"],
        cache["context_signals"],
    )

    began = time.perf_counter()
    progress = tqdm(
        range(start_step, resolved.training.max_steps + 1),
        desc=f"{dataset}-{model_name}-fold{fold}",
        unit="step",
    )
    last_step = start_step - 1
    total_amp_overflows = sum(
        1
        for item in records
        if pd.notna(item.get("optimizer_step_skipped", False))
        and bool(item.get("optimizer_step_skipped", False))
    )
    successful_updates = len(records) - total_amp_overflows
    consecutive_amp_overflows = 0
    for step in progress:
        step_started = time.perf_counter()
        episodes = []
        attempts = 0
        while len(episodes) < resolved.training.batch_size:
            attempts += 1
            if attempts > resolved.training.batch_size * 50:
                raise RuntimeError(
                    f"could not sample enough valid {dataset} training episodes"
                )
            cell = train_cells[int(rng.integers(0, len(train_cells)))]
            try:
                episodes.append(sampler.sample_training(cell, rng))
            except EpisodeUnavailable:
                continue
        sampling_finished = time.perf_counter()
        batch = collate_episodes(episodes)
        collation_finished = time.perf_counter()
        batch.to(device)
        transfer_finished = time.perf_counter()
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
                context_signal=batch.context_signal,
                context_signal_mask=batch.context_signal_mask,
            )
        # Keep the network forward in AMP, but evaluate log/std/division terms
        # of the probabilistic ELBO in float32 for numerical stability.
        loss_output = {name: value.float() for name, value in output.items()}
        losses = anp_elbo_loss(
            loss_output,
            batch.target_y.float(),
            batch.target_mask,
            kl_weight,
        )
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(
                f"non-finite ANP loss at step {step}: "
                f"loss={float(losses['loss'].detach().cpu())}, "
                f"nll={float(losses['nll'].detach().cpu())}, "
                f"kl={float(losses['kl'].detach().cpu())}"
            )
        amp_scale_before = float(amp_scaler.get_scale())
        amp_scaler.scale(losses["loss"]).backward()
        amp_scaler.unscale_(optimizer)
        nonfinite_gradient_names = _nonfinite_gradient_names(model)
        if nonfinite_gradient_names:
            gradient_norm = float("inf")
        else:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), resolved.training.gradient_clip_norm
                ).detach().cpu()
            )

        raw_targets = np.concatenate([episode.target_soh_raw for episode in episodes])
        target_min, target_max = _masked_range(batch.target_y, batch.target_mask)
        prediction_min, prediction_max = _masked_range(output["mean"], batch.target_mask)
        prediction_std_min, prediction_std_max = _masked_range(
            output["std"], batch.target_mask
        )
        allocated_mb, reserved_mb = _gpu_memory_mb(device)
        record: dict[str, Any] = {
            "step": step,
            "loss": float(losses["loss"].detach().cpu()),
            "nll": float(losses["nll"].detach().cpu()),
            "kl": float(losses["kl"].detach().cpu()),
            "kl_weight": kl_weight,
            "gradient_norm": gradient_norm,
            "optimizer_step_skipped": False,
            "amp_scale_before": amp_scale_before,
            "amp_scale_after": np.nan,
            "batch_sampling_attempts": attempts,
            "batch_sampling_seconds": sampling_finished - step_started,
            "batch_collation_seconds": collation_finished - sampling_finished,
            "batch_transfer_seconds": transfer_finished - collation_finished,
            "batch_cells": "|".join(episode.cell_id for episode in episodes),
            "batch_current_cycles": "|".join(
                str(episode.current_cycle) for episode in episodes
            ),
            "batch_betas": "|".join(f"{episode.beta:g}" for episode in episodes),
            "context_points_min": min(len(episode.context_x) for episode in episodes),
            "context_points_max": max(len(episode.context_x) for episode in episodes),
            "target_points_min": min(len(episode.target_x) for episode in episodes),
            "target_points_max": max(len(episode.target_x) for episode in episodes),
            "target_soh_raw_min": float(np.min(raw_targets)),
            "target_soh_raw_max": float(np.max(raw_targets)),
            "target_normalized_min": target_min,
            "target_normalized_max": target_max,
            "prediction_mean_min": prediction_min,
            "prediction_mean_max": prediction_max,
            "prediction_std_min": prediction_std_min,
            "prediction_std_max": prediction_std_max,
            "prior_std_min": float(output["prior_std"].min().detach().cpu()),
            "prior_std_max": float(output["prior_std"].max().detach().cpu()),
            "posterior_std_min": float(output["posterior_std"].min().detach().cpu()),
            "posterior_std_max": float(output["posterior_std"].max().detach().cpu()),
            "iv_observed_fraction_mean": float(
                batch.iv_feature[:, 2, :].mean().detach().cpu()
            ),
            "gpu_allocated_mb": allocated_mb,
            "gpu_reserved_mb": reserved_mb,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_rmse": np.nan,
            "elapsed_seconds": time.perf_counter() - began,
        }

        if not math.isfinite(gradient_norm):
            record["optimizer_step_skipped"] = True
            record["nonfinite_gradient_parameters"] = "|".join(
                nonfinite_gradient_names
            )
            if not amp_enabled:
                logger.error(
                    "Non-finite gradient without AMP at step=%d loss=%.7g nll=%.7g "
                    "kl=%.7g target_raw=[%.7g,%.7g] prediction_std=[%.7g,%.7g] "
                    "parameters=%s",
                    step,
                    record["loss"],
                    record["nll"],
                    record["kl"],
                    record["target_soh_raw_min"],
                    record["target_soh_raw_max"],
                    record["prediction_std_min"],
                    record["prediction_std_max"],
                    nonfinite_gradient_names,
                )
                raise FloatingPointError(f"non-finite gradient norm at step {step}")

            # A first-step fp16 overflow is expected occasionally. GradScaler's
            # normal policy is to skip that optimizer update and lower its scale.
            amp_scaler.update(new_scale=max(1.0, amp_scale_before / 2.0))
            optimizer.zero_grad(set_to_none=True)
            record["amp_scale_after"] = float(amp_scaler.get_scale())
            total_amp_overflows += 1
            consecutive_amp_overflows += 1
            records.append(record)
            last_step = step
            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                amp_scaler,
                step,
                best_step,
                best_validation_rmse,
                stale_validations,
                resolved,
                model_spec,
                split,
                scalers,
                rng,
            )
            _atomic_save(checkpoint, run_dir / "checkpoints/last.pt")
            pd.DataFrame(records).to_csv(history_path, index=False)
            logger.warning(
                "AMP overflow step=%d/%d update=SKIPPED scale=%.7g->%.7g "
                "consecutive=%d/%d loss=%.7g nll=%.7g kl=%.7g "
                "target_raw=[%.7g,%.7g] target_norm=[%.7g,%.7g] "
                "pred_mean=[%.7g,%.7g] pred_std=[%.7g,%.7g] bad_gradients=%s",
                step,
                resolved.training.max_steps,
                amp_scale_before,
                record["amp_scale_after"],
                consecutive_amp_overflows,
                resolved.training.max_consecutive_amp_overflows,
                record["loss"],
                record["nll"],
                record["kl"],
                record["target_soh_raw_min"],
                record["target_soh_raw_max"],
                target_min,
                target_max,
                prediction_min,
                prediction_max,
                prediction_std_min,
                prediction_std_max,
                nonfinite_gradient_names,
            )
            logger.warning(
                "overflow_batch cells=%s cycles=%s betas=%s context_points=%d..%d "
                "target_points=%d..%d iv_observed_mean=%.5f gpu_mb=%.1f/%.1f "
                "checkpoint=%s",
                record["batch_cells"],
                record["batch_current_cycles"],
                record["batch_betas"],
                record["context_points_min"],
                record["context_points_max"],
                record["target_points_min"],
                record["target_points_max"],
                record["iv_observed_fraction_mean"],
                allocated_mb,
                reserved_mb,
                run_dir / "checkpoints/last.pt",
            )
            progress.set_postfix(
                loss=f"{record['loss']:.4g}",
                status="amp-overflow-skip",
                scale=f"{record['amp_scale_after']:.0f}",
                refresh=False,
            )
            if (
                consecutive_amp_overflows
                >= resolved.training.max_consecutive_amp_overflows
            ):
                manifest.update(
                    {
                        "status": "failed",
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "last_step": step,
                        "successful_optimizer_updates": successful_updates,
                        "amp_overflow_skips": total_amp_overflows,
                        "failure": "consecutive non-finite AMP gradients",
                    }
                )
                write_json(run_dir / "run_manifest.json", manifest)
                raise FloatingPointError(
                    f"AMP gradients stayed non-finite for "
                    f"{consecutive_amp_overflows} consecutive steps; see "
                    f"{run_dir / 'logs/train.log'} and "
                    f"{run_dir / 'training/history.csv'}"
                )
            continue

        amp_scaler.step(optimizer)
        amp_scaler.update()
        record["amp_scale_after"] = float(amp_scaler.get_scale())
        successful_updates += 1
        consecutive_amp_overflows = 0

        validate_now = (
            step % resolved.training.validation_interval == 0
            or step == resolved.training.max_steps
        )
        improved = False
        validation_by_cell: dict[str, float] = {}
        if validate_now:
            validation_rmse, validation_by_cell = _validation_rmse(
                model, validation_cells, sampler, model_name, resolved, device
            )
            record["validation_rmse"] = validation_rmse
            record["validation_cell_rmse"] = "|".join(
                f"{cell_id}:{value:.7g}"
                for cell_id, value in sorted(validation_by_cell.items())
            )
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
            scale=f"{record['amp_scale_after']:.0f}" if amp_enabled else "off",
            refresh=False,
        )
        if step % resolved.training.log_interval == 0 or validate_now or step == start_step:
            completed_attempts = step - start_step + 1
            eta_seconds = (
                record["elapsed_seconds"]
                / max(1, completed_attempts)
                * (resolved.training.max_steps - step)
            )
            logger.info(
                "step=%d/%d loss=%.7g nll=%.7g kl=%.7g kl_weight=%.4f "
                "grad=%.6g update=APPLIED successful_updates=%d amp_scale=%.7g->%.7g "
                "validation_rmse=%s best=%.7g@%d stale=%d overflows=%d "
                "elapsed=%.1fs eta=%.1fs gpu_mb=%.1f/%.1f "
                "data_ms=%.2f collate_ms=%.2f transfer_ms=%.2f",
                step, resolved.training.max_steps, record["loss"], record["nll"],
                record["kl"], kl_weight, gradient_norm, successful_updates,
                record["amp_scale_before"], record["amp_scale_after"],
                f"{record['validation_rmse']:.7g}" if validate_now else "-",
                best_validation_rmse, best_step, stale_validations,
                total_amp_overflows, record["elapsed_seconds"], eta_seconds,
                allocated_mb, reserved_mb,
                1000.0 * record["batch_sampling_seconds"],
                1000.0 * record["batch_collation_seconds"],
                1000.0 * record["batch_transfer_seconds"],
            )
            logger.info(
                "batch step=%d cells=%s cycles=%s betas=%s attempts=%d "
                "context_points=%d..%d target_points=%d..%d "
                "target_raw=[%.7g,%.7g] target_norm=[%.7g,%.7g] "
                "pred_mean=[%.7g,%.7g] pred_std=[%.7g,%.7g] "
                "prior_std=[%.7g,%.7g] posterior_std=[%.7g,%.7g] "
                "iv_observed_mean=%.5f lr=%.7g",
                step,
                record["batch_cells"],
                record["batch_current_cycles"],
                record["batch_betas"],
                record["batch_sampling_attempts"],
                record["context_points_min"],
                record["context_points_max"],
                record["target_points_min"],
                record["target_points_max"],
                record["target_soh_raw_min"],
                record["target_soh_raw_max"],
                record["target_normalized_min"],
                record["target_normalized_max"],
                record["prediction_mean_min"],
                record["prediction_mean_max"],
                record["prediction_std_min"],
                record["prediction_std_max"],
                record["prior_std_min"],
                record["prior_std_max"],
                record["posterior_std_min"],
                record["posterior_std_max"],
                record["iv_observed_fraction_mean"],
                record["learning_rate"],
            )
            if validate_now:
                logger.info(
                    "validation_per_cell step=%d mean_rmse=%.7g values=%s",
                    step,
                    record["validation_rmse"],
                    record["validation_cell_rmse"],
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
            "successful_optimizer_updates": successful_updates,
            "amp_overflow_skips": total_amp_overflows,
            "best_step": best_step,
            "best_validation_rmse": best_validation_rmse,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    logger.info("Completed %s ANP run: %s", dataset, run_dir)
    return run_dir


def parse_args(default_config: str = "configs/matr_partial_iv_anp.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one cell-level ANP fold")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--device")
    return parser.parse_args()


def main(default_config: str = "configs/matr_partial_iv_anp.yaml") -> None:
    args = parse_args(default_config)
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    data_root = resolve_data_root(config, args.data_root)
    run_dir = train_run(
        config, args.model, args.fold, data_root,
        resume=args.resume, max_steps=args.max_steps, batch_size=args.batch_size,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

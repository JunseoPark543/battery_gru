"""Train the horizon-conditioned lifetime I-V ANP."""

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
from torch.nn import functional as F
from tqdm.auto import tqdm

from battery_weighted_maml.matr_anp.runtime import (
    configure_logger,
    git_commit,
    resolve_device,
    seed_everything,
    write_json,
)
from battery_weighted_maml.matr_anp.splits import FoldSplit, make_splits, save_splits

from .config import (
    LifetimeIVConfig,
    TrainingConfig,
    load_config,
    resolve_data_root,
    save_config,
)
from .data import (
    LabeledCell,
    LifetimeIVPrefixStore,
    LifetimeIVScalers,
    load_labeled_cells,
)
from .losses import lifetime_elbo
from .model import LifetimeIVANP, ModelSpec, build_model
from .tasks import LifetimeTaskSampler, TaskUnavailable, collate_tasks


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
            [value.cpu().to(torch.uint8) for value in state["cuda"]]
        )


def split_cells(
    cells: Sequence[LabeledCell], split: FoldSplit
) -> tuple[list[LabeledCell], list[LabeledCell], list[LabeledCell]]:
    by_id = {item.cell_id: item for item in cells}
    return (
        [by_id[cell_id] for cell_id in split.train_cells],
        [by_id[cell_id] for cell_id in split.validation_cells],
        [by_id[cell_id] for cell_id in split.test_cells],
    )


def _model_arguments(batch) -> tuple[torch.Tensor, ...]:
    return (
        batch.context_cycles, batch.context_cycle_mask,
        batch.context_curves, batch.context_curve_mask,
        batch.context_point_mask, batch.context_y,
        batch.query_cycles, batch.query_cycle_mask,
        batch.query_curves, batch.query_curve_mask,
        batch.query_point_mask,
    )


def validation_rmse(
    model: LifetimeIVANP,
    train_cells: Sequence[LabeledCell],
    validation_cells: Sequence[LabeledCell],
    sampler: LifetimeTaskSampler,
    config: LifetimeIVConfig,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for horizon in config.evaluation.horizons:
            try:
                context_seed = (
                    config.evaluation.context_seed
                    if config.evaluation.context_seed is not None
                    else config.seed
                )
                task = sampler.evaluation(
                    horizon, train_cells, validation_cells,
                    context_size=config.evaluation.context_size,
                    seed=context_seed + 400_009,
                    nested_context_selection=(
                        config.evaluation.context_seed is not None
                    ),
                )
            except TaskUnavailable as exc:
                rows.append({"horizon": horizon, "status": "skipped", "reason": str(exc)})
                continue
            batch = collate_tasks([task]).to(device)
            # Validation never passes query lifetime to the model.
            output = model(*_model_arguments(batch), sample_latent=False)
            predicted = sampler.scalers.inverse_lifetime(
                output["mean"][0, :, 0].float().cpu().numpy()
            )
            for index, point in enumerate(task.query):
                error = float(predicted[index] - point.lifetime_cycles)
                rows.append({
                    "horizon": horizon,
                    "status": "ok",
                    "reason": "",
                    "cell_id": point.cell_id,
                    "true_lifetime_cycles": point.lifetime_cycles,
                    "predicted_lifetime_cycles": float(predicted[index]),
                    "true_rul_cycles": point.lifetime_cycles - horizon,
                    "predicted_rul_cycles": float(predicted[index]) - horizon,
                    "squared_error": error * error,
                    "num_context_cells": len(task.context),
                })
    if was_training:
        model.train()
    errors = [row["squared_error"] for row in rows if row.get("status") == "ok"]
    if not errors:
        raise ValueError("validation produced no usable predictions")
    return float(np.sqrt(np.mean(errors))), rows


def _checkpoint_payload(
    model: LifetimeIVANP,
    spec: ModelSpec,
    optimizer: torch.optim.Optimizer,
    amp_scaler: Any,
    step: int,
    best_step: int,
    best_rmse: float,
    stale: int,
    config: LifetimeIVConfig,
    split: FoldSplit,
    scalers: LifetimeIVScalers,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "algorithm": spec.algorithm,
        "dataset": "MATR",
        "prediction_target": "lifetime",
        "input_features": ["cycle", "soh", "voltage_q_256", "current_q_256"],
        "step": int(step),
        "best_step": int(best_step),
        "best_validation_rmse": float(best_rmse),
        "stale_validations": int(stale),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": amp_scaler.state_dict(),
        "model_spec": spec.to_dict(),
        "config": config.to_dict(),
        "fold_split": asdict(split),
        "scalers": scalers.to_dict(),
        "rng_states": _capture_rng(rng),
        "git_commit": git_commit(),
    }


def train_run(
    config: LifetimeIVConfig,
    fold: int,
    data_root: str | Path,
    *,
    resume: str | Path | None = None,
    max_steps: int | None = None,
    task_batch_size: int | None = None,
    output_root: str | Path | None = None,
) -> Path:
    resolved = copy.deepcopy(config)
    if max_steps is not None:
        resolved.training.max_steps = int(max_steps)
    if task_batch_size is not None:
        resolved.training.task_batch_size = int(task_batch_size)
    resolved.validate()
    seed_everything(resolved.seed, resolved.training.deterministic)
    cells, audit = load_labeled_cells(data_root, resolved.data)  # type: ignore[arg-type]
    splits = make_splits([item.cell_id for item in cells], resolved.split)
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must lie in [0,{len(splits)-1}]")
    split = splits[fold]
    train_cells, validation_cells, _ = split_cells(cells, split)
    maximum_training_horizon = max(resolved.task.horizons)
    scalers = LifetimeIVScalers.fit(train_cells, maximum_training_horizon)
    if set(scalers.fit_cell_ids) != set(split.train_cells):
        raise RuntimeError("scalers were not fit on exactly the train split")
    store = LifetimeIVPrefixStore(
        scalers,
        resolved.q_grid,
        max(maximum_training_horizon, max(resolved.evaluation.horizons)),
    )
    sampler = LifetimeTaskSampler(resolved.task, scalers, store)
    model, spec = build_model(resolved.model, resolved.q_grid.num_points)

    payload = None
    if resume is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mode = "base"
        if resolved.training.paired_horizon_training:
            weight = f"{resolved.training.consistency_weight:g}".replace(".", "p")
            mode = f"pair-w{weight}-g{resolved.training.consistency_horizon_gap}"
        run_dir = Path(output_root or resolved.paths.output_root).resolve() / (
            f"{timestamp}_f{fold}_life_iv_{mode}_s{resolved.seed}"
        )
    else:
        checkpoint_path = Path(resume).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        run_dir = checkpoint_path.parent.parent
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != spec.algorithm:
            raise ValueError("resume checkpoint algorithm differs")
        if payload["fold_split"] != asdict(split):
            raise ValueError("resume split differs")
        if payload["scalers"] != scalers.to_dict():
            raise ValueError("resume train-only scalers differ")
        saved, current = copy.deepcopy(payload["config"]), resolved.to_dict()
        # Checkpoints created before consistency training existed represent the
        # exact weight=0 baseline and remain resumable with the baseline config.
        defaults = asdict(TrainingConfig())
        for key in (
            "paired_horizon_training",
            "consistency_weight", "consistency_horizon_gap",
            "consistency_warmup_steps", "consistency_huber_beta",
        ):
            saved["training"].setdefault(key, defaults[key])
        saved["evaluation"].setdefault("context_seed", None)
        saved["training"].pop("max_steps", None)
        current["training"].pop("max_steps", None)
        if saved != current:
            raise ValueError("resume config differs except for max_steps")

    for name in ("checkpoints", "logs", "training", "audit", "scalers", "evaluation"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "logs/train.log")
    save_config(resolved, run_dir / "resolved_config.yaml")
    save_splits(splits, run_dir / "splits.json", resolved.split)
    scalers.save(run_dir / "scalers/lifetime_iv_scalers.json")
    audit.to_csv(run_dir / "audit/data_and_label_audit.csv", index=False)

    device = resolve_device(resolved.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved.training.learning_rate)
    amp_enabled = bool(resolved.training.use_amp and device.type == "cuda")
    try:
        amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        amp_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    rng = np.random.default_rng(resolved.seed + fold * 100_003)
    start_step, best_step, stale = 1, 0, 0
    best_rmse = float("inf")
    history_path = run_dir / "training/history.csv"
    records: list[dict[str, Any]] = []
    if payload is not None:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        amp_scaler.load_state_dict(payload["amp_scaler_state_dict"])
        _restore_rng(payload["rng_states"], rng)
        start_step = int(payload["step"]) + 1
        best_step = int(payload["best_step"])
        best_rmse = float(payload["best_validation_rmse"])
        stale = int(payload["stale_validations"])
        if history_path.is_file():
            records = pd.read_csv(history_path).to_dict("records")

    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": spec.algorithm,
        "prediction_target": "lifetime",
        "input_features": ["cycle", "soh", "voltage_q_256", "current_q_256"],
        "delta_soh_used": False,
        "consistency": {
            "paired_horizon_training": resolved.training.paired_horizon_training,
            "weight": resolved.training.consistency_weight,
            "horizon_gap": resolved.training.consistency_horizon_gap,
            "warmup_steps": resolved.training.consistency_warmup_steps,
            "huber_beta": resolved.training.consistency_huber_beta,
            "teacher": "later_prior_mean_stop_gradient",
        },
        "fold": fold,
        "device": str(device),
        "model_spec": spec.to_dict(),
        "split": asdict(split),
        "data_root": str(Path(data_root).resolve()),
        "git_commit": git_commit(),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    logger.info(
        "start fold=%d device=%s parameters=%d train=%d validation=%d "
        "horizons=%s q_points=%d context=[%d,%d] query=%d target=lifetime "
        "delta_soh=false paired_horizons=%s consistency_weight=%.5g "
        "consistency_gap=%d",
        fold, device, spec.parameter_count, len(train_cells), len(validation_cells),
        resolved.task.horizons, resolved.q_grid.num_points,
        resolved.task.context_size_min, resolved.task.context_size_max,
        resolved.task.query_size,
        resolved.training.paired_horizon_training,
        resolved.training.consistency_weight,
        resolved.training.consistency_horizon_gap,
    )
    use_paired_horizons = resolved.training.paired_horizon_training
    use_consistency = resolved.training.consistency_weight > 0
    logger.info(
        "train_scalers cycle=%.5g soh=%.5g+-%.5g voltage=%.5g+-%.5g "
        "current=%.5g+-%.5g lifetime=%.5g+-%.5g",
        scalers.cycle_scale, scalers.soh_mean, scalers.soh_std,
        scalers.voltage_mean, scalers.voltage_std,
        scalers.current_mean, scalers.current_std,
        scalers.lifetime_mean, scalers.lifetime_std,
    )

    began = time.perf_counter()
    last_step = start_step - 1
    progress = tqdm(
        range(start_step, resolved.training.max_steps + 1),
        desc=f"MATR-life-IV-ANP-f{fold}", unit="step",
    )
    for step in progress:
        if use_paired_horizons:
            pairs = [
                sampler.sample_training_pair(
                    train_cells, rng,
                    resolved.training.consistency_horizon_gap,
                )
                for _ in range(resolved.training.task_batch_size)
            ]
            early_tasks = [pair[0] for pair in pairs]
            late_tasks = [pair[1] for pair in pairs]
            early_batch = collate_tasks(early_tasks).to(device)
            late_batch = collate_tasks(late_tasks).to(device)
            tasks = early_tasks
        else:
            tasks = [
                sampler.sample_training(train_cells, rng)
                for _ in range(resolved.training.task_batch_size)
            ]
            early_batch = collate_tasks(tasks).to(device)
            late_batch = None
        warmup = resolved.training.kl_warmup_steps
        beta = resolved.training.beta_kl * (
            1.0 if warmup == 0 else min(1.0, step / warmup)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            early_output = model(
                *_model_arguments(early_batch),
                query_y=early_batch.query_y,
                sample_latent=True,
                return_representations=use_consistency,
            )
            if use_paired_horizons and late_batch is not None:
                late_output = model(
                    *_model_arguments(late_batch),
                    query_y=late_batch.query_y,
                    sample_latent=True,
                    return_representations=use_consistency,
                )
            else:
                late_output = None
        early_losses = lifetime_elbo(
            {name: value.float() for name, value in early_output.items()},
            early_batch.query_y.float(), early_batch.query_point_mask, beta,
        )
        if use_paired_horizons and late_batch is not None and late_output is not None:
            late_losses = lifetime_elbo(
                {name: value.float() for name, value in late_output.items()},
                late_batch.query_y.float(), late_batch.query_point_mask, beta,
            )
            base_loss = 0.5 * (early_losses["loss"] + late_losses["loss"])
            nll_value = 0.5 * (early_losses["nll"] + late_losses["nll"])
            kl_value = 0.5 * (early_losses["kl"] + late_losses["kl"])
        else:
            base_loss = early_losses["loss"]
            nll_value = early_losses["nll"]
            kl_value = early_losses["kl"]
        consistency_loss = early_losses["loss"].new_zeros(())
        effective_consistency_weight = 0.0
        if use_consistency and late_batch is not None and late_output is not None:
            was_training = model.training
            model.eval()
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                early_prior = model.forward_from_embeddings(
                    early_output["context_h"], early_batch.context_point_mask,
                    early_batch.context_y, early_output["query_h"],
                    early_batch.query_point_mask, query_y=None, sample_latent=False,
                )
                late_prior = model.forward_from_embeddings(
                    late_output["context_h"], late_batch.context_point_mask,
                    late_batch.context_y, late_output["query_h"],
                    late_batch.query_point_mask, query_y=None, sample_latent=False,
                )
            model.train(was_training)
            selected = early_batch.query_point_mask.unsqueeze(-1)
            pointwise = F.smooth_l1_loss(
                early_prior["mean"].float(),
                late_prior["mean"].float().detach(),
                beta=resolved.training.consistency_huber_beta,
                reduction="none",
            )
            consistency_loss = pointwise.masked_select(selected).mean()
            warmup = resolved.training.consistency_warmup_steps
            effective_consistency_weight = resolved.training.consistency_weight * (
                1.0 if warmup == 0 else min(1.0, step / warmup)
            )
            total_loss = base_loss + effective_consistency_weight * consistency_loss
        else:
            total_loss = base_loss
        amp_scaler.scale(total_loss).backward()
        amp_scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), resolved.training.gradient_clip_norm
        ).detach().cpu())
        skipped = not math.isfinite(gradient_norm)
        if skipped:
            optimizer.zero_grad(set_to_none=True)
            if not amp_enabled:
                raise FloatingPointError(f"non-finite gradient at step {step}")
            amp_scaler.update(new_scale=max(1.0, float(amp_scaler.get_scale()) / 2.0))
        else:
            amp_scaler.step(optimizer)
            amp_scaler.update()
        true_values = early_batch.query_lifetime_cycles.masked_select(
            early_batch.query_point_mask
        )
        horizon_text = "|".join(str(task.horizon) for task in tasks)
        if use_paired_horizons and late_batch is not None:
            horizon_text = "|".join(
                f"{early.horizon}->{late.horizon}"
                for early, late in zip(early_tasks, late_tasks)
            )
        record: dict[str, Any] = {
            "step": step,
            "loss": float(total_loss.detach().cpu()),
            "nll": float(nll_value.detach().cpu()),
            "kl": float(kl_value.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "consistency_weight": effective_consistency_weight,
            "beta_kl": beta,
            "gradient_norm": gradient_norm,
            "optimizer_step_skipped": skipped,
            "horizons": horizon_text,
            "context_cells": "|".join(",".join(p.cell_id for p in task.context) for task in tasks),
            "query_cells": "|".join(",".join(p.cell_id for p in task.query) for task in tasks),
            "true_lifetime_min": float(true_values.min().cpu()),
            "true_lifetime_max": float(true_values.max().cpu()),
            "validation_rmse_cycles": np.nan,
            "elapsed_seconds": time.perf_counter() - began,
        }
        last_step = step
        if step % resolved.training.validation_interval == 0 and not skipped:
            value, rows = validation_rmse(
                model, train_cells, validation_cells, sampler, resolved, device
            )
            record["validation_rmse_cycles"] = value
            pd.DataFrame(rows).to_csv(
                run_dir / f"training/validation_step{step}.csv", index=False
            )
            if value < best_rmse:
                best_rmse, best_step, stale = value, step, 0
                _atomic_save(
                    _checkpoint_payload(
                        model, spec, optimizer, amp_scaler, step, best_step,
                        best_rmse, stale, resolved, split, scalers, rng,
                    ),
                    run_dir / "checkpoints/best.pt",
                )
            else:
                stale += 1
            logger.info(
                "validation step=%d rmse=%.7g best=%.7g@%d stale=%d/%d",
                step, value, best_rmse, best_step, stale,
                resolved.training.early_stopping_patience,
            )
        records.append(record)
        if step == 1 or step % resolved.training.log_interval == 0:
            logger.info(
                "step=%d/%d loss=%.7g nll=%.7g kl=%.7g consistency=%.7g "
                "lambda=%.5g beta=%.5g grad=%.7g horizon=%s "
                "lifetime=[%.0f,%.0f] skipped=%s",
                step, resolved.training.max_steps, record["loss"], record["nll"],
                record["kl"], record["consistency_loss"],
                record["consistency_weight"], beta, gradient_norm, record["horizons"],
                record["true_lifetime_min"], record["true_lifetime_max"], skipped,
            )
        if step % resolved.training.checkpoint_interval == 0:
            _atomic_save(
                _checkpoint_payload(
                    model, spec, optimizer, amp_scaler, step, best_step,
                    best_rmse, stale, resolved, split, scalers, rng,
                ),
                run_dir / "checkpoints/last.pt",
            )
            pd.DataFrame(records).to_csv(history_path, index=False)
        progress.set_postfix(loss=f"{record['loss']:.4g}", best=f"{best_rmse:.4g}")
        if stale >= resolved.training.early_stopping_patience:
            logger.info("early stopping at step=%d best=%.7g@%d", step, best_rmse, best_step)
            break

    if best_step == 0 and last_step > 0:
        best_rmse, rows = validation_rmse(
            model, train_cells, validation_cells, sampler, resolved, device
        )
        best_step = last_step
        pd.DataFrame(rows).to_csv(
            run_dir / f"training/validation_step{last_step}.csv", index=False
        )
    final = _checkpoint_payload(
        model, spec, optimizer, amp_scaler, last_step, best_step,
        best_rmse, stale, resolved, split, scalers, rng,
    )
    _atomic_save(final, run_dir / "checkpoints/last.pt")
    if not (run_dir / "checkpoints/best.pt").is_file():
        _atomic_save(final, run_dir / "checkpoints/best.pt")
    pd.DataFrame(records).to_csv(history_path, index=False)
    manifest.update({
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_step": last_step,
        "best_step": best_step,
        "best_validation_rmse_cycles": best_rmse,
    })
    write_json(run_dir / "run_manifest.json", manifest)
    logger.info("completed run=%s", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MATR lifetime I-V ANP")
    parser.add_argument("--config", default="configs/matr_horizon_lifetime_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--task-batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    run_dir = train_run(
        config, args.fold, resolve_data_root(config, args.data_root),
        resume=args.resume, max_steps=args.max_steps,
        task_batch_size=args.task_batch_size,
    )
    print(f"Lifetime I-V ANP run: {run_dir}")


if __name__ == "__main__":
    main()

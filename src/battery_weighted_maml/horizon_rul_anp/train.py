"""Train horizon-conditioned inter-cell RUL ANP on leakage-safe MATR folds."""

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
    HorizonRULConfig,
    load_config,
    resolve_data_root,
    save_config,
)
from .data import LabeledCell, RULScalers, load_labeled_cells
from .losses import horizon_rul_elbo
from .model import HorizonRULANP, ModelSpec, build_model
from .tasks import HorizonTaskSampler, TaskUnavailable, collate_tasks


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


def _checkpoint_payload(
    model: HorizonRULANP,
    spec: ModelSpec,
    optimizer: torch.optim.Optimizer,
    amp_scaler: Any,
    step: int,
    best_step: int,
    best_validation_rmse: float,
    stale_validations: int,
    config: HorizonRULConfig,
    split: FoldSplit,
    scalers: RULScalers,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "algorithm": spec.algorithm,
        "dataset": "MATR",
        "step": int(step),
        "best_step": int(best_step),
        "best_validation_rmse": float(best_validation_rmse),
        "stale_validations": int(stale_validations),
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


def _split_cells(
    cells: Sequence[LabeledCell], split: FoldSplit
) -> tuple[list[LabeledCell], list[LabeledCell], list[LabeledCell]]:
    by_id = {item.cell_id: item for item in cells}
    return (
        [by_id[cell_id] for cell_id in split.train_cells],
        [by_id[cell_id] for cell_id in split.validation_cells],
        [by_id[cell_id] for cell_id in split.test_cells],
    )


def validation_rmse(
    model: HorizonRULANP,
    train_cells: Sequence[LabeledCell],
    validation_cells: Sequence[LabeledCell],
    sampler: HorizonTaskSampler,
    config: HorizonRULConfig,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    """Validate with train cells as context and validation cells as queries."""
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for horizon in config.training.validation_horizons:
            try:
                task = sampler.evaluation(
                    horizon,
                    train_cells,
                    validation_cells,
                    context_size=config.evaluation.context_size,
                    seed=config.seed + 400_009,
                )
            except TaskUnavailable as exc:
                rows.append(
                    {
                        "horizon": horizon,
                        "status": "skipped",
                        "reason": str(exc),
                        "cell_id": "",
                        "true_rul_cycles": np.nan,
                        "predicted_rul_cycles": np.nan,
                        "squared_error": np.nan,
                    }
                )
                continue
            batch = collate_tasks([task]).to(device)
            # No query_y here: validation RUL is never an inference input.
            output = model(
                batch.context_prefix,
                batch.context_prefix_mask,
                batch.context_mask,
                batch.context_y,
                batch.query_prefix,
                batch.query_prefix_mask,
                batch.query_mask,
                sample_latent=False,
            )
            predicted = sampler.scalers.inverse_rul(
                output["mean"][0, :, 0].float().cpu().numpy()
            )
            truth = batch.query_rul_cycles[0].cpu().numpy()
            count = len(task.query)
            for index, point in enumerate(task.query):
                error = float(predicted[index] - truth[index])
                rows.append(
                    {
                        "horizon": horizon,
                        "status": "ok",
                        "reason": "",
                        "cell_id": point.cell_id,
                        "true_rul_cycles": float(truth[index]),
                        "predicted_rul_cycles": float(predicted[index]),
                        "squared_error": error * error,
                        "num_context_cells": len(task.context),
                        "num_query_cells": count,
                    }
                )
    if was_training:
        model.train()
    valid = [row["squared_error"] for row in rows if row["status"] == "ok"]
    if not valid:
        raise ValueError("validation produced no usable horizon/cell predictions")
    return float(np.sqrt(np.mean(valid))), rows


def train_run(
    config: HorizonRULConfig,
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
    cells, audit = load_labeled_cells(data_root, resolved.data)
    splits = make_splits([item.cell_id for item in cells], resolved.split)
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must lie in [0,{len(splits) - 1}]")
    split = splits[fold]
    train_cells, validation_cells, _ = _split_cells(cells, split)
    scalers = RULScalers.fit(train_cells, resolved.task)
    if set(scalers.fit_cell_ids) != set(split.train_cells):
        raise RuntimeError("RUL scalers were not fit on exactly the train split")
    sampler = HorizonTaskSampler(resolved.task, scalers)
    model, spec = build_model(resolved.model)

    if resume is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = Path(output_root or resolved.paths.output_root).resolve()
        run_dir = root / f"{timestamp}_fold{fold}_horizon_rul_anp_s{resolved.seed}"
        payload = None
    else:
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
        run_dir = checkpoint.parent.parent
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != spec.algorithm or payload.get("dataset") != "MATR":
            raise ValueError("resume checkpoint algorithm/dataset mismatch")
        if payload["fold_split"] != asdict(split):
            raise ValueError("resume checkpoint fold split differs")
        if payload["scalers"] != scalers.to_dict():
            raise ValueError("resume train-only scaler values differ")
        saved = copy.deepcopy(payload["config"])
        current = resolved.to_dict()
        saved["training"].pop("max_steps", None)
        current["training"].pop("max_steps", None)
        if saved != current:
            raise ValueError("resume config differs except for max_steps")

    for name in ("checkpoints", "logs", "training", "audit", "scalers", "evaluation"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "logs/train.log")
    save_config(resolved, run_dir / "resolved_config.yaml")
    save_splits(splits, run_dir / "splits.json", resolved.split)
    scalers.save(run_dir / "scalers/rul_scalers.json")
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
    start_step = 1
    best_step = 0
    best_rmse = float("inf")
    stale = 0
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
        logger.info("resuming at step=%d", start_step)

    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": spec.algorithm,
        "dataset": "MATR",
        "fold": fold,
        "model_spec": spec.to_dict(),
        "split": asdict(split),
        "lifetime_source": resolved.data.lifetime_source,
        "label_path": resolved.data.label_path,
        "device": str(device),
        "data_root": str(Path(data_root).resolve()),
        "git_commit": git_commit(),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    logger.info(
        "start fold=%d device=%s parameters=%d train=%d validation=%d test=%d "
        "horizon=[%d,%d] task_batch=%d context=[%d,%d] query=%d",
        fold,
        device,
        spec.parameter_count,
        len(split.train_cells),
        len(split.validation_cells),
        len(split.test_cells),
        resolved.task.min_horizon,
        resolved.task.max_horizon,
        resolved.training.task_batch_size,
        resolved.task.context_size_min,
        resolved.task.context_size_max,
        resolved.task.query_size,
    )
    logger.info(
        "train_only_scalers cycle_scale=%.7g soh_mean=%.7g soh_std=%.7g "
        "delta_soh_mean=%.7g delta_soh_std=%.7g rul_mean=%.7g rul_std=%.7g",
        scalers.cycle_scale,
        scalers.soh_mean,
        scalers.soh_std,
        scalers.delta_soh_mean,
        scalers.delta_soh_std,
        scalers.rul_mean,
        scalers.rul_std,
    )

    began = time.perf_counter()
    last_step = start_step - 1
    progress = tqdm(
        range(start_step, resolved.training.max_steps + 1),
        desc=f"MATR-horizon-rul-anp-fold{fold}",
        unit="step",
    )
    for step in progress:
        tasks = [
            sampler.sample_training(train_cells, rng)
            for _ in range(resolved.training.task_batch_size)
        ]
        batch = collate_tasks(tasks).to(device)
        warmup = resolved.training.kl_warmup_steps
        beta = resolved.training.beta_kl * (
            1.0 if warmup == 0 else min(1.0, step / warmup)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(
                batch.context_prefix,
                batch.context_prefix_mask,
                batch.context_mask,
                batch.context_y,
                batch.query_prefix,
                batch.query_prefix_mask,
                batch.query_mask,
                query_y=batch.query_y,
                sample_latent=True,
            )
        losses = horizon_rul_elbo(
            {name: value.float() for name, value in output.items()},
            batch.query_y.float(),
            batch.query_mask,
            beta,
        )
        amp_scaler.scale(losses["loss"]).backward()
        amp_scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), resolved.training.gradient_clip_norm
            ).detach().cpu()
        )
        skipped = not math.isfinite(gradient_norm)
        if skipped:
            optimizer.zero_grad(set_to_none=True)
            if not amp_enabled:
                raise FloatingPointError(f"non-finite gradient at step {step}")
            current_scale = float(amp_scaler.get_scale())
            amp_scaler.update(new_scale=max(1.0, current_scale / 2.0))
        else:
            amp_scaler.step(optimizer)
            amp_scaler.update()

        query_values = batch.query_rul_cycles.masked_select(batch.query_mask)
        predicted_norm = output["mean"].detach().float().masked_select(
            batch.query_mask.unsqueeze(-1)
        )
        record: dict[str, Any] = {
            "step": step,
            "loss": float(losses["loss"].detach().cpu()),
            "nll": float(losses["nll"].detach().cpu()),
            "kl": float(losses["kl"].detach().cpu()),
            "beta_kl": beta,
            "gradient_norm": gradient_norm,
            "optimizer_step_skipped": skipped,
            "horizons": "|".join(str(task.horizon) for task in tasks),
            "context_cells": "|".join(
                ",".join(point.cell_id for point in task.context) for task in tasks
            ),
            "query_cells": "|".join(
                ",".join(point.cell_id for point in task.query) for task in tasks
            ),
            "true_rul_min": float(query_values.min().cpu()),
            "true_rul_max": float(query_values.max().cpu()),
            "prediction_normalized_min": float(predicted_norm.min().cpu()),
            "prediction_normalized_max": float(predicted_norm.max().cpu()),
            "validation_rmse_cycles": np.nan,
            "elapsed_seconds": time.perf_counter() - began,
        }
        last_step = step

        if step % resolved.training.validation_interval == 0 and not skipped:
            value, validation_rows = validation_rmse(
                model,
                train_cells,
                validation_cells,
                sampler,
                resolved,
                device,
            )
            record["validation_rmse_cycles"] = value
            pd.DataFrame(validation_rows).to_csv(
                run_dir / f"training/validation_step{step}.csv", index=False
            )
            if value < best_rmse:
                best_rmse = value
                best_step = step
                stale = 0
                checkpoint = _checkpoint_payload(
                    model, spec, optimizer, amp_scaler, step, best_step,
                    best_rmse, stale, resolved, split, scalers, rng,
                )
                _atomic_save(checkpoint, run_dir / "checkpoints/best.pt")
            else:
                stale += 1
            logger.info(
                "validation step=%d rmse_cycles=%.7g best=%.7g@%d stale=%d/%d",
                step,
                value,
                best_rmse,
                best_step,
                stale,
                resolved.training.early_stopping_patience,
            )

        records.append(record)
        if step == 1 or step % resolved.training.log_interval == 0:
            logger.info(
                "step=%d/%d loss=%.7g nll=%.7g kl=%.7g beta=%.5g "
                "grad=%.7g horizons=%s rul=[%.1f,%.1f] skipped=%s",
                step,
                resolved.training.max_steps,
                record["loss"],
                record["nll"],
                record["kl"],
                beta,
                gradient_norm,
                record["horizons"],
                record["true_rul_min"],
                record["true_rul_max"],
                skipped,
            )
        if step % resolved.training.checkpoint_interval == 0:
            checkpoint = _checkpoint_payload(
                model, spec, optimizer, amp_scaler, step, best_step,
                best_rmse, stale, resolved, split, scalers, rng,
            )
            _atomic_save(checkpoint, run_dir / "checkpoints/last.pt")
            pd.DataFrame(records).to_csv(history_path, index=False)
        progress.set_postfix(loss=f"{record['loss']:.4g}", best=f"{best_rmse:.4g}")
        if stale >= resolved.training.early_stopping_patience:
            logger.info("early stopping at step=%d best=%.7g@%d", step, best_rmse, best_step)
            break

    if best_step == 0 and last_step > 0:
        best_rmse, validation_rows = validation_rmse(
            model,
            train_cells,
            validation_cells,
            sampler,
            resolved,
            device,
        )
        best_step = last_step
        pd.DataFrame(validation_rows).to_csv(
            run_dir / f"training/validation_step{last_step}.csv", index=False
        )
        logger.info(
            "final validation step=%d rmse_cycles=%.7g",
            last_step,
            best_rmse,
        )
    final_checkpoint = _checkpoint_payload(
        model, spec, optimizer, amp_scaler, last_step, best_step,
        best_rmse, stale, resolved, split, scalers, rng,
    )
    _atomic_save(final_checkpoint, run_dir / "checkpoints/last.pt")
    pd.DataFrame(records).to_csv(history_path, index=False)
    if not (run_dir / "checkpoints/best.pt").is_file():
        _atomic_save(final_checkpoint, run_dir / "checkpoints/best.pt")
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_step": last_step,
            "best_step": best_step,
            "best_validation_rmse_cycles": best_rmse,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    logger.info("completed run=%s", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train horizon-conditioned MATR RUL ANP")
    parser.add_argument("--config", default="configs/matr_horizon_rul_anp.yaml")
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
        config,
        args.fold,
        resolve_data_root(config, args.data_root),
        resume=args.resume,
        max_steps=args.max_steps,
        task_batch_size=args.task_batch_size,
    )
    print(f"Horizon RUL ANP run: {run_dir}")


if __name__ == "__main__":
    main()

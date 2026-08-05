"""Full second-order MAML training over the five fixed CALCE cells."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import higher
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ExperimentConfig, save_config
from .data import (
    CellTask,
    QuerySequenceDataset,
    RecursivePairDataset,
    sample_support_batch,
    variable_length_collate,
)
from .losses import get_loss
from .model import GRUEncoderDecoder


@dataclass
class MetaTrainingResult:
    model: GRUEncoderDecoder
    history: pd.DataFrame
    best_epoch: int
    best_meta_loss: float
    last_epoch: int
    run_dir: Path


def _capture_rng_state() -> dict[str, Any]:
    import random

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    import random

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu().to(torch.uint8))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.detach().cpu().to(torch.uint8) for item in state["cuda"]])


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _payload(
    model: GRUEncoderDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_meta_loss: float,
    stale_epochs: int,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_meta_loss": best_meta_loss,
        "stale_epochs": stale_epochs,
        "config": config.to_dict(),
        "train_cells": list(config.data.train_cells),
        "history_length": config.data.history_length,
        "algorithm": "full_second_order_maml",
        "track_higher_grads": True,
        "rng_states": _capture_rng_state(),
    }


def load_meta_checkpoint(
    checkpoint: str | Path,
    model: GRUEncoderDecoder,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    source = Path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(f"meta checkpoint not found: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    required = {
        "model_state_dict", "epoch", "best_epoch", "best_meta_loss", "config",
        "train_cells", "history_length", "algorithm", "track_higher_grads", "rng_states",
    }
    if optimizer is not None:
        required.add("optimizer_state_dict")
    missing = required - set(payload)
    if missing:
        raise ValueError(f"meta checkpoint missing keys: {sorted(missing)}")
    if payload["algorithm"] != "full_second_order_maml" or not payload["track_higher_grads"]:
        raise ValueError("checkpoint is not a full second-order MAML checkpoint")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        _restore_rng_state(payload["rng_states"])
    return payload


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _model_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    loss_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    predicted_input_probability: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    prediction = model(
        history=batch["history"],
        input_lengths=batch["input_lengths"],
        target=batch["target"],
        prediction_length=batch["target"].shape[1],
        predicted_input_probability=predicted_input_probability,
        generator=generator,
    )
    return loss_function(prediction, batch["target"], batch["target_mask"])


def _task_post_adaptation_loss(
    meta_model: GRUEncoderDecoder,
    task: CellTask,
    config: ExperimentConfig,
    epoch: int,
    task_index: int,
    device: torch.device,
    loss_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    support, query = task.split(config.data.history_length)
    support_dataset = RecursivePairDataset(support)
    query_loader = DataLoader(
        QuerySequenceDataset(support, query),
        batch_size=config.maml.query_batch_size,
        shuffle=False,
        collate_fn=variable_length_collate,
    )
    inner_optimizer = torch.optim.SGD(
        meta_model.parameters(), lr=config.maml.inner_learning_rate
    )
    support_generator = torch.Generator(device="cpu").manual_seed(
        config.seed + epoch * 10007 + task_index * 101
    )
    teacher_device = "cuda" if device.type == "cuda" else "cpu"
    teacher_generator = torch.Generator(device=teacher_device).manual_seed(
        config.seed + epoch * 20011 + task_index * 211
    )
    support_losses: list[torch.Tensor] = []
    query_losses: list[torch.Tensor] = []
    with higher.innerloop_ctx(
        meta_model,
        inner_optimizer,
        copy_initial_weights=False,
        track_higher_grads=True,
    ) as (task_model, differentiable_optimizer):
        task_model.train()
        for _ in range(config.maml.inner_steps):
            support_batch = sample_support_batch(
                support_dataset,
                config.maml.inner_batch_size,
                support_generator,
                device,
            )
            support_loss = _model_loss(
                task_model,
                support_batch,
                loss_function,
                config.model.predicted_input_probability,
                teacher_generator,
            )
            if not torch.isfinite(support_loss):
                raise FloatingPointError(f"{task.name}: non-finite support loss")
            differentiable_optimizer.step(support_loss)
            support_losses.append(support_loss)
        # The exact query definition in the request is support[1:L] -> all
        # SOH after L. query_batch_size is configurable, though this dataset
        # intentionally contains one complete query sample per task.
        for raw_batch in query_loader:
            query_batch = _move_batch(raw_batch, device)
            query_loss = _model_loss(
                task_model,
                query_batch,
                loss_function,
                config.model.predicted_input_probability,
                teacher_generator,
            )
            if not torch.isfinite(query_loss):
                raise FloatingPointError(f"{task.name}: non-finite query loss")
            query_losses.append(query_loss)
    return torch.stack(support_losses).mean(), torch.stack(query_losses).mean()


def train_meta_model(
    model: GRUEncoderDecoder,
    train_tasks: Sequence[CellTask],
    config: ExperimentConfig,
    device: torch.device,
    run_dir: str | Path,
    logger: logging.Logger,
    resume: str | Path | None = None,
) -> MetaTrainingResult:
    """Train full second-order MAML and select by training-task query loss.

    The paper/request provides no separate meta-validation cells. Therefore
    using the mean post-adaptation query loss of the five training tasks for
    checkpoint selection is an explicit implementation choice. Test cells are
    never consulted here.
    """
    if [task.name for task in train_tasks] != list(config.data.train_cells):
        raise ValueError("train tasks must exactly match the five configured cells in order")
    root = Path(run_dir)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "training").mkdir(parents=True, exist_ok=True)
    save_config(config, root / "config_resolved.yaml")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.maml.outer_learning_rate)
    loss_function = get_loss(config.loss.kind)
    start_epoch = 1
    best_epoch = 0
    best_meta_loss = float("inf")
    stale_epochs = 0
    records: list[dict[str, Any]] = []
    history_path = root / "training/meta_history.csv"
    if resume is not None:
        payload = load_meta_checkpoint(
            resume, model, device, optimizer=optimizer, restore_rng=True
        )
        if list(payload["train_cells"]) != list(config.data.train_cells):
            raise ValueError("resume training-cell split does not match")
        if int(payload["history_length"]) != config.data.history_length:
            raise ValueError("resume history length does not match")
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload["best_epoch"])
        best_meta_loss = float(payload["best_meta_loss"])
        stale_epochs = int(payload.get("stale_epochs", 0))
        if history_path.is_file():
            old = pd.read_csv(history_path)
            records = old[old["epoch"] < start_epoch].to_dict("records")
        logger.info("Resuming full MAML at epoch %d", start_epoch)

    began = time.perf_counter()
    last_epoch = start_epoch - 1
    progress = tqdm(
        range(start_epoch, config.maml.max_epochs + 1),
        desc="second-order-maml",
        unit="epoch",
    )
    for epoch in progress:
        model.train()
        task_support_losses: list[torch.Tensor] = []
        task_query_losses: list[torch.Tensor] = []
        # CuDNN RNN kernels lack the double backward used by second-order MAML.
        # CUDA tensors remain on GPU; only CuDNN's RNN implementation is disabled.
        rnn_backend = (
            torch.backends.cudnn.flags(enabled=False)
            if device.type == "cuda"
            else nullcontext()
        )
        with rnn_backend:
            for task_index, task in enumerate(train_tasks):
                support_loss, query_loss = _task_post_adaptation_loss(
                    model, task, config, epoch, task_index, device, loss_function
                )
                task_support_losses.append(support_loss)
                task_query_losses.append(query_loss)
            # Unweighted arithmetic mean: every one of the five tasks contributes 1/5.
            meta_loss = torch.stack(task_query_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            meta_loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.maml.gradient_clip_norm
        )
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite outer gradient at epoch {epoch}")
        optimizer.step()
        value = float(meta_loss.detach().cpu())
        improved = value < best_meta_loss - config.maml.early_stopping_min_delta
        if improved:
            best_meta_loss = value
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        elapsed = time.perf_counter() - began
        record: dict[str, Any] = {
            "epoch": epoch,
            "meta_loss": value,
            "mean_support_loss": float(torch.stack(task_support_losses).mean().detach().cpu()),
            "gradient_norm": grad_norm,
            "elapsed_seconds": elapsed,
            "outer_learning_rate": optimizer.param_groups[0]["lr"],
        }
        for task, support_loss, query_loss in zip(
            train_tasks, task_support_losses, task_query_losses
        ):
            stem = Path(task.name).stem
            record[f"{stem}_support_loss"] = float(support_loss.detach().cpu())
            record[f"{stem}_query_loss"] = float(query_loss.detach().cpu())
        records.append(record)
        checkpoint = _payload(
            model, optimizer, epoch, best_epoch, best_meta_loss,
            stale_epochs, config,
        )
        if improved:
            _atomic_save(checkpoint, root / "checkpoints/best_meta_model.pt")
        if epoch % config.maml.checkpoint_interval == 0 or epoch == config.maml.max_epochs:
            _atomic_save(checkpoint, root / "checkpoints/last.pt")
            pd.DataFrame(records).to_csv(history_path, index=False)
        progress.set_postfix(meta_loss=f"{value:.6g}", best=f"{best_meta_loss:.6g}")
        if epoch % config.maml.log_interval == 0 or epoch in {start_epoch, config.maml.max_epochs}:
            logger.info(
                "epoch=%d/%d meta_loss=%.8g best=%.8g@%d support=%s query=%s "
                "grad=%.6g stale=%d elapsed=%.1fs",
                epoch, config.maml.max_epochs, value, best_meta_loss, best_epoch,
                {task.name: float(loss.detach().cpu()) for task, loss in zip(train_tasks, task_support_losses)},
                {task.name: float(loss.detach().cpu()) for task, loss in zip(train_tasks, task_query_losses)},
                grad_norm, stale_epochs, elapsed,
            )
        last_epoch = epoch
        if config.maml.early_stopping and stale_epochs >= config.maml.early_stopping_patience:
            logger.info(
                "Meta early stopping at epoch %d; best %.8g at epoch %d",
                epoch, best_meta_loss, best_epoch,
            )
            break
    frame = pd.DataFrame(records)
    frame.to_csv(history_path, index=False)
    best_path = root / "checkpoints/best_meta_model.pt"
    if not best_path.is_file():
        raise RuntimeError("MAML training produced no best checkpoint")
    load_meta_checkpoint(best_path, model, device)
    model.eval()
    return MetaTrainingResult(model, frame, best_epoch, best_meta_loss, last_epoch, root)


def run_optuna_search(
    base_config: ExperimentConfig,
    train_tasks: Sequence[CellTask],
    device: torch.device,
    output_dir: str | Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Optional TPE search for the paper-unspecified outer learning rate."""
    if base_config.maml.optuna_trials <= 0:
        raise ValueError("maml.optuna_trials must be positive for Optuna mode")
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna mode requires `python -m pip install optuna`") from exc
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    def objective(trial: Any) -> float:
        trial_config = copy.deepcopy(base_config)
        trial_config.maml.outer_learning_rate = trial.suggest_float(
            "outer_learning_rate",
            trial_config.maml.optuna_lr_low,
            trial_config.maml.optuna_lr_high,
            log=True,
        )
        trial_dir = root / f"trial_{trial.number:04d}"
        model = GRUEncoderDecoder(
            trial_config.model.hidden_size, trial_config.model.num_layers
        )
        result = train_meta_model(
            model, train_tasks, trial_config, device, trial_dir, logger
        )
        return result.best_meta_loss

    sampler = optuna.samplers.TPESampler(seed=base_config.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=base_config.maml.optuna_trials)
    result = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "trial_count": len(study.trials),
    }
    (root / "optuna_best.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

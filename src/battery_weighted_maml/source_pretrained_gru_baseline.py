"""Non-meta GRU baseline: supervised source pretraining and target fine-tuning."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from .baseline_paths import new_baseline_run_dir, transfer_gru_run_name
from .data.collate import sample_support_batch
from .data.preprocess import preprocess_dataset, summary_record
from .data.support_dataset import PrefixFutureDataset
from .data.task_views import (
    FullCellTrajectory,
    SourceTaskView,
    TargetEvaluationView,
    TargetSupportView,
)
from .evaluation.evaluator import EvaluationResult, evaluate_target
from .evaluation.plots import plot_target_prediction
from .logging_utils import configure_logging, parameter_counts, resolve_device
from .models.gru_seq2seq import GRUSeq2Seq, masked_mse
from .seed import capture_rng_state, make_generator, restore_rng_state, seed_everything


@dataclass
class TransferDataConfig:
    calce_dir: str = "data/CALCE"
    label_path: str = "data/Life labels/CALCE_labels.json"
    history_length: int = 100
    max_forecast_cycle: int | None = None
    eol_threshold: float = 0.8
    source_mode: str = "same_family"


@dataclass
class TransferModelConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    # Probability that the decoder receives the previous ground-truth SOH
    # during source pretraining and target-support fine-tuning.
    teacher_forcing_ratio: float = 0.5


@dataclass
class SourcePretrainingConfig:
    max_epochs: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 5.0
    early_stopping: bool = True
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1.0e-7


@dataclass
class TargetFineTuningConfig:
    learning_rate: float = 0.05
    batch_size: int = 64
    fast_steps: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 15, 20])
    full_max_steps: int = 200
    full_patience: int = 20


@dataclass
class TransferLoggingConfig:
    log_interval: int = 1
    checkpoint_interval: int = 10


@dataclass
class SourcePretrainedGRUConfig:
    seed: int = 42
    device: str = "auto"
    data: TransferDataConfig = field(default_factory=TransferDataConfig)
    model: TransferModelConfig = field(default_factory=TransferModelConfig)
    pretraining: SourcePretrainingConfig = field(default_factory=SourcePretrainingConfig)
    fine_tuning: TargetFineTuningConfig = field(default_factory=TargetFineTuningConfig)
    logging: TransferLoggingConfig = field(default_factory=TransferLoggingConfig)

    def validate(self) -> None:
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be auto, cpu, or a CUDA device")
        if self.data.history_length < 2:
            raise ValueError("data.history_length must be at least 2")
        if self.data.source_mode not in {"same_family", "all_calce"}:
            raise ValueError("data.source_mode must be same_family or all_calce")
        if (
            self.data.max_forecast_cycle is not None
            and self.data.max_forecast_cycle <= self.data.history_length
        ):
            raise ValueError("max_forecast_cycle must exceed history_length")
        if not 0.0 < self.data.eol_threshold < 2.0:
            raise ValueError("data.eol_threshold must be between 0 and 2")
        if self.model.hidden_size <= 0 or self.model.num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0,1)")
        if not 0.0 <= self.model.teacher_forcing_ratio <= 1.0:
            raise ValueError("model.teacher_forcing_ratio must be in [0,1]")
        pretraining = self.pretraining
        if any(
            value <= 0
            for value in (
                pretraining.max_epochs,
                pretraining.learning_rate,
                pretraining.gradient_clip_norm,
                pretraining.early_stopping_patience,
            )
        ):
            raise ValueError("pretraining epoch/rate/clipping/patience must be positive")
        if pretraining.weight_decay < 0 or pretraining.early_stopping_min_delta < 0:
            raise ValueError("pretraining weight decay/min delta cannot be negative")
        fine_tuning = self.fine_tuning
        if any(
            value <= 0
            for value in (
                fine_tuning.learning_rate,
                fine_tuning.batch_size,
                fine_tuning.full_max_steps,
                fine_tuning.full_patience,
            )
        ):
            raise ValueError("fine-tuning rate/batch/steps/patience must be positive")
        if (
            not fine_tuning.fast_steps
            or any(step <= 0 for step in fine_tuning.fast_steps)
            or sorted(set(fine_tuning.fast_steps)) != fine_tuning.fast_steps
            or max(fine_tuning.fast_steps) > fine_tuning.full_max_steps
        ):
            raise ValueError(
                "fine_tuning.fast_steps must be unique, increasing, positive, "
                "and no larger than full_max_steps"
            )
        if self.logging.log_interval <= 0 or self.logging.checkpoint_interval <= 0:
            raise ValueError("logging intervals must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FineTuningResult:
    model: GRUSeq2Seq
    history: pd.DataFrame
    best_loss: float
    best_step: int
    snapshots: dict[int, GRUSeq2Seq] = field(default_factory=dict)


def load_source_pretrained_gru_config(
    path: str | Path,
) -> SourcePretrainedGRUConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"source-pretrained baseline config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("source-pretrained config root must be a mapping")
    unknown = set(raw) - {
        "seed",
        "device",
        "data",
        "model",
        "pretraining",
        "fine_tuning",
        "logging",
    }
    if unknown:
        raise ValueError(f"unknown source-pretrained config keys: {sorted(unknown)}")
    try:
        config = SourcePretrainedGRUConfig(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            data=TransferDataConfig(**raw.get("data", {})),
            model=TransferModelConfig(**raw.get("model", {})),
            pretraining=SourcePretrainingConfig(**raw.get("pretraining", {})),
            fine_tuning=TargetFineTuningConfig(**raw.get("fine_tuning", {})),
            logging=TransferLoggingConfig(**raw.get("logging", {})),
        )
    except TypeError as exc:
        raise ValueError(f"invalid source-pretrained config field: {exc}") from exc
    config.validate()
    return config


def _rooted(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _select_source_names(
    trajectories: dict[str, FullCellTrajectory],
    target_name: str,
    source_mode: str,
) -> list[str]:
    if target_name not in trajectories:
        raise FileNotFoundError(f"target cell was not found: {target_name}")
    if source_mode == "same_family":
        target_family = trajectories[target_name].family
        names = sorted(
            name
            for name, trajectory in trajectories.items()
            if name != target_name and trajectory.family == target_family
        )
    elif source_mode == "all_calce":
        names = sorted(name for name in trajectories if name != target_name)
    else:
        raise ValueError("source_mode must be same_family or all_calce")
    if not names:
        raise ValueError(f"no source cells are available for {target_name}")
    return names


def _make_run_tree(run_dir: Path) -> None:
    for child in (
        "logs",
        "checkpoints",
        "preprocessing",
        "pretraining",
        "adaptation",
        "predictions",
        "metrics",
        "figures",
    ):
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _pretraining_checkpoint_payload(
    model: GRUSeq2Seq,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_selection_loss: float,
    best_epoch: int,
    stale_epochs: int,
    config: SourcePretrainedGRUConfig,
    target_name: str,
    source_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_selection_loss": best_selection_loss,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "config": config.to_dict(),
        "target_file_name": target_name,
        "source_file_names": list(source_names),
        "history_length": config.data.history_length,
        "source_mode": config.data.source_mode,
        "training_strategy": "equal_source_supervised_pretraining",
        "rng_states": capture_rng_state(),
    }


def _load_pretraining_checkpoint(
    path: str | Path,
    model: GRUSeq2Seq,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    restore_rng: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"pretraining checkpoint not found: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    required = {
        "model_state_dict",
        "epoch",
        "best_selection_loss",
        "best_epoch",
        "config",
        "target_file_name",
        "source_file_names",
        "history_length",
        "source_mode",
        "training_strategy",
        "rng_states",
    }
    if optimizer is not None:
        required.add("optimizer_state_dict")
    missing = required - set(payload)
    if missing:
        raise ValueError(f"pretraining checkpoint missing keys: {sorted(missing)}")
    if payload["training_strategy"] != "equal_source_supervised_pretraining":
        raise ValueError("checkpoint is not a source-pretrained GRU baseline")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_states"])
    return payload


def _task_loss(
    model: GRUSeq2Seq,
    task: SourceTaskView,
    device: torch.device,
    teacher_forcing_ratio: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    history = torch.tensor(
        task.support_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    target = torch.tensor(task.query_soh, dtype=torch.float32, device=device).view(
        1, -1, 1
    )
    lengths = torch.tensor([len(task.support_soh)], dtype=torch.long, device=device)
    prediction = model(
        history,
        lengths,
        future_targets=target,
        teacher_forcing_ratio=teacher_forcing_ratio,
        generator=generator,
    )
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite source loss for {task.file_name}: {loss}")
    return loss


@torch.no_grad()
def _recursive_source_losses(
    model: GRUSeq2Seq,
    source_tasks: Sequence[SourceTaskView],
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses = {
        task.file_name: float(
            _task_loss(model, task, device, teacher_forcing_ratio=0.0, generator=None)
            .detach()
            .cpu()
        )
        for task in source_tasks
    }
    model.train(was_training)
    return losses


def train_on_sources(
    model: GRUSeq2Seq,
    source_tasks: Sequence[SourceTaskView],
    config: SourcePretrainedGRUConfig,
    device: torch.device,
    run_dir: Path,
    logger: Any,
    target_name: str,
    resume: str | Path | None = None,
) -> tuple[GRUSeq2Seq, pd.DataFrame, pd.DataFrame, Path]:
    """Train every epoch on every source with equal per-cell loss weight."""
    if not source_tasks:
        raise ValueError("source pretraining requires at least one source task")
    source_names = [task.file_name for task in source_tasks]
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.pretraining.learning_rate,
        weight_decay=config.pretraining.weight_decay,
    )
    epoch_path = run_dir / "pretraining/epoch_history.csv"
    source_path = run_dir / "pretraining/source_loss_history.csv"
    epoch_records: list[dict[str, float | int]] = []
    source_records: list[dict[str, float | int | str]] = []
    start_epoch = 1
    best_selection_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    if resume is not None:
        payload = _load_pretraining_checkpoint(
            resume, model, optimizer, device, restore_rng=True
        )
        if payload["target_file_name"] != target_name:
            raise ValueError("resume checkpoint target does not match")
        if list(payload["source_file_names"]) != source_names:
            raise ValueError("resume checkpoint source cells do not match")
        if int(payload["history_length"]) != config.data.history_length:
            raise ValueError("resume checkpoint history length does not match")
        if payload["source_mode"] != config.data.source_mode:
            raise ValueError("resume checkpoint source mode does not match")
        start_epoch = int(payload["epoch"]) + 1
        best_selection_loss = float(payload["best_selection_loss"])
        best_epoch = int(payload["best_epoch"])
        stale_epochs = int(payload.get("stale_epochs", 0))
        if epoch_path.is_file():
            previous = pd.read_csv(epoch_path)
            epoch_records = previous[previous["epoch"] < start_epoch].to_dict("records")
        if source_path.is_file():
            previous = pd.read_csv(source_path)
            source_records = previous[previous["epoch"] < start_epoch].to_dict("records")
        logger.info("Resuming source pretraining at epoch %d", start_epoch)

    began = time.perf_counter()
    last_epoch = start_epoch - 1
    progress = tqdm(
        range(start_epoch, config.pretraining.max_epochs + 1),
        desc="source-supervised-pretraining",
        unit="epoch",
    )
    for epoch in progress:
        model.train()
        order_generator = torch.Generator(device="cpu").manual_seed(
            config.seed + epoch * 1009
        )
        order = torch.randperm(len(source_tasks), generator=order_generator).tolist()
        teacher_generator = make_generator(config.seed + epoch * 2027, device)
        optimizer.zero_grad(set_to_none=True)
        teacher_losses: dict[str, float] = {}
        for index in order:
            task = source_tasks[int(index)]
            loss = _task_loss(
                model,
                task,
                device,
                teacher_forcing_ratio=config.model.teacher_forcing_ratio,
                generator=teacher_generator,
            )
            # Equal task weighting prevents long-lived cells from dominating.
            (loss / len(source_tasks)).backward()
            teacher_losses[task.file_name] = float(loss.detach().cpu())
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.pretraining.gradient_clip_norm
            )
            .detach()
            .cpu()
        )
        optimizer.step()

        recursive_losses = _recursive_source_losses(model, source_tasks, device)
        train_loss = sum(teacher_losses.values()) / len(teacher_losses)
        selection_loss = sum(recursive_losses.values()) / len(recursive_losses)
        improved = (
            selection_loss
            < best_selection_loss - config.pretraining.early_stopping_min_delta
        )
        if improved:
            best_selection_loss = selection_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        elapsed = time.perf_counter() - began
        epoch_records.append(
            {
                "epoch": epoch,
                "task_balanced_teacher_forced_mse": train_loss,
                "task_balanced_recursive_mse": selection_loss,
                "gradient_norm": gradient_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed,
            }
        )
        for task in source_tasks:
            source_records.append(
                {
                    "epoch": epoch,
                    "source": task.file_name,
                    "teacher_forced_mse": teacher_losses[task.file_name],
                    "recursive_mse": recursive_losses[task.file_name],
                    "source_weight": 1.0 / len(source_tasks),
                }
            )
        payload = _pretraining_checkpoint_payload(
            model,
            optimizer,
            epoch,
            best_selection_loss,
            best_epoch,
            stale_epochs,
            config,
            target_name,
            source_names,
        )
        if improved:
            _save_checkpoint(payload, run_dir / "checkpoints/pretrain_best.pt")
        if (
            epoch % config.logging.checkpoint_interval == 0
            or epoch == config.pretraining.max_epochs
        ):
            _save_checkpoint(payload, run_dir / "checkpoints/pretrain_last.pt")
            pd.DataFrame(epoch_records).to_csv(epoch_path, index=False)
            pd.DataFrame(source_records).to_csv(source_path, index=False)
        last_epoch = epoch
        progress.set_postfix(
            train=f"{train_loss:.5g}", recursive=f"{selection_loss:.5g}"
        )
        if epoch % config.logging.log_interval == 0 or epoch in {
            start_epoch,
            config.pretraining.max_epochs,
        }:
            logger.info(
                "source_epoch=%d/%d teacher_forced_task_mean=%.8g "
                "recursive_task_mean=%.8g best=%.8g@%d stale=%d grad=%.6g elapsed=%.1fs",
                epoch,
                config.pretraining.max_epochs,
                train_loss,
                selection_loss,
                best_selection_loss,
                best_epoch,
                stale_epochs,
                gradient_norm,
                elapsed,
            )
        if (
            config.pretraining.early_stopping
            and stale_epochs >= config.pretraining.early_stopping_patience
        ):
            _save_checkpoint(payload, run_dir / "checkpoints/pretrain_last.pt")
            logger.info(
                "Source pretraining early stopping at epoch %d; best recursive "
                "source MSE %.8g at epoch %d",
                epoch,
                best_selection_loss,
                best_epoch,
            )
            break

    epoch_frame = pd.DataFrame(epoch_records)
    source_frame = pd.DataFrame(source_records)
    epoch_frame.to_csv(epoch_path, index=False)
    source_frame.to_csv(source_path, index=False)
    if last_epoch < 1 and not (run_dir / "checkpoints/pretrain_last.pt").is_file():
        raise RuntimeError("source pretraining did not run and has no last checkpoint")
    selected_path = (
        run_dir / "checkpoints/pretrain_best.pt"
        if config.pretraining.early_stopping
        else run_dir / "checkpoints/pretrain_last.pt"
    )
    if not selected_path.is_file():
        raise RuntimeError(f"selected pretraining checkpoint is missing: {selected_path}")
    _load_pretraining_checkpoint(selected_path, model, None, device)
    model.eval()
    return model, epoch_frame, source_frame, selected_path


def _plot_pretraining(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        frame["epoch"],
        frame["task_balanced_teacher_forced_mse"],
        label="teacher-forced task mean",
    )
    axis.plot(
        frame["epoch"],
        frame["task_balanced_recursive_mse"],
        label="recursive task mean",
    )
    axis.set(xlabel="Epoch", ylabel="MSE", title="Equal-source supervised pretraining")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def fine_tune_on_target_support(
    model: GRUSeq2Seq,
    target: TargetSupportView,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    teacher_forcing_ratio: float,
    device: torch.device,
    generator: torch.Generator,
    patience: int | None = None,
    capture_steps: Sequence[int] | None = None,
) -> FineTuningResult:
    """Ordinary full-model SGD using only prefix pairs inside target cycles 1..L."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    requested_steps = set(int(step) for step in (capture_steps or ()))
    if any(step <= 0 or step > max_steps for step in requested_steps):
        raise ValueError("capture_steps must lie between 1 and max_steps")
    adapted = copy.deepcopy(model).to(device)
    adapted.train()
    optimizer = torch.optim.SGD(adapted.parameters(), lr=learning_rate)
    dataset = PrefixFutureDataset(target.soh, target.features)
    best_state = copy.deepcopy(adapted.state_dict())
    best_loss = float("inf")
    best_step = 0
    stale = 0
    records: list[dict[str, float | int]] = []
    snapshots: dict[int, GRUSeq2Seq] = {}
    for step in range(1, max_steps + 1):
        batch = sample_support_batch(dataset, batch_size, generator, device)
        prediction = adapted(
            batch["history"],
            batch["history_lengths"],
            future_targets=batch["future"],
            teacher_forcing_ratio=teacher_forcing_ratio,
            generator=generator,
        )
        loss = masked_mse(prediction, batch["future"], batch["future_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite target fine-tuning loss at step {step}: {loss}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        records.append({"step": step, "support_loss": value})
        if step in requested_steps:
            snapshot = copy.deepcopy(adapted)
            snapshot.eval()
            snapshots[step] = snapshot
        if value < best_loss - 1.0e-12:
            best_loss = value
            best_step = step
            best_state = copy.deepcopy(adapted.state_dict())
            stale = 0
        else:
            stale += 1
        if patience is not None and stale >= patience:
            break
    adapted.load_state_dict(best_state)
    adapted.eval()
    return FineTuningResult(
        model=adapted,
        history=pd.DataFrame(records, columns=["step", "support_loss"]),
        best_loss=best_loss,
        best_step=best_step,
        snapshots=snapshots,
    )


def _save_adapted_checkpoint(
    result: FineTuningResult,
    path: Path,
    config: SourcePretrainedGRUConfig,
    mode: str,
) -> None:
    _save_checkpoint(
        {
            "model_state_dict": result.model.state_dict(),
            "mode": mode,
            "best_support_loss": result.best_loss,
            "best_step": result.best_step,
            "config": config.to_dict(),
        },
        path,
    )


def _evaluate_and_plot(
    model: GRUSeq2Seq,
    evaluation_view: TargetEvaluationView,
    config: SourcePretrainedGRUConfig,
    mode: str,
    title: str,
    run_dir: Path,
    logger: Any,
) -> EvaluationResult:
    result = evaluate_target(
        model,
        evaluation_view,
        config.data.history_length,
        config.data.max_forecast_cycle,
        config.data.eol_threshold,
        mode,
        run_dir,
        logger,
    )
    plot_target_prediction(
        result.predictions,
        run_dir / f"figures/target_soh_{mode}.png",
        title,
        metrics=result.metrics,
    )
    return result


def run_source_pretrained_gru_baseline(
    config: SourcePretrainedGRUConfig,
    target_name: str,
    project_root: str | Path = ".",
    resume: str | Path | None = None,
    smoke_test: bool = False,
    trajectories: dict[str, FullCellTrajectory] | None = None,
    run_name: str | None = None,
) -> Path:
    """Pretrain on source futures, fine-tune on target support, then evaluate."""
    root = Path(project_root).resolve()
    resolved = copy.deepcopy(config)
    if smoke_test:
        resolved.pretraining.max_epochs = 2
        resolved.pretraining.early_stopping = False
        resolved.fine_tuning.fast_steps = [1, 2]
        resolved.fine_tuning.full_max_steps = 2
        resolved.fine_tuning.full_patience = 2
        resolved.logging.log_interval = 1
        resolved.logging.checkpoint_interval = 1
        resolved.data.max_forecast_cycle = resolved.data.history_length + 3
    resolved.validate()
    seed_everything(resolved.seed)
    if trajectories is None:
        trajectories = preprocess_dataset(
            _rooted(resolved.data.calce_dir, root),
            _rooted(resolved.data.label_path, root),
            root / "outputs/preprocessed",
            configure_logging(None),
        )
    source_names = _select_source_names(
        trajectories, target_name, resolved.data.source_mode
    )
    target_full = trajectories[target_name]
    target_support = target_full.target_support(resolved.data.history_length, ("soh",))
    source_tasks = [
        trajectories[name].source_task(resolved.data.history_length, ("soh",))
        for name in source_names
    ]
    if resume is not None:
        if run_name is not None:
            raise ValueError("--run-name cannot be combined with --resume")
        run_dir = Path(resume).resolve().parent.parent
    else:
        run_dir = new_baseline_run_dir(
            root,
            automatic_name=transfer_gru_run_name(
                target_name,
                resolved.data.history_length,
                resolved.data.source_mode,
                resolved.seed,
            ),
            requested_name=run_name,
        )
    _make_run_tree(run_dir)
    logger = configure_logging(run_dir / "logs/train.log")
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    pd.DataFrame([summary_record(trajectories[name]) for name in source_names]).to_csv(
        run_dir / "preprocessing/source_summary.csv", index=False
    )
    pd.DataFrame(
        {
            "file_name": target_name,
            "cycle": target_support.cycles,
            "soh": target_support.soh,
        }
    ).to_csv(run_dir / "preprocessing/target_support.csv", index=False)

    device = resolve_device(resolved.device)
    model = GRUSeq2Seq(
        input_size=1,
        hidden_size=resolved.model.hidden_size,
        num_layers=resolved.model.num_layers,
        dropout=resolved.model.dropout,
    )
    total, trainable = parameter_counts(model)
    manifest: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "experiment": "source_pretrained_target_finetuned_gru",
        "run_name": run_dir.name,
        "output_group": "baseline",
        "meta_learning": False,
        "weighted_meta_learning": False,
        "source_weighting": "uniform_task_balanced",
        "source_pretraining": True,
        "target_fine_tuning": True,
        "target_future_used_for_training_or_selection": False,
        "target": target_name,
        "sources": source_names,
        "source_mode": resolved.data.source_mode,
        "history_length": resolved.data.history_length,
        "input_features": ["soh"],
        "device": str(device),
        "seed": resolved.seed,
        "parameters": total,
        "trainable_parameters": trainable,
        "resolved_config": resolved.to_dict(),
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )
    logger.info(
        "Starting non-meta transfer baseline target=%s sources=%s L=%d "
        "input=['soh'] teacher_forcing=%.3f device=%s parameters=%d",
        target_name,
        source_names,
        resolved.data.history_length,
        resolved.model.teacher_forcing_ratio,
        device,
        total,
    )

    model, pretrain_history, _, selected_pretrain = train_on_sources(
        model,
        source_tasks,
        resolved,
        device,
        run_dir,
        logger,
        target_name,
        resume=resume,
    )
    _plot_pretraining(pretrain_history, run_dir / "figures/pretraining_loss.png")

    # Fine-tuning can only see TargetSupportView, which physically has no
    # target future or EOL label.
    fast_steps = list(resolved.fine_tuning.fast_steps)
    fast = fine_tune_on_target_support(
        model,
        target_support,
        max_steps=max(fast_steps),
        learning_rate=resolved.fine_tuning.learning_rate,
        batch_size=resolved.fine_tuning.batch_size,
        teacher_forcing_ratio=resolved.model.teacher_forcing_ratio,
        device=device,
        generator=make_generator(resolved.seed + 2001, device),
        patience=None,
        capture_steps=fast_steps,
    )
    full = fine_tune_on_target_support(
        model,
        target_support,
        max_steps=resolved.fine_tuning.full_max_steps,
        learning_rate=resolved.fine_tuning.learning_rate,
        batch_size=resolved.fine_tuning.batch_size,
        teacher_forcing_ratio=resolved.model.teacher_forcing_ratio,
        device=device,
        generator=make_generator(resolved.seed + 3001, device),
        patience=resolved.fine_tuning.full_patience,
    )
    fast.history.to_csv(run_dir / "adaptation/fast_history.csv", index=False)
    full.history.to_csv(run_dir / "adaptation/full_history.csv", index=False)
    _save_adapted_checkpoint(
        full, run_dir / "checkpoints/target_full_best.pt", resolved, "transfer_full"
    )
    for step, snapshot in fast.snapshots.items():
        _save_checkpoint(
            {
                "model_state_dict": snapshot.state_dict(),
                "mode": f"transfer_fast_{step}",
                "step": step,
                "config": resolved.to_dict(),
            },
            run_dir / f"checkpoints/target_fast_{step}.pt",
        )

    # Reveal the target future only after pretraining and every fine-tuned
    # checkpoint have already been selected using source/target-support data.
    evaluation_view = TargetEvaluationView.after_training(
        target_full, resolved.data.history_length, ("soh",)
    )
    results: dict[str, EvaluationResult] = {}
    results["transfer_0"] = _evaluate_and_plot(
        model,
        evaluation_view,
        resolved,
        "transfer_0",
        f"{target_name} source-pretrained, 0-step target fine-tuning",
        run_dir,
        logger,
    )
    for step in fast_steps:
        mode = f"transfer_fast_{step}"
        results[mode] = _evaluate_and_plot(
            fast.snapshots[step],
            evaluation_view,
            resolved,
            mode,
            f"{target_name} source-pretrained + {step}-step target fine-tuning",
            run_dir,
            logger,
        )
    results["transfer_full"] = _evaluate_and_plot(
        full.model,
        evaluation_view,
        resolved,
        "transfer_full",
        f"{target_name} source-pretrained + full target fine-tuning",
        run_dir,
        logger,
    )
    metric_rows = [
        {"mode": mode, **result.metrics} for mode, result in results.items()
    ]
    pd.DataFrame(metric_rows).to_csv(
        run_dir / "metrics/transfer_metrics_summary.csv", index=False
    )
    selected_payload = torch.load(
        selected_pretrain, map_location="cpu", weights_only=False
    )
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_pretraining_checkpoint": str(selected_pretrain),
            "selected_pretraining_epoch": int(selected_payload["epoch"]),
            "best_pretraining_epoch": int(selected_payload["best_epoch"]),
            "best_source_recursive_mse": float(
                selected_payload["best_selection_loss"]
            ),
            "fast_steps": fast_steps,
            "full_fine_tuning_best_step": full.best_step,
            "full_fine_tuning_best_support_loss": full.best_loss,
            "metrics_by_mode": {
                mode: result.metrics for mode, result in results.items()
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )
    logger.info("Completed source-pretrained GRU baseline: %s", run_dir)
    return run_dir


def source_pretrained_baseline_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Supervised source pretraining plus target-support fine-tuning, "
            "without MAML or source weighting"
        )
    )
    parser.add_argument("--target", default="CALCE_CX2_37.pkl")
    parser.add_argument(
        "--config",
        default="configs/baseline/source_pretrained_gru_l100_soh.yaml",
    )
    parser.add_argument(
        "--source-mode", choices=["same_family", "all_calce"]
    )
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--run-name")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_source_pretrained_gru_config(args.config)
    if args.source_mode is not None:
        config.data.source_mode = args.source_mode
    if args.device is not None:
        config.device = args.device
    run_source_pretrained_gru_baseline(
        config,
        target_name=args.target,
        project_root=Path.cwd(),
        resume=args.resume,
        smoke_test=args.smoke_test,
        run_name=args.run_name,
    )

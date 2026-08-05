"""Training and CLI orchestration for the plain SOH-only GRU baseline."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from .data.calce_loader import load_calce_pickle, load_eol_labels
from .data.preprocess import preprocess_cell
from .data.soh_window_dataset import SOHWindowDataset
from .data.task_views import TargetEvaluationView, TargetSupportView
from .evaluation.evaluator import EvaluationResult, evaluate_target
from .evaluation.plots import plot_target_prediction
from .logging_utils import configure_logging, parameter_counts, resolve_device
from .models.gru_encoder_decoder_baseline import SOHGRUEncoderDecoder
from .seed import capture_rng_state, make_generator, restore_rng_state, seed_everything


@dataclass
class BaselineDataConfig:
    calce_dir: str = "data/CALCE"
    label_path: str = "data/Life labels/CALCE_labels.json"
    history_length: int = 500
    # None means forecast through the target trajectory's final observed cycle.
    max_forecast_cycle: int | None = None
    eol_threshold: float = 0.8


@dataclass
class BaselineModelConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0


@dataclass
class BaselineTrainingConfig:
    max_epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    encoder_window: int = 100
    forecast_horizon: int = 25
    window_stride: int = 1
    validation_cycles: int = 50
    teacher_forcing_ratio: float = 0.5
    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1.0e-7


@dataclass
class BaselineLoggingConfig:
    log_interval: int = 1
    checkpoint_interval: int = 1


@dataclass
class GRUBaselineConfig:
    seed: int = 42
    device: str = "auto"
    data: BaselineDataConfig = field(default_factory=BaselineDataConfig)
    model: BaselineModelConfig = field(default_factory=BaselineModelConfig)
    training: BaselineTrainingConfig = field(default_factory=BaselineTrainingConfig)
    logging: BaselineLoggingConfig = field(default_factory=BaselineLoggingConfig)

    def validate(self) -> None:
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device")
        if self.data.history_length < 3:
            raise ValueError("history_length must be at least 3")
        if (
            self.data.max_forecast_cycle is not None
            and self.data.max_forecast_cycle <= self.data.history_length
        ):
            raise ValueError("max_forecast_cycle must exceed history_length")
        if not 0.0 < self.data.eol_threshold < 2.0:
            raise ValueError("eol_threshold must be between 0 and 2")
        if self.model.hidden_size <= 0 or self.model.num_layers <= 0:
            raise ValueError("model sizes must be positive")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        training = self.training
        positive = {
            "max_epochs": training.max_epochs,
            "batch_size": training.batch_size,
            "learning_rate": training.learning_rate,
            "encoder_window": training.encoder_window,
            "forecast_horizon": training.forecast_horizon,
            "window_stride": training.window_stride,
            "validation_cycles": training.validation_cycles,
            "gradient_clip_norm": training.gradient_clip_norm,
            "early_stopping_patience": training.early_stopping_patience,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"training values must be positive: {positive}")
        if training.weight_decay < 0 or training.early_stopping_min_delta < 0:
            raise ValueError("weight_decay and early_stopping_min_delta cannot be negative")
        if not 0.0 <= training.teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        train_cycles = self.data.history_length - training.validation_cycles
        required = training.encoder_window + training.forecast_horizon
        if train_cycles < required:
            raise ValueError(
                "training prefix is too short: history_length - validation_cycles "
                f"must be at least encoder_window + forecast_horizon ({required})"
            )
        if self.logging.log_interval <= 0 or self.logging.checkpoint_interval <= 0:
            raise ValueError("logging intervals must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_gru_baseline_config(path: str | Path) -> GRUBaselineConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"baseline configuration file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("baseline configuration root must be a mapping")
    unknown = set(raw) - {"seed", "device", "data", "model", "training", "logging"}
    if unknown:
        raise ValueError(f"unknown baseline config keys: {sorted(unknown)}")
    try:
        config = GRUBaselineConfig(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            data=BaselineDataConfig(**raw.get("data", {})),
            model=BaselineModelConfig(**raw.get("model", {})),
            training=BaselineTrainingConfig(**raw.get("training", {})),
            logging=BaselineLoggingConfig(**raw.get("logging", {})),
        )
    except TypeError as exc:
        raise ValueError(f"invalid baseline config field: {exc}") from exc
    config.validate()
    return config


def _rooted(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _make_run_tree(run_dir: Path) -> None:
    for child in ("logs", "checkpoints", "preprocessing", "training", "predictions", "metrics", "figures"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    model: SOHGRUEncoderDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    stale_epochs: int,
    config: GRUBaselineConfig,
    target_name: str,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "config": config.to_dict(),
        "target_file_name": target_name,
        "history_length": config.data.history_length,
        "rng_states": capture_rng_state(),
    }


def _load_baseline_checkpoint(
    path: str | Path,
    model: SOHGRUEncoderDecoder,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    restore_rng: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    required = {
        "model_state_dict", "epoch", "best_validation_loss", "best_epoch",
        "config", "target_file_name", "history_length", "rng_states",
    }
    missing = required - set(payload)
    if optimizer is not None and "optimizer_state_dict" not in payload:
        missing.add("optimizer_state_dict")
    if missing:
        raise ValueError(f"baseline checkpoint missing keys: {sorted(missing)}")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_states"])
    return payload


def _validation_loss(
    model: SOHGRUEncoderDecoder,
    support_soh: np.ndarray,
    split: int,
    device: torch.device,
) -> float:
    history = torch.tensor(
        support_soh[:split], dtype=torch.float32, device=device
    ).view(1, -1, 1)
    target = torch.tensor(
        support_soh[split:], dtype=torch.float32, device=device
    ).view(1, -1, 1)
    model.eval()
    with torch.no_grad():
        prediction = model(history, horizon=target.shape[1], teacher_forcing_ratio=0.0)
        loss = torch.mean((prediction - target).square())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite validation loss: {loss}")
    return float(loss.cpu())


def train_gru_baseline(
    model: SOHGRUEncoderDecoder,
    target_support: TargetSupportView,
    config: GRUBaselineConfig,
    device: torch.device,
    run_dir: Path,
    logger: Any,
    resume: str | Path | None = None,
) -> tuple[SOHGRUEncoderDecoder, pd.DataFrame, int, float]:
    """Train only on the target's initial-L SOH; no source or meta objective exists."""
    training = config.training
    split = target_support.history_length - training.validation_cycles
    dataset = SOHWindowDataset(
        target_support.soh[:split],
        encoder_window=training.encoder_window,
        forecast_horizon=training.forecast_horizon,
        stride=training.window_stride,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    history_path = run_dir / "training/epoch_history.csv"
    records: list[dict[str, float | int]] = []
    start_epoch = 1
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    if resume is not None:
        payload = _load_baseline_checkpoint(
            resume, model, optimizer, device, restore_rng=True
        )
        if payload["target_file_name"] != target_support.file_name:
            raise ValueError("resume checkpoint target does not match")
        if int(payload["history_length"]) != target_support.history_length:
            raise ValueError("resume checkpoint history length does not match")
        start_epoch = int(payload["epoch"]) + 1
        best_validation_loss = float(payload["best_validation_loss"])
        best_epoch = int(payload["best_epoch"])
        stale_epochs = int(payload.get("stale_epochs", 0))
        if history_path.is_file():
            previous = pd.read_csv(history_path)
            previous = previous[previous["epoch"] < start_epoch]
            records = previous.to_dict("records")
        logger.info("Resuming plain GRU training at epoch %d", start_epoch)

    began = time.perf_counter()
    progress = tqdm(
        range(start_epoch, training.max_epochs + 1), desc="gru-training", unit="epoch"
    )
    last_epoch = start_epoch - 1
    for epoch in progress:
        model.train()
        permutation_generator = torch.Generator(device="cpu").manual_seed(
            config.seed + epoch * 1009
        )
        indices = torch.randperm(len(dataset), generator=permutation_generator)
        teacher_generator = make_generator(config.seed + epoch * 2027, device)
        total_squared_error = 0.0
        total_values = 0
        epoch_grad_norm = 0.0
        for offset in range(0, len(indices), training.batch_size):
            selected = indices[offset:offset + training.batch_size].tolist()
            samples = [dataset[int(index)] for index in selected]
            histories = torch.stack([sample[0] for sample in samples]).to(device)
            futures = torch.stack([sample[1] for sample in samples]).to(device)
            prediction = model(
                histories,
                future_targets=futures,
                teacher_forcing_ratio=training.teacher_forcing_ratio,
                generator=teacher_generator,
            )
            loss = torch.mean((prediction - futures).square())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}: {loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip_norm
            )
            epoch_grad_norm = max(epoch_grad_norm, float(grad_norm_tensor.detach().cpu()))
            optimizer.step()
            value_count = futures.numel()
            total_squared_error += float(loss.detach().cpu()) * value_count
            total_values += value_count
        train_loss = total_squared_error / total_values
        validation_loss = _validation_loss(model, target_support.soh, split, device)
        improved = validation_loss < best_validation_loss - training.early_stopping_min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        elapsed = time.perf_counter() - began
        record = {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": validation_loss,
            "gradient_norm_max": epoch_grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        records.append(record)
        payload = _checkpoint_payload(
            model, optimizer, epoch, best_validation_loss, best_epoch,
            stale_epochs, config, target_support.file_name,
        )
        if improved:
            _save_checkpoint(payload, run_dir / "checkpoints/best_validation.pt")
        if epoch % config.logging.checkpoint_interval == 0 or epoch == training.max_epochs:
            _save_checkpoint(payload, run_dir / "checkpoints/last.pt")
            pd.DataFrame(records).to_csv(history_path, index=False)
        last_epoch = epoch
        progress.set_postfix(train=f"{train_loss:.5g}", val=f"{validation_loss:.5g}")
        if epoch % config.logging.log_interval == 0 or epoch in {start_epoch, training.max_epochs}:
            logger.info(
                "epoch=%d/%d train_mse=%.7g validation_mse=%.7g best=%.7g@%d "
                "stale=%d grad=%.5g elapsed=%.1fs",
                epoch, training.max_epochs, train_loss, validation_loss,
                best_validation_loss, best_epoch, stale_epochs, epoch_grad_norm, elapsed,
            )
        if stale_epochs >= training.early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d; best validation MSE %.7g at epoch %d",
                epoch, best_validation_loss, best_epoch,
            )
            break
    frame = pd.DataFrame(records)
    frame.to_csv(history_path, index=False)
    best_path = run_dir / "checkpoints/best_validation.pt"
    if not best_path.is_file():
        raise RuntimeError("plain GRU training produced no best validation checkpoint")
    _load_baseline_checkpoint(best_path, model, None, device)
    model.eval()
    return model, frame, last_epoch, best_validation_loss


def _plot_baseline_history(frame: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["epoch"], frame["train_mse"], label="train MSE")
    axis.plot(frame["epoch"], frame["validation_mse"], label="validation MSE")
    axis.set(xlabel="Epoch", ylabel="MSE", title="Plain GRU encoder-decoder loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_gru_baseline(
    config: GRUBaselineConfig,
    target_name: str,
    project_root: str | Path = ".",
    resume: str | Path | None = None,
    smoke_test: bool = False,
    target_trajectory: Any | None = None,
) -> Path:
    """Train and evaluate one target-specific, non-meta GRU baseline."""
    root = Path(project_root).resolve()
    resolved = copy.deepcopy(config)
    if smoke_test:
        resolved.training.max_epochs = 2
        resolved.training.early_stopping_patience = 2
        resolved.logging.log_interval = 1
        resolved.logging.checkpoint_interval = 1
        resolved.data.max_forecast_cycle = resolved.data.history_length + 10
    resolved.validate()
    seed_everything(resolved.seed)
    if resume is not None:
        run_dir = Path(resume).resolve().parent.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root / "outputs/runs" / (
            f"{timestamp}_gru_baseline_{Path(target_name).stem}_"
            f"L{resolved.data.history_length}_seed{resolved.seed}"
        )
    _make_run_tree(run_dir)
    logger = configure_logging(run_dir / "logs/train.log")
    if target_trajectory is None:
        labels = load_eol_labels(_rooted(resolved.data.label_path, root))
        if target_name not in labels:
            raise ValueError(f"missing EOL label for target: {target_name}")
        target_path = _rooted(resolved.data.calce_dir, root) / target_name
        target_trajectory = preprocess_cell(
            load_calce_pickle(target_path), labels[target_name], logger=logger
        )
    target_support = target_trajectory.target_support(
        resolved.data.history_length, ("soh",)
    )
    config_path = run_dir / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(resolved.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "file_name": target_name,
            "cycle": target_support.cycles,
            "soh": target_support.soh,
        }
    ).to_csv(run_dir / "preprocessing/target_support.csv", index=False)
    device = resolve_device(resolved.device)
    model = SOHGRUEncoderDecoder(
        hidden_size=resolved.model.hidden_size,
        num_layers=resolved.model.num_layers,
        dropout=resolved.model.dropout,
    )
    total, trainable = parameter_counts(model)
    manifest: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "experiment": "plain_gru_encoder_decoder",
        "weighted_meta_learning": False,
        "target": target_name,
        "history_length": resolved.data.history_length,
        "input_features": ["soh"],
        "device": str(device),
        "seed": resolved.seed,
        "parameters": total,
        "trainable_parameters": trainable,
        "resolved_config": resolved.to_dict(),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )
    logger.info(
        "Starting plain GRU baseline target=%s L=%d input=['soh'] device=%s "
        "parameters=%d trainable=%d weighted_meta_learning=False config=%s",
        target_name, resolved.data.history_length, device, total, trainable,
        resolved.to_dict(),
    )
    model, history, last_epoch, best_validation_loss = train_gru_baseline(
        model, target_support, resolved, device, run_dir, logger, resume=resume
    )
    # Future SOH and EOL become accessible only after model selection is finished.
    evaluation_view = TargetEvaluationView.after_training(
        target_trajectory, resolved.data.history_length, ("soh",)
    )
    result: EvaluationResult = evaluate_target(
        model,
        evaluation_view,
        resolved.data.history_length,
        resolved.data.max_forecast_cycle,
        resolved.data.eol_threshold,
        "gru_baseline",
        run_dir,
        logger,
    )
    _plot_baseline_history(history, run_dir / "figures/training_loss.png")
    plot_target_prediction(
        result.predictions,
        run_dir / "figures/target_soh_gru_baseline.png",
        f"{target_name} plain GRU encoder-decoder",
        metrics=result.metrics,
    )
    best_payload = torch.load(
        run_dir / "checkpoints/best_validation.pt",
        map_location="cpu",
        weights_only=False,
    )
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_epoch": last_epoch,
            "best_epoch": int(best_payload["best_epoch"]),
            "best_validation_mse": best_validation_loss,
            "metrics": result.metrics,
        }
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )
    logger.info("Completed plain GRU baseline: %s", run_dir)
    return run_dir


def baseline_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a plain SOH-only GRU encoder-decoder without meta-learning"
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--config", default="configs/gru_baseline_l500.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_gru_baseline_config(args.config)
    run_gru_baseline(
        config,
        target_name=args.target,
        project_root=Path.cwd(),
        resume=args.resume,
        smoke_test=args.smoke_test,
    )

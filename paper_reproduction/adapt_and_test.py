"""Meta-test adaptation, recursive rollout, and result serialization."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import CellTask, RecursivePairDataset, sample_support_batch, variable_length_collate
from .losses import get_loss
from .metrics import evaluate_prediction
from .model import GRUEncoderDecoder


@dataclass
class AdaptationResult:
    model: GRUEncoderDecoder
    history: pd.DataFrame
    best_step: int
    best_support_loss: float


def _loss_on_batch(
    model: GRUEncoderDecoder,
    batch: dict[str, torch.Tensor],
    loss_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    predicted_input_probability: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    prediction = model(
        batch["history"],
        input_lengths=batch["input_lengths"],
        target=batch["target"],
        prediction_length=batch["target"].shape[1],
        predicted_input_probability=predicted_input_probability,
        generator=generator,
    )
    return loss_function(prediction, batch["target"], batch["target_mask"])


def _fixed_validation_batch(
    dataset: RecursivePairDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count = min(batch_size, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, steps=count).round().long().unique().tolist()
    batch = variable_length_collate([dataset[int(index)] for index in indices])
    return {key: value.to(device) for key, value in batch.items()}


def adapt_model(
    meta_model: GRUEncoderDecoder,
    support_soh: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    max_steps: int,
    seed_offset: int,
    patience: int | None = None,
) -> AdaptationResult:
    """Adapt all encoder, decoder, and output parameters on support pairs only."""
    adapted = copy.deepcopy(meta_model).to(device)
    dataset = RecursivePairDataset(support_soh)
    loss_function = get_loss(config.loss.kind)
    optimizer = torch.optim.SGD(
        adapted.parameters(), lr=config.adaptation.learning_rate
    )
    support_generator = torch.Generator(device="cpu").manual_seed(
        config.seed + seed_offset
    )
    teacher_device = "cuda" if device.type == "cuda" else "cpu"
    teacher_generator = torch.Generator(device=teacher_device).manual_seed(
        config.seed + seed_offset + 1
    )
    validation_batch = _fixed_validation_batch(
        dataset, config.adaptation.batch_size, device
    )
    best_state = copy.deepcopy(adapted.state_dict())
    best_loss = float("inf")
    best_step = 0
    stale = 0
    records: list[dict[str, float | int]] = []
    if max_steps == 0:
        adapted.eval()
        return AdaptationResult(
            adapted,
            pd.DataFrame(columns=["step", "train_support_loss", "recursive_support_loss"]),
            0,
            float("nan"),
        )
    for step in range(1, max_steps + 1):
        adapted.train()
        batch = sample_support_batch(
            dataset,
            config.adaptation.batch_size,
            support_generator,
            device,
        )
        train_loss = _loss_on_batch(
            adapted,
            batch,
            loss_function,
            config.model.predicted_input_probability,
            teacher_generator,
        )
        if not torch.isfinite(train_loss):
            raise FloatingPointError(f"non-finite adaptation loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), config.maml.gradient_clip_norm)
        optimizer.step()
        # Paper does not define complete-adaptation early stopping. This fixed
        # support-only probe batch and fully recursive loss are implementation choices.
        adapted.eval()
        with torch.no_grad():
            recursive_loss = _loss_on_batch(
                adapted, validation_batch, loss_function, 1.0, None
            )
        value = float(recursive_loss.cpu())
        records.append(
            {
                "step": step,
                "train_support_loss": float(train_loss.detach().cpu()),
                "recursive_support_loss": value,
            }
        )
        if value < best_loss - config.adaptation.complete_min_delta:
            best_loss = value
            best_step = step
            best_state = copy.deepcopy(adapted.state_dict())
            stale = 0
        else:
            stale += 1
        if patience is not None and stale >= patience:
            break
    if patience is not None:
        # Complete adaptation returns the support-only early-stopping winner.
        adapted.load_state_dict(best_state)
    else:
        # Fast adaptation must mean exactly N SGD updates, even if an earlier
        # intermediate step happened to have a smaller probe loss.
        best_step = max_steps
        best_loss = float(records[-1]["recursive_support_loss"])
    adapted.eval()
    return AdaptationResult(adapted, pd.DataFrame(records), best_step, best_loss)


def _evaluate_one(
    model: GRUEncoderDecoder,
    task: CellTask,
    history_length: int,
    mode: str,
    config: ExperimentConfig,
    output_dir: Path,
) -> dict[str, object]:
    support, _ = task.split(history_length)
    current_cycle = int(task.cycles[history_length - 1])
    horizon = config.evaluation.max_forecast_cycle - current_cycle
    if horizon <= 0:
        raise ValueError("max_forecast_cycle must be after the current cycle")
    forecast_tensor = model.recursive_forecast(support, horizon)
    forecast = forecast_tensor[0, :, 0].detach().cpu().numpy().astype(float)
    if not np.all(np.isfinite(forecast)):
        first_bad = int(np.flatnonzero(~np.isfinite(forecast))[0])
        forecast = forecast[:first_bad]
    forecast_cycles = np.arange(
        current_cycle + 1, current_cycle + 1 + len(forecast), dtype=np.int64
    )
    metrics = evaluate_prediction(
        task.cycles,
        task.soh,
        forecast_cycles,
        forecast,
        current_cycle,
        float(support[-1]),
        config.evaluation.eol_threshold,
    )
    metrics.update({"cell": task.name, "mode": mode})
    actual_by_cycle = dict(zip(task.cycles.tolist(), task.soh.tolist()))
    observed = np.asarray(
        [actual_by_cycle.get(int(cycle), float("nan")) for cycle in forecast_cycles],
        dtype=float,
    )
    support_frame = pd.DataFrame(
        {
            "cycle": task.cycles[:history_length],
            "observed_soh": support,
            "predicted_soh": support,
            "split": "support",
        }
    )
    future_frame = pd.DataFrame(
        {
            "cycle": forecast_cycles,
            "observed_soh": observed,
            "predicted_soh": forecast,
            "split": "future",
        }
    )
    predictions = pd.concat([support_frame, future_frame], ignore_index=True)
    predictions["eol_threshold"] = config.evaluation.eol_threshold
    predictions["mode"] = mode
    cell_root = output_dir / Path(task.name).stem
    (cell_root / "predictions").mkdir(parents=True, exist_ok=True)
    (cell_root / "metrics").mkdir(parents=True, exist_ok=True)
    (cell_root / "figures").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(cell_root / f"predictions/{mode}.csv", index=False)
    (cell_root / f"metrics/{mode}.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    fig, axis = plt.subplots(figsize=(9, 5))
    available = predictions["observed_soh"].notna()
    future = predictions["split"] == "future"
    axis.plot(
        predictions.loc[available, "cycle"],
        predictions.loc[available, "observed_soh"],
        label="actual SOH",
    )
    axis.plot(
        predictions.loc[future, "cycle"],
        predictions.loc[future, "predicted_soh"],
        label="recursive forecast",
    )
    axis.axhline(config.evaluation.eol_threshold, color="tab:red", linestyle="--", label="EOL")
    axis.axvline(current_cycle, color="gray", linestyle=":", label=f"L={history_length}")
    axis.set(xlabel="Cycle", ylabel="SOH", title=f"{task.name} {mode}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(cell_root / f"figures/{mode}.png", dpi=150)
    plt.close(fig)
    return metrics


def adapt_and_evaluate_cell(
    meta_model: GRUEncoderDecoder,
    task: CellTask,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    """Run paper fast-step comparisons and complete support-only adaptation."""
    support, _ = task.split(config.data.history_length)
    root = Path(output_dir)
    cell_root = root / Path(task.name).stem
    (cell_root / "adaptation").mkdir(parents=True, exist_ok=True)
    (cell_root / "models").mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    for steps in config.adaptation.fast_steps:
        mode = f"fast_{steps}_steps"
        result = adapt_model(
            meta_model,
            support,
            config,
            device,
            max_steps=steps,
            # The same random stream makes 1-step adaptation the exact prefix
            # of the 3- and 5-step comparisons.
            seed_offset=1000,
            patience=None,
        )
        result.history.to_csv(cell_root / f"adaptation/{mode}.csv", index=False)
        torch.save(result.model.state_dict(), cell_root / f"models/{mode}.pt")
        row = _evaluate_one(
            result.model, task, config.data.history_length, mode, config, root
        )
        row.update(
            {
                "adaptation_best_step": result.best_step,
                "adaptation_best_support_loss": result.best_support_loss,
            }
        )
        metrics.append(row)
        logger.info("Meta-test %s %s metrics=%s", task.name, mode, row)
    complete = adapt_model(
        meta_model,
        support,
        config,
        device,
        max_steps=config.adaptation.complete_max_steps,
        seed_offset=5000,
        patience=config.adaptation.complete_patience,
    )
    complete.history.to_csv(cell_root / "adaptation/complete.csv", index=False)
    torch.save(complete.model.state_dict(), cell_root / "models/complete.pt")
    row = _evaluate_one(
        complete.model, task, config.data.history_length, "complete", config, root
    )
    row.update(
        {
            "adaptation_best_step": complete.best_step,
            "adaptation_best_support_loss": complete.best_support_loss,
        }
    )
    metrics.append(row)
    logger.info("Meta-test %s complete metrics=%s", task.name, row)
    return metrics


def evaluate_test_tasks(
    meta_model: GRUEncoderDecoder,
    test_tasks: Sequence[CellTask],
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    if [task.name for task in test_tasks] != list(config.data.test_cells):
        raise ValueError("test tasks must exactly match configured meta-test cells")
    rows: list[dict[str, object]] = []
    for task in test_tasks:
        rows.extend(
            adapt_and_evaluate_cell(meta_model, task, config, device, output_dir, logger)
        )
    frame = pd.DataFrame(rows)
    destination = Path(output_dir) / "meta_test_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame

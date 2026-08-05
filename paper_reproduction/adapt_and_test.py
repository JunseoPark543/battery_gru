"""Meta-test adaptation trajectories, diagnostics, and safe model selection."""

from __future__ import annotations

import copy
import json
import logging
import math
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
from .data import CellTask, RecursivePairDataset, sample_support_batch
from .losses import get_loss
from .metrics import evaluate_prediction
from .model import GRUEncoderDecoder


StateDict = dict[str, torch.Tensor]


@dataclass
class AdaptationResult:
    """Backward-compatible result returned by :func:`adapt_model`."""

    model: GRUEncoderDecoder
    history: pd.DataFrame
    best_step: int
    best_support_loss: float


@dataclass
class AdaptationTrajectory:
    """One continuous SGD path and its deployment/oracle selections."""

    diagnostics: pd.DataFrame
    captured_states: dict[int, StateDict]
    captured_forecasts: dict[int, np.ndarray]
    deployment_best_state: StateDict
    deployment_best_step: int
    deployment_best_mae: float
    oracle_best_state: StateDict
    oracle_best_step: int
    oracle_best_query_mae: float
    final_state: StateDict
    final_step: int


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


def _clone_state(model: GRUEncoderDecoder) -> StateDict:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def model_from_state(
    meta_model: GRUEncoderDecoder,
    state: StateDict,
    device: torch.device,
) -> GRUEncoderDecoder:
    model = copy.deepcopy(meta_model).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _tensor_global_norm(tensors: Sequence[torch.Tensor]) -> float:
    squared = sum(float(tensor.detach().double().square().sum().cpu()) for tensor in tensors)
    return math.sqrt(squared)


def _gradient_norm(model: GRUEncoderDecoder) -> float:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return _tensor_global_norm(gradients) if gradients else 0.0


def _parameter_update_metrics(
    before: dict[str, torch.Tensor],
    model: GRUEncoderDecoder,
) -> tuple[float, float, float, float]:
    differences: list[torch.Tensor] = []
    previous: list[torch.Tensor] = []
    hidden_difference: list[torch.Tensor] = []
    output_bias_difference: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        old = before[name].to(parameter.device)
        difference = parameter.detach() - old
        differences.append(difference)
        previous.append(old)
        if "weight_hh" in name:
            hidden_difference.append(difference)
        if name == "output_layer.bias":
            output_bias_difference.append(difference)
    update_norm = _tensor_global_norm(differences)
    parameter_norm = _tensor_global_norm(previous)
    relative = update_norm / max(parameter_norm, 1.0e-12)
    hidden_norm = _tensor_global_norm(hidden_difference) if hidden_difference else 0.0
    output_bias_norm = (
        _tensor_global_norm(output_bias_difference) if output_bias_difference else 0.0
    )
    return update_norm, relative, hidden_norm, output_bias_norm


def _validation_length(history_length: int, config: ExperimentConfig) -> int:
    adaptation = config.adaptation
    proposed = max(
        adaptation.minimum_validation_length,
        int(adaptation.validation_ratio * history_length),
    )
    length = min(adaptation.maximum_validation_length, proposed, history_length - 2)
    if length < 1 or history_length - length < 2:
        raise ValueError(
            "chronological adaptation split leaves fewer than two training SOH points"
        )
    return length


def _support_validation_mae(
    model: GRUEncoderDecoder,
    train_soh: np.ndarray,
    validation_soh: np.ndarray | None,
) -> float:
    if validation_soh is None:
        return float("nan")
    prediction = model.recursive_forecast(train_soh, len(validation_soh))
    values = prediction[0, :, 0].detach().cpu().numpy().astype(float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("support validation forecast contains NaN or infinity")
    return float(np.mean(np.abs(values - validation_soh)))


def _query_diagnostics(
    model: GRUEncoderDecoder,
    task: CellTask,
    history_length: int,
    config: ExperimentConfig,
) -> tuple[dict[str, object], np.ndarray]:
    support, query = task.split(history_length)
    forecast_tensor = model.recursive_forecast(support, len(query))
    forecast = forecast_tensor[0, :, 0].detach().cpu().numpy().astype(float)
    if not np.all(np.isfinite(forecast)):
        raise FloatingPointError("query diagnostic forecast contains NaN or infinity")
    metrics = evaluate_prediction(
        task.cycles,
        task.soh,
        task.cycles[history_length:],
        forecast,
        int(task.cycles[history_length - 1]),
        float(support[-1]),
        config.evaluation.eol_threshold,
    )
    return metrics, forecast


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    adaptation = config.adaptation
    if adaptation.scheduler == "constant":
        return None
    if adaptation.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=adaptation.scheduler_step_size,
            gamma=adaptation.scheduler_gamma,
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=adaptation.scheduler_gamma,
        patience=adaptation.scheduler_patience,
    )


def run_adaptation_trajectory(
    meta_model: GRUEncoderDecoder,
    task: CellTask,
    config: ExperimentConfig,
    device: torch.device,
    training_soh: np.ndarray,
    validation_soh: np.ndarray | None,
    learning_rate: float,
    max_steps: int,
    sampling_mode: str,
    seed_offset: int = 1000,
    patience: int | None = None,
    capture_steps: Sequence[int] = (0, 1, 2, 3, 5, 10),
    query_diagnostics: bool = True,
) -> AdaptationTrajectory:
    """Run one reproducible trajectory; query labels never affect its updates.

    Deployment selection uses chronological support validation only. Query MAE
    is calculated under ``torch.no_grad`` for explicitly labeled oracle
    diagnostics and is never passed to ``backward`` or the optimizer.
    """
    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    adapted = copy.deepcopy(meta_model).to(device)
    dataset = RecursivePairDataset(training_soh)
    loss_function = get_loss(
        config.loss.kind, config.adaptation.recursive_loss_reduction
    )
    optimizer = torch.optim.SGD(adapted.parameters(), lr=learning_rate)
    scheduler = _make_scheduler(optimizer, config)
    support_generator = torch.Generator(device="cpu").manual_seed(
        config.seed + seed_offset
    )
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    teacher_generator = torch.Generator(device=generator_device).manual_seed(
        config.seed + seed_offset + 1
    )
    fixed_generator = torch.Generator(device="cpu").manual_seed(
        config.seed + seed_offset + 2
    )
    evaluation_batch = sample_support_batch(
        dataset,
        config.adaptation.batch_size,
        fixed_generator,
        device,
        mode="length_stratified",
        length_bins=config.adaptation.length_stratified_bins,
    )
    requested_captures = set(int(step) for step in capture_steps)
    captured_states: dict[int, StateDict] = {}
    captured_forecasts: dict[int, np.ndarray] = {}
    deployment_best_state = _clone_state(adapted)
    deployment_best_step = 0
    deployment_best_mae = float("inf")
    oracle_best_state = _clone_state(adapted)
    oracle_best_step = 0
    oracle_best_query_mae = float("inf")
    records: list[dict[str, object]] = []
    stale = 0

    def diagnose(
        step: int,
        train_loss: float,
        gradient_before_clip: float,
        gradient_after_clip: float,
        update_norm: float,
        relative_update_norm: float,
        hidden_update_norm: float,
        output_bias_update_norm: float,
        split_indices: list[int],
    ) -> None:
        nonlocal deployment_best_state, deployment_best_step, deployment_best_mae
        nonlocal oracle_best_state, oracle_best_step, oracle_best_query_mae, stale
        adapted.eval()
        with torch.no_grad():
            support_eval = _loss_on_batch(
                adapted, evaluation_batch, loss_function, 1.0, None
            )
        support_eval_value = float(support_eval.cpu())
        validation_mae = _support_validation_mae(
            adapted, training_soh, validation_soh
        )
        selection_value = (
            validation_mae if validation_soh is not None else support_eval_value
        )
        if selection_value < deployment_best_mae - config.adaptation.complete_min_delta:
            deployment_best_mae = selection_value
            deployment_best_step = step
            deployment_best_state = _clone_state(adapted)
            stale = 0
        elif step > 0:
            stale += 1

        if query_diagnostics:
            query_metrics, forecast = _query_diagnostics(
                adapted, task, config.data.history_length, config
            )
            query_mae = float(query_metrics["mae"])
            if query_mae < oracle_best_query_mae:
                oracle_best_query_mae = query_mae
                oracle_best_step = step
                oracle_best_state = _clone_state(adapted)
        else:
            query_metrics = {
                "mae": float("nan"), "mae_percent": float("nan"),
                "rmse_percent": float("nan"), "r2": float("nan"),
                "predicted_eol_cycle_last_hitting": None, "predicted_rul": None,
            }
            forecast = np.asarray([], dtype=float)
        if step in requested_captures:
            captured_states[step] = _clone_state(adapted)
            if forecast.size:
                captured_forecasts[step] = forecast.copy()
        records.append(
            {
                "step": step,
                "support_train_loss": train_loss,
                "support_eval_loss": support_eval_value,
                "support_validation_mae_fraction": validation_mae,
                "support_validation_mae_percent": 100.0 * validation_mae,
                "query_mae_fraction": query_metrics["mae"],
                "query_mae_percent": query_metrics["mae_percent"],
                "query_rmse_percent": query_metrics["rmse_percent"],
                "query_r2": query_metrics["r2"],
                "gradient_norm": gradient_before_clip,
                "gradient_norm_before_clip": gradient_before_clip,
                "gradient_norm_after_clip": gradient_after_clip,
                "parameter_update_norm": update_norm,
                "relative_update_norm": relative_update_norm,
                "hidden_to_hidden_update_norm": hidden_update_norm,
                "output_bias_update_norm": output_bias_update_norm,
                "relative_update_warning": (
                    relative_update_norm
                    > config.adaptation.relative_update_warning_threshold
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "sampling_mode": sampling_mode,
                "selected_split_indices": json.dumps(split_indices),
                "first_predicted_soh": (
                    float(forecast[0]) if forecast.size else float("nan")
                ),
                "last_predicted_soh": (
                    float(forecast[-1]) if forecast.size else float("nan")
                ),
                "predicted_eol": query_metrics["predicted_eol_cycle_last_hitting"],
                "predicted_rul": query_metrics["predicted_rul"],
                "oracle_query_selection": bool(query_diagnostics),
            }
        )

    diagnose(0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
    final_step = 0
    for step in range(1, max_steps + 1):
        adapted.train()
        batch = sample_support_batch(
            dataset,
            config.adaptation.batch_size,
            support_generator,
            device,
            mode=sampling_mode,
            length_bins=config.adaptation.length_stratified_bins,
        )
        before_parameters = {
            name: parameter.detach().clone()
            for name, parameter in adapted.named_parameters()
        }
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
        gradient_before = _gradient_norm(adapted)
        if not math.isfinite(gradient_before):
            raise FloatingPointError(f"non-finite adaptation gradient at step {step}")
        if config.adaptation.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                adapted.parameters(), config.adaptation.gradient_clip_norm
            )
        gradient_after = _gradient_norm(adapted)
        if not math.isfinite(gradient_after):
            raise FloatingPointError(f"non-finite clipped gradient at step {step}")
        optimizer.step()
        update_norm, relative_update, hidden_update, bias_update = (
            _parameter_update_metrics(before_parameters, adapted)
        )
        split_indices = batch["split_indices"].detach().cpu().tolist()
        diagnose(
            step,
            float(train_loss.detach().cpu()),
            gradient_before,
            gradient_after,
            update_norm,
            relative_update,
            hidden_update,
            bias_update,
            split_indices,
        )
        latest_selection = float(
            records[-1][
                "support_validation_mae_fraction"
                if validation_soh is not None
                else "support_eval_loss"
            ]
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(latest_selection)
        elif scheduler is not None:
            scheduler.step()
        final_step = step
        if patience is not None and stale >= patience:
            break
    final_state = _clone_state(adapted)
    if final_step not in captured_states:
        captured_states[final_step] = final_state
        if query_diagnostics:
            _, final_forecast = _query_diagnostics(
                adapted, task, config.data.history_length, config
            )
            captured_forecasts[final_step] = final_forecast
    return AdaptationTrajectory(
        diagnostics=pd.DataFrame(records),
        captured_states=captured_states,
        captured_forecasts=captured_forecasts,
        deployment_best_state=deployment_best_state,
        deployment_best_step=deployment_best_step,
        deployment_best_mae=deployment_best_mae,
        oracle_best_state=oracle_best_state,
        oracle_best_step=oracle_best_step,
        oracle_best_query_mae=oracle_best_query_mae,
        final_state=final_state,
        final_step=final_step,
    )


def adapt_model(
    meta_model: GRUEncoderDecoder,
    support_soh: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    max_steps: int,
    seed_offset: int,
    patience: int | None = None,
) -> AdaptationResult:
    """Compatibility wrapper around the new continuous adaptation trajectory."""
    values = np.asarray(support_soh, dtype=float)
    fake_task = CellTask(
        "support_only.pkl",
        np.arange(1, len(values) + 2),
        np.concatenate([values, values[-1:]]),
    )
    learning_rate = (
        config.adaptation.resolved_complete_learning_rate()
        if patience is not None
        else config.adaptation.resolved_fast_learning_rate()
    )
    trajectory = run_adaptation_trajectory(
        meta_model,
        fake_task,
        config,
        device,
        training_soh=values,
        validation_soh=None,
        learning_rate=learning_rate,
        max_steps=max_steps,
        sampling_mode=(
            config.adaptation.sampling_mode
            if patience is not None
            else config.adaptation.fast_sampling_mode
        ),
        seed_offset=seed_offset,
        patience=patience,
        capture_steps=[max_steps],
        query_diagnostics=False,
    )
    state = (
        trajectory.deployment_best_state
        if patience is not None
        else trajectory.final_state
    )
    model = model_from_state(meta_model, state, device)
    return AdaptationResult(
        model,
        trajectory.diagnostics,
        trajectory.deployment_best_step if patience is not None else max_steps,
        trajectory.deployment_best_mae,
    )


def _evaluate_one(
    model: GRUEncoderDecoder,
    task: CellTask,
    history_length: int,
    mode: str,
    config: ExperimentConfig,
    output_dir: Path,
    flat_output: bool = False,
) -> dict[str, object]:
    support, query = task.split(history_length)
    current_cycle = int(task.cycles[history_length - 1])
    if config.evaluation.forecast_mode == "paper":
        horizon = len(query)
        forecast_cycles = task.cycles[history_length:].copy()
    else:
        horizon = (
            config.evaluation.max_prediction_length
            if config.evaluation.max_prediction_length is not None
            else config.evaluation.max_forecast_cycle - current_cycle
        )
        forecast_cycles = np.arange(
            current_cycle + 1, current_cycle + 1 + horizon, dtype=np.int64
        )
    if horizon <= 0:
        raise ValueError("prediction length must be positive")
    forecast_tensor = model.recursive_forecast(support, horizon)
    forecast = forecast_tensor[0, :, 0].detach().cpu().numpy().astype(float)
    if not np.all(np.isfinite(forecast)):
        raise FloatingPointError("recursive evaluation contains NaN or infinity")
    forecast_cycles = forecast_cycles[:len(forecast)]
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
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "cycle": task.cycles[:history_length],
                    "observed_soh": support,
                    "predicted_soh": support,
                    "split": "support",
                }
            ),
            pd.DataFrame(
                {
                    "cycle": forecast_cycles,
                    "observed_soh": observed,
                    "predicted_soh": forecast,
                    "split": "future",
                }
            ),
        ],
        ignore_index=True,
    )
    predictions["eol_threshold"] = config.evaluation.eol_threshold
    predictions["mode"] = mode
    cell_root = output_dir if flat_output else output_dir / Path(task.name).stem
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
    axis.set(
        xlabel="Cycle",
        ylabel="SOH",
        title=(
            f"{task.name} {mode} | "
            f"MAE={metrics['mae_percent']:.3f}% RMSE={metrics['rmse_percent']:.3f}%"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(cell_root / f"figures/{mode}.png", dpi=150)
    plt.close(fig)
    return metrics


def _plot_diagnostics(
    trajectory: AdaptationTrajectory,
    task: CellTask,
    config: ExperimentConfig,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = trajectory.diagnostics
    plots = [
        (
            "adaptation_step_vs_support_loss.png",
            [("support_train_loss", "train"), ("support_eval_loss", "recursive eval")],
            "Loss",
        ),
        (
            "adaptation_step_vs_query_mae.png",
            [("query_mae_percent", "query MAE (%)"), ("support_validation_mae_percent", "support validation MAE (%)")],
            "MAE (%)",
        ),
        (
            "adaptation_step_vs_gradient_norm.png",
            [("gradient_norm_before_clip", "before clip"), ("gradient_norm_after_clip", "after clip")],
            "Gradient L2 norm",
        ),
        (
            "adaptation_step_vs_update_norm.png",
            [("parameter_update_norm", "update norm"), ("relative_update_norm", "relative update")],
            "Update norm",
        ),
    ]
    for file_name, series, ylabel in plots:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for column, label in series:
            axis.plot(frame["step"], frame[column], label=label)
        axis.set(xlabel="Adaptation step", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / file_name, dpi=150)
        plt.close(fig)

    support, query = task.split(config.data.history_length)
    query_cycles = task.cycles[config.data.history_length:]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(query_cycles, query, color="black", linewidth=1.5, label="actual query")
    desired = {0, 1, 2, 3, 5, 10, trajectory.deployment_best_step, trajectory.oracle_best_step, trajectory.final_step}
    for step in sorted(desired):
        forecast = trajectory.captured_forecasts.get(step)
        if forecast is not None:
            axis.plot(query_cycles[:len(forecast)], forecast, label=f"step {step}", alpha=0.8)
    axis.axhline(config.evaluation.eol_threshold, color="tab:red", linestyle="--")
    axis.set(xlabel="Cycle", ylabel="SOH", title="Recursive forecast by adaptation step")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "recursive_forecast_by_step.png", dpi=150)
    plt.close(fig)


def _write_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")


def adapt_and_evaluate_cell(
    meta_model: GRUEncoderDecoder,
    task: CellTask,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
    logger: logging.Logger,
    flat_output: bool = False,
) -> list[dict[str, object]]:
    """Evaluate shared fast trajectory plus deployment-safe/oracle complete models."""
    support, _ = task.split(config.data.history_length)
    root = Path(output_dir)
    cell_root = root if flat_output else root / Path(task.name).stem
    adaptation_dir = cell_root / "adaptation"
    checkpoint_dir = cell_root / "checkpoints"
    plot_dir = cell_root / "plots"
    for directory in (adaptation_dir, checkpoint_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []

    fast_steps = sorted(set(config.adaptation.fast_steps))
    fast_trajectory = run_adaptation_trajectory(
        meta_model,
        task,
        config,
        device,
        training_soh=support,
        validation_soh=None,
        learning_rate=config.adaptation.resolved_fast_learning_rate(),
        max_steps=max(fast_steps),
        sampling_mode=config.adaptation.fast_sampling_mode,
        seed_offset=1000,
        patience=None,
        capture_steps=fast_steps,
        query_diagnostics=True,
    )
    fast_trajectory.diagnostics.to_csv(
        adaptation_dir / "fast_adaptation_diagnostics.csv", index=False
    )
    for step in fast_steps:
        state = fast_trajectory.captured_states[step]
        model = model_from_state(meta_model, state, device)
        mode = f"fast_{step}_steps"
        row = _evaluate_one(
            model,
            task,
            config.data.history_length,
            mode,
            config,
            root,
            flat_output=flat_output,
        )
        row.update(
            {
                "adaptation_best_step": step,
                "adaptation_best_support_loss": float(
                    fast_trajectory.diagnostics.loc[
                        fast_trajectory.diagnostics["step"] == step, "support_eval_loss"
                    ].iloc[0]
                ),
                "oracle_query_selection": False,
                "learning_rate": config.adaptation.resolved_fast_learning_rate(),
                "loss_reduction": config.adaptation.recursive_loss_reduction,
                "sampling_mode": config.adaptation.fast_sampling_mode,
            }
        )
        torch.save(state, checkpoint_dir / f"fast_{step}_model.pt")
        _write_metrics(adaptation_dir / f"fast_{step}_metrics.json", row)
        metrics.append(row)
        logger.info("Meta-test %s %s metrics=%s", task.name, mode, row)

    validation_length = _validation_length(config.data.history_length, config)
    training_soh = support[:-validation_length]
    validation_soh = support[-validation_length:]
    complete_trajectory = run_adaptation_trajectory(
        meta_model,
        task,
        config,
        device,
        training_soh=training_soh,
        validation_soh=validation_soh,
        learning_rate=config.adaptation.resolved_complete_learning_rate(),
        max_steps=config.adaptation.complete_max_steps,
        sampling_mode=config.adaptation.sampling_mode,
        seed_offset=1000,
        patience=config.adaptation.complete_patience,
        capture_steps=(0, 1, 2, 3, 5, 10),
        query_diagnostics=config.adaptation.oracle_diagnostics,
    )
    complete_trajectory.captured_states[
        complete_trajectory.deployment_best_step
    ] = complete_trajectory.deployment_best_state
    special_states = [
        (complete_trajectory.deployment_best_step, complete_trajectory.deployment_best_state)
    ]
    if config.adaptation.oracle_diagnostics:
        complete_trajectory.captured_states[
            complete_trajectory.oracle_best_step
        ] = complete_trajectory.oracle_best_state
        special_states.append(
            (complete_trajectory.oracle_best_step, complete_trajectory.oracle_best_state)
        )
    for special_step, state in special_states:
        special_model = model_from_state(meta_model, state, device)
        _, forecast = _query_diagnostics(
            special_model, task, config.data.history_length, config
        )
        complete_trajectory.captured_forecasts[special_step] = forecast
    complete_trajectory.diagnostics.to_csv(
        adaptation_dir / "adaptation_diagnostics.csv", index=False
    )
    torch.save(
        complete_trajectory.deployment_best_state,
        checkpoint_dir / "complete_best_model.pt",
    )
    torch.save(
        complete_trajectory.final_state,
        checkpoint_dir / "complete_final_model.pt",
    )
    if config.adaptation.oracle_diagnostics:
        torch.save(
            complete_trajectory.oracle_best_state,
            checkpoint_dir / "complete_oracle_model.pt",
        )

    deployment_model = model_from_state(
        meta_model, complete_trajectory.deployment_best_state, device
    )
    deployment_metrics = _evaluate_one(
        deployment_model,
        task,
        config.data.history_length,
        "complete_deployment_safe",
        config,
        root,
        flat_output=flat_output,
    )
    deployment_metrics.update(
        {
            "adaptation_best_step": complete_trajectory.deployment_best_step,
            "support_validation_mae_fraction": complete_trajectory.deployment_best_mae,
            "support_validation_mae_percent": 100.0 * complete_trajectory.deployment_best_mae,
            "oracle_query_selection": False,
            "checkpoint_selection": "support_recursive_validation",
            "learning_rate": config.adaptation.resolved_complete_learning_rate(),
            "loss_reduction": config.adaptation.recursive_loss_reduction,
            "sampling_mode": config.adaptation.sampling_mode,
        }
    )
    _write_metrics(
        adaptation_dir / "complete_deployment_safe_metrics.json",
        deployment_metrics,
    )
    metrics.append(deployment_metrics)

    oracle_metrics: dict[str, object] | None = None
    if config.adaptation.oracle_diagnostics:
        oracle_model = model_from_state(
            meta_model, complete_trajectory.oracle_best_state, device
        )
        oracle_metrics = _evaluate_one(
            oracle_model,
            task,
            config.data.history_length,
            "complete_oracle_diagnostic",
            config,
            root,
            flat_output=flat_output,
        )
        oracle_metrics.update(
            {
                "adaptation_best_step": complete_trajectory.oracle_best_step,
                "oracle_query_mae_fraction": complete_trajectory.oracle_best_query_mae,
                "oracle_query_selection": True,
                "deployment_safe": False,
                "query_used_for_gradient": False,
                "learning_rate": config.adaptation.resolved_complete_learning_rate(),
                "loss_reduction": config.adaptation.recursive_loss_reduction,
                "sampling_mode": config.adaptation.sampling_mode,
            }
        )
        _write_metrics(
            adaptation_dir / "complete_oracle_diagnostic_metrics.json",
            oracle_metrics,
        )
        metrics.append(oracle_metrics)
    _plot_diagnostics(complete_trajectory, task, config, plot_dir)
    logger.info("Meta-test %s deployment-safe metrics=%s", task.name, deployment_metrics)
    if oracle_metrics is not None:
        logger.info("Meta-test %s ORACLE diagnostic metrics=%s", task.name, oracle_metrics)
    return metrics


def evaluate_test_tasks(
    meta_model: GRUEncoderDecoder,
    test_tasks: Sequence[CellTask],
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
    logger: logging.Logger,
    flat_output: bool = False,
) -> pd.DataFrame:
    names = [task.name for task in test_tasks]
    expected_order = [name for name in config.data.test_cells if name in names]
    if not names or names != expected_order or any(
        name not in config.data.test_cells for name in names
    ):
        raise ValueError("test tasks must be an ordered subset of configured meta-test cells")
    if flat_output and len(test_tasks) != 1:
        raise ValueError("flat output is available only for one test cell")
    rows: list[dict[str, object]] = []
    for task in test_tasks:
        rows.extend(
            adapt_and_evaluate_cell(
                meta_model,
                task,
                config,
                device,
                output_dir,
                logger,
                flat_output=flat_output,
            )
        )
    frame = pd.DataFrame(rows)
    destination = Path(output_dir) / "meta_test_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame

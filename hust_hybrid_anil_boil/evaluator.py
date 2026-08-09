"""Few-shot unseen-protocol evaluation and component diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from hust_direct_rul_boil.data import CellSample
from hust_direct_rul_boil.metrics import regression_metrics

from .config import ExperimentConfig
from .meta import adapt_task, forward_with_parameters, representation_change
from .trainer import TrainingResult


LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_payload: dict[str, np.ndarray]
    support_splits: list[dict[str, Any]]


def _target_split(
    samples: Sequence[CellSample], support_count: int, seed: int
) -> tuple[list[CellSample], list[CellSample]]:
    ordered = sorted(samples, key=lambda sample: (sample.replicate, sample.file_name))
    if len(ordered) <= support_count:
        raise ValueError(
            f"target protocol has {len(ordered)} cells; needs more than {support_count} support cells"
        )
    offset = seed % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    support = rotated[:support_count]
    query = rotated[support_count:]
    if {item.file_name for item in support} & {item.file_name for item in query}:
        raise RuntimeError("target support/query leakage detected")
    return support, query


def _batch(
    samples: Sequence[CellSample], result: TrainingResult, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    waveforms = torch.as_tensor(
        np.stack([result.normalizer.transform_waveforms(sample.waveforms) for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    scalars = torch.as_tensor(
        np.stack([result.normalizer.transform_scalars(sample.scalars) for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    targets = torch.as_tensor(
        [sample.rul_cycles for sample in samples], dtype=torch.float32, device=device
    )
    return waveforms, scalars, targets


def _select_deployment_step_from_support(
    training: TrainingResult,
    support: Sequence[CellSample],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[int, dict[int, float]]:
    """Select a step without query labels using leave-one-support-cell-out MAE."""
    if config.method == "supervised":
        return 0, {0: 0.0}
    if config.evaluation.deployment_step_selection == "fixed":
        return config.evaluation.primary_adaptation_step, {}
    if len(support) < 2:
        LOGGER.warning("support_loo needs at least two cells; falling back to step 0")
        return 0, {0: 0.0}
    candidates = (
        list(config.evaluation.deployment_candidate_steps)
        or list(config.evaluation.adaptation_steps)
    )
    max_steps = max(candidates)
    absolute_errors: dict[int, list[float]] = {step: [] for step in candidates}
    model = training.model
    for validation_index, validation_sample in enumerate(support):
        adaptation_support = [
            sample for index, sample in enumerate(support) if index != validation_index
        ]
        support_waveforms, support_scalars, support_targets = _batch(
            adaptation_support, training, device
        )
        validation_waveforms, validation_scalars, validation_target = _batch(
            [validation_sample], training, device
        )
        with torch.enable_grad():
            adapted = adapt_task(
                model,
                support_waveforms,
                support_scalars,
                support_targets,
                method=config.method,
                steps=max_steps,
                config=config,
                second_order=False,
            )
            for step in candidates:
                output = forward_with_parameters(
                    model,
                    adapted.parameter_trajectory[step],
                    validation_waveforms,
                    validation_scalars,
                    grl_strength=0.0,
                )
                prediction = output.prediction.detach()
                if config.evaluation.clip_negative_rul:
                    prediction = prediction.clamp_min(0.0)
                error = torch.abs(prediction - validation_target).mean()
                absolute_errors[step].append(float(error.cpu()))
    scores = {
        step: float(np.mean(values)) for step, values in absolute_errors.items()
    }
    best_step = min(candidates, key=lambda step: (scores[step], step))
    baseline = scores.get(0, float("inf"))
    improvement = baseline - scores[best_step]
    if improvement < config.evaluation.support_loo_min_improvement_cycles:
        best_step = 0
    return best_step, scores


def evaluate_unseen_protocol(
    training: TrainingResult,
    targets: Sequence[CellSample],
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> EvaluationResult:
    if len({sample.protocol for sample in targets}) != 1:
        raise ValueError("target evaluation requires exactly one held-out protocol")
    held_out = targets[0].protocol
    model = training.model
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    support_splits: list[dict[str, Any]] = []
    feature_parts: dict[str, list[np.ndarray]] = {
        "general": [], "specific": [], "domain": [], "normalized_rul": [], "file_name": []
    }
    for repeat in range(config.evaluation.target_support_repeats):
        split_seed = seed + repeat * config.evaluation.target_support_cells
        support, query = _target_split(
            targets, config.evaluation.target_support_cells, split_seed
        )
        support_splits.append(
            {
                "repeat": repeat,
                "support_files": [sample.file_name for sample in support],
                "query_files": [sample.file_name for sample in query],
            }
        )
        support_waveforms, support_scalars, support_targets = _batch(
            support, training, device
        )
        query_waveforms, query_scalars, query_targets = _batch(query, training, device)
        selected_step, support_step_scores = _select_deployment_step_from_support(
            training, support, config, device
        )
        support_splits[-1].update(
            {
                "deployment_step_selection": config.evaluation.deployment_step_selection,
                "selected_adaptation_step": selected_step,
                "support_loo_mae_by_step": {
                    str(step): score for step, score in support_step_scores.items()
                },
            }
        )
        with torch.no_grad():
            before = model(query_waveforms, query_scalars, grl_strength=0.0)
        for steps in sorted(config.evaluation.adaptation_steps):
            effective_steps = 0 if config.method == "supervised" else steps
            with torch.enable_grad():
                adapted = adapt_task(
                    model,
                    support_waveforms,
                    support_scalars,
                    support_targets,
                    method=config.method,
                    steps=effective_steps,
                    config=config,
                    second_order=False,
                )
                output = forward_with_parameters(
                    model,
                    adapted.parameters,
                    query_waveforms,
                    query_scalars,
                    grl_strength=0.0,
                )
            raw_prediction = output.prediction.detach().cpu().numpy()
            prediction = (
                np.maximum(raw_prediction, 0.0)
                if config.evaluation.clip_negative_rul
                else raw_prediction
            )
            actual = query_targets.detach().cpu().numpy()
            y_general = output.general_prediction.detach().cpu().numpy()
            residual = output.specific_residual.detach().cpu().numpy()
            general_change = representation_change(
                before.general_embedding, output.general_embedding
            )
            specific_change = representation_change(
                before.specific_embedding, output.specific_embedding
            )
            metrics = regression_metrics(actual, prediction)
            general_metrics = regression_metrics(actual, np.maximum(y_general, 0.0))
            mean_general = float(np.mean(np.abs(y_general)))
            mean_residual = float(np.mean(np.abs(residual)))
            residual_ratio = mean_residual / max(mean_general, 1.0e-8)
            warning = residual_ratio > config.evaluation.residual_ratio_warning
            is_deployment_selection = steps == selected_step
            if warning:
                LOGGER.warning(
                    "method=%s fold=%s repeat=%d step=%d residual/general ratio %.3f exceeds %.3f",
                    config.method,
                    held_out,
                    repeat,
                    steps,
                    residual_ratio,
                    config.evaluation.residual_ratio_warning,
                )
            metric_rows.append(
                {
                    "method": config.method,
                    "seed": seed,
                    "held_out_protocol": held_out,
                    "support_repeat": repeat,
                    "adaptation_step": steps,
                    "effective_adaptation_step": effective_steps,
                    "deployment_step_selection": config.evaluation.deployment_step_selection,
                    "deployment_selected_step": selected_step,
                    "is_deployment_selection": is_deployment_selection,
                    "support_loo_selected_mae_cycles": support_step_scores.get(selected_step),
                    "support_loo_zero_step_mae_cycles": support_step_scores.get(0),
                    **metrics,
                    "y_general_mae_cycles": general_metrics["mae_cycles"],
                    "y_general_rmse_cycles": general_metrics["rmse_cycles"],
                    "mean_absolute_y_general": mean_general,
                    "mean_absolute_specific_residual": mean_residual,
                    "residual_to_general_ratio": residual_ratio,
                    "residual_ratio_warning": warning,
                    "general_representation_cosine_distance": general_change["cosine_distance"],
                    "general_representation_l2_distance": general_change["l2_distance"],
                    "specific_representation_cosine_distance": specific_change["cosine_distance"],
                    "specific_representation_l2_distance": specific_change["l2_distance"],
                    "source_validation_general_domain_accuracy": training.source_domain_metrics["general_domain_accuracy"],
                    "source_validation_specific_domain_accuracy": training.source_domain_metrics["specific_domain_accuracy"],
                    "target_support_labels_used_for_adaptation": effective_steps > 0,
                    "target_query_labels_used_for_adaptation_or_checkpoint_selection": False,
                    "prediction_mode": config.ablation.prediction_mode,
                }
            )
            for sample, target_value, general_value, residual_value, raw_value, final_value in zip(
                query, actual, y_general, residual, raw_prediction, prediction
            ):
                prediction_rows.append(
                    {
                        "method": config.method,
                        "seed": seed,
                        "held_out_protocol": held_out,
                        "support_repeat": repeat,
                        "adaptation_step": steps,
                        "deployment_step_selection": config.evaluation.deployment_step_selection,
                        "deployment_selected_step": selected_step,
                        "is_deployment_selection": is_deployment_selection,
                        "file_name": sample.file_name,
                        "cell_id": sample.cell_id,
                        "domain": sample.protocol,
                        "target_y": float(target_value),
                        "y_G": float(general_value),
                        "delta_y_S": float(residual_value),
                        "raw_y_hat": float(raw_value),
                        "y_hat": float(final_value),
                        "absolute_error": float(abs(final_value - target_value)),
                        "prediction_mode": config.ablation.prediction_mode,
                    }
                )
            if is_deployment_selection:
                feature_parts["general"].append(output.general_embedding.detach().cpu().numpy())
                feature_parts["specific"].append(output.specific_embedding.detach().cpu().numpy())
                feature_parts["domain"].append(np.asarray([sample.protocol for sample in query]))
                feature_parts["normalized_rul"].append(actual / config.loss.rul_scale_cycles)
                feature_parts["file_name"].append(np.asarray([sample.file_name for sample in query]))
    if not feature_parts["general"]:
        raise RuntimeError("primary adaptation step did not produce feature output")
    return EvaluationResult(
        predictions=pd.DataFrame(prediction_rows),
        metrics=pd.DataFrame(metric_rows),
        feature_payload={
            key: np.concatenate(values) for key, values in feature_parts.items()
        },
        support_splits=support_splits,
    )

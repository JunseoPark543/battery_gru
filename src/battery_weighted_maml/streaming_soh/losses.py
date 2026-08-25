"""Multi-task losses for streaming SOH trajectory prediction."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .episodes import StreamingBatch


def streaming_soh_loss(
    output: dict[str, torch.Tensor],
    batch: StreamingBatch,
    *,
    soh_huber_delta: float,
    voltage_huber_delta: float,
    endpoint_huber_delta: float,
    uncertainty_weight: float,
    voltage_completion_weight: float,
    endpoint_weight: float,
    monotonic_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mean = output["soh_mean"]
    std = output["soh_std"]
    if mean.shape != batch.target_soh.shape or std.shape != batch.target_soh.shape:
        raise ValueError("SOH prediction and target shapes differ")
    soh_huber = F.huber_loss(
        mean, batch.target_soh, reduction="none", delta=soh_huber_delta
    ).masked_select(batch.query_mask).mean()
    residual = (batch.target_soh - mean) / std
    gaussian_nll = (0.5 * residual.square() + torch.log(std)).masked_select(
        batch.query_mask
    ).mean()
    voltage = F.huber_loss(
        output["voltage"],
        batch.target_voltage,
        reduction="none",
        delta=voltage_huber_delta,
    ).masked_select(batch.future_q_mask).mean()
    endpoint = F.huber_loss(
        output["endpoint_fraction"],
        batch.endpoint_fraction,
        reduction="mean",
        delta=endpoint_huber_delta,
    )
    adjacent = batch.query_mask[:, 1:] & batch.query_mask[:, :-1]
    increases = F.relu(mean[:, 1:] - mean[:, :-1])
    monotonic = (
        increases.masked_select(adjacent).mean() if adjacent.any() else mean.new_zeros(())
    )
    total = (
        soh_huber
        + uncertainty_weight * gaussian_nll
        + voltage_completion_weight * voltage
        + endpoint_weight * endpoint
        + monotonic_weight * monotonic
    )
    return total, {
        "soh_huber": soh_huber,
        "uncertainty_nll": gaussian_nll,
        "future_voltage": voltage,
        "endpoint": endpoint,
        "monotonic": monotonic,
    }


def soh_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    error = prediction_array - target_array
    return {
        "soh_mae": float(np.mean(np.abs(error))),
        "soh_rmse": float(np.sqrt(np.mean(np.square(error)))),
    }

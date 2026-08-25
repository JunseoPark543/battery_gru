"""Losses and metrics for future V-Q curve completion."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def forecasting_loss(
    output: dict[str, torch.Tensor],
    target_voltage: torch.Tensor,
    observed_mask: torch.Tensor,
    future_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    endpoint_fraction: torch.Tensor,
    *,
    voltage_huber_delta: float,
    endpoint_huber_delta: float,
    endpoint_weight: float,
    observed_reconstruction_weight: float,
    monotonic_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted = output["voltage"]
    if predicted.shape != target_voltage.shape:
        raise ValueError("predicted and target voltage shapes differ")
    future = F.huber_loss(
        predicted, target_voltage, reduction="none", delta=voltage_huber_delta
    ).masked_select(future_mask).mean()
    observed = F.huber_loss(
        predicted, target_voltage, reduction="none", delta=voltage_huber_delta
    ).masked_select(observed_mask).mean()
    endpoint = F.huber_loss(
        output["endpoint_fraction"],
        endpoint_fraction,
        reduction="mean",
        delta=endpoint_huber_delta,
    )
    adjacent = valid_mask[:, 1:] & valid_mask[:, :-1]
    increases = F.relu(predicted[:, 1:] - predicted[:, :-1])
    monotonic = (
        increases.masked_select(adjacent).mean()
        if adjacent.any()
        else predicted.new_zeros(())
    )
    total = (
        future
        + observed_reconstruction_weight * observed
        + endpoint_weight * endpoint
        + monotonic_weight * monotonic
    )
    return total, {
        "future_voltage": future,
        "observed_reconstruction": observed,
        "endpoint": endpoint,
        "monotonic": monotonic,
    }


def voltage_metrics(target_v: np.ndarray, prediction_v: np.ndarray) -> dict[str, float]:
    target = np.asarray(target_v, dtype=np.float64)
    prediction = np.asarray(prediction_v, dtype=np.float64)
    error = prediction - target
    return {
        "voltage_mae_v": float(np.mean(np.abs(error))),
        "voltage_rmse_v": float(np.sqrt(np.mean(np.square(error)))),
    }

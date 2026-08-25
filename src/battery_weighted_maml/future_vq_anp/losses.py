"""Variational, reconstruction, and physical regularization losses."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from .episodes import FutureVQBatch


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.log(prior_std / posterior_std)
        + (posterior_std.square() + (posterior_mean - prior_mean).square())
        / (2.0 * prior_std.square())
        - 0.5
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value.masked_select(mask)
    if selected.numel() == 0:
        return value.new_zeros(())
    return selected.mean()


def future_vq_loss(
    output: dict[str, torch.Tensor],
    batch: FutureVQBatch,
    *,
    voltage_huber_delta: float,
    voltage_huber_weight: float,
    endpoint_huber_delta: float,
    endpoint_weight: float,
    kl_coefficient: float,
    kl_free_bits: float,
    q_monotonic_weight: float,
    endpoint_monotonic_weight: float,
    temporal_smoothness_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required = {"posterior_mean", "posterior_std", "prior_mean", "prior_std"}
    if not required.issubset(output):
        raise ValueError("training requires posterior and prior distributions")
    voltage_mean = output["sample_voltage_mean"][0]
    voltage_std = output["sample_voltage_std"][0]
    voltage_residual = (batch.target_voltage - voltage_mean) / voltage_std
    voltage_nll = _masked_mean(
        0.5 * voltage_residual.square()
        + torch.log(voltage_std)
        + 0.5 * math.log(2.0 * math.pi),
        batch.target_q_mask & batch.query_mask.unsqueeze(-1),
    )
    voltage_huber = _masked_mean(
        F.huber_loss(
            voltage_mean,
            batch.target_voltage,
            reduction="none",
            delta=voltage_huber_delta,
        ),
        batch.target_q_mask & batch.query_mask.unsqueeze(-1),
    )
    endpoint_mean = output["sample_endpoint_mean"][0]
    endpoint_std = output["sample_endpoint_std"][0]
    endpoint_residual = (batch.target_endpoint_fraction - endpoint_mean) / endpoint_std
    endpoint_nll = _masked_mean(
        0.5 * endpoint_residual.square()
        + torch.log(endpoint_std)
        + 0.5 * math.log(2.0 * math.pi),
        batch.query_mask,
    )
    endpoint_huber = _masked_mean(
        F.huber_loss(
            endpoint_mean,
            batch.target_endpoint_fraction,
            reduction="none",
            delta=endpoint_huber_delta,
        ),
        batch.query_mask,
    )
    kl_dimensions = diagonal_gaussian_kl(
        output["posterior_mean"], output["posterior_std"],
        output["prior_mean"], output["prior_std"],
    )
    kl = torch.clamp(kl_dimensions, min=kl_free_bits).sum(dim=-1).mean()
    adjacent_q = batch.target_q_mask[..., 1:] & batch.target_q_mask[..., :-1]
    q_increases = F.relu(voltage_mean[..., 1:] - voltage_mean[..., :-1])
    q_monotonic = _masked_mean(q_increases, adjacent_q & batch.query_mask.unsqueeze(-1))
    adjacent_cycles = batch.query_mask[:, 1:] & batch.query_mask[:, :-1]
    endpoint_increases = F.relu(endpoint_mean[:, 1:] - endpoint_mean[:, :-1])
    endpoint_monotonic = _masked_mean(endpoint_increases, adjacent_cycles)
    if voltage_mean.shape[1] >= 3:
        second_difference = (
            voltage_mean[:, 2:] - 2.0 * voltage_mean[:, 1:-1] + voltage_mean[:, :-2]
        ).abs()
        smooth_mask = (
            batch.query_mask[:, 2:]
            & batch.query_mask[:, 1:-1]
            & batch.query_mask[:, :-2]
        ).unsqueeze(-1)
        smooth_mask = smooth_mask & batch.target_q_mask[:, 1:-1]
        temporal_smoothness = _masked_mean(second_difference, smooth_mask)
    else:
        temporal_smoothness = voltage_mean.new_zeros(())
    total = (
        voltage_nll
        + voltage_huber_weight * voltage_huber
        + endpoint_weight * (endpoint_nll + endpoint_huber)
        + kl_coefficient * kl
        + q_monotonic_weight * q_monotonic
        + endpoint_monotonic_weight * endpoint_monotonic
        + temporal_smoothness_weight * temporal_smoothness
    )
    return total, {
        "voltage_nll": voltage_nll,
        "voltage_huber": voltage_huber,
        "endpoint_nll": endpoint_nll,
        "endpoint_huber": endpoint_huber,
        "kl": kl,
        "q_monotonic": q_monotonic,
        "endpoint_monotonic": endpoint_monotonic,
        "temporal_smoothness": temporal_smoothness,
    }


def gaussian_metrics(
    target: np.ndarray, mean: np.ndarray, std: np.ndarray, *, prefix: str
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1.0e-8)
    error = mean - target
    standardized = (target - mean) / std
    nll = np.mean(
        0.5 * np.square(error / std) + np.log(std) + 0.5 * np.log(2.0 * np.pi)
    )
    normal_pdf = np.exp(-0.5 * standardized**2) / np.sqrt(2.0 * np.pi)
    normal_cdf = 0.5 * (
        1.0
        + np.vectorize(math.erf, otypes=[np.float64])(standardized / np.sqrt(2.0))
    )
    crps = std * (
        standardized * (2.0 * normal_cdf - 1.0)
        + 2.0 * normal_pdf
        - 1.0 / np.sqrt(np.pi)
    )
    return {
        f"{prefix}_mae": float(np.mean(np.abs(error))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(error)))),
        f"{prefix}_nll": float(nll),
        f"{prefix}_crps": float(np.mean(crps)),
    }

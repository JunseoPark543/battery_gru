"""Variational and auxiliary losses for latent streaming SOH ANP."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from battery_weighted_maml.streaming_soh.episodes import StreamingBatch


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    """KL(q||p) per latent dimension."""
    return (
        torch.log(prior_std / posterior_std)
        + (posterior_std.square() + (posterior_mean - prior_mean).square())
        / (2.0 * prior_std.square())
        - 0.5
    )


def latent_anp_loss(
    output: dict[str, torch.Tensor],
    batch: StreamingBatch,
    *,
    soh_huber_delta: float,
    soh_huber_weight: float,
    voltage_huber_delta: float,
    endpoint_huber_delta: float,
    kl_coefficient: float,
    kl_free_bits: float,
    voltage_completion_weight: float,
    endpoint_weight: float,
    monotonic_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required = {"posterior_mean", "posterior_std", "prior_mean", "prior_std"}
    if not required.issubset(output):
        raise ValueError("training loss requires posterior and prior distributions")
    sample_mean = output["sample_soh_mean"][0]
    observation_std = output["sample_observation_std"][0]
    residual = (batch.target_soh - sample_mean) / observation_std
    gaussian_nll = (
        0.5 * residual.square()
        + torch.log(observation_std)
        + 0.5 * math.log(2.0 * math.pi)
    ).masked_select(batch.query_mask).mean()
    soh_huber = F.huber_loss(
        sample_mean,
        batch.target_soh,
        reduction="none",
        delta=soh_huber_delta,
    ).masked_select(batch.query_mask).mean()
    kl_dimensions = diagonal_gaussian_kl(
        output["posterior_mean"],
        output["posterior_std"],
        output["prior_mean"],
        output["prior_std"],
    )
    kl = torch.clamp(kl_dimensions, min=kl_free_bits).sum(dim=-1).mean()
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
    increases = F.relu(sample_mean[:, 1:] - sample_mean[:, :-1])
    monotonic = (
        increases.masked_select(adjacent).mean()
        if adjacent.any()
        else sample_mean.new_zeros(())
    )
    total = (
        gaussian_nll
        + soh_huber_weight * soh_huber
        + kl_coefficient * kl
        + voltage_completion_weight * voltage
        + endpoint_weight * endpoint
        + monotonic_weight * monotonic
    )
    return total, {
        "soh_nll": gaussian_nll,
        "soh_huber": soh_huber,
        "kl": kl,
        "future_voltage": voltage,
        "endpoint": endpoint,
        "monotonic": monotonic,
    }


def regression_metrics(
    target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    target_array = np.asarray(target, dtype=np.float64)
    mean_array = np.asarray(mean, dtype=np.float64)
    std_array = np.maximum(np.asarray(std, dtype=np.float64), 1.0e-8)
    error = mean_array - target_array
    gaussian_nll = np.mean(
        0.5 * np.square(error / std_array)
        + np.log(std_array)
        + 0.5 * np.log(2.0 * np.pi)
    )
    standardized = (target_array - mean_array) / std_array
    # Closed-form Gaussian CRPS.
    normal_pdf = np.exp(-0.5 * standardized**2) / np.sqrt(2.0 * np.pi)
    normal_cdf = 0.5 * (
        1.0
        + np.vectorize(math.erf, otypes=[np.float64])(
            standardized / np.sqrt(2.0)
        )
    )
    crps = std_array * (
        standardized * (2.0 * normal_cdf - 1.0)
        + 2.0 * normal_pdf
        - 1.0 / np.sqrt(np.pi)
    )
    return {
        "soh_mae": float(np.mean(np.abs(error))),
        "soh_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "soh_nll": float(gaussian_nll),
        "soh_crps": float(np.mean(crps)),
    }

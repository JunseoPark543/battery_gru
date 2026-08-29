"""MC latent inference with lifetime output and deterministically derived RUL."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import torch

from .data import LifetimeIVScalers
from .model import LifetimeIVANP
from .tasks import LifetimeBatch


@dataclass(frozen=True)
class LifetimePrediction:
    lifetime_mean: np.ndarray
    lifetime_std: np.ndarray
    lifetime_lower: np.ndarray
    lifetime_upper: np.ndarray
    rul_mean: np.ndarray
    rul_lower: np.ndarray
    rul_upper: np.ndarray


def predict_batch(
    model: LifetimeIVANP,
    batch: LifetimeBatch,
    scalers: LifetimeIVScalers,
    *,
    mc_samples: int,
    interval_level: float,
    seed: int,
) -> LifetimePrediction:
    if mc_samples <= 0 or not 0 < interval_level < 1:
        raise ValueError("invalid MC sample count or interval level")
    device = batch.context_cycles.device
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    was_training = model.training
    model.eval()
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        # Prefix encoding is deterministic in eval mode and dominates runtime.
        # Reuse h while sampling only the ANP latent variable.
        context_h = model._encode(
            batch.context_cycles,
            batch.context_cycle_mask,
            batch.context_curves,
            batch.context_curve_mask,
            batch.context_point_mask,
        )
        query_h = model._encode(
            batch.query_cycles,
            batch.query_cycle_mask,
            batch.query_curves,
            batch.query_curve_mask,
            batch.query_point_mask,
        )
        for _ in range(mc_samples):
            # Query lifetime is deliberately omitted at inference.
            output = model.forward_from_embeddings(
                context_h,
                batch.context_point_mask,
                batch.context_y,
                query_h,
                batch.query_point_mask,
                sample_latent=True,
            )
            means.append(output["mean"].float())
            variances.append(output["std"].float().square())
    if was_training:
        model.train()
    mean_stack = torch.stack(means)
    variance_stack = torch.stack(variances)
    normalized_mean = mean_stack.mean(dim=0)
    normalized_variance = (
        (variance_stack + mean_stack.square()).mean(dim=0)
        - normalized_mean.square()
    ).clamp_min(0.0)
    lifetime_mean = scalers.inverse_lifetime(
        normalized_mean.cpu().numpy()[..., 0]
    )
    lifetime_std = scalers.std_to_cycles(
        normalized_variance.sqrt().cpu().numpy()[..., 0]
    )
    z_value = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    lifetime_lower = lifetime_mean - z_value * lifetime_std
    lifetime_upper = lifetime_mean + z_value * lifetime_std
    horizons = batch.horizons.cpu().numpy()[:, None]
    return LifetimePrediction(
        lifetime_mean=lifetime_mean,
        lifetime_std=lifetime_std,
        lifetime_lower=lifetime_lower,
        lifetime_upper=lifetime_upper,
        rul_mean=lifetime_mean - horizons,
        rul_lower=lifetime_lower - horizons,
        rul_upper=lifetime_upper - horizons,
    )

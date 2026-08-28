"""Inference helpers that never pass unseen query RUL into the ANP."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import torch

from .data import RULScalers
from .model import HorizonRULANP
from .tasks import HorizonBatch


@dataclass(frozen=True)
class RULPrediction:
    mean_cycles: np.ndarray
    std_cycles: np.ndarray
    lower_cycles: np.ndarray
    upper_cycles: np.ndarray


def predict_batch(
    model: HorizonRULANP,
    batch: HorizonBatch,
    scalers: RULScalers,
    *,
    mc_samples: int,
    interval_level: float,
    seed: int,
) -> RULPrediction:
    """Predict query RUL using context labels and query prefixes only."""
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must lie in (0,1)")
    device = batch.context_prefix.device
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    was_training = model.training
    model.eval()
    means = []
    variances = []
    with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        for _ in range(mc_samples):
            # Deliberately omit query_y: unseen query lifetime/RUL is not an
            # inference input. Sampling therefore uses q(z|context) only.
            output = model(
                batch.context_prefix,
                batch.context_prefix_mask,
                batch.context_mask,
                batch.context_y,
                batch.query_prefix,
                batch.query_prefix_mask,
                batch.query_mask,
                sample_latent=True,
            )
            means.append(output["mean"].float())
            variances.append(output["std"].float().square())
    if was_training:
        model.train()
    mean_stack = torch.stack(means)
    variance_stack = torch.stack(variances)
    predictive_mean = mean_stack.mean(dim=0)
    predictive_variance = (
        (variance_stack + mean_stack.square()).mean(dim=0)
        - predictive_mean.square()
    ).clamp_min(0.0)
    mean_normalized = predictive_mean.cpu().numpy()[..., 0]
    std_normalized = predictive_variance.sqrt().cpu().numpy()[..., 0]
    mean_cycles = scalers.inverse_rul(mean_normalized)
    std_cycles = scalers.std_to_cycles(std_normalized)
    z_value = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    return RULPrediction(
        mean_cycles=mean_cycles,
        std_cycles=std_cycles,
        lower_cycles=mean_cycles - z_value * std_cycles,
        upper_cycles=mean_cycles + z_value * std_cycles,
    )

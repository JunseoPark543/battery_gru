"""Query-only Gaussian ELBO for horizon-conditioned RUL ANP."""

from __future__ import annotations

import math

import torch

from battery_weighted_maml.matr_anp.losses import diagonal_gaussian_kl


def horizon_rul_elbo(
    output: dict[str, torch.Tensor],
    query_y: torch.Tensor,
    query_mask: torch.Tensor,
    beta_kl: float,
) -> dict[str, torch.Tensor]:
    """Optimize query NLL plus KL[q(z|C union Q) || q(z|C)]."""
    mean, std = output["mean"], output["std"]
    if mean.shape != query_y.shape or std.shape != query_y.shape:
        raise ValueError("prediction/query RUL tensor shape mismatch")
    if query_mask.shape != query_y.shape[:2]:
        raise ValueError("query mask shape mismatch")
    selected = query_mask.unsqueeze(-1)
    y = query_y.masked_select(selected)
    mu = mean.masked_select(selected)
    sigma = std.masked_select(selected)
    if y.numel() == 0:
        raise ValueError("query mask selects no RUL targets")
    nll = (
        0.5 * math.log(2.0 * math.pi)
        + sigma.log()
        + 0.5 * ((y - mu) / sigma).square()
    ).mean()
    kl = diagonal_gaussian_kl(
        output["posterior_mean"],
        output["posterior_std"],
        output["prior_mean"],
        output["prior_std"],
    ).mean()
    loss = nll + float(beta_kl) * kl
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"non-finite horizon RUL ELBO: loss={loss}, nll={nll}, kl={kl}"
        )
    return {"loss": loss, "nll": nll, "kl": kl}

"""Masked Gaussian ELBO for Attentive Neural Processes."""

from __future__ import annotations

import math

import torch


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    if not (
        posterior_mean.shape
        == posterior_std.shape
        == prior_mean.shape
        == prior_std.shape
    ):
        raise ValueError("prior/posterior Gaussian tensors must share shape")
    variance_ratio = posterior_std.square() / prior_std.square()
    mean_term = (posterior_mean - prior_mean).square() / prior_std.square()
    return 0.5 * (
        variance_ratio + mean_term - 1.0 + 2.0 * (prior_std.log() - posterior_std.log())
    ).sum(dim=-1)


def anp_elbo_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    mean, std = output["mean"], output["std"]
    if mean.shape != target.shape or std.shape != target.shape:
        raise ValueError("ANP prediction and target shapes must match")
    if mask.shape != target.shape[:2]:
        raise ValueError("target mask shape mismatch")
    expanded_mask = mask.unsqueeze(-1)
    selected_mean = mean.masked_select(expanded_mask)
    selected_std = std.masked_select(expanded_mask)
    selected_target = target.masked_select(expanded_mask)
    if selected_target.numel() == 0:
        raise ValueError("ELBO target mask selects no observations")
    nll_values = (
        0.5 * math.log(2.0 * math.pi)
        + selected_std.log()
        + 0.5 * ((selected_target - selected_mean) / selected_std).square()
    )
    nll = nll_values.mean()
    kl = diagonal_gaussian_kl(
        output["posterior_mean"],
        output["posterior_std"],
        output["prior_mean"],
        output["prior_std"],
    ).mean()
    loss = nll + float(kl_weight) * kl
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"non-finite ANP loss: total={loss}, nll={nll}, kl={kl}, "
            f"kl_weight={kl_weight}"
        )
    return {"loss": loss, "nll": nll, "kl": kl}

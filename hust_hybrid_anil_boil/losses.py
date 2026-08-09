"""Independent prediction, domain, reconstruction, and separation losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import AblationConfig, LossConfig
from .model import HybridOutput


def scaled_prediction_mse(prediction: Tensor, target: Tensor, scale: float) -> Tensor:
    """Raw-cycle MSE divided by a fixed physical scale squared for stability."""
    return F.mse_loss(prediction / scale, target / scale)


def general_domain_loss(logits: Tensor, domain: Tensor) -> Tensor:
    return F.cross_entropy(logits, domain)


def specific_domain_loss(logits: Tensor, domain: Tensor) -> Tensor:
    return F.cross_entropy(logits, domain)


def reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(prediction, target)


def orthogonality_loss(general: Tensor, specific: Tensor) -> Tensor:
    """Squared normalized cross-covariance between the two representations."""
    general = F.normalize(general - general.mean(dim=0, keepdim=True), dim=-1, eps=1.0e-8)
    specific = F.normalize(specific - specific.mean(dim=0, keepdim=True), dim=-1, eps=1.0e-8)
    cross = general.transpose(0, 1) @ specific / max(1, general.shape[0])
    return cross.square().mean()


def general_consistency_loss(
    general_prediction: Tensor,
    target_rul: Tensor,
    domain: Tensor,
    *,
    rul_scale: float,
    sigma: float,
) -> Tensor:
    """Align only cross-domain samples with similar normalized degradation state."""
    if target_rul.numel() < 2:
        return general_prediction.new_zeros(())
    state = target_rul / rul_scale
    state_distance = state[:, None] - state[None, :]
    weights = torch.exp(-state_distance.square() / (sigma * sigma))
    cross_domain = domain[:, None].ne(domain[None, :])
    pair_mask = cross_domain & ~torch.eye(
        len(domain), dtype=torch.bool, device=domain.device
    )
    weights = weights * pair_mask.to(weights.dtype)
    denominator = weights.sum()
    if not bool((denominator > 0).detach()):
        return general_prediction.new_zeros(())
    prediction_distance = (
        general_prediction[:, None] - general_prediction[None, :]
    ) / rul_scale
    return torch.sum(weights * prediction_distance.square()) / denominator


def residual_regularization(residual: Tensor, scale: float) -> Tensor:
    return torch.mean((residual / scale).square())


@dataclass
class LossBreakdown:
    total: Tensor
    query_loss: Tensor
    total_prediction: Tensor
    general_prediction: Tensor
    general_domain: Tensor
    specific_domain: Tensor
    reconstruction: Tensor
    consistency: Tensor
    orthogonal: Tensor
    residual: Tensor
    general_domain_accuracy: Tensor
    specific_domain_accuracy: Tensor
    mean_absolute_residual: Tensor


def outer_objective(
    output: HybridOutput,
    target: Tensor,
    domain: Tensor,
    config: LossConfig,
    ablation: AblationConfig,
) -> LossBreakdown:
    zero = output.prediction.new_zeros(())
    total_prediction = scaled_prediction_mse(
        output.prediction, target, config.rul_scale_cycles
    )
    general_prediction = (
        scaled_prediction_mse(
            output.general_prediction, target, config.rul_scale_cycles
        )
        if ablation.use_general_prediction_loss
        else zero
    )
    general_domain = general_domain_loss(output.general_domain_logits, domain)
    specific_domain = (
        specific_domain_loss(output.specific_domain_logits, domain)
        if ablation.use_specific_domain_classifier
        and output.specific_domain_logits is not None
        else zero
    )
    reconstruction = (
        reconstruction_loss(output.reconstruction, output.reconstruction_target)
        if ablation.use_reconstruction and output.reconstruction is not None
        else zero
    )
    consistency = (
        general_consistency_loss(
            output.general_prediction,
            target,
            domain,
            rul_scale=config.rul_scale_cycles,
            sigma=config.consistency_sigma,
        )
        if ablation.use_consistency
        else zero
    )
    orthogonal = (
        orthogonality_loss(output.general_embedding, output.specific_embedding)
        if ablation.use_orthogonality
        else zero
    )
    residual = (
        residual_regularization(output.specific_residual, config.rul_scale_cycles)
        if ablation.use_residual_regularization
        else zero
    )
    total = (
        config.lambda_total_prediction * total_prediction
        + config.lambda_general_prediction * general_prediction
        + config.lambda_general_domain * general_domain
        + config.lambda_specific_domain * specific_domain
        + config.lambda_reconstruction * reconstruction
        + config.lambda_consistency * consistency
        + config.lambda_orthogonal * orthogonal
        + config.lambda_residual * residual
    )
    general_accuracy = (
        output.general_domain_logits.argmax(dim=-1).eq(domain).float().mean()
    )
    specific_accuracy = (
        output.specific_domain_logits.argmax(dim=-1).eq(domain).float().mean()
        if output.specific_domain_logits is not None
        else zero
    )
    return LossBreakdown(
        total=total,
        query_loss=total_prediction,
        total_prediction=total_prediction,
        general_prediction=general_prediction,
        general_domain=general_domain,
        specific_domain=specific_domain,
        reconstruction=reconstruction,
        consistency=consistency,
        orthogonal=orthogonal,
        residual=residual,
        general_domain_accuracy=general_accuracy,
        specific_domain_accuracy=specific_accuracy,
        mean_absolute_residual=output.specific_residual.abs().mean(),
    )


def inner_objective(
    output: HybridOutput,
    target: Tensor,
    config: LossConfig,
) -> Tensor:
    total = scaled_prediction_mse(output.prediction, target, config.rul_scale_cycles)
    if config.inner_general_prediction_beta:
        total = total + config.inner_general_prediction_beta * scaled_prediction_mse(
            output.general_prediction, target, config.rul_scale_cycles
        )
    return total


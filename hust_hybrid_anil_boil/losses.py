"""Independent prediction, domain, reconstruction, and separation losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import AblationConfig, LossConfig
from .model import HybridOutput


def scaled_prediction_mse(prediction: Tensor, target: Tensor, scale: float) -> Tensor:
    """Raw-cycle MSE divided by a fixed physical scale squared for stability."""
    return F.mse_loss(prediction / scale, target / scale)


def general_domain_loss(
    logits: Tensor, domain: Tensor, label_smoothing: float = 0.0
) -> Tensor:
    return F.cross_entropy(logits, domain, label_smoothing=label_smoothing)


def specific_domain_loss(
    logits: Tensor, domain: Tensor, label_smoothing: float = 0.0
) -> Tensor:
    return F.cross_entropy(logits, domain, label_smoothing=label_smoothing)


def specific_supervised_contrastive_loss(
    embedding: Tensor, domain: Tensor, temperature: float
) -> Tensor:
    """Cluster different cells from the same protocol without target-domain data."""
    if len(embedding) < 2:
        return embedding.new_zeros(())
    features = F.normalize(embedding, dim=-1, eps=1.0e-8)
    logits = features @ features.transpose(0, 1) / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positives = domain[:, None].eq(domain[None, :]) & ~identity
    valid_anchor = positives.any(dim=1)
    if not bool(valid_anchor.any().detach()):
        return embedding.new_zeros(())
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~identity).to(logits.dtype)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    positive_count = positives.sum(dim=1).clamp_min(1)
    per_anchor = -(
        (log_probability * positives.to(log_probability.dtype)).sum(dim=1)
        / positive_count
    )
    return per_anchor[valid_anchor].mean()


def reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(prediction, target)


def orthogonality_loss(
    general: Tensor, specific: Tensor, reduction: str = "mean"
) -> Tensor:
    """Squared normalized cross-covariance between the two representations."""
    general = F.normalize(general - general.mean(dim=0, keepdim=True), dim=-1, eps=1.0e-8)
    specific = F.normalize(specific - specific.mean(dim=0, keepdim=True), dim=-1, eps=1.0e-8)
    cross = general.transpose(0, 1) @ specific / max(1, general.shape[0])
    if reduction == "sum":
        return cross.square().sum()
    if reduction == "mean":
        return cross.square().mean()
    raise ValueError(f"unknown orthogonality reduction: {reduction}")


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


def specific_residual_fit_loss(
    residual: Tensor,
    general_prediction: Tensor,
    target: Tensor,
    scale: float,
) -> Tensor:
    """Train Specific as a correction without letting this term move General."""
    residual_target = target - general_prediction.detach()
    return F.mse_loss(residual / scale, residual_target / scale)


def within_domain_rul_difference_loss(
    prediction: Tensor,
    target: Tensor,
    domain: Tensor,
    scale: float,
) -> Tensor:
    """Preserve cell-to-cell RUL ordering and spacing inside each protocol."""
    if len(prediction) < 2:
        return prediction.new_zeros(())
    same_domain = domain[:, None].eq(domain[None, :])
    upper_triangle = torch.triu(
        torch.ones_like(same_domain, dtype=torch.bool), diagonal=1
    )
    pair_mask = same_domain & upper_triangle
    if not bool(pair_mask.any().detach()):
        return prediction.new_zeros(())
    prediction_difference = (prediction[:, None] - prediction[None, :]) / scale
    target_difference = (target[:, None] - target[None, :]) / scale
    return F.smooth_l1_loss(
        prediction_difference[pair_mask],
        target_difference[pair_mask],
    )


def adaptation_path_objective(
    predictions: Sequence[Tensor],
    target: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Mean path error and positive regret relative to no adaptation.

    The first prediction must be step 0.  Regret only penalizes steps that are
    worse than the initialization; useful adaptation is never penalized.
    """
    if not predictions:
        zero = target.new_zeros(())
        return zero, zero
    losses = torch.stack(
        [scaled_prediction_mse(prediction, target, scale) for prediction in predictions]
    )
    mean_path = losses.mean()
    regret = (
        torch.relu(losses[1:] - losses[0]).mean()
        if len(losses) > 1
        else losses.new_zeros(())
    )
    return mean_path, regret


@dataclass
class LossBreakdown:
    total: Tensor
    query_loss: Tensor
    total_prediction: Tensor
    general_prediction: Tensor
    general_domain: Tensor
    specific_domain: Tensor
    specific_contrastive: Tensor
    reconstruction: Tensor
    consistency: Tensor
    orthogonal: Tensor
    residual: Tensor
    specific_residual_fit: Tensor
    within_domain_difference: Tensor
    adaptation_path: Tensor
    adaptation_regret: Tensor
    general_domain_accuracy: Tensor
    specific_domain_accuracy: Tensor
    mean_absolute_residual: Tensor


def outer_objective(
    output: HybridOutput,
    target: Tensor,
    domain: Tensor,
    config: LossConfig,
    ablation: AblationConfig,
    auxiliary_scale: float = 1.0,
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
    general_domain = general_domain_loss(
        output.general_domain_logits, domain, config.domain_label_smoothing
    )
    specific_domain = (
        specific_domain_loss(
            output.specific_domain_logits, domain, config.domain_label_smoothing
        )
        if ablation.use_specific_domain_classifier
        and output.specific_domain_logits is not None
        else zero
    )
    specific_contrastive = (
        specific_supervised_contrastive_loss(
            output.specific_embedding,
            domain,
            config.specific_contrastive_temperature,
        )
        if config.lambda_specific_contrastive > 0
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
        orthogonality_loss(
            output.general_embedding,
            output.specific_embedding,
            config.orthogonality_reduction,
        )
        if ablation.use_orthogonality
        else zero
    )
    residual = (
        residual_regularization(output.specific_residual, config.rul_scale_cycles)
        if ablation.use_residual_regularization
        else zero
    )
    specific_residual_fit = (
        specific_residual_fit_loss(
            output.specific_residual,
            output.general_prediction,
            target,
            config.rul_scale_cycles,
        )
        if ablation.prediction_mode == "residual"
        else zero
    )
    within_domain_difference = within_domain_rul_difference_loss(
        output.prediction,
        target,
        domain,
        config.rul_scale_cycles,
    )
    total = (
        config.lambda_total_prediction * total_prediction
        + config.lambda_general_prediction * general_prediction
        + config.lambda_specific_residual_fit * specific_residual_fit
        + config.lambda_within_domain_difference * within_domain_difference
        + auxiliary_scale * (
            config.lambda_general_domain * general_domain
            + config.lambda_specific_domain * specific_domain
            + config.lambda_specific_contrastive * specific_contrastive
            + config.lambda_reconstruction * reconstruction
            + config.lambda_consistency * consistency
            + config.lambda_orthogonal * orthogonal
            + config.lambda_residual * residual
        )
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
        specific_contrastive=specific_contrastive,
        reconstruction=reconstruction,
        consistency=consistency,
        orthogonal=orthogonal,
        residual=residual,
        specific_residual_fit=specific_residual_fit,
        within_domain_difference=within_domain_difference,
        adaptation_path=zero,
        adaptation_regret=zero,
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

"""Losses for direct RUL regression and representation disentanglement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import LossConfig
from .model import ModelOutput


def task_loss(prediction: Tensor, target: Tensor, delta: float) -> Tensor:
    return F.huber_loss(prediction, target, delta=delta)


def fuzzy_uniform_loss(logits: Tensor) -> Tensor:
    """KL(model domain distribution || uniform), minimized at maximum ambiguity."""
    log_probability = F.log_softmax(logits, dim=-1)
    probability = log_probability.exp()
    log_uniform = -torch.log(
        torch.as_tensor(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    )
    return torch.sum(probability * (log_probability - log_uniform), dim=-1).mean()


def paired_orthogonality_loss(invariant: Tensor, specific: Tensor) -> Tensor:
    """Penalize per-cell cosine alignment without requiring a large batch."""
    invariant = F.normalize(invariant, p=2, dim=-1, eps=1.0e-8)
    specific = F.normalize(specific, p=2, dim=-1, eps=1.0e-8)
    return torch.sum(invariant * specific, dim=-1).square().mean()


@dataclass
class JointLossOutput:
    total: Tensor
    task: Tensor
    domain: Tensor
    fuzzy: Tensor
    orthogonality: Tensor


def joint_loss(
    output: ModelOutput,
    target: Tensor,
    domain_target: Tensor,
    config: LossConfig,
) -> JointLossOutput:
    regression = task_loss(output.prediction, target, config.huber_delta)
    domain = F.cross_entropy(output.domain_logits, domain_target)
    fuzzy = fuzzy_uniform_loss(output.fuzzy_logits)
    orthogonality = paired_orthogonality_loss(
        output.invariant_embedding, output.specific_embedding
    )
    total = (
        regression
        + config.domain_adversarial_weight * domain
        + config.domain_fuzzy_weight * fuzzy
        + config.orthogonality_weight * orthogonality
    )
    return JointLossOutput(total, regression, domain, fuzzy, orthogonality)


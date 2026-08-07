"""Raw-cycle RUL and representation losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .config import LossConfig
from .model import ModelOutput


def raw_rul_loss(prediction: Tensor, target: Tensor, config: LossConfig) -> Tensor:
    """Smooth-L1 on raw cycles, scaled only by a fixed physical cycle constant."""
    return F.smooth_l1_loss(
        prediction,
        target,
        beta=config.raw_rul_huber_beta_cycles,
    ) / config.raw_rul_loss_scale_cycles


def fuzzy_uniform_loss(logits: Tensor) -> Tensor:
    log_probability = F.log_softmax(logits, dim=-1)
    probability = log_probability.exp()
    log_uniform = -torch.log(
        torch.as_tensor(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    )
    return torch.sum(probability * (log_probability - log_uniform), dim=-1).mean()


def orthogonality_loss(invariant: Tensor, specific: Tensor) -> Tensor:
    invariant = F.normalize(invariant, dim=-1, eps=1.0e-8)
    specific = F.normalize(specific, dim=-1, eps=1.0e-8)
    return torch.sum(invariant * specific, dim=-1).square().mean()


@dataclass
class JointLoss:
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
) -> JointLoss:
    task = raw_rul_loss(output.prediction, target, config)
    domain = F.cross_entropy(output.domain_logits, domain_target)
    fuzzy = fuzzy_uniform_loss(output.fuzzy_logits)
    orthogonality = orthogonality_loss(
        output.invariant_embedding, output.specific_embedding
    )
    total = (
        task
        + config.domain_adversarial_weight * domain
        + config.domain_fuzzy_weight * fuzzy
        + config.orthogonality_weight * orthogonality
    )
    return JointLoss(total, task, domain, fuzzy, orthogonality)


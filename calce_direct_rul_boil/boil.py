"""BOIL inner/outer computations for the domain-specific representation body."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import DirectRULBOILModel
from .losses import task_loss


@dataclass
class BOILResult:
    total_loss: Tensor
    support_loss: Tensor
    query_loss: Tensor
    body_update_norm: Tensor
    query_prediction: Tensor


def adapt_specific_body(
    model: DirectRULBOILModel,
    support_features: Tensor,
    support_targets: Tensor,
    inner_steps: int,
    inner_lr: float,
    huber_delta: float,
    second_order: bool,
) -> tuple[OrderedDict[str, Tensor], Tensor, Tensor]:
    """Return fast F2 weights; the fusion prediction head is never updated here."""
    fast = model.initial_specific_parameters()
    support_losses: list[Tensor] = []
    update_squares: list[Tensor] = []
    for _ in range(inner_steps):
        prediction, _ = model.forward_meta(support_features, fast)
        loss = task_loss(prediction, support_targets, huber_delta)
        gradients = torch.autograd.grad(
            loss,
            tuple(fast.values()),
            create_graph=second_order,
            retain_graph=second_order,
            allow_unused=False,
        )
        fast = OrderedDict(
            (name, parameter - inner_lr * gradient)
            for (name, parameter), gradient in zip(fast.items(), gradients)
        )
        support_losses.append(loss)
        update_squares.extend((inner_lr * gradient).square().sum() for gradient in gradients)
    update_norm = torch.sqrt(torch.stack(update_squares).sum().clamp_min(0.0))
    return fast, torch.stack(support_losses).mean(), update_norm


def boil_episode(
    model: DirectRULBOILModel,
    support_features: Tensor,
    support_targets: Tensor,
    query_features: Tensor,
    query_targets: Tensor,
    *,
    inner_steps: int,
    inner_lr: float,
    huber_delta: float,
    second_order: bool,
    support_weight: float,
    query_weight: float,
) -> BOILResult:
    fast, support, update_norm = adapt_specific_body(
        model,
        support_features,
        support_targets,
        inner_steps,
        inner_lr,
        huber_delta,
        second_order,
    )
    query_prediction, _ = model.forward_meta(query_features, fast)
    query = task_loss(query_prediction, query_targets, huber_delta)
    total = support_weight * support + query_weight * query
    return BOILResult(total, support, query, update_norm, query_prediction)


def adapted_prediction(
    model: DirectRULBOILModel,
    support_features: Tensor,
    support_targets: Tensor,
    query_features: Tensor,
    *,
    inner_steps: int,
    inner_lr: float,
    huber_delta: float,
) -> Tensor:
    """Source-only validation helper; never used for held-out target evaluation."""
    with torch.enable_grad():
        fast, _, _ = adapt_specific_body(
            model,
            support_features,
            support_targets,
            inner_steps,
            inner_lr,
            huber_delta,
            second_order=False,
        )
        prediction, _ = model.forward_meta(query_features, fast)
    return prediction.detach()


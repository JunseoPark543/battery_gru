"""Second-order BOIL with a fixed prediction head in the inner loop."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import LossConfig
from .losses import raw_rul_loss
from .model import HUSTDirectRULModel


@dataclass
class BOILResult:
    total: Tensor
    support: Tensor
    query: Tensor
    update_norm: Tensor


def adapt_specific_body(
    model: HUSTDirectRULModel,
    support_waveforms: Tensor,
    support_scalars: Tensor,
    support_targets: Tensor,
    *,
    inner_steps: int,
    inner_lr: float,
    second_order: bool,
    loss_config: LossConfig,
) -> tuple[OrderedDict[str, Tensor], Tensor, Tensor]:
    fast = model.initial_specific_parameters()
    losses: list[Tensor] = []
    update_squares: list[Tensor] = []
    for _ in range(inner_steps):
        prediction = model.forward_meta(support_waveforms, support_scalars, fast)
        loss = raw_rul_loss(prediction, support_targets, loss_config)
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
        losses.append(loss)
        update_squares.extend((inner_lr * gradient).square().sum() for gradient in gradients)
    update_norm = torch.sqrt(torch.stack(update_squares).sum().clamp_min(0.0))
    return fast, torch.stack(losses).mean(), update_norm


def boil_episode(
    model: HUSTDirectRULModel,
    support_waveforms: Tensor,
    support_scalars: Tensor,
    support_targets: Tensor,
    query_waveforms: Tensor,
    query_scalars: Tensor,
    query_targets: Tensor,
    *,
    inner_steps: int,
    inner_lr: float,
    second_order: bool,
    loss_config: LossConfig,
) -> BOILResult:
    fast, support, update_norm = adapt_specific_body(
        model,
        support_waveforms,
        support_scalars,
        support_targets,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        second_order=second_order,
        loss_config=loss_config,
    )
    query_prediction = model.forward_meta(query_waveforms, query_scalars, fast)
    query = raw_rul_loss(query_prediction, query_targets, loss_config)
    total = loss_config.meta_support_weight * support + loss_config.meta_query_weight * query
    return BOILResult(total, support, query, update_norm)


def adapted_prediction(
    model: HUSTDirectRULModel,
    support_waveforms: Tensor,
    support_scalars: Tensor,
    support_targets: Tensor,
    query_waveforms: Tensor,
    query_scalars: Tensor,
    *,
    inner_steps: int,
    inner_lr: float,
    loss_config: LossConfig,
) -> Tensor:
    """Source-only selection helper; final target evaluation never calls this."""
    with torch.enable_grad():
        fast, _, _ = adapt_specific_body(
            model,
            support_waveforms,
            support_scalars,
            support_targets,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            second_order=False,
            loss_config=loss_config,
        )
        prediction = model.forward_meta(query_waveforms, query_scalars, fast)
    return prediction.detach()


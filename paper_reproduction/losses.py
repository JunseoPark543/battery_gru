"""Masked losses for padded recursive SOH targets."""

from __future__ import annotations

from collections.abc import Callable

import torch


def _validated_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError("prediction and target must share shape [B,H,1]")
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    if mask.shape != prediction.shape:
        try:
            mask = mask.expand_as(prediction)
        except RuntimeError as exc:
            raise ValueError("mask cannot be expanded over prediction") from exc
    mask = mask.to(device=prediction.device, dtype=torch.bool)
    if int(mask.sum()) == 0:
        raise ValueError("loss mask contains no valid target positions")
    selected_prediction = prediction.masked_select(mask)
    selected_target = target.to(prediction.device).masked_select(mask)
    if not torch.isfinite(selected_prediction).all() or not torch.isfinite(selected_target).all():
        raise FloatingPointError("valid loss positions contain NaN or infinity")
    return selected_prediction, selected_target


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over non-padding positions only."""
    predicted, actual = _validated_values(prediction, target, mask)
    return torch.mean((predicted - actual).square())


def masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error over non-padding positions only."""
    predicted, actual = _validated_values(prediction, target, mask)
    return torch.mean(torch.abs(predicted - actual))


def get_loss(kind: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if kind == "mse":
        return masked_mse
    if kind == "mae":
        return masked_mae
    raise ValueError("loss kind must be mse or mae")


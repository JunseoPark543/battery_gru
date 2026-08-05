"""Masked losses for padded recursive SOH targets."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch


def _validated_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    valid_per_sample = mask.sum(dim=(1, 2))
    if int(mask.sum()) == 0:
        raise ValueError("loss mask contains no valid target positions")
    if torch.any(valid_per_sample <= 0):
        raise ValueError("Every sample must contain at least one valid target.")
    target = target.to(prediction.device)
    selected_prediction = prediction.masked_select(mask)
    selected_target = target.masked_select(mask)
    if not torch.isfinite(selected_prediction).all() or not torch.isfinite(selected_target).all():
        raise FloatingPointError("valid loss positions contain NaN or infinity")
    return prediction, target, mask


def _reduce(error: torch.Tensor, mask: torch.Tensor, reduction: str) -> torch.Tensor:
    weights = mask.to(error.dtype)
    if reduction == "point_balanced":
        return (error * weights).sum() / weights.sum()
    if reduction == "sample_balanced":
        valid_count = weights.sum(dim=(1, 2))
        per_sample = (error * weights).sum(dim=(1, 2)) / valid_count
        return per_sample.mean()
    raise ValueError("recursive reduction must be point_balanced or sample_balanced")


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "point_balanced",
) -> torch.Tensor:
    """Masked MSE with global-point or equal-per-sample reduction."""
    predicted, actual, expanded_mask = _validated_tensors(prediction, target, mask)
    return _reduce((predicted - actual).square(), expanded_mask, reduction)


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "point_balanced",
) -> torch.Tensor:
    """Masked MAE with global-point or equal-per-sample reduction."""
    predicted, actual, expanded_mask = _validated_tensors(prediction, target, mask)
    return _reduce(torch.abs(predicted - actual), expanded_mask, reduction)


def get_loss(
    kind: str,
    reduction: str = "point_balanced",
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if kind == "mse":
        return partial(masked_mse, reduction=reduction)
    if kind == "mae":
        return partial(masked_mae, reduction=reduction)
    raise ValueError("loss kind must be mse or mae")

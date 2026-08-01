"""Prefix-to-future support pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SupportPair:
    history: torch.Tensor
    future: torch.Tensor


class PrefixFutureDataset(Dataset[SupportPair]):
    """All ``([y_1..y_j], [y_j+1..y_L])`` pairs for j=1..L-1."""

    def __init__(self, soh: np.ndarray | list[float] | torch.Tensor) -> None:
        if isinstance(soh, torch.Tensor):
            values = soh.detach().to(dtype=torch.float32).flatten().clone()
        else:
            values = torch.tensor(np.asarray(soh, dtype=np.float32).copy()).flatten()
        if len(values) < 2:
            raise ValueError("at least two SOH values are required for prefix pairs")
        if not torch.isfinite(values).all():
            raise ValueError("support SOH values must all be finite")
        self._values = values

    def __len__(self) -> int:
        return len(self._values) - 1

    def __getitem__(self, index: int) -> SupportPair:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        split = index + 1
        return SupportPair(
            history=self._values[:split].unsqueeze(-1),
            future=self._values[split:].unsqueeze(-1),
        )

    def sample_indices(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        count = min(batch_size, len(self))
        if count == len(self):
            return torch.arange(len(self), dtype=torch.long)
        return torch.randperm(len(self), generator=generator)[:count]

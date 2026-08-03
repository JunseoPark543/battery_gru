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
    """All feature-prefix to future-SOH pairs for j=1..L-1."""

    def __init__(
        self,
        soh: np.ndarray | list[float] | torch.Tensor,
        history_features: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        if isinstance(soh, torch.Tensor):
            values = soh.detach().to(dtype=torch.float32).flatten().clone()
        else:
            values = torch.tensor(np.asarray(soh, dtype=np.float32).copy()).flatten()
        if len(values) < 2:
            raise ValueError("at least two SOH values are required for prefix pairs")
        if not torch.isfinite(values).all():
            raise ValueError("support SOH values must all be finite")
        self._values = values
        if history_features is None:
            features = values.unsqueeze(-1).clone()
        elif isinstance(history_features, torch.Tensor):
            features = history_features.detach().to(dtype=torch.float32).clone()
        else:
            features = torch.tensor(
                np.asarray(history_features, dtype=np.float32).copy(), dtype=torch.float32
            )
        if features.ndim != 2 or features.shape[0] != len(values):
            raise ValueError("history features must have shape [L, feature_count]")
        if not torch.isfinite(features).all():
            raise ValueError("support history features must all be finite")
        if not torch.allclose(features[:, 0], values):
            raise ValueError("the first history feature must be SOH")
        self._features = features

    def __len__(self) -> int:
        return len(self._values) - 1

    def __getitem__(self, index: int) -> SupportPair:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        split = index + 1
        return SupportPair(
            history=self._features[:split],
            future=self._values[split:].unsqueeze(-1),
        )

    def sample_indices(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        count = min(batch_size, len(self))
        if count == len(self):
            return torch.arange(len(self), dtype=torch.long)
        # A CUDA generator cannot drive a CPU randperm. Sample on the
        # generator's own device, then return CPU indices for Dataset access.
        return torch.randperm(
            len(self), generator=generator, device=generator.device
        )[:count].cpu()

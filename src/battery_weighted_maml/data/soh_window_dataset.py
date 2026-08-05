"""Fixed-window supervised samples for the plain GRU baseline."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SOHWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map observed SOH windows to their immediately following SOH horizon."""

    def __init__(
        self,
        soh: np.ndarray | list[float] | torch.Tensor,
        encoder_window: int,
        forecast_horizon: int,
        stride: int = 1,
    ) -> None:
        values = (
            soh.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
            if isinstance(soh, torch.Tensor)
            else torch.tensor(np.asarray(soh, dtype=np.float32).copy()).flatten()
        )
        if not torch.isfinite(values).all():
            raise ValueError("SOH window data must contain only finite values")
        if encoder_window <= 0 or forecast_horizon <= 0 or stride <= 0:
            raise ValueError("window, horizon, and stride must be positive")
        required = encoder_window + forecast_horizon
        if len(values) < required:
            raise ValueError(
                f"at least {required} SOH values are required, found {len(values)}"
            )
        self.values = values
        self.encoder_window = encoder_window
        self.forecast_horizon = forecast_horizon
        self.starts = list(range(0, len(values) - required + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        split = start + self.encoder_window
        end = split + self.forecast_horizon
        return (
            self.values[start:split].unsqueeze(-1),
            self.values[split:end].unsqueeze(-1),
        )

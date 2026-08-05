"""Plain SOH-only GRU encoder-decoder baseline without meta-learning."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class SOHGRUEncoderDecoder(nn.Module):
    """Encode observed SOH and recursively decode future scalar SOH."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    @staticmethod
    def _validate_history(history: torch.Tensor) -> None:
        if history.ndim != 3 or history.shape[-1] != 1:
            raise ValueError(f"history must have shape [B,T,1], got {tuple(history.shape)}")
        if history.shape[1] < 1:
            raise ValueError("history cannot be empty")
        if not torch.isfinite(history).all():
            raise ValueError("history must contain only finite SOH values")

    def forward(
        self,
        history: torch.Tensor,
        horizon: int | None = None,
        future_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return recursive predictions with shape ``[B,horizon,1]``."""
        self._validate_history(history)
        if horizon is None:
            if future_targets is None:
                raise ValueError("horizon is required when future_targets is absent")
            horizon = future_targets.shape[1]
        if horizon < 0:
            raise ValueError("horizon cannot be negative")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        if future_targets is not None and (
            future_targets.ndim != 3
            or future_targets.shape[0] != history.shape[0]
            or future_targets.shape[1] < horizon
            or future_targets.shape[2] != 1
        ):
            raise ValueError("future_targets must have shape [B,at least horizon,1]")

        _, state = self.encoder(history)
        decoder_input = history[:, -1:, :]
        predictions: list[torch.Tensor] = []
        for step in range(horizon):
            decoder_output, state = self.decoder(decoder_input, state)
            prediction = self.output(decoder_output[:, -1, :]).unsqueeze(1)
            predictions.append(prediction)
            if future_targets is not None and teacher_forcing_ratio > 0.0:
                if teacher_forcing_ratio == 1.0:
                    use_teacher = torch.ones(
                        history.shape[0], 1, 1, dtype=torch.bool, device=history.device
                    )
                else:
                    use_teacher = torch.rand(
                        history.shape[0], 1, 1,
                        device=history.device,
                        generator=generator,
                    ) < teacher_forcing_ratio
                decoder_input = torch.where(
                    use_teacher, future_targets[:, step:step + 1, :], prediction
                )
            else:
                decoder_input = prediction
        if not predictions:
            return history.new_empty((history.shape[0], 0, 1))
        return torch.cat(predictions, dim=1)

    @torch.no_grad()
    def recursive_forecast(
        self,
        history: torch.Tensor | Sequence[float],
        horizon: int,
    ) -> torch.Tensor:
        """Forecast without teacher forcing from ``[T]``, ``[T,1]``, or ``[B,T,1]``."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        if isinstance(history, torch.Tensor):
            values = history.detach().to(device=device, dtype=torch.float32).clone()
        else:
            values = torch.tensor(history, dtype=torch.float32, device=device)
        if values.ndim == 1:
            values = values.view(1, -1, 1)
        elif values.ndim == 2 and values.shape[-1] == 1:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[-1] != 1:
            raise ValueError("history must represent [T], [T,1], or [B,T,1] SOH")
        result = self.forward(values, horizon=horizon, teacher_forcing_ratio=0.0)
        self.train(was_training)
        return result


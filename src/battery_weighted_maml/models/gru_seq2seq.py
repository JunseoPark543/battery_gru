"""Recursive GRU encoder-decoder for multivariate history and scalar SOH output."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over true mask entries only.

    Args:
        prediction: ``[batch, horizon, 1]``.
        target: ``[batch, horizon, 1]``.
        mask: boolean ``[batch, horizon]`` or ``[batch, horizon, 1]``.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {prediction.shape} vs {target.shape}")
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(-1)
    if mask.shape != prediction.shape:
        try:
            mask = mask.expand_as(prediction)
        except RuntimeError as exc:
            raise ValueError(f"mask shape {mask.shape} cannot cover {prediction.shape}") from exc
    mask = mask.to(device=prediction.device, dtype=torch.bool)
    count = mask.sum()
    if int(count) == 0:
        raise ValueError("masked_mse received a mask with no valid positions")
    squared = (prediction - target).square()
    return squared.masked_select(mask).mean()


class GRUSeq2Seq(nn.Module):
    """GRU encoder-decoder with multivariate encoder and scalar SOH decoder."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.GRU(
            input_size=input_size,
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

    def encode(
        self, sequence: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded ``[B,T,F]`` sequences into outputs and ``[layers,B,H]`` state."""
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_size:
            raise ValueError(
                f"sequence must have shape [B,T,{self.input_size}], got {tuple(sequence.shape)}"
            )
        if lengths.ndim != 1 or len(lengths) != sequence.shape[0]:
            raise ValueError("lengths must be a [B] tensor matching sequence batch size")
        if torch.any(lengths <= 0) or torch.any(lengths > sequence.shape[1]):
            raise ValueError("all sequence lengths must be within [1, T]")
        packed = pack_padded_sequence(
            sequence,
            lengths.detach().to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.encoder(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=sequence.shape[1]
        )
        return output, hidden

    def decode(
        self,
        hidden: torch.Tensor,
        first_value: torch.Tensor,
        horizon: int,
        targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode recursively and return ``[B,horizon,1]`` predictions."""
        if horizon < 0:
            raise ValueError("horizon cannot be negative")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        if first_value.ndim == 1:
            first_value = first_value.unsqueeze(-1)
        if first_value.ndim == 3 and first_value.shape[1] == 1:
            first_value = first_value[:, 0, :]
        if first_value.ndim != 2 or first_value.shape[-1] != 1:
            raise ValueError("first_value must have shape [B,1]")
        if targets is not None and (
            targets.ndim != 3
            or targets.shape[0] != first_value.shape[0]
            or targets.shape[1] < horizon
            or targets.shape[2] != 1
        ):
            raise ValueError("targets must have shape [B,at least horizon,1]")
        decoder_input = first_value.unsqueeze(1)
        state = hidden
        predictions: list[torch.Tensor] = []
        for step in range(horizon):
            decoder_output, state = self.decoder(decoder_input, state)
            prediction = self.output(decoder_output[:, -1, :])
            predictions.append(prediction.unsqueeze(1))
            if targets is not None and teacher_forcing_ratio > 0:
                if teacher_forcing_ratio == 1.0:
                    use_teacher = torch.ones(
                        first_value.shape[0], 1, dtype=torch.bool, device=first_value.device
                    )
                else:
                    use_teacher = torch.rand(
                        first_value.shape[0], 1,
                        device=first_value.device,
                        generator=generator,
                    ) < teacher_forcing_ratio
                next_value = torch.where(use_teacher, targets[:, step, :], prediction)
            else:
                next_value = prediction
            decoder_input = next_value.unsqueeze(1)
        if not predictions:
            return first_value.new_empty((first_value.shape[0], 0, 1))
        return torch.cat(predictions, dim=1)

    def forward(
        self,
        history: torch.Tensor,
        history_lengths: torch.Tensor,
        future_targets: torch.Tensor | None = None,
        horizon: int | None = None,
        teacher_forcing_ratio: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Forecast SOH from padded ``history [B,T,F]`` to ``[B,horizon,1]``."""
        _, hidden = self.encode(history, history_lengths)
        batch_index = torch.arange(history.shape[0], device=history.device)
        # SOH is always feature zero. Future voltage is deliberately not needed:
        # the decoder recursively consumes scalar SOH values only.
        last_value = history[
            batch_index, history_lengths.to(history.device) - 1, 0:1
        ]
        if horizon is None:
            if future_targets is None:
                raise ValueError("horizon is required when future_targets is absent")
            horizon = future_targets.shape[1]
        return self.decode(
            hidden,
            last_value,
            horizon,
            targets=future_targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            generator=generator,
        )

    @torch.no_grad()
    def recursive_forecast(
        self,
        history: torch.Tensor | Sequence[float],
        horizon: int,
    ) -> torch.Tensor:
        """Forecast without teacher forcing, returning ``[B,horizon,1]``."""
        was_training = self.training
        self.eval()
        model_device = next(self.parameters()).device
        if isinstance(history, torch.Tensor):
            values = history.detach().to(device=model_device, dtype=torch.float32).clone()
        else:
            values = torch.tensor(history, dtype=torch.float32, device=model_device)
        if self.input_size == 1:
            if values.ndim == 1:
                values = values.unsqueeze(0).unsqueeze(-1)
            elif values.ndim == 2:
                values = (
                    values.unsqueeze(0)
                    if values.shape[-1] == 1
                    else values.unsqueeze(-1)
                )
        elif values.ndim == 2 and values.shape[-1] == self.input_size:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[-1] != self.input_size:
            raise ValueError(
                f"history must end with {self.input_size} feature(s); "
                "use [T,F] or [B,T,F] for multivariate input"
            )
        lengths = torch.full(
            (values.shape[0],), values.shape[1], dtype=torch.long, device=values.device
        )
        result = self.forward(values, lengths, horizon=horizon, teacher_forcing_ratio=0.0)
        self.train(was_training)
        return result

    def empirical_points(self, support_features: torch.Tensor) -> torch.Tensor:
        """Build detached-ready empirical points ``[L-1,H+1]`` from support only."""
        values = support_features
        if values.ndim == 1 and self.input_size == 1:
            values = values.unsqueeze(-1)
        if values.ndim != 2 or values.shape[-1] != self.input_size:
            raise ValueError(
                f"support_features must have shape [L,{self.input_size}]"
            )
        if len(values) < 2:
            raise ValueError("at least two support observations are required")
        history = values.unsqueeze(0)
        lengths = torch.tensor([len(values)], dtype=torch.long, device=values.device)
        outputs, _ = self.encode(history, lengths)
        normalized = F.normalize(outputs[0, :-1, :], p=2, dim=-1, eps=1e-12)
        labels = values[1:, 0:1]
        return torch.cat([normalized, labels], dim=-1)

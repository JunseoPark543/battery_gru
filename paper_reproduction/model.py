"""Paper GRU encoder-decoder with predicted-input probability semantics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoder(nn.Module):
    """Encode padded scalar SOH ``[B,T,1]`` into hidden ``[layers,B,H]``."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

    def forward(self, history: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            history,
            lengths.detach().to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden


class GRUDecoder(nn.Module):
    """Run one scalar decoder step ``[B,1,1]``."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

    def forward(
        self, value: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gru(value, hidden)


class GRUEncoderDecoder(nn.Module):
    """Forecast future SOH from ``history [B,T,1]`` to ``[B,H,1]``."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        self.encoder = GRUEncoder(hidden_size, num_layers)
        self.decoder = GRUDecoder(hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(
        self,
        history: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        prediction_length: int | None = None,
        predicted_input_probability: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode SOH, where probability p selects the previous prediction.

        ``predicted_input_probability`` follows the paper: p=0 uses ground
        truth whenever available, whereas p=1 always uses model predictions.
        In eval mode, predictions are always recursive regardless of p.
        """
        if history.ndim != 3 or history.shape[-1] != 1 or history.shape[1] < 1:
            raise ValueError(f"history must have shape [B,T,1], got {tuple(history.shape)}")
        if not torch.isfinite(history).all():
            raise ValueError("history contains NaN or infinity")
        if input_lengths is None:
            input_lengths = torch.full(
                (history.shape[0],), history.shape[1], dtype=torch.long, device=history.device
            )
        if input_lengths.ndim != 1 or len(input_lengths) != history.shape[0]:
            raise ValueError("input_lengths must be [B]")
        if torch.any(input_lengths <= 0) or torch.any(input_lengths > history.shape[1]):
            raise ValueError("input lengths must lie in [1,T]")
        if not 0.0 <= predicted_input_probability <= 1.0:
            raise ValueError("predicted_input_probability must be in [0,1]")
        if prediction_length is None:
            if target is None:
                raise ValueError("prediction_length is required when target is None")
            prediction_length = target.shape[1]
        if prediction_length < 0:
            raise ValueError("prediction_length cannot be negative")
        if target is not None and (
            target.ndim != 3
            or target.shape[0] != history.shape[0]
            or target.shape[1] < prediction_length
            or target.shape[2] != 1
        ):
            raise ValueError("target must have shape [B,at least H,1]")

        hidden = self.encoder(history, input_lengths)
        batch_index = torch.arange(history.shape[0], device=history.device)
        last_indices = input_lengths.to(history.device) - 1
        decoder_input = history[batch_index, last_indices, :].unsqueeze(1)
        predictions: list[torch.Tensor] = []
        for step in range(prediction_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            prediction = self.output_layer(decoder_output)
            predictions.append(prediction)
            if self.training and target is not None and predicted_input_probability < 1.0:
                if predicted_input_probability == 0.0:
                    use_prediction = torch.zeros(
                        history.shape[0], 1, 1, dtype=torch.bool, device=history.device
                    )
                else:
                    use_prediction = torch.rand(
                        history.shape[0], 1, 1,
                        device=history.device,
                        generator=generator,
                    ) < predicted_input_probability
                decoder_input = torch.where(
                    use_prediction, prediction, target[:, step:step + 1, :]
                )
            else:
                decoder_input = prediction
        if not predictions:
            return history.new_empty((history.shape[0], 0, 1))
        return torch.cat(predictions, dim=1)

    @torch.no_grad()
    def recursive_forecast(
        self, history: torch.Tensor | Sequence[float], prediction_length: int
    ) -> torch.Tensor:
        """Teacher-forcing-free rollout returning ``[B,H,1]``."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        values = (
            history.detach().to(device=device, dtype=torch.float32).clone()
            if isinstance(history, torch.Tensor)
            else torch.tensor(history, dtype=torch.float32, device=device)
        )
        if values.ndim == 1:
            values = values.view(1, -1, 1)
        elif values.ndim == 2 and values.shape[-1] == 1:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[-1] != 1:
            raise ValueError("history must represent [T], [T,1], or [B,T,1]")
        result = self.forward(values, prediction_length=prediction_length)
        self.train(was_training)
        return result


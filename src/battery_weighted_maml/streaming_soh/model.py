"""Hierarchical curve encoder + recurrent degradation state + SOH decoder."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(1, output_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class PartialCycleEncoder(nn.Module):
    """Encode a full or partial V/I-Q curve using masked point attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = [4, *config.convolution_channels]
        self.convolution = nn.Sequential(
            *[
                ConvBlock(channels[index], channels[index + 1], config.kernel_size, config.dropout)
                for index in range(len(channels) - 1)
            ]
        )
        self.point_projection = nn.Linear(channels[-1], config.curve_embedding_dim)
        self.attention_score = nn.Sequential(
            nn.Linear(config.curve_embedding_dim, config.curve_embedding_dim),
            nn.Tanh(),
            nn.Linear(config.curve_embedding_dim, 1),
        )
        self.output_norm = nn.LayerNorm(config.curve_embedding_dim)

    def forward(self, feature: torch.Tensor, q_coordinate: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 3 or feature.shape[-1] != 3:
            raise ValueError("curve feature must have shape [batch,q,3]")
        if q_coordinate.shape != feature.shape[:2]:
            raise ValueError("q_coordinate must have shape [batch,q]")
        observed = feature[..., 2] > 0.5
        convolution_input = torch.cat([feature, q_coordinate.unsqueeze(-1)], dim=-1)
        tokens = self.convolution(convolution_input.transpose(1, 2)).transpose(1, 2)
        tokens = self.point_projection(tokens)
        logits = self.attention_score(tokens).squeeze(-1).masked_fill(~observed, -1.0e4)
        weights = torch.softmax(logits, dim=1) * observed.to(tokens.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        available = observed.any(dim=1, keepdim=True).to(tokens.dtype)
        return self.output_norm(pooled) * available


class StreamingSOHForecaster(nn.Module):
    """Update an SOH trajectory from a causally observed within-cycle V/I prefix.

    A completed-history state is computed once conceptually. Every new prefix
    of the same current cycle is applied to that immutable state to form a
    candidate state. The current cycle is therefore never counted repeatedly.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.curve_encoder = PartialCycleEncoder(config)
        scalar_features = 4  # SOH, cycle gap, absolute cycle, curve available/prefix.
        self.completed_cycle_projection = nn.Sequential(
            nn.Linear(config.curve_embedding_dim + scalar_features, config.cycle_feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.current_cycle_projection = nn.Sequential(
            nn.Linear(config.curve_embedding_dim + scalar_features, config.cycle_feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        cells: list[nn.GRUCell] = []
        for layer in range(config.gru_layers):
            input_dim = config.cycle_feature_dim if layer == 0 else config.gru_hidden_dim
            cells.append(nn.GRUCell(input_dim, config.gru_hidden_dim))
        self.gru_cells = nn.ModuleList(cells)
        self.recurrent_dropout = nn.Dropout(config.dropout)
        trajectory_input = config.gru_hidden_dim + 4
        self.trajectory_backbone = nn.Sequential(
            nn.Linear(trajectory_input, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
        )
        self.soh_delta_head = nn.Linear(config.decoder_hidden_dim, 1)
        self.soh_std_head = nn.Linear(config.decoder_hidden_dim, 1)
        # Persistence is a stable initial prediction before training.
        nn.init.zeros_(self.soh_delta_head.weight)
        nn.init.zeros_(self.soh_delta_head.bias)
        self.q_embedding = nn.Sequential(
            nn.Linear(1, config.curve_embedding_dim),
            nn.GELU(),
            nn.Linear(config.curve_embedding_dim, config.curve_embedding_dim),
        )
        self.voltage_decoder = nn.Sequential(
            nn.Linear(
                config.gru_hidden_dim + config.curve_embedding_dim + 1,
                config.decoder_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, 1),
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(config.gru_hidden_dim + 1, config.gru_hidden_dim),
            nn.GELU(),
            nn.Linear(config.gru_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _advance(
        self,
        cycle_feature: torch.Tensor,
        state: Sequence[torch.Tensor],
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        next_state: list[torch.Tensor] = []
        layer_input = cycle_feature
        for layer, cell in enumerate(self.gru_cells):
            candidate = cell(layer_input, state[layer])
            updated = torch.where(active.unsqueeze(-1), candidate, state[layer])
            next_state.append(updated)
            layer_input = self.recurrent_dropout(updated) if layer + 1 < len(self.gru_cells) else updated
        return tuple(next_state)

    def encode_history(
        self,
        history_curve: torch.Tensor,
        history_soh: torch.Tensor,
        history_gap: torch.Tensor,
        history_cycle_scaled: torch.Tensor,
        history_mask: torch.Tensor,
        q_coordinate: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if history_curve.ndim != 4:
            raise ValueError("history_curve must have shape [batch,history,q,3]")
        batch, history, q_points, channels = history_curve.shape
        if channels != 3 or q_coordinate.shape != (batch, q_points):
            raise ValueError("history curve/q shapes are incompatible")
        expanded_q = q_coordinate[:, None, :].expand(-1, history, -1).reshape(
            batch * history, q_points
        )
        curve_embedding = self.curve_encoder(
            history_curve.reshape(batch * history, q_points, channels), expanded_q
        ).reshape(batch, history, -1)
        curve_available = history_curve[..., 2].any(dim=-1).to(history_curve.dtype)
        cycle_feature = self.completed_cycle_projection(
            torch.cat(
                [
                    curve_embedding,
                    history_soh.unsqueeze(-1),
                    history_gap.unsqueeze(-1),
                    history_cycle_scaled.unsqueeze(-1),
                    curve_available.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        state = tuple(
            history_curve.new_zeros((batch, self.config.gru_hidden_dim))
            for _ in range(self.config.gru_layers)
        )
        for time_index in range(history):
            state = self._advance(
                cycle_feature[:, time_index], state, history_mask[:, time_index]
            )
        if not history_mask.any(dim=1).all():
            raise ValueError("every sample must contain completed history")
        # Histories are left padded, so translate the valid count to the actual
        # last occupied column rather than indexing by the count itself.
        last_position = history - 1 - torch.flip(history_mask, dims=[1]).long().argmax(dim=1)
        last_soh = history_soh.gather(1, last_position.unsqueeze(1)).squeeze(1)
        return state, last_soh

    def condition_current(
        self,
        history_state: Sequence[torch.Tensor],
        last_soh: torch.Tensor,
        current_curve: torch.Tensor,
        q_coordinate: torch.Tensor,
        current_gap: torch.Tensor,
        current_cycle_scaled: torch.Tensor,
        prefix_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        current_embedding = self.curve_encoder(current_curve, q_coordinate)
        cycle_feature = self.current_cycle_projection(
            torch.cat(
                [
                    current_embedding,
                    last_soh.unsqueeze(-1),
                    current_gap.unsqueeze(-1),
                    current_cycle_scaled.unsqueeze(-1),
                    prefix_fraction.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        active = torch.ones(len(current_curve), dtype=torch.bool, device=current_curve.device)
        return self._advance(cycle_feature, history_state, active)

    def decode_trajectory(
        self,
        current_state: Sequence[torch.Tensor],
        last_soh: torch.Tensor,
        current_cycle_scaled: torch.Tensor,
        query_cycle_scaled: torch.Tensor,
        prefix_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = current_state[-1].unsqueeze(1).expand(-1, query_cycle_scaled.shape[1], -1)
        relative = query_cycle_scaled - current_cycle_scaled.unsqueeze(1)
        decoder_input = torch.cat(
            [
                state,
                query_cycle_scaled.unsqueeze(-1),
                relative.unsqueeze(-1),
                last_soh[:, None, None].expand(-1, query_cycle_scaled.shape[1], -1),
                prefix_fraction[:, None, None].expand(-1, query_cycle_scaled.shape[1], -1),
            ],
            dim=-1,
        )
        hidden = self.trajectory_backbone(decoder_input)
        mean = last_soh.unsqueeze(1) + self.soh_delta_head(hidden).squeeze(-1)
        std = self.config.minimum_soh_std + F.softplus(self.soh_std_head(hidden).squeeze(-1))
        return mean, std

    def forward(
        self,
        history_curve: torch.Tensor,
        history_soh: torch.Tensor,
        history_gap: torch.Tensor,
        history_cycle_scaled: torch.Tensor,
        history_mask: torch.Tensor,
        current_curve: torch.Tensor,
        q_coordinate: torch.Tensor,
        current_gap: torch.Tensor,
        current_cycle_scaled: torch.Tensor,
        prefix_fraction: torch.Tensor,
        query_cycle_scaled: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_state, last_soh = self.encode_history(
            history_curve,
            history_soh,
            history_gap,
            history_cycle_scaled,
            history_mask,
            q_coordinate,
        )
        current_state = self.condition_current(
            history_state,
            last_soh,
            current_curve,
            q_coordinate,
            current_gap,
            current_cycle_scaled,
            prefix_fraction,
        )
        soh_mean, soh_std = self.decode_trajectory(
            current_state,
            last_soh,
            current_cycle_scaled,
            query_cycle_scaled,
            prefix_fraction,
        )
        state = current_state[-1]
        q_embedding = self.q_embedding(q_coordinate.unsqueeze(-1))
        voltage_state = state.unsqueeze(1).expand(-1, q_coordinate.shape[1], -1)
        voltage = self.voltage_decoder(
            torch.cat(
                [
                    voltage_state,
                    q_embedding,
                    prefix_fraction[:, None, None].expand(-1, q_coordinate.shape[1], -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        remaining = self.endpoint_head(
            torch.cat([state, prefix_fraction.unsqueeze(-1)], dim=-1)
        ).squeeze(-1)
        endpoint_fraction = prefix_fraction + (1.0 - prefix_fraction) * remaining
        return {
            "soh_mean": soh_mean,
            "soh_std": soh_std,
            "voltage": voltage,
            "endpoint_fraction": endpoint_fraction,
            "completed_state": history_state[-1],
            "candidate_state": current_state[-1],
        }


def build_model(config: ModelConfig) -> StreamingSOHForecaster:
    return StreamingSOHForecaster(config)

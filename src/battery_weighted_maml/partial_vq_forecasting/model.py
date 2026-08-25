"""CNN/attention prefix encoder with coordinate-conditioned V(Q) decoder."""

from __future__ import annotations

import torch
from torch import nn

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


class PartialVQForecaster(nn.Module):
    """Complete the current discharge V-Q curve from its observed prefix.

    The network predicts voltage at every configured q query and separately
    predicts q_end. No SOH, cycle index, future voltage, or future endpoint is
    included in the input.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = [2, *config.convolution_channels]
        self.convolution = nn.Sequential(
            *[
                ConvBlock(channels[index], channels[index + 1], config.kernel_size, config.dropout)
                for index in range(len(channels) - 1)
            ]
        )
        self.token_projection = nn.Linear(channels[-1], config.hidden_dim)
        self.q_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.use_attention = bool(config.use_attention)
        if self.use_attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.attention = nn.TransformerEncoder(
                encoder_layer, num_layers=config.attention_layers
            )
            self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.attention = None
            self.register_parameter("cls_token", None)
        self.summary_norm = nn.LayerNorm(config.hidden_dim)
        self.voltage_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, 1),
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def encode(self, input_feature: torch.Tensor, q_coordinate: torch.Tensor) -> torch.Tensor:
        if input_feature.ndim != 3 or input_feature.shape[-1] != 2:
            raise ValueError("input_feature must have shape [batch,q,2]")
        if q_coordinate.shape != input_feature.shape[:2]:
            raise ValueError("q_coordinate must have shape [batch,q]")
        observed = input_feature[..., 1] > 0.5
        if not observed.any(dim=1).all():
            raise ValueError("every sample must contain at least one observed point")
        tokens = self.convolution(input_feature.transpose(1, 2)).transpose(1, 2)
        tokens = self.token_projection(tokens) + self.q_embedding(q_coordinate.unsqueeze(-1))
        if self.use_attention:
            assert self.attention is not None and self.cls_token is not None
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            padding = torch.cat(
                [
                    torch.zeros(
                        observed.shape[0], 1, dtype=torch.bool, device=observed.device
                    ),
                    ~observed,
                ],
                dim=1,
            )
            encoded = self.attention(tokens, src_key_padding_mask=padding)
            summary = encoded[:, 0]
        else:
            weights = observed.unsqueeze(-1).to(tokens.dtype)
            summary = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.summary_norm(summary)

    def forward(
        self,
        input_feature: torch.Tensor,
        q_coordinate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        summary = self.encode(input_feature, q_coordinate)
        query = self.q_embedding(q_coordinate.unsqueeze(-1))
        state = summary.unsqueeze(1).expand(-1, q_coordinate.shape[1], -1)
        voltage = self.voltage_decoder(torch.cat([state, query], dim=-1)).squeeze(-1)
        observed = input_feature[..., 1] > 0.5
        q_cut_fraction = q_coordinate.masked_fill(~observed, 0.0).amax(dim=1)
        remaining_fraction = self.endpoint_head(summary).squeeze(-1)
        # A discharge cannot end before the latest already observed q point.
        endpoint_fraction = q_cut_fraction + (1.0 - q_cut_fraction) * remaining_fraction
        return {"voltage": voltage, "endpoint_fraction": endpoint_fraction}


def build_model(config: ModelConfig) -> PartialVQForecaster:
    return PartialVQForecaster(config)

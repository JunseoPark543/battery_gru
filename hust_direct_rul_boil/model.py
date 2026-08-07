"""Hierarchical waveform/cycle model with a BOIL-adaptable specific branch."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.func import functional_call
import torch.nn.functional as F

from .config import ModelConfig


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, inputs: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(inputs: Tensor, strength: float) -> Tensor:
    return _GradientReversal.apply(inputs, strength)


class ResidualWaveformBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.conv1(F.gelu(self.norm1(inputs)))
        hidden = self.conv2(self.dropout(F.gelu(self.norm2(hidden))))
        return inputs + self.dropout(hidden)


class WaveformEncoder(nn.Module):
    """Encode one charge waveform for every early cycle."""

    def __init__(self, channels: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(channels, config.waveform_hidden, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            [
                ResidualWaveformBlock(config.waveform_hidden, dilation, config.dropout)
                for dilation in (1, 2, 4)
            ]
        )
        self.attention = nn.Conv1d(config.waveform_hidden, 1, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Linear(2 * config.waveform_hidden, config.waveform_embedding),
            nn.LayerNorm(config.waveform_embedding),
            nn.GELU(),
        )

    def forward(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim != 4:
            raise ValueError("waveforms must be [batch, cycle, point, channel]")
        batch, cycles, points, channels = waveforms.shape
        hidden = waveforms.reshape(batch * cycles, points, channels).transpose(1, 2)
        hidden = self.input_projection(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        weights = torch.softmax(self.attention(hidden), dim=-1)
        mean = torch.sum(weights * hidden, dim=-1)
        second = torch.sum(weights * hidden.square(), dim=-1)
        std = torch.sqrt(torch.clamp(second - mean.square(), min=1.0e-6))
        embedding = self.projection(torch.cat((mean, std), dim=-1))
        return embedding.reshape(batch, cycles, -1)


class MultiScaleCycleBlock(nn.Module):
    """Capture short-, medium-, and long-range changes before self-attention."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2)
                for kernel in (3, 7, 15)
            ]
        )
        self.fusion = nn.Conv1d(3 * d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        channels_first = inputs.transpose(1, 2)
        branches = [F.gelu(layer(channels_first)) for layer in self.convolutions]
        fused = self.fusion(torch.cat(branches, dim=1)).transpose(1, 2)
        return self.norm(inputs + self.dropout(fused))


class MultiHeadSelfAttention(nn.Module):
    """Explicit attention path retained for reliable second-order derivatives."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, length, 3, self.nhead, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(torch.softmax(logits, dim=-1))
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.output(context)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(
            config.d_model, config.nhead, config.dropout
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.dropout(self.attention(self.norm1(inputs)))
        return hidden + self.dropout(self.feed_forward(self.norm2(hidden)))


class CBAM1d(nn.Module):
    def __init__(self, channels: int, reduction: int, kernel_size: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
        )
        self.temporal = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        channel = torch.sigmoid(
            self.channel_mlp(inputs.mean(dim=1))
            + self.channel_mlp(inputs.amax(dim=1))
        ).unsqueeze(1)
        hidden = inputs * channel
        temporal_input = torch.stack(
            (hidden.mean(dim=-1), hidden.amax(dim=-1)), dim=1
        )
        temporal = torch.sigmoid(self.temporal(temporal_input)).transpose(1, 2)
        return hidden * temporal


class CycleAttentionStage(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.transformer = TransformerBlock(config)
        self.cbam = CBAM1d(
            config.d_model, config.cbam_reduction, config.cbam_kernel_size
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.transformer(inputs)
        return self.norm(inputs + self.cbam(hidden))


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, d_model: int, embedding_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.Tanh(),
            nn.Linear(max(16, d_model // 2), 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * d_model, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        weights = torch.softmax(self.attention(inputs), dim=1)
        mean = torch.sum(weights * inputs, dim=1)
        second = torch.sum(weights * inputs.square(), dim=1)
        std = torch.sqrt(torch.clamp(second - mean.square(), min=1.0e-6))
        return self.projection(torch.cat((mean, std), dim=-1))


class HierarchicalFeatureExtractor(nn.Module):
    def __init__(
        self,
        waveform_channels: int,
        scalar_features: int,
        history_length: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.waveform_channels = waveform_channels
        self.scalar_features = scalar_features
        self.history_length = history_length
        self.waveform_encoder = WaveformEncoder(waveform_channels, config)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_features, config.scalar_embedding),
            nn.LayerNorm(config.scalar_embedding),
            nn.GELU(),
        )
        self.fusion = nn.Linear(
            config.waveform_embedding + config.scalar_embedding, config.d_model
        )
        self.position = nn.Parameter(torch.zeros(1, history_length, config.d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.multi_scale = MultiScaleCycleBlock(config.d_model, config.dropout)
        self.stages = nn.ModuleList(
            [CycleAttentionStage(config) for _ in range(config.attention_stages)]
        )
        self.pooling = AttentiveStatisticsPooling(config.d_model, config.embedding_dim)

    def forward(self, waveforms: Tensor, scalars: Tensor) -> Tensor:
        if waveforms.shape[1] != self.history_length:
            raise ValueError("waveform history length mismatch")
        if waveforms.shape[-1] != self.waveform_channels:
            raise ValueError("waveform channel mismatch")
        if scalars.ndim != 3 or scalars.shape[1:] != (
            self.history_length,
            self.scalar_features,
        ):
            raise ValueError("scalar feature shape mismatch")
        waveform_embedding = self.waveform_encoder(waveforms)
        scalar_embedding = self.scalar_encoder(scalars)
        hidden = self.fusion(torch.cat((waveform_embedding, scalar_embedding), dim=-1))
        hidden = self.multi_scale(hidden + self.position)
        for stage in self.stages:
            hidden = stage(hidden)
        return self.pooling(hidden)


class RawRULPredictor(nn.Module):
    """Positive raw-cycle prediction with a fixed, non-dataset-dependent scale."""

    def __init__(self, embedding: int, hidden: int, dropout: float, scale: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * embedding, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(scale)

    def forward(self, invariant: Tensor, specific: Tensor) -> Tensor:
        hidden = F.gelu(self.fc1(torch.cat((invariant, specific), dim=-1)))
        score = self.fc2(self.dropout(hidden)).squeeze(-1)
        return self.scale * F.softplus(score)

    def initialize_bias(self, source_rul_cycles: float) -> None:
        """Start near a source-only RUL median without transforming labels."""
        scaled = max(float(source_rul_cycles) / self.scale, 1.0e-6)
        inverse_softplus = scaled + math.log(-math.expm1(-scaled))
        nn.init.constant_(self.fc2.bias, inverse_softplus)


class DomainClassifier(nn.Module):
    def __init__(self, embedding: int, hidden: int, domains: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding, hidden)
        self.fc2 = nn.Linear(hidden, domains)
        self.dropout = dropout

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.dropout(F.gelu(self.fc1(inputs)), p=self.dropout, training=self.training)
        return self.fc2(hidden)

    def detached_classifier_forward(self, inputs: Tensor) -> Tensor:
        hidden = F.linear(inputs, self.fc1.weight.detach(), self.fc1.bias.detach())
        hidden = F.dropout(F.gelu(hidden), p=self.dropout, training=self.training)
        return F.linear(hidden, self.fc2.weight.detach(), self.fc2.bias.detach())


@dataclass
class ModelOutput:
    prediction: Tensor
    invariant_embedding: Tensor
    specific_embedding: Tensor
    domain_logits: Tensor
    fuzzy_logits: Tensor


class HUSTDirectRULModel(nn.Module):
    def __init__(
        self,
        waveform_channels: int,
        scalar_features: int,
        history_length: int,
        source_domains: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        extractor_args = (
            waveform_channels,
            scalar_features,
            history_length,
            config,
        )
        self.domain_invariant = HierarchicalFeatureExtractor(*extractor_args)
        self.domain_specific = HierarchicalFeatureExtractor(*extractor_args)
        self.predictor = RawRULPredictor(
            config.embedding_dim,
            config.predictor_hidden,
            config.dropout,
            config.rul_output_scale_cycles,
        )
        self.domain_classifier = DomainClassifier(
            config.embedding_dim,
            config.domain_hidden,
            source_domains,
            config.dropout,
        )

    def forward(
        self, waveforms: Tensor, scalars: Tensor, grl_strength: float = 1.0
    ) -> ModelOutput:
        invariant = self.domain_invariant(waveforms, scalars)
        specific = self.domain_specific(waveforms, scalars)
        return ModelOutput(
            prediction=self.predictor(invariant, specific),
            invariant_embedding=invariant,
            specific_embedding=specific,
            domain_logits=self.domain_classifier(
                gradient_reverse(invariant, grl_strength)
            ),
            fuzzy_logits=self.domain_classifier.detached_classifier_forward(invariant),
        )

    def forward_meta(
        self,
        waveforms: Tensor,
        scalars: Tensor,
        fast_specific_parameters: Mapping[str, Tensor],
    ) -> Tensor:
        with torch.no_grad():
            invariant = self.domain_invariant(waveforms, scalars)
        specific = functional_call(
            self.domain_specific,
            fast_specific_parameters,
            (waveforms, scalars),
            strict=False,
        )
        return self.predictor(invariant, specific)

    def initial_specific_parameters(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(self.domain_specific.named_parameters())

    def meta_parameters(self) -> list[nn.Parameter]:
        return list(self.domain_specific.parameters()) + list(self.predictor.parameters())

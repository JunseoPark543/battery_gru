"""Paper-inspired dual-branch network for direct RUL regression.

The domain-specific branch is deliberately a separate module so BOIL can
adapt only that body in its inner loop while keeping the prediction head fixed.
"""

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


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, inputs: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * grad_output, None


def gradient_reverse(inputs: Tensor, strength: float) -> Tensor:
    return _GradientReversalFunction.apply(inputs, float(strength))


class MultiHeadSelfAttention(nn.Module):
    """Unfused attention whose operations support second-order gradients."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, length, 3, self.nhead, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.out(context)


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
    """Channel and temporal attention for tensors shaped [batch, cycle, channel]."""

    def __init__(self, channels: int, reduction: int, kernel_size: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, channels, bias=False),
        )
        self.temporal = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        average = inputs.mean(dim=1)
        maximum = inputs.amax(dim=1)
        channel_weight = torch.sigmoid(
            self.channel_mlp(average) + self.channel_mlp(maximum)
        ).unsqueeze(1)
        hidden = inputs * channel_weight
        temporal_input = torch.stack(
            (hidden.mean(dim=-1), hidden.amax(dim=-1)), dim=1
        )
        temporal_weight = torch.sigmoid(self.temporal(temporal_input)).transpose(1, 2)
        return hidden * temporal_weight


class AttentionStage(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.transformer_layers_per_stage)]
        )
        self.cbam = CBAM1d(
            config.d_model, config.cbam_reduction, config.cbam_kernel_size
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_norm(inputs + self.cbam(hidden))


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, d_model: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(8, d_model // 2)
        self.attention = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * d_model, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        weight = torch.softmax(self.attention(inputs), dim=1)
        mean = torch.sum(weight * inputs, dim=1)
        second_moment = torch.sum(weight * inputs.square(), dim=1)
        std = torch.sqrt(torch.clamp(second_moment - mean.square(), min=1.0e-6))
        return self.projection(torch.cat((mean, std), dim=-1))


class FeatureExtractor(nn.Module):
    def __init__(self, input_size: int, history_length: int, config: ModelConfig) -> None:
        super().__init__()
        self.input_size = input_size
        self.history_length = history_length
        self.input_projection = nn.Linear(input_size, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, history_length, config.d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.stages = nn.ModuleList(
            [AttentionStage(config) for _ in range(config.attention_stages)]
        )
        self.pooling = AttentiveStatisticsPooling(config.d_model, config.embedding_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("features must have shape [batch, cycle, feature]")
        if inputs.shape[1:] != (self.history_length, self.input_size):
            raise ValueError(
                f"expected [batch, {self.history_length}, {self.input_size}], "
                f"got {tuple(inputs.shape)}"
            )
        hidden = self.input_projection(inputs) + self.position
        for stage in self.stages:
            hidden = stage(hidden)
        return self.pooling(hidden)


class FusionPredictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, invariant: Tensor, specific: Tensor) -> Tensor:
        hidden = F.gelu(self.fc1(torch.cat((invariant, specific), dim=-1)))
        return self.fc2(self.dropout(hidden)).squeeze(-1)


class DomainClassifier(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, domains: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, domains)
        self.dropout_probability = dropout

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.gelu(self.fc1(inputs))
        hidden = F.dropout(hidden, p=self.dropout_probability, training=self.training)
        return self.fc2(hidden)

    def detached_classifier_forward(self, inputs: Tensor) -> Tensor:
        """Propagate into inputs but not classifier weights for the fuzzy loss."""
        hidden = F.linear(inputs, self.fc1.weight.detach(), self.fc1.bias.detach())
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout_probability, training=self.training)
        return F.linear(hidden, self.fc2.weight.detach(), self.fc2.bias.detach())


@dataclass
class ModelOutput:
    prediction: Tensor
    invariant_embedding: Tensor
    specific_embedding: Tensor
    domain_logits: Tensor
    fuzzy_logits: Tensor


class DirectRULBOILModel(nn.Module):
    """Dual representation model with an explicitly BOIL-adaptable body."""

    def __init__(
        self,
        input_size: int,
        history_length: int,
        num_source_domains: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.domain_invariant = FeatureExtractor(input_size, history_length, config)
        self.domain_specific = FeatureExtractor(input_size, history_length, config)
        self.predictor = FusionPredictor(
            config.embedding_dim, config.predictor_hidden, config.dropout
        )
        self.domain_classifier = DomainClassifier(
            config.embedding_dim,
            config.domain_hidden,
            num_source_domains,
            config.dropout,
        )

    def forward(self, inputs: Tensor, grl_strength: float = 1.0) -> ModelOutput:
        invariant = self.domain_invariant(inputs)
        specific = self.domain_specific(inputs)
        prediction = self.predictor(invariant, specific)
        domain_logits = self.domain_classifier(
            gradient_reverse(invariant, grl_strength)
        )
        fuzzy_logits = self.domain_classifier.detached_classifier_forward(invariant)
        return ModelOutput(
            prediction=prediction,
            invariant_embedding=invariant,
            specific_embedding=specific,
            domain_logits=domain_logits,
            fuzzy_logits=fuzzy_logits,
        )

    def forward_meta(
        self, inputs: Tensor, fast_specific_parameters: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """BOIL path: invariant branch is context; only the specific body is fast."""
        with torch.no_grad():
            invariant = self.domain_invariant(inputs)
        specific = functional_call(
            self.domain_specific, fast_specific_parameters, (inputs,), strict=False
        )
        return self.predictor(invariant, specific), specific

    def initial_specific_parameters(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(self.domain_specific.named_parameters())

    def specific_parameters(self) -> list[nn.Parameter]:
        return list(self.domain_specific.parameters())

    def predictor_parameters(self) -> list[nn.Parameter]:
        return list(self.predictor.parameters())

    def meta_parameters(self) -> list[nn.Parameter]:
        return self.specific_parameters() + self.predictor_parameters()

"""SOH-only and partial I-V conditioned Attentive Neural Processes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


MODEL_NAMES = ("soh_only_anp", "soh_only_anp_wide", "partial_iv_anp")


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
    if layers < 2:
        raise ValueError("MLPs require at least two linear layers")
    modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(layers - 2):
        modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class MaskedAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def self_attention(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(
            values, values, values, key_padding_mask=~mask, need_weights=False
        )
        output = self.norm(values + output)
        return output * mask.unsqueeze(-1)

    def cross_attention(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = self.attention(
            query,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        return self.norm(query + output)


class IVEncoder(nn.Module):
    """Small 1-D CNN with masked pooling and a learned beta=0 embedding."""

    def __init__(self, channels: list[int], embedding_dim: int):
        super().__init__()
        dimensions = [3, *channels]
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(dimensions[:-1], dimensions[1:]):
            blocks.extend(
                [
                    nn.Conv1d(input_channels, output_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                ]
            )
        self.convolution = nn.Sequential(*blocks)
        self.projection = _mlp(channels[-1], max(channels[-1], embedding_dim), embedding_dim, 2)
        self.null_embedding = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.null_embedding, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 3:
            raise ValueError("I-V input must have shape [batch,3,q_length]")
        mask = values[:, 2:3, :].clamp(0.0, 1.0)
        # Remove unobserved delta-V/current before convolution. Bias activations
        # outside the mask are removed again by mask-aware pooling.
        masked_input = torch.cat([values[:, :2, :] * mask, mask], dim=1)
        encoded = self.convolution(masked_input)
        denominator = mask.sum(dim=-1).clamp_min(1.0)
        pooled = (encoded * mask).sum(dim=-1) / denominator
        embedding = self.projection(pooled)
        present = mask.sum(dim=(-1, -2)) > 0
        null = self.null_embedding.unsqueeze(0).expand_as(embedding)
        return torch.where(present.unsqueeze(-1), embedding, null)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    hidden_dim: int
    latent_dim: int
    attention_heads: int
    conditional_iv: bool
    parameter_count: int
    parameter_match_target: int | None = None
    parameter_match_relative_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AttentiveNeuralProcess(nn.Module):
    """Latent ANP with optional fast I-V conditioning in every path."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        hidden_dim: int,
        conditional_iv: bool,
    ) -> None:
        super().__init__()
        if hidden_dim % config.attention_heads != 0:
            raise ValueError("ANP hidden dimension must be divisible by attention heads")
        self.hidden_dim = hidden_dim
        self.latent_dim = config.latent_dim
        self.conditional_iv = conditional_iv
        self.minimum_std = config.minimum_std
        self.context_encoder = _mlp(2, hidden_dim, hidden_dim, config.mlp_layers)
        self.context_attention = MaskedAttention(hidden_dim, config.attention_heads)
        self.target_query = _mlp(1, hidden_dim, hidden_dim, 2)
        self.cross_attention = MaskedAttention(hidden_dim, config.attention_heads)
        self.latent_encoder = _mlp(2, hidden_dim, hidden_dim, config.mlp_layers)
        self.latent_attention = MaskedAttention(hidden_dim, config.attention_heads)

        iv_dim = config.iv_embedding_dim if conditional_iv else 0
        self.iv_encoder = (
            IVEncoder(config.iv_channels, config.iv_embedding_dim)
            if conditional_iv
            else None
        )
        latent_input = hidden_dim + iv_dim
        self.latent_fusion = _mlp(latent_input, hidden_dim, hidden_dim, 2)
        self.latent_parameters = nn.Linear(hidden_dim, 2 * config.latent_dim)
        self.deterministic_fusion = (
            _mlp(hidden_dim + iv_dim, hidden_dim, hidden_dim, 2)
            if conditional_iv
            else nn.Identity()
        )
        decoder_input = 1 + hidden_dim + config.latent_dim + iv_dim
        self.decoder = _mlp(decoder_input, hidden_dim, 2, config.mlp_layers)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _latent_distribution(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        iv_embedding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.latent_encoder(torch.cat([x, y], dim=-1))
        encoded = self.latent_attention.self_attention(encoded, mask)
        aggregate = self._masked_mean(encoded, mask)
        if iv_embedding is not None:
            aggregate = torch.cat([aggregate, iv_embedding], dim=-1)
        fused = self.latent_fusion(aggregate)
        mean, raw_std = self.latent_parameters(fused).chunk(2, dim=-1)
        std = F.softplus(raw_std) + self.minimum_std
        return mean, std

    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        context_mask: torch.Tensor,
        target_x: torch.Tensor,
        *,
        target_y: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        iv_feature: torch.Tensor | None = None,
        sample_latent: bool = True,
    ) -> dict[str, torch.Tensor]:
        if context_x.shape != context_y.shape or context_x.ndim != 3:
            raise ValueError("context x/y must share shape [batch,points,1]")
        if context_mask.shape != context_x.shape[:2] or not context_mask.any(dim=1).all():
            raise ValueError("every ANP example needs at least one valid context point")
        if target_x.ndim != 3 or target_x.shape[-1] != 1:
            raise ValueError("target_x must have shape [batch,target_points,1]")
        if self.conditional_iv:
            if iv_feature is None:
                raise ValueError("partial_iv_anp requires iv_feature")
            assert self.iv_encoder is not None
            iv_embedding = self.iv_encoder(iv_feature)
        else:
            iv_embedding = None

        context_pairs = torch.cat([context_x, context_y], dim=-1)
        deterministic_context = self.context_encoder(context_pairs)
        deterministic_context = self.context_attention.self_attention(
            deterministic_context, context_mask
        )
        queries = self.target_query(target_x)
        deterministic = self.cross_attention.cross_attention(
            queries, deterministic_context, context_mask
        )
        if iv_embedding is not None:
            repeated_iv = iv_embedding[:, None, :].expand(-1, target_x.shape[1], -1)
            deterministic = self.deterministic_fusion(
                torch.cat([deterministic, repeated_iv], dim=-1)
            )

        prior_mean, prior_std = self._latent_distribution(
            context_x, context_y, context_mask, iv_embedding
        )
        if target_y is not None:
            if target_mask is None or target_y.shape != target_x.shape:
                raise ValueError("training posterior requires matching target_y and mask")
            posterior_x = torch.cat([context_x, target_x], dim=1)
            posterior_y = torch.cat([context_y, target_y], dim=1)
            posterior_mask = torch.cat([context_mask, target_mask], dim=1)
            posterior_mean, posterior_std = self._latent_distribution(
                posterior_x, posterior_y, posterior_mask, iv_embedding
            )
            latent_mean, latent_std = posterior_mean, posterior_std
        else:
            posterior_mean, posterior_std = prior_mean, prior_std
            latent_mean, latent_std = prior_mean, prior_std
        latent = (
            latent_mean + latent_std * torch.randn_like(latent_std)
            if sample_latent
            else latent_mean
        )
        repeated_latent = latent[:, None, :].expand(-1, target_x.shape[1], -1)
        decoder_parts = [target_x, deterministic, repeated_latent]
        if iv_embedding is not None:
            decoder_parts.append(iv_embedding[:, None, :].expand(-1, target_x.shape[1], -1))
        decoded = self.decoder(torch.cat(decoder_parts, dim=-1))
        prediction_mean, raw_prediction_std = decoded.chunk(2, dim=-1)
        prediction_std = F.softplus(raw_prediction_std) + self.minimum_std
        return {
            "mean": prediction_mean,
            "std": prediction_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _make(config: ModelConfig, hidden: int, conditional: bool) -> AttentiveNeuralProcess:
    return AttentiveNeuralProcess(config, hidden_dim=hidden, conditional_iv=conditional)


def build_model(
    model_name: str,
    config: ModelConfig,
    *,
    resolved_hidden_dim: int | None = None,
) -> tuple[AttentiveNeuralProcess, ModelSpec]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {model_name}")
    if model_name == "partial_iv_anp":
        model = _make(config, config.hidden_dim, True)
        spec = ModelSpec(
            model_name, config.hidden_dim, config.latent_dim,
            config.attention_heads, True, parameter_count(model),
        )
        return model, spec
    if model_name == "soh_only_anp":
        model = _make(config, config.hidden_dim, False)
        spec = ModelSpec(
            model_name, config.hidden_dim, config.latent_dim,
            config.attention_heads, False, parameter_count(model),
        )
        return model, spec

    target_model = _make(config, config.hidden_dim, True)
    target_count = parameter_count(target_model)
    if resolved_hidden_dim is None:
        candidates = range(config.wide_hidden_min, config.wide_hidden_max + 1)
        valid = [value for value in candidates if value % config.attention_heads == 0]
        if not valid:
            raise ValueError("wide hidden search contains no attention-compatible dimension")
        scored: list[tuple[int, int, AttentiveNeuralProcess]] = []
        for hidden in valid:
            candidate = _make(config, hidden, False)
            scored.append((abs(parameter_count(candidate) - target_count), hidden, candidate))
        _, resolved_hidden_dim, model = min(scored, key=lambda item: item[0])
    else:
        model = _make(config, resolved_hidden_dim, False)
    count = parameter_count(model)
    relative_error = abs(count - target_count) / target_count
    if relative_error > 0.05:
        raise ValueError(
            f"wide SOH-only parameter count {count} is not within 5% of "
            f"partial I-V count {target_count}; relative error={relative_error:.3%}"
        )
    spec = ModelSpec(
        model_name,
        resolved_hidden_dim,
        config.latent_dim,
        config.attention_heads,
        False,
        count,
        parameter_match_target=target_count,
        parameter_match_relative_error=relative_error,
    )
    return model, spec

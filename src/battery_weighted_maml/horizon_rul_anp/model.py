"""Prefix encoder and standard inter-cell ANP for direct RUL prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
    if layers < 2:
        raise ValueError("MLP requires at least two linear layers")
    modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(layers - 2):
        modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


def _sinusoidal(length: int, dimension: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / max(dimension, 1))
    )
    result = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * frequency)
    if dimension > 1:
        result[:, 1::2] = torch.cos(
            position * frequency[: result[:, 1::2].shape[1]]
        )
    return result


class PrefixEncoder(nn.Module):
    """Encode cycles 1:k of each battery into one vector.

    Inputs are ``prefix=[B,N,K,F]``, ``prefix_mask=[B,N,K]`` and
    ``point_mask=[B,N]``. Output is ``[B,N,D]``. Padding never contributes to
    self-attention or attention pooling.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.prefix_feature_dim, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.prefix_layers,
            enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(config.d_model, 1)
        self.output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

    def forward(
        self,
        prefix: torch.Tensor,
        prefix_mask: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prefix.ndim != 4:
            raise ValueError("prefix must have shape [batch,points,cycles,features]")
        if prefix_mask.shape != prefix.shape[:3]:
            raise ValueError("prefix_mask shape mismatch")
        if point_mask.shape != prefix.shape[:2]:
            raise ValueError("point_mask shape mismatch")
        batch, points, cycles, features = prefix.shape
        if features != self.input_projection.in_features:
            raise ValueError("prefix feature dimension mismatch")
        mask = prefix_mask & point_mask.unsqueeze(-1)
        flat_mask = mask.reshape(batch * points, cycles)
        flat_prefix = prefix.reshape(batch * points, cycles, features)
        present = flat_mask.any(dim=1)
        safe_mask = flat_mask.clone()
        safe_mask[~present, 0] = True
        values = flat_prefix * flat_mask.unsqueeze(-1).to(prefix.dtype)
        tokens = self.input_projection(values)
        positions = _sinusoidal(cycles, tokens.shape[-1], tokens.device)
        tokens = tokens + positions.to(dtype=tokens.dtype).unsqueeze(0)
        encoded = self.temporal_encoder(
            tokens,
            src_key_padding_mask=~safe_mask,
        )
        scores = self.pool_score(encoded).squeeze(-1)
        scores = scores.masked_fill(~safe_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * flat_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        pooled = torch.einsum("bk,bkd->bd", weights, encoded)
        pooled = self.output(pooled)
        pooled = pooled * present.unsqueeze(-1).to(pooled.dtype)
        return pooled.reshape(batch, points, -1)


@dataclass(frozen=True)
class ModelSpec:
    algorithm: str
    prefix_feature_dim: int
    d_model: int
    latent_dim: int
    parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HorizonRULANP(nn.Module):
    """ANP across batteries at a shared observation horizon tau_k.

    Context/query x values are learned prefix representations, not cycle
    coordinates. Query RUL is accepted only to construct q(z|C union Q) during
    training and is never required by inference.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.minimum_std = config.minimum_std
        dimension = config.d_model
        self.prefix_encoder = PrefixEncoder(config)
        self.context_pair_encoder = _mlp(
            dimension + 1, dimension, dimension, config.mlp_layers
        )
        self.context_self_attention = nn.MultiheadAttention(
            dimension,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(dimension)
        self.query_projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.context_key = nn.Linear(dimension, dimension)
        self.deterministic_cross_attention = nn.MultiheadAttention(
            dimension,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.deterministic_norm = nn.LayerNorm(dimension)

        self.latent_point_encoder = _mlp(
            dimension + 1, dimension, dimension, config.mlp_layers
        )
        self.latent_self_attention = nn.MultiheadAttention(
            dimension,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.latent_norm = nn.LayerNorm(dimension)
        self.latent_parameters = nn.Linear(dimension, 2 * config.latent_dim)
        self.decoder = _mlp(
            2 * dimension + config.latent_dim,
            dimension,
            2,
            config.mlp_layers,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _latent_distribution(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if representations.shape[:2] != labels.shape[:2] or labels.shape[-1] != 1:
            raise ValueError("latent representations/RUL labels are misaligned")
        encoded = self.latent_point_encoder(
            torch.cat([representations, labels], dim=-1)
        )
        attended, _ = self.latent_self_attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=~mask,
            need_weights=False,
        )
        encoded = self.latent_norm(encoded + attended)
        aggregate = self._masked_mean(encoded, mask)
        mean, raw_std = self.latent_parameters(aggregate).chunk(2, dim=-1)
        return mean, F.softplus(raw_std) + self.minimum_std

    def forward(
        self,
        context_prefix: torch.Tensor,
        context_prefix_mask: torch.Tensor,
        context_mask: torch.Tensor,
        context_y: torch.Tensor,
        query_prefix: torch.Tensor,
        query_prefix_mask: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        query_y: torch.Tensor | None = None,
        sample_latent: bool = True,
        return_representations: bool = False,
    ) -> dict[str, torch.Tensor]:
        if context_y.shape != (*context_mask.shape, 1):
            raise ValueError("context_y must have shape [batch,context,1]")
        if not context_mask.any(dim=1).all() or not query_mask.any(dim=1).all():
            raise ValueError("every task requires valid context and query cells")
        context_x = self.prefix_encoder(
            context_prefix, context_prefix_mask, context_mask
        )
        query_x = self.prefix_encoder(query_prefix, query_prefix_mask, query_mask)

        context_values = self.context_pair_encoder(
            torch.cat([context_x, context_y], dim=-1)
        )
        self_attended, _ = self.context_self_attention(
            context_values,
            context_values,
            context_values,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        context_values = self.context_norm(context_values + self_attended)
        query = self.query_projection(query_x)
        deterministic, attention = self.deterministic_cross_attention(
            query,
            self.context_key(context_x),
            context_values,
            key_padding_mask=~context_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        deterministic = self.deterministic_norm(query + deterministic)

        prior_mean, prior_std = self._latent_distribution(
            context_x, context_y, context_mask
        )
        if query_y is not None:
            if query_y.shape != (*query_mask.shape, 1):
                raise ValueError("query_y must have shape [batch,query,1]")
            posterior_mean, posterior_std = self._latent_distribution(
                torch.cat([context_x, query_x], dim=1),
                torch.cat([context_y, query_y], dim=1),
                torch.cat([context_mask, query_mask], dim=1),
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
        repeated_latent = latent[:, None, :].expand(-1, query_x.shape[1], -1)
        decoded = self.decoder(
            torch.cat([query_x, deterministic, repeated_latent], dim=-1)
        )
        prediction_mean, raw_prediction_std = decoded.chunk(2, dim=-1)
        output = {
            "mean": prediction_mean,
            "std": F.softplus(raw_prediction_std) + self.minimum_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "inter_cell_attention": attention,
        }
        if return_representations:
            output.update(
                {
                    "context_x": context_x,
                    "query_x": query_x,
                    "deterministic": deterministic,
                }
            )
        return output


def build_model(config: ModelConfig) -> tuple[HorizonRULANP, ModelSpec]:
    model = HorizonRULANP(config)
    spec = ModelSpec(
        algorithm="horizon_conditioned_rul_anp",
        prefix_feature_dim=config.prefix_feature_dim,
        d_model=config.d_model,
        latent_dim=config.latent_dim,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    return model, spec

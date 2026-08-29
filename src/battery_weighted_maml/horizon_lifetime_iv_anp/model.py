"""Hierarchical self-attention encoder and inter-cell lifetime ANP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
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
    output = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    output[:, 0::2] = torch.sin(position * frequency)
    if dimension > 1:
        output[:, 1::2] = torch.cos(position * frequency[: output[:, 1::2].shape[1]])
    return output


class CurveSelfAttentionEncoder(nn.Module):
    """Encode each 256-point [q,V,I] curve after non-learned masking."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_size = config.curve_patch_size
        self.gradient_checkpointing = config.gradient_checkpoint_curves
        self.patch_projection = nn.Conv1d(
            config.curve_input_dim,
            config.curve_d_model,
            kernel_size=config.curve_patch_size,
            stride=config.curve_patch_size,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.curve_d_model,
            nhead=config.curve_attention_heads,
            dim_feedforward=4 * config.curve_d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.attention = nn.TransformerEncoder(
            layer,
            num_layers=config.curve_layers,
            enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(config.curve_d_model, 1)
        self.output = nn.Sequential(
            nn.Linear(config.curve_d_model, config.curve_d_model),
            nn.GELU(),
            nn.LayerNorm(config.curve_d_model),
        )

    def forward(
        self,
        curves: torch.Tensor,
        q_mask: torch.Tensor,
        cycle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if curves.ndim != 5 or q_mask.shape != curves.shape[:-1]:
            raise ValueError("curves/q_mask shapes must be [B,N,K,Q,F]/[B,N,K,Q]")
        if cycle_mask.shape != curves.shape[:3]:
            raise ValueError("curve cycle mask shape mismatch")
        batch, points, cycles, q_points, features = curves.shape
        flat_count = batch * points * cycles
        flat_mask = (q_mask & cycle_mask.unsqueeze(-1)).reshape(flat_count, q_points)
        flat_curves = curves.reshape(flat_count, q_points, features)
        flat_curves = flat_curves * flat_mask.unsqueeze(-1).to(flat_curves.dtype)
        tokens = self.patch_projection(flat_curves.transpose(1, 2)).transpose(1, 2)
        patch_mask = F.max_pool1d(
            flat_mask.to(tokens.dtype).unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).squeeze(1).bool()
        present = patch_mask.any(dim=1)
        safe_mask = patch_mask.clone()
        safe_mask[~present, 0] = True
        tokens = tokens + _sinusoidal(
            tokens.shape[1], tokens.shape[2], tokens.device
        ).to(tokens.dtype).unsqueeze(0)

        def attend(values: torch.Tensor) -> torch.Tensor:
            return self.attention(values, src_key_padding_mask=~safe_mask)

        if self.gradient_checkpointing and self.training and tokens.requires_grad:
            encoded = checkpoint(attend, tokens, use_reentrant=False)
        else:
            encoded = attend(tokens)
        scores = self.pool_score(encoded).squeeze(-1).masked_fill(~safe_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1) * patch_mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        pooled = torch.einsum("bp,bpd->bd", weights, encoded)
        pooled = self.output(pooled) * present.unsqueeze(-1).to(pooled.dtype)
        return (
            pooled.reshape(batch, points, cycles, -1),
            present.reshape(batch, points, cycles),
        )


class BatteryPrefixEncoder(nn.Module):
    """Combine curve embeddings with [cycle,SOH] and self-attend over 1:k."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.curve_encoder = CurveSelfAttentionEncoder(config)
        self.input_projection = nn.Linear(
            config.curve_d_model + 3,  # curve h, normalized cycle/SOH, curve available
            config.temporal_d_model,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.temporal_d_model,
            nhead=config.temporal_attention_heads,
            dim_feedforward=4 * config.temporal_d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_attention = nn.TransformerEncoder(
            layer,
            num_layers=config.temporal_layers,
            enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(config.temporal_d_model, 1)
        self.output = nn.Sequential(
            nn.Linear(config.temporal_d_model, config.temporal_d_model),
            nn.GELU(),
            nn.LayerNorm(config.temporal_d_model),
        )

    def forward(
        self,
        cycle_features: torch.Tensor,
        cycle_mask: torch.Tensor,
        curves: torch.Tensor,
        curve_mask: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        if cycle_features.shape[:-1] != cycle_mask.shape or cycle_features.shape[-1] != 2:
            raise ValueError("cycle features must be [B,N,K,2]")
        if point_mask.shape != cycle_mask.shape[:2]:
            raise ValueError("point mask shape mismatch")
        valid_cycles = cycle_mask & point_mask.unsqueeze(-1)
        curve_h, curve_available = self.curve_encoder(
            curves, curve_mask, valid_cycles
        )
        combined = torch.cat(
            [curve_h, cycle_features, curve_available.unsqueeze(-1).to(curve_h.dtype)],
            dim=-1,
        )
        batch, points, cycles, _ = combined.shape
        tokens = self.input_projection(combined).reshape(
            batch * points, cycles, -1
        )
        flat_mask = valid_cycles.reshape(batch * points, cycles)
        present = flat_mask.any(dim=1)
        safe_mask = flat_mask.clone()
        safe_mask[~present, 0] = True
        tokens = tokens + _sinusoidal(
            cycles, tokens.shape[-1], tokens.device
        ).to(tokens.dtype).unsqueeze(0)
        encoded = self.temporal_attention(tokens, src_key_padding_mask=~safe_mask)
        scores = self.pool_score(encoded).squeeze(-1).masked_fill(~safe_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1) * flat_mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        pooled = torch.einsum("bk,bkd->bd", weights, encoded)
        pooled = self.output(pooled) * present.unsqueeze(-1).to(pooled.dtype)
        return pooled.reshape(batch, points, -1)


@dataclass(frozen=True)
class ModelSpec:
    algorithm: str
    q_points: int
    curve_d_model: int
    temporal_d_model: int
    latent_dim: int
    parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LifetimeIVANP(nn.Module):
    """ANP whose point x is a learned battery prefix h and y is lifetime."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.minimum_std = config.minimum_std
        dimension = config.temporal_d_model
        self.prefix_encoder = BatteryPrefixEncoder(config)
        self.context_pair_encoder = _mlp(
            dimension + 1, dimension, dimension, config.anp_mlp_layers
        )
        self.context_self_attention = nn.MultiheadAttention(
            dimension, config.temporal_attention_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.context_norm = nn.LayerNorm(dimension)
        self.query_projection = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.LayerNorm(dimension)
        )
        self.context_key = nn.Linear(dimension, dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension, config.temporal_attention_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.deterministic_norm = nn.LayerNorm(dimension)
        self.latent_point_encoder = _mlp(
            dimension + 1, dimension, dimension, config.anp_mlp_layers
        )
        self.latent_attention = nn.MultiheadAttention(
            dimension, config.temporal_attention_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.latent_norm = nn.LayerNorm(dimension)
        self.latent_parameters = nn.Linear(dimension, 2 * config.latent_dim)
        self.decoder = _mlp(
            2 * dimension + config.latent_dim,
            dimension,
            2,
            config.anp_mlp_layers,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _latent(
        self, h: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.latent_point_encoder(torch.cat([h, y], dim=-1))
        attended, _ = self.latent_attention(
            encoded, encoded, encoded, key_padding_mask=~mask, need_weights=False
        )
        encoded = self.latent_norm(encoded + attended)
        mean, raw_std = self.latent_parameters(
            self._masked_mean(encoded, mask)
        ).chunk(2, dim=-1)
        return mean, F.softplus(raw_std) + self.minimum_std

    def _encode(
        self,
        cycles: torch.Tensor,
        cycle_mask: torch.Tensor,
        curves: torch.Tensor,
        curve_mask: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.prefix_encoder(cycles, cycle_mask, curves, curve_mask, point_mask)

    def forward_from_embeddings(
        self,
        context_h: torch.Tensor,
        context_point_mask: torch.Tensor,
        context_y: torch.Tensor,
        query_h: torch.Tensor,
        query_point_mask: torch.Tensor,
        *,
        query_y: torch.Tensor | None = None,
        sample_latent: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run the inter-cell ANP after the expensive prefix encoding step."""
        if context_h.shape[:2] != context_point_mask.shape:
            raise ValueError("context embedding/mask shape mismatch")
        if query_h.shape[:2] != query_point_mask.shape:
            raise ValueError("query embedding/mask shape mismatch")
        if context_y.shape != (*context_point_mask.shape, 1):
            raise ValueError("context lifetime shape mismatch")
        if query_y is not None and query_y.shape != (*query_point_mask.shape, 1):
            raise ValueError("query lifetime shape mismatch")
        context_values = self.context_pair_encoder(torch.cat([context_h, context_y], dim=-1))
        self_attended, _ = self.context_self_attention(
            context_values, context_values, context_values,
            key_padding_mask=~context_point_mask, need_weights=False,
        )
        context_values = self.context_norm(context_values + self_attended)
        query = self.query_projection(query_h)
        deterministic, attention = self.cross_attention(
            query,
            self.context_key(context_h),
            context_values,
            key_padding_mask=~context_point_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        deterministic = self.deterministic_norm(query + deterministic)
        prior_mean, prior_std = self._latent(context_h, context_y, context_point_mask)
        if query_y is None:
            posterior_mean, posterior_std = prior_mean, prior_std
            latent_mean, latent_std = prior_mean, prior_std
        else:
            posterior_mean, posterior_std = self._latent(
                torch.cat([context_h, query_h], dim=1),
                torch.cat([context_y, query_y], dim=1),
                torch.cat([context_point_mask, query_point_mask], dim=1),
            )
            latent_mean, latent_std = posterior_mean, posterior_std
        latent = (
            latent_mean + latent_std * torch.randn_like(latent_std)
            if sample_latent else latent_mean
        )
        latent = latent[:, None, :].expand(-1, query_h.shape[1], -1)
        mean, raw_std = self.decoder(
            torch.cat([query_h, deterministic, latent], dim=-1)
        ).chunk(2, dim=-1)
        return {
            "mean": mean,
            "std": F.softplus(raw_std) + self.minimum_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "inter_cell_attention": attention,
        }

    def forward(
        self,
        context_cycles: torch.Tensor,
        context_cycle_mask: torch.Tensor,
        context_curves: torch.Tensor,
        context_curve_mask: torch.Tensor,
        context_point_mask: torch.Tensor,
        context_y: torch.Tensor,
        query_cycles: torch.Tensor,
        query_cycle_mask: torch.Tensor,
        query_curves: torch.Tensor,
        query_curve_mask: torch.Tensor,
        query_point_mask: torch.Tensor,
        *,
        query_y: torch.Tensor | None = None,
        sample_latent: bool = True,
        return_representations: bool = False,
    ) -> dict[str, torch.Tensor]:
        context_h = self._encode(
            context_cycles, context_cycle_mask, context_curves,
            context_curve_mask, context_point_mask,
        )
        query_h = self._encode(
            query_cycles, query_cycle_mask, query_curves,
            query_curve_mask, query_point_mask,
        )
        output = self.forward_from_embeddings(
            context_h, context_point_mask, context_y,
            query_h, query_point_mask,
            query_y=query_y, sample_latent=sample_latent,
        )
        if return_representations:
            output.update({"context_h": context_h, "query_h": query_h})
        return output


def build_model(config: ModelConfig, q_points: int = 256) -> tuple[LifetimeIVANP, ModelSpec]:
    model = LifetimeIVANP(config)
    return model, ModelSpec(
        algorithm="horizon_conditioned_lifetime_iv_anp",
        q_points=int(q_points),
        curve_d_model=config.curve_d_model,
        temporal_d_model=config.temporal_d_model,
        latent_dim=config.latent_dim,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )

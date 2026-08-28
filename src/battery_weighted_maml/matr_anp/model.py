"""SOH-only and partial I-V conditioned Attentive Neural Processes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


MODEL_NAMES = (
    "soh_only_anp",
    "soh_only_anp_wide",
    "partial_iv_anp",
    "hs_anp_pooled",
    "hs_anp_add",
    "hs_anp",
)
HS_MODEL_NAMES = ("hs_anp_pooled", "hs_anp_add", "hs_anp")


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


def _sinusoidal_positions(
    length: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return deterministic intra-cycle positional features ``[L,D]``."""
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / max(dimension, 1))
    )
    output = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    output[:, 0::2] = torch.sin(positions * frequencies)
    if dimension > 1:
        output[:, 1::2] = torch.cos(
            positions * frequencies[: output[:, 1::2].shape[1]]
        )
    return output.to(dtype=dtype)


class IntraCycleSignalEncoder(nn.Module):
    """Encode historical V/I without collapsing its q-position sequence.

    Input shapes are ``signal=[B,C,L,2]`` and ``mask=[B,C,L]``. The returned
    full representation ``H`` is ``[B,C,L,D]`` and the masked mean ``hbar`` is
    ``[B,C,D]``. All-missing historical curves remain exactly zero.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        dimension = config.hs_d_model
        self.cycle_chunk_size = config.hs_intra_cycle_chunk_size
        self.gradient_checkpointing = config.hs_gradient_checkpointing
        self.input_projection = nn.Linear(2, dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=config.hs_attention_heads,
            dim_feedforward=4 * dimension,
            dropout=config.hs_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.hs_intra_layers)
        self.output_norm = nn.LayerNorm(dimension)

    def forward(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if signal.ndim != 4 or signal.shape[-1] != 2:
            raise ValueError("context_signal must have shape [batch,context,q,2]")
        if mask.shape != signal.shape[:3]:
            raise ValueError("context_signal_mask must match [batch,context,q]")
        batch, context, length, _ = signal.shape
        flat_signal = signal.reshape(batch * context, length, 2)
        flat_mask = mask.reshape(batch * context, length)
        positions = _sinusoidal_positions(
            length,
            self.input_projection.out_features,
            device=signal.device,
            dtype=signal.dtype,
        ).unsqueeze(0)
        encoded_chunks: list[torch.Tensor] = []
        pooled_chunks: list[torch.Tensor] = []
        # Self-attention is independent between cycles. Chunking this flattened
        # axis gives the same model operation while avoiding an attention
        # workspace proportional to B*C*heads*L^2 (about 1 GiB for the default
        # B=16, C=128, heads=4, L=256 even in fp16).
        for start in range(0, batch * context, self.cycle_chunk_size):
            stop = min(start + self.cycle_chunk_size, batch * context)
            chunk_signal = flat_signal[start:stop]
            chunk_mask = flat_mask[start:stop]
            present = chunk_mask.any(dim=1)
            safe_mask = chunk_mask.clone()
            safe_mask[~present, 0] = True
            masked_signal = chunk_signal * chunk_mask.unsqueeze(-1).to(signal.dtype)
            tokens = self.input_projection(masked_signal)
            tokens = tokens + positions.to(dtype=tokens.dtype)
            padding_mask = ~safe_mask
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                encoded = checkpoint(
                    lambda inputs, key_padding_mask: self.encoder(
                        inputs,
                        src_key_padding_mask=key_padding_mask,
                    ),
                    tokens,
                    padding_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
            encoded = self.output_norm(encoded)
            encoded = encoded * chunk_mask.unsqueeze(-1).to(encoded.dtype)
            denominator = (
                chunk_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
            )
            pooled = encoded.sum(dim=1) / denominator
            pooled = pooled * present.unsqueeze(-1).to(pooled.dtype)
            encoded_chunks.append(encoded)
            pooled_chunks.append(pooled)
        encoded = torch.cat(encoded_chunks, dim=0)
        pooled = torch.cat(pooled_chunks, dim=0)
        return (
            encoded.reshape(batch, context, length, -1),
            pooled.reshape(batch, context, -1),
        )


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
    context_signal: bool = False
    signal_cross_attention: bool = False
    gated_fusion: bool = False

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
        context_signal: torch.Tensor | None = None,
        context_signal_mask: torch.Tensor | None = None,
        sample_latent: bool = True,
        return_attention: bool = False,
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


class HierarchicalSignalANP(AttentiveNeuralProcess):
    """Signal-aware deterministic path with the baseline ANP latent path.

    Only historical ``context_signal`` is accepted. There is deliberately no
    target-signal argument: future target Voltage/Current cannot enter the
    deterministic encoder, latent prior, posterior, decoder, or inference.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        use_signal_cross_attention: bool,
        use_gated_fusion: bool,
    ) -> None:
        # This preserves the SOH-only latent encoder and Gaussian decoder sizes.
        super().__init__(config, hidden_dim=config.hidden_dim, conditional_iv=False)
        self.use_signal_cross_attention = bool(use_signal_cross_attention)
        self.use_gated_fusion = bool(use_gated_fusion)
        dimension = config.hs_d_model
        self.signal_target_chunk_size = config.hs_signal_target_chunk_size
        self.signal_encoder = IntraCycleSignalEncoder(config)
        self.xy_projection = nn.Linear(config.hidden_dim, dimension)
        self.context_cycle_key = _mlp(1, dimension, dimension, 2)
        self.target_cycle_query = nn.Sequential(
            nn.Linear(config.hidden_dim, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.cycle_token_fusion = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.GELU(),
            nn.Dropout(config.hs_dropout),
            nn.Linear(dimension, dimension),
            nn.LayerNorm(dimension),
        )
        self.cycle_cross_attention = nn.MultiheadAttention(
            dimension,
            config.hs_attention_heads,
            dropout=config.hs_dropout,
            batch_first=True,
        )
        self.cycle_attention_norm = nn.LayerNorm(dimension)
        self.signal_query = _mlp(2 * dimension, dimension, dimension, 2)
        self.signal_key = nn.Linear(dimension, dimension, bias=False)
        self.signal_value = nn.Linear(dimension, dimension, bias=False)
        self.signal_scale = dimension**-0.5
        self.signal_add_norm = nn.LayerNorm(dimension)
        self.gate = _mlp(3 * dimension, dimension, dimension, 2)
        self.deterministic_projection = nn.Sequential(
            nn.Linear(dimension, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )

    def _signal_cross_path(
        self,
        query: torch.Tensor,
        cycle_representation: torch.Tensor,
        cycle_attention: torch.Tensor,
        full_signal: torch.Tensor,
        signal_mask: torch.Tensor,
        *,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, targets, dimension = query.shape
        context, length = full_signal.shape[1:3]
        keys = self.signal_key(full_signal)
        values = self.signal_value(full_signal)
        valid = signal_mask[:, None, :, :]
        representations: list[torch.Tensor] = []
        attention_chunks: list[torch.Tensor] = []
        for start in range(0, targets, self.signal_target_chunk_size):
            stop = min(start + self.signal_target_chunk_size, targets)
            signal_query = self.signal_query(
                torch.cat(
                    [query[:, start:stop], cycle_representation[:, start:stop]],
                    dim=-1,
                )
            )
            scores = torch.einsum("btd,bcld->btcl", signal_query, keys)
            scores = scores * self.signal_scale
            scores = scores.masked_fill(~valid, -1.0e4)
            beta = torch.softmax(scores, dim=-1)
            beta = beta * valid.to(beta.dtype)
            beta = beta / beta.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            per_cycle = torch.einsum("btcl,bcld->btcd", beta, values)
            representations.append(
                torch.einsum(
                    "btc,btcd->btd",
                    cycle_attention[:, start:stop],
                    per_cycle,
                )
            )
            if return_attention:
                attention_chunks.append(beta)
        representation = torch.cat(representations, dim=1)
        beta_output = torch.cat(attention_chunks, dim=1) if return_attention else None
        if representation.shape != (batch, targets, dimension):
            raise RuntimeError("internal HS-ANP signal attention shape failure")
        if beta_output is not None and beta_output.shape != (
            batch,
            targets,
            context,
            length,
        ):
            raise RuntimeError("internal HS-ANP beta shape failure")
        return representation, beta_output

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
        context_signal: torch.Tensor | None = None,
        context_signal_mask: torch.Tensor | None = None,
        sample_latent: bool = True,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        if context_x.shape != context_y.shape or context_x.ndim != 3:
            raise ValueError("context x/y must share shape [batch,points,1]")
        if context_mask.shape != context_x.shape[:2] or not context_mask.any(dim=1).all():
            raise ValueError("every HS-ANP example needs a valid context cycle")
        if target_x.ndim != 3 or target_x.shape[-1] != 1:
            raise ValueError("target_x must have shape [batch,target_points,1]")
        if context_signal is None or context_signal_mask is None:
            raise ValueError("HS-ANP requires historical context_signal and its mask")
        if context_signal.shape[:2] != context_x.shape[:2]:
            raise ValueError("context_signal context dimension must match context_x")
        combined_signal_mask = context_signal_mask & context_mask.unsqueeze(-1)

        full_signal, pooled_signal = self.signal_encoder(
            context_signal,
            combined_signal_mask,
        )
        xy = self.context_encoder(torch.cat([context_x, context_y], dim=-1))
        xy = self.context_attention.self_attention(xy, context_mask)
        cycle_tokens = self.cycle_token_fusion(
            torch.cat([self.xy_projection(xy), pooled_signal], dim=-1)
        )
        cycle_tokens = cycle_tokens * context_mask.unsqueeze(-1).to(cycle_tokens.dtype)
        target_query = self.target_cycle_query(self.target_query(target_x))
        # Keys and values both originate from the fused cycle token; the
        # normalized cycle coordinate is added as a positional feature.
        cycle_keys = cycle_tokens + self.context_cycle_key(context_x)
        attended, alpha = self.cycle_cross_attention(
            target_query,
            cycle_keys,
            cycle_tokens,
            key_padding_mask=~context_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        r_cyc = self.cycle_attention_norm(target_query + attended)

        beta: torch.Tensor | None = None
        r_sig = torch.zeros_like(r_cyc)
        gate = torch.zeros_like(r_cyc)
        if self.use_signal_cross_attention:
            r_sig, beta = self._signal_cross_path(
                target_query,
                r_cyc,
                alpha,
                full_signal,
                combined_signal_mask,
                return_attention=return_attention,
            )
            if self.use_gated_fusion:
                gate = torch.sigmoid(
                    self.gate(torch.cat([target_query, r_cyc, r_sig], dim=-1))
                )
                fused = self.signal_add_norm(r_cyc + gate * r_sig)
            else:
                fused = self.signal_add_norm(r_cyc + r_sig)
        else:
            fused = r_cyc
        deterministic = self.deterministic_projection(fused)

        # The latent path is intentionally identical to SOH-only ANP and never
        # receives context_signal, iv_feature, or any target Voltage/Current.
        prior_mean, prior_std = self._latent_distribution(
            context_x,
            context_y,
            context_mask,
            None,
        )
        if target_y is not None:
            if target_mask is None or target_y.shape != target_x.shape:
                raise ValueError("training posterior requires matching target_y and mask")
            posterior_mean, posterior_std = self._latent_distribution(
                torch.cat([context_x, target_x], dim=1),
                torch.cat([context_y, target_y], dim=1),
                torch.cat([context_mask, target_mask], dim=1),
                None,
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
        decoded = self.decoder(
            torch.cat([target_x, deterministic, repeated_latent], dim=-1)
        )
        prediction_mean, raw_prediction_std = decoded.chunk(2, dim=-1)
        output = {
            "mean": prediction_mean,
            "std": F.softplus(raw_prediction_std) + self.minimum_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
        }
        if return_attention:
            output.update(
                {
                    "H": full_signal,
                    "hbar": pooled_signal,
                    "cycle_tokens": cycle_tokens,
                    "cycle_attention": alpha,
                    "signal_attention": (
                        beta
                        if beta is not None
                        else full_signal.new_zeros(
                            target_x.shape[0],
                            target_x.shape[1],
                            context_x.shape[1],
                            context_signal.shape[2],
                        )
                    ),
                    "r_cyc": r_cyc,
                    "r_sig": r_sig,
                    "r_det": fused,
                    "fusion_gate": gate,
                }
            )
        return output


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _make(config: ModelConfig, hidden: int, conditional: bool) -> AttentiveNeuralProcess:
    return AttentiveNeuralProcess(config, hidden_dim=hidden, conditional_iv=conditional)


def build_model(
    model_name: str,
    config: ModelConfig,
    *,
    resolved_hidden_dim: int | None = None,
) -> tuple[nn.Module, ModelSpec]:
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
    if model_name in HS_MODEL_NAMES:
        use_signal_cross_attention = model_name != "hs_anp_pooled"
        use_gated_fusion = model_name == "hs_anp"
        model = HierarchicalSignalANP(
            config,
            use_signal_cross_attention=use_signal_cross_attention,
            use_gated_fusion=use_gated_fusion,
        )
        spec = ModelSpec(
            model_name,
            config.hidden_dim,
            config.latent_dim,
            config.hs_attention_heads,
            False,
            parameter_count(model),
            context_signal=True,
            signal_cross_attention=use_signal_cross_attention,
            gated_fusion=use_gated_fusion,
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

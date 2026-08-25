"""Hierarchical curve encoder, GRU, and latent ANP V(cycle,Q) decoder."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class CurveEncoder(nn.Module):
    """Encode one masked complete V-Q curve without using its future labels."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 3  # normalized voltage, valid mask, fixed q coordinate
        for output_channels in config.convolution_channels:
            groups = min(8, output_channels)
            while output_channels % groups:
                groups -= 1
            layers.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        config.kernel_size,
                        padding=config.kernel_size // 2,
                    ),
                    nn.GroupNorm(groups, output_channels),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_channels = output_channels
        self.convolution = nn.Sequential(*layers)
        self.point_projection = nn.Linear(input_channels, config.curve_embedding_dim)
        self.score = nn.Linear(config.curve_embedding_dim, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(config.curve_embedding_dim),
            nn.Linear(config.curve_embedding_dim, config.curve_embedding_dim),
            nn.GELU(),
        )

    def forward(self, curve: torch.Tensor, q_coordinate: torch.Tensor) -> torch.Tensor:
        if curve.ndim != 3 or curve.shape[-1] != 2:
            raise ValueError("curve must have shape [batch,q,2]")
        if q_coordinate.shape != curve.shape[:2]:
            raise ValueError("q_coordinate must match curve batch/q dimensions")
        valid = curve[..., 1] > 0.5
        if not valid.any(dim=1).all():
            raise ValueError("every curve must contain at least one valid q point")
        inputs = torch.cat([curve, q_coordinate.unsqueeze(-1)], dim=-1).transpose(1, 2)
        tokens = self.point_projection(self.convolution(inputs).transpose(1, 2))
        scores = self.score(torch.tanh(tokens)).squeeze(-1).masked_fill(~valid, -1.0e4)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return self.output(pooled)


class FutureVQLatentANP(nn.Module):
    """Directly generate a coherent distribution over all future V-Q curves."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.curve_encoder = CurveEncoder(config)
        self.history_projection = nn.Sequential(
            nn.Linear(config.curve_embedding_dim + 4, config.cycle_feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.history_gru = nn.GRU(
            config.cycle_feature_dim,
            config.gru_hidden_dim,
            num_layers=config.gru_layers,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.query_embedding = nn.Sequential(
            nn.Linear(2, config.gru_hidden_dim),
            nn.GELU(),
            nn.Linear(config.gru_hidden_dim, config.gru_hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            config.gru_hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(config.gru_hidden_dim)
        self.prior_network = nn.Sequential(
            nn.Linear(2 * config.gru_hidden_dim, config.latent_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_hidden_dim, 2 * config.latent_dim),
        )
        self.target_projection = nn.Sequential(
            nn.Linear(config.curve_embedding_dim + 3, config.gru_hidden_dim),
            nn.GELU(),
            nn.Linear(config.gru_hidden_dim, config.gru_hidden_dim),
        )
        self.posterior_network = nn.Sequential(
            nn.Linear(3 * config.gru_hidden_dim, config.latent_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_hidden_dim, 2 * config.latent_dim),
        )
        self.q_embedding = nn.Sequential(
            nn.Linear(1, config.q_embedding_dim),
            nn.GELU(),
            nn.Linear(config.q_embedding_dim, config.q_embedding_dim),
        )
        coordinate_input = (
            2 * config.gru_hidden_dim
            + config.latent_dim
            + config.q_embedding_dim
            + 4
        )
        self.coordinate_decoder = nn.Sequential(
            nn.Linear(coordinate_input, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
        )
        self.voltage_mean_head = nn.Linear(config.decoder_hidden_dim, 1)
        self.voltage_std_head = nn.Linear(config.decoder_hidden_dim, 1)
        endpoint_input = 2 * config.gru_hidden_dim + config.latent_dim + 3
        self.endpoint_decoder = nn.Sequential(
            nn.Linear(endpoint_input, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
        )
        self.endpoint_mean_head = nn.Linear(config.decoder_hidden_dim, 1)
        self.endpoint_std_head = nn.Linear(config.decoder_hidden_dim, 1)
        self._initialize_distribution_heads()

    def _initialize_distribution_heads(self) -> None:
        latent_raw = math.log(
            math.expm1(max(0.1 - self.config.minimum_latent_std, 1.0e-4))
        )
        for network in (self.prior_network, self.posterior_network):
            head = network[-1]
            assert isinstance(head, nn.Linear)
            nn.init.normal_(head.weight, mean=0.0, std=1.0e-2)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[self.config.latent_dim :].fill_(latent_raw)
        nn.init.normal_(self.voltage_mean_head.weight, mean=0.0, std=1.0e-2)
        nn.init.zeros_(self.voltage_mean_head.bias)
        voltage_raw = math.log(
            math.expm1(max(0.02 - self.config.minimum_voltage_std, 1.0e-4))
        )
        nn.init.normal_(self.voltage_std_head.weight, mean=0.0, std=1.0e-2)
        nn.init.constant_(self.voltage_std_head.bias, voltage_raw)
        nn.init.normal_(self.endpoint_mean_head.weight, mean=0.0, std=1.0e-2)
        nn.init.zeros_(self.endpoint_mean_head.bias)
        endpoint_raw = math.log(
            math.expm1(max(0.02 - self.config.minimum_endpoint_std, 1.0e-4))
        )
        nn.init.normal_(self.endpoint_std_head.weight, mean=0.0, std=1.0e-2)
        nn.init.constant_(self.endpoint_std_head.bias, endpoint_raw)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return torch.sum(values * weights, dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _distribution(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, unconstrained = raw.chunk(2, dim=-1)
        return mean, self.config.minimum_latent_std + F.softplus(unconstrained)

    def encode_history(
        self,
        history_curve: torch.Tensor,
        history_endpoint_fraction: torch.Tensor,
        history_cycle_scaled: torch.Tensor,
        history_gap_scaled: torch.Tensor,
        history_mask: torch.Tensor,
        q_coordinate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_curve.ndim != 4 or history_curve.shape[-1] != 2:
            raise ValueError("history_curve must have shape [batch,history,q,2]")
        batch, history, q_points, _ = history_curve.shape
        if q_coordinate.shape != (batch, q_points):
            raise ValueError("q_coordinate shape is incompatible with history_curve")
        if not history_mask.any(dim=1).all():
            raise ValueError("every item requires at least one history cycle")
        expanded_q = q_coordinate[:, None].expand(-1, history, -1).reshape(
            batch * history, q_points
        )
        embeddings = self.curve_encoder(
            history_curve.reshape(batch * history, q_points, 2), expanded_q
        ).reshape(batch, history, -1)
        available = history_curve[..., 1].any(dim=-1).to(history_curve.dtype)
        features = self.history_projection(
            torch.cat(
                [
                    embeddings,
                    history_endpoint_fraction.unsqueeze(-1),
                    history_cycle_scaled.unsqueeze(-1),
                    history_gap_scaled.unsqueeze(-1),
                    available.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        tokens, _ = self.history_gru(features)
        positions = history - 1 - torch.flip(history_mask, dims=[1]).long().argmax(dim=1)
        rows = torch.arange(batch, device=history_curve.device)
        last_state = tokens[rows, positions]
        last_cycle = history_cycle_scaled[rows, positions]
        last_endpoint = history_endpoint_fraction[rows, positions]
        mean_state = self._masked_mean(tokens, history_mask)
        context_summary = torch.cat([last_state, mean_state], dim=-1)
        prior_mean, prior_std = self._distribution(self.prior_network(context_summary))
        return {
            "tokens": tokens,
            "history_mask": history_mask,
            "last_state": last_state,
            "last_cycle_scaled": last_cycle,
            "last_endpoint_fraction": last_endpoint,
            "context_summary": context_summary,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
        }

    def posterior_parameters(
        self,
        encoded: dict[str, torch.Tensor],
        target_voltage: torch.Tensor,
        target_q_mask: torch.Tensor,
        target_endpoint_fraction: torch.Tensor,
        query_cycle_scaled: torch.Tensor,
        query_mask: torch.Tensor,
        q_coordinate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, future, q_points = target_voltage.shape
        safe_mask = target_q_mask.clone()
        inactive_rows = ~query_mask
        if inactive_rows.any():
            safe_mask[..., 0] = safe_mask[..., 0] | inactive_rows
        target_feature = torch.stack(
            [target_voltage, safe_mask.to(target_voltage.dtype)], dim=-1
        )
        expanded_q = q_coordinate[:, None].expand(-1, future, -1).reshape(
            batch * future, q_points
        )
        curve_embedding = self.curve_encoder(
            target_feature.reshape(batch * future, q_points, 2), expanded_q
        ).reshape(batch, future, -1)
        relative = query_cycle_scaled - encoded["last_cycle_scaled"].unsqueeze(1)
        target_tokens = self.target_projection(
            torch.cat(
                [
                    curve_embedding,
                    target_endpoint_fraction.unsqueeze(-1),
                    query_cycle_scaled.unsqueeze(-1),
                    relative.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        target_summary = self._masked_mean(target_tokens, query_mask)
        raw = self.posterior_network(
            torch.cat([encoded["context_summary"], target_summary], dim=-1)
        )
        return self._distribution(raw)

    @staticmethod
    def sample_latent(
        mean: torch.Tensor, std: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        noise = torch.randn(
            (num_samples,) + tuple(mean.shape), device=mean.device, dtype=mean.dtype
        )
        return mean.unsqueeze(0) + std.unsqueeze(0) * noise

    def deterministic_path(
        self,
        encoded: dict[str, torch.Tensor],
        query_cycle_scaled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative = query_cycle_scaled - encoded["last_cycle_scaled"].unsqueeze(1)
        query = self.query_embedding(
            torch.stack([query_cycle_scaled, relative], dim=-1)
        )
        attended, _ = self.cross_attention(
            query,
            encoded["tokens"],
            encoded["tokens"],
            key_padding_mask=~encoded["history_mask"],
            need_weights=False,
        )
        attended = self.attention_norm(attended + query)
        last = encoded["last_state"].unsqueeze(1).expand(-1, query.shape[1], -1)
        return torch.cat([attended, last], dim=-1), relative

    def decode_queries(
        self,
        encoded: dict[str, torch.Tensor],
        query_cycle_scaled: torch.Tensor,
        q_coordinate: torch.Tensor,
        latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode query chunks; the same latent may be reused across all chunks."""
        deterministic, relative = self.deterministic_path(encoded, query_cycle_scaled)
        samples, batch, latent_dim = latent.shape
        future = query_cycle_scaled.shape[1]
        q_points = q_coordinate.shape[1]
        det = deterministic.unsqueeze(0).expand(samples, -1, -1, -1)
        z = latent[:, :, None, :].expand(-1, -1, future, -1)
        absolute = query_cycle_scaled[None, :, :, None].expand(samples, -1, -1, -1)
        relative_feature = relative[None, :, :, None].expand(samples, -1, -1, -1)
        last_endpoint = encoded["last_endpoint_fraction"][None, :, None, None].expand(
            samples, -1, future, -1
        )
        endpoint_input = torch.cat(
            [det, z, absolute, relative_feature, last_endpoint], dim=-1
        )
        endpoint_hidden = self.endpoint_decoder(endpoint_input)
        endpoint_mean = torch.sigmoid(self.endpoint_mean_head(endpoint_hidden)).squeeze(-1)
        endpoint_std = self.config.minimum_endpoint_std + F.softplus(
            self.endpoint_std_head(endpoint_hidden).squeeze(-1)
        )
        q_embed = self.q_embedding(q_coordinate.unsqueeze(-1))
        q_embed = q_embed[None, :, None].expand(samples, -1, future, -1, -1)
        q_value = q_coordinate[None, :, None, :, None].expand(
            samples, -1, future, -1, -1
        )
        coordinate_context = torch.cat(
            [det, z, absolute, relative_feature, last_endpoint], dim=-1
        )
        coordinate_context = coordinate_context.unsqueeze(3).expand(-1, -1, -1, q_points, -1)
        hidden = self.coordinate_decoder(
            torch.cat([coordinate_context, q_embed, q_value], dim=-1)
        )
        voltage_mean = self.voltage_mean_head(hidden).squeeze(-1)
        voltage_std = self.config.minimum_voltage_std + F.softplus(
            self.voltage_std_head(hidden).squeeze(-1)
        )
        return {
            "sample_voltage_mean": voltage_mean,
            "sample_voltage_std": voltage_std,
            "sample_endpoint_mean": endpoint_mean,
            "sample_endpoint_std": endpoint_std,
        }

    @staticmethod
    def summarize_samples(decoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = dict(decoded)
        for prefix in ("voltage", "endpoint"):
            sample_mean = decoded[f"sample_{prefix}_mean"]
            sample_std = decoded[f"sample_{prefix}_std"]
            mean = sample_mean.mean(dim=0)
            epistemic = sample_mean.var(dim=0, unbiased=False).sqrt()
            aleatoric = sample_std.square().mean(dim=0).sqrt()
            output[f"{prefix}_mean"] = mean
            output[f"{prefix}_epistemic_std"] = epistemic
            output[f"{prefix}_aleatoric_std"] = aleatoric
            output[f"{prefix}_std"] = torch.sqrt(epistemic.square() + aleatoric.square())
        return output

    def forward(
        self,
        *,
        history_curve: torch.Tensor,
        history_endpoint_fraction: torch.Tensor,
        history_cycle_scaled: torch.Tensor,
        history_gap_scaled: torch.Tensor,
        history_mask: torch.Tensor,
        q_coordinate: torch.Tensor,
        query_cycle_scaled: torch.Tensor,
        query_mask: torch.Tensor,
        target_voltage: torch.Tensor | None = None,
        target_q_mask: torch.Tensor | None = None,
        target_endpoint_fraction: torch.Tensor | None = None,
        use_posterior: bool = False,
        num_latent_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_history(
            history_curve,
            history_endpoint_fraction,
            history_cycle_scaled,
            history_gap_scaled,
            history_mask,
            q_coordinate,
        )
        mean, std = encoded["prior_mean"], encoded["prior_std"]
        output: dict[str, torch.Tensor] = {
            "prior_mean": mean,
            "prior_std": std,
            "completed_state": encoded["last_state"],
        }
        if use_posterior:
            if target_voltage is None or target_q_mask is None or target_endpoint_fraction is None:
                raise ValueError("posterior training requires all future targets")
            posterior_mean, posterior_std = self.posterior_parameters(
                encoded,
                target_voltage,
                target_q_mask,
                target_endpoint_fraction,
                query_cycle_scaled,
                query_mask,
                q_coordinate,
            )
            output["posterior_mean"] = posterior_mean
            output["posterior_std"] = posterior_std
            mean, std = posterior_mean, posterior_std
        latent = self.sample_latent(mean, std, num_latent_samples)
        output.update(self.summarize_samples(
            self.decode_queries(encoded, query_cycle_scaled, q_coordinate, latent)
        ))
        return output


def build_model(config: ModelConfig) -> FutureVQLatentANP:
    return FutureVQLatentANP(config)

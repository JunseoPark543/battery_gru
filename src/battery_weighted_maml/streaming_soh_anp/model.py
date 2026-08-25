"""Hierarchical GRU + latent attentive neural process SOH forecaster."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from battery_weighted_maml.streaming_soh.model import PartialCycleEncoder

from .config import ModelConfig


class StreamingSOHLatentANP(nn.Module):
    """Model a distribution over entire SOH trajectories from causal context.

    Completed cycles and the current V/I-Q prefix form the context. Future SOH
    targets are used only by the variational posterior during training. At
    validation/deployment, predictions are sampled from the context prior.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.curve_encoder = PartialCycleEncoder(config)  # structural config compatibility
        scalar_features = 4
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
        self.gru_cells = nn.ModuleList(
            [
                nn.GRUCell(
                    config.cycle_feature_dim if layer == 0 else config.gru_hidden_dim,
                    config.gru_hidden_dim,
                )
                for layer in range(config.gru_layers)
            ]
        )
        self.recurrent_dropout = nn.Dropout(config.dropout)
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
            nn.Linear(config.gru_hidden_dim, config.latent_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_hidden_dim, 2 * config.latent_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(3, config.latent_hidden_dim),
            nn.GELU(),
            nn.Linear(config.latent_hidden_dim, config.gru_hidden_dim),
        )
        self.posterior_network = nn.Sequential(
            nn.Linear(2 * config.gru_hidden_dim, config.latent_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_hidden_dim, 2 * config.latent_dim),
        )
        latent_raw_std = math.log(
            math.expm1(max(0.1 - config.minimum_latent_std, 1.0e-4))
        )
        for distribution_network in (self.prior_network, self.posterior_network):
            distribution_head = distribution_network[-1]
            assert isinstance(distribution_head, nn.Linear)
            nn.init.normal_(distribution_head.weight, mean=0.0, std=1.0e-2)
            nn.init.zeros_(distribution_head.bias)
            with torch.no_grad():
                distribution_head.bias[config.latent_dim :].fill_(latent_raw_std)
        decoder_input = 2 * config.gru_hidden_dim + config.latent_dim + 4
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(decoder_input, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
        )
        self.soh_delta_head = nn.Linear(config.decoder_hidden_dim, 1)
        self.observation_std_head = nn.Linear(config.decoder_hidden_dim, 1)
        # A small non-zero initialization lets latent samples produce distinct
        # trajectories from the first optimization step without destabilizing
        # the persistence baseline.
        nn.init.normal_(self.soh_delta_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.soh_delta_head.bias)
        observation_raw_std = math.log(
            math.expm1(max(0.02 - config.minimum_observation_std, 1.0e-4))
        )
        nn.init.normal_(self.observation_std_head.weight, mean=0.0, std=1.0e-2)
        nn.init.constant_(self.observation_std_head.bias, observation_raw_std)
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
            layer_input = (
                self.recurrent_dropout(updated) if layer + 1 < len(self.gru_cells) else updated
            )
        return tuple(next_state)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _distribution_parameters(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, unconstrained = raw.chunk(2, dim=-1)
        std = self.config.minimum_latent_std + F.softplus(unconstrained)
        return mean, std

    def encode_completed_context(
        self,
        history_curve: torch.Tensor,
        history_soh: torch.Tensor,
        history_gap: torch.Tensor,
        history_cycle_scaled: torch.Tensor,
        history_mask: torch.Tensor,
        q_coordinate: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        if history_curve.ndim != 4:
            raise ValueError("history_curve must have shape [batch,history,q,3]")
        batch, history, q_points, channels = history_curve.shape
        if channels != 3 or q_coordinate.shape != (batch, q_points):
            raise ValueError("history curve/q shapes are incompatible")
        if not history_mask.any(dim=1).all():
            raise ValueError("every sample must have at least one completed cycle")
        expanded_q = q_coordinate[:, None, :].expand(-1, history, -1).reshape(
            batch * history, q_points
        )
        curve_embedding = self.curve_encoder(
            history_curve.reshape(batch * history, q_points, channels), expanded_q
        ).reshape(batch, history, -1)
        curve_available = history_curve[..., 2].any(dim=-1).to(history_curve.dtype)
        features = self.completed_cycle_projection(
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
        recurrent_tokens: list[torch.Tensor] = []
        for time_index in range(history):
            state = self._advance(features[:, time_index], state, history_mask[:, time_index])
            recurrent_tokens.append(state[-1])
        last_position = history - 1 - torch.flip(history_mask, dims=[1]).long().argmax(dim=1)
        last_soh = history_soh.gather(1, last_position.unsqueeze(1)).squeeze(1)
        return state, last_soh, torch.stack(recurrent_tokens, dim=1)

    def condition_current(
        self,
        completed_state: Sequence[torch.Tensor],
        last_soh: torch.Tensor,
        current_curve: torch.Tensor,
        q_coordinate: torch.Tensor,
        current_gap: torch.Tensor,
        current_cycle_scaled: torch.Tensor,
        prefix_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        current_embedding = self.curve_encoder(current_curve, q_coordinate)
        feature = self.current_cycle_projection(
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
        return self._advance(feature, completed_state, active)

    def build_context(
        self,
        completed_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        current_state: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = torch.cat([completed_tokens, current_state[-1].unsqueeze(1)], dim=1)
        current_mask = torch.ones(
            (history_mask.shape[0], 1), dtype=torch.bool, device=history_mask.device
        )
        return memory, torch.cat([history_mask, current_mask], dim=1)

    def prior_parameters(
        self, context_memory: torch.Tensor, context_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summary = self._masked_mean(context_memory, context_mask)
        return self._distribution_parameters(self.prior_network(summary))

    def posterior_parameters(
        self,
        context_memory: torch.Tensor,
        context_mask: torch.Tensor,
        query_cycle_scaled: torch.Tensor,
        current_cycle_scaled: torch.Tensor,
        target_soh: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative = query_cycle_scaled - current_cycle_scaled.unsqueeze(1)
        target_tokens = self.target_encoder(
            torch.stack([query_cycle_scaled, relative, target_soh], dim=-1)
        )
        context_summary = self._masked_mean(context_memory, context_mask)
        target_summary = self._masked_mean(target_tokens, query_mask)
        return self._distribution_parameters(
            self.posterior_network(torch.cat([context_summary, target_summary], dim=-1))
        )

    def deterministic_path(
        self,
        context_memory: torch.Tensor,
        context_mask: torch.Tensor,
        query_cycle_scaled: torch.Tensor,
        current_cycle_scaled: torch.Tensor,
    ) -> torch.Tensor:
        relative = query_cycle_scaled - current_cycle_scaled.unsqueeze(1)
        query = self.query_embedding(torch.stack([query_cycle_scaled, relative], dim=-1))
        attended, _ = self.cross_attention(
            query,
            context_memory,
            context_memory,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        return self.attention_norm(query + attended)

    def decode_trajectory(
        self,
        deterministic: torch.Tensor,
        current_state: Sequence[torch.Tensor],
        last_soh: torch.Tensor,
        current_cycle_scaled: torch.Tensor,
        query_cycle_scaled: torch.Tensor,
        prefix_fraction: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # latent: [samples,batch,latent]; outputs: [samples,batch,queries]
        samples, batch, _ = latent.shape
        queries = query_cycle_scaled.shape[1]
        recurrent = current_state[-1].unsqueeze(1).expand(-1, queries, -1)
        relative = query_cycle_scaled - current_cycle_scaled.unsqueeze(1)
        base = torch.cat(
            [
                deterministic,
                recurrent,
                query_cycle_scaled.unsqueeze(-1),
                relative.unsqueeze(-1),
                last_soh[:, None, None].expand(-1, queries, -1),
                prefix_fraction[:, None, None].expand(-1, queries, -1),
            ],
            dim=-1,
        )
        decoder_input = torch.cat(
            [
                base.unsqueeze(0).expand(samples, -1, -1, -1),
                latent[:, :, None, :].expand(-1, -1, queries, -1),
            ],
            dim=-1,
        )
        hidden = self.trajectory_decoder(decoder_input)
        mean = last_soh[None, :, None] + self.soh_delta_head(hidden).squeeze(-1)
        observation_std = self.config.minimum_observation_std + F.softplus(
            self.observation_std_head(hidden).squeeze(-1)
        )
        if mean.shape[:2] != (samples, batch):
            raise RuntimeError("latent decoder shape failure")
        return mean, observation_std

    def _auxiliary_heads(
        self,
        current_state: Sequence[torch.Tensor],
        q_coordinate: torch.Tensor,
        prefix_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = current_state[-1]
        q_embedding = self.q_embedding(q_coordinate.unsqueeze(-1))
        voltage = self.voltage_decoder(
            torch.cat(
                [
                    state.unsqueeze(1).expand(-1, q_coordinate.shape[1], -1),
                    q_embedding,
                    prefix_fraction[:, None, None].expand(-1, q_coordinate.shape[1], -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        remaining = self.endpoint_head(
            torch.cat([state, prefix_fraction.unsqueeze(-1)], dim=-1)
        ).squeeze(-1)
        endpoint = prefix_fraction + (1.0 - prefix_fraction) * remaining
        return voltage, endpoint

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
        *,
        target_soh: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        num_latent_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        if num_latent_samples <= 0:
            raise ValueError("num_latent_samples must be positive")
        completed_state, last_soh, completed_tokens = self.encode_completed_context(
            history_curve,
            history_soh,
            history_gap,
            history_cycle_scaled,
            history_mask,
            q_coordinate,
        )
        current_state = self.condition_current(
            completed_state,
            last_soh,
            current_curve,
            q_coordinate,
            current_gap,
            current_cycle_scaled,
            prefix_fraction,
        )
        context_memory, context_mask = self.build_context(
            completed_tokens, history_mask, current_state
        )
        prior_mean, prior_std = self.prior_parameters(context_memory, context_mask)
        posterior_mean: torch.Tensor | None = None
        posterior_std: torch.Tensor | None = None
        latent_mean, latent_std = prior_mean, prior_std
        if target_soh is not None:
            if query_mask is None:
                raise ValueError("query_mask is required with target_soh")
            posterior_mean, posterior_std = self.posterior_parameters(
                context_memory,
                context_mask,
                query_cycle_scaled,
                current_cycle_scaled,
                target_soh,
                query_mask,
            )
            latent_mean, latent_std = posterior_mean, posterior_std
        noise = torch.randn(
            num_latent_samples,
            *latent_mean.shape,
            dtype=latent_mean.dtype,
            device=latent_mean.device,
        )
        latent = latent_mean.unsqueeze(0) + latent_std.unsqueeze(0) * noise
        deterministic = self.deterministic_path(
            context_memory, context_mask, query_cycle_scaled, current_cycle_scaled
        )
        sample_mean, sample_observation_std = self.decode_trajectory(
            deterministic,
            current_state,
            last_soh,
            current_cycle_scaled,
            query_cycle_scaled,
            prefix_fraction,
            latent,
        )
        predictive_mean = sample_mean.mean(dim=0)
        epistemic_variance = sample_mean.var(dim=0, unbiased=False)
        aleatoric_variance = sample_observation_std.square().mean(dim=0)
        predictive_std = torch.sqrt(
            (epistemic_variance + aleatoric_variance).clamp_min(1.0e-12)
        )
        voltage, endpoint = self._auxiliary_heads(
            current_state, q_coordinate, prefix_fraction
        )
        output = {
            "soh_mean": predictive_mean,
            "soh_std": predictive_std,
            "soh_epistemic_std": torch.sqrt(epistemic_variance.clamp_min(0.0)),
            "soh_aleatoric_std": torch.sqrt(aleatoric_variance.clamp_min(1.0e-12)),
            "sample_soh_mean": sample_mean,
            "sample_observation_std": sample_observation_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "voltage": voltage,
            "endpoint_fraction": endpoint,
            "completed_state": completed_state[-1],
            "candidate_state": current_state[-1],
        }
        if posterior_mean is not None and posterior_std is not None:
            output["posterior_mean"] = posterior_mean
            output["posterior_std"] = posterior_std
        return output


def build_model(config: ModelConfig) -> StreamingSOHLatentANP:
    return StreamingSOHLatentANP(config)

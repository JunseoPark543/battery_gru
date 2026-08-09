"""Dual-representation direct-RUL model used by all comparison methods."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from hust_direct_rul_boil.model import (
    DomainClassifier,
    HierarchicalFeatureExtractor,
    gradient_reverse,
)

from .config import AblationConfig, ModelConfig


class GeneralRULHead(nn.Module):
    """Positive baseline RUL prediction in physical cycle units."""

    def __init__(self, embedding: int, hidden: int, dropout: float, scale: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.scale = float(scale)

    def forward(self, embedding: Tensor) -> Tensor:
        score = self.network(embedding).squeeze(-1)
        return self.scale * F.softplus(score)

    def initialize_bias(self, source_rul_cycles: float) -> None:
        scaled = max(float(source_rul_cycles) / self.scale, 1.0e-6)
        inverse_softplus = scaled + math.log(-math.expm1(-scaled))
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("general RUL head has an unexpected final layer")
        nn.init.constant_(final.bias, inverse_softplus)


class SpecificResidualHead(nn.Module):
    """Signed, bounded domain/task-specific correction in cycle units."""

    def __init__(self, embedding: int, hidden: int, dropout: float, limit: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.limit = float(limit)
        final = self.network[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.limit * torch.tanh(self.network(embedding).squeeze(-1))


class ConcatRULHead(nn.Module):
    """Single-head prediction used only for the requested architecture ablation."""

    def __init__(self, embedding: int, hidden: int, dropout: float, scale: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * embedding, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.scale = float(scale)

    def forward(self, general: Tensor, specific: Tensor) -> Tensor:
        score = self.network(torch.cat((general, specific), dim=-1)).squeeze(-1)
        return self.scale * F.softplus(score)

    def initialize_bias(self, source_rul_cycles: float) -> None:
        scaled = max(float(source_rul_cycles) / self.scale, 1.0e-6)
        inverse_softplus = scaled + math.log(-math.expm1(-scaled))
        final = self.network[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.bias, inverse_softplus)


class ReconstructionDecoder(nn.Module):
    """Reconstruct normalized per-cycle input summaries, not raw waveforms.

    The target at each cycle is [profile mean, profile std, scalar features].
    This preserves temporal degradation information while avoiding a very large
    100 x 64 x 8 raw-waveform decoder.
    """

    def __init__(
        self,
        embedding: int,
        hidden: int,
        history_length: int,
        target_features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.context = nn.Linear(2 * embedding, hidden)
        self.position = nn.Parameter(torch.zeros(1, history_length, hidden))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, target_features),
        )

    def forward(self, general: Tensor, specific: Tensor) -> Tensor:
        context = self.context(torch.cat((general, specific), dim=-1)).unsqueeze(1)
        return self.decoder(context + self.position)


def reconstruction_target(waveforms: Tensor, scalars: Tensor) -> Tensor:
    """Return [B, 100, 2*C_profile + C_scalar] normalized summaries."""
    profile_mean = waveforms.mean(dim=2)
    profile_std = waveforms.std(dim=2, unbiased=False)
    return torch.cat((profile_mean, profile_std, scalars), dim=-1)


@dataclass
class HybridOutput:
    prediction: Tensor
    general_prediction: Tensor
    specific_residual: Tensor
    general_embedding: Tensor
    specific_embedding: Tensor
    general_domain_logits: Tensor
    specific_domain_logits: Tensor | None
    reconstruction: Tensor | None
    reconstruction_target: Tensor


class GeneralSpecificRULModel(nn.Module):
    """General representation reuse plus specific representation change."""

    def __init__(
        self,
        waveform_channels: int,
        scalar_features: int,
        history_length: int,
        source_domains: int,
        model_config: ModelConfig,
        ablation: AblationConfig,
    ) -> None:
        super().__init__()
        extractor_args = (
            waveform_channels,
            scalar_features,
            history_length,
            model_config,
        )
        self.general_encoder = HierarchicalFeatureExtractor(*extractor_args)
        self.specific_encoder = HierarchicalFeatureExtractor(*extractor_args)
        self.general_head = GeneralRULHead(
            model_config.embedding_dim,
            model_config.predictor_hidden,
            model_config.dropout,
            model_config.rul_output_scale_cycles,
        )
        self.specific_head = SpecificResidualHead(
            model_config.embedding_dim,
            model_config.predictor_hidden,
            model_config.dropout,
            model_config.residual_limit_cycles,
        )
        self.concat_head = ConcatRULHead(
            model_config.embedding_dim,
            model_config.predictor_hidden,
            model_config.dropout,
            model_config.rul_output_scale_cycles,
        )
        self.general_domain_classifier = DomainClassifier(
            model_config.embedding_dim,
            model_config.domain_hidden,
            source_domains,
            model_config.dropout,
        )
        self.specific_domain_classifier = (
            DomainClassifier(
                model_config.embedding_dim,
                model_config.domain_hidden,
                source_domains,
                model_config.dropout,
            )
            if ablation.use_specific_domain_classifier
            else None
        )
        target_features = 2 * waveform_channels + scalar_features
        self.reconstruction_decoder = (
            ReconstructionDecoder(
                model_config.embedding_dim,
                model_config.reconstruction_hidden,
                history_length,
                target_features,
                model_config.dropout,
            )
            if ablation.use_reconstruction
            else None
        )
        self.ablation = ablation

    def initialize_prediction_bias(self, source_rul_cycles: float) -> None:
        self.general_head.initialize_bias(source_rul_cycles)
        self.concat_head.initialize_bias(source_rul_cycles)

    def forward(
        self,
        waveforms: Tensor,
        scalars: Tensor,
        grl_strength: float = 1.0,
    ) -> HybridOutput:
        general = self.general_encoder(waveforms, scalars)
        specific = self.specific_encoder(waveforms, scalars)
        y_general = self.general_head(general)
        residual = self.specific_head(specific)
        if self.ablation.prediction_mode == "residual":
            prediction = y_general + residual
        else:
            prediction = self.concat_head(general, specific)
        domain_input = (
            gradient_reverse(general, grl_strength)
            if self.ablation.use_grl
            else general
        )
        return HybridOutput(
            prediction=prediction,
            general_prediction=y_general,
            specific_residual=residual,
            general_embedding=general,
            specific_embedding=specific,
            general_domain_logits=self.general_domain_classifier(domain_input),
            specific_domain_logits=(
                None
                if self.specific_domain_classifier is None
                else self.specific_domain_classifier(specific)
            ),
            reconstruction=(
                None
                if self.reconstruction_decoder is None
                else self.reconstruction_decoder(general, specific)
            ),
            reconstruction_target=reconstruction_target(waveforms, scalars),
        )

    def module_parameter_counts(self) -> dict[str, int]:
        names = (
            "general_encoder",
            "specific_encoder",
            "general_head",
            "specific_head",
            "concat_head",
            "general_domain_classifier",
            "specific_domain_classifier",
            "reconstruction_decoder",
        )
        result: dict[str, int] = {}
        for name in names:
            module = getattr(self, name)
            result[name] = 0 if module is None else sum(
                parameter.numel() for parameter in module.parameters()
            )
        result["total"] = sum(parameter.numel() for parameter in self.parameters())
        return result


"""Strict configuration for the hybrid General-ANIL/Specific-BOIL study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hust_direct_rul_boil.config import (
    DataConfig,
    PROFILE_CHANNELS,
    SCALAR_FEATURES,
)


METHODS = ("supervised", "maml", "anil", "boil", "hybrid")
PREDICTION_MODES = ("residual", "concat")


@dataclass
class ModelConfig:
    waveform_hidden: int = 32
    waveform_embedding: int = 48
    scalar_embedding: int = 32
    d_model: int = 64
    nhead: int = 4
    attention_stages: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.10
    cbam_reduction: int = 4
    cbam_kernel_size: int = 7
    embedding_dim: int = 64
    predictor_hidden: int = 64
    domain_hidden: int = 64
    reconstruction_hidden: int = 96
    rul_output_scale_cycles: float = 500.0
    residual_limit_cycles: float = 500.0


@dataclass
class LossConfig:
    rul_scale_cycles: float = 500.0
    lambda_total_prediction: float = 1.0
    lambda_general_prediction: float = 0.5
    lambda_general_domain: float = 0.05
    lambda_specific_domain: float = 0.05
    lambda_reconstruction: float = 0.01
    lambda_consistency: float = 0.01
    lambda_orthogonal: float = 0.01
    lambda_residual: float = 0.001
    inner_general_prediction_beta: float = 0.0
    consistency_sigma: float = 0.15
    grl_max_strength: float = 1.0


@dataclass
class AblationConfig:
    use_grl: bool = True
    use_specific_domain_classifier: bool = True
    use_reconstruction: bool = True
    use_consistency: bool = True
    use_orthogonality: bool = True
    use_general_prediction_loss: bool = True
    use_residual_regularization: bool = True
    prediction_mode: str = "residual"


@dataclass
class AugmentationConfig:
    enabled: bool = True
    waveform_gaussian_std: float = 0.01
    scalar_gaussian_std: float = 0.02
    profile_channel_mask_probability: float = 0.03
    cycle_mask_probability: float = 0.03


@dataclass
class TrainConfig:
    iterations: int = 1500
    validation_cells_per_protocol: int = 2
    support_cells_per_task: int = 2
    query_cells_per_task: int = 2
    tasks_per_iteration: int = 4
    supervised_cells_per_domain: int = 4
    inner_steps: int = 1
    inner_lr_general_head: float = 0.01
    inner_lr_specific_encoder: float = 0.01
    inner_lr_other: float = 0.01
    outer_lr: float = 0.001
    weight_decay: float = 1.0e-4
    first_order: bool = True
    gradient_clip_norm: float = 5.0
    evaluation_interval: int = 25
    checkpoint_interval: int = 100
    early_stopping_patience_evaluations: int = 12
    log_interval: int = 10
    validation_adaptation_steps: int = 1


@dataclass
class EvaluationConfig:
    held_out_protocols: list[str] | str = "all"
    target_support_cells: int = 2
    adaptation_steps: list[int] = field(default_factory=lambda: [0, 1, 2, 5, 10])
    primary_adaptation_step: int = 1
    clip_negative_rul: bool = True
    residual_ratio_warning: float = 0.5


@dataclass
class OutputConfig:
    output_dir: str = "outputs/hust_hybrid_anil_boil"


@dataclass
class ExperimentConfig:
    seeds: list[int] = field(default_factory=lambda: [42])
    device: str = "auto"
    method: str = "hybrid"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique list")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be auto, cpu, or cuda[:index]")
        if self.data.history_length != 100:
            raise ValueError("this direct-RUL experiment is fixed to L=100")
        if not 1 <= self.data.reference_cycle <= self.data.history_length:
            raise ValueError("reference_cycle must lie inside the history window")
        if self.data.waveform_points < 16:
            raise ValueError("waveform_points must be at least 16")
        if self.data.expected_protocol_count < 2:
            raise ValueError("expected_protocol_count must be at least two")
        if self.data.current_threshold_c_rate <= 0 or self.data.minimum_phase_points < 2:
            raise ValueError("invalid charge/discharge phase extraction settings")
        if self.data.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        if tuple(self.data.profile_channels) != PROFILE_CHANNELS:
            raise ValueError("profile_channels do not match the reused HUST loader")
        if tuple(self.data.scalar_features) != SCALAR_FEATURES:
            raise ValueError("scalar_features do not match the reused HUST loader")
        if self.model.d_model % self.model.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if self.model.cbam_kernel_size % 2 == 0:
            raise ValueError("cbam_kernel_size must be odd")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        model_values = asdict(self.model)
        if any(
            float(value) <= 0
            for key, value in model_values.items()
            if key != "dropout"
        ):
            raise ValueError("model dimensions and physical output scales must be positive")
        if self.ablation.prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
        loss_values = asdict(self.loss)
        if self.loss.rul_scale_cycles <= 0 or self.loss.consistency_sigma <= 0:
            raise ValueError("loss scale and consistency sigma must be positive")
        if any(float(value) < 0 for key, value in loss_values.items() if key != "consistency_sigma"):
            raise ValueError("loss coefficients must be non-negative")
        train = self.train
        positive_train = (
            train.iterations,
            train.validation_cells_per_protocol,
            train.support_cells_per_task,
            train.query_cells_per_task,
            train.tasks_per_iteration,
            train.supervised_cells_per_domain,
            train.inner_lr_general_head,
            train.inner_lr_specific_encoder,
            train.inner_lr_other,
            train.outer_lr,
            train.gradient_clip_norm,
            train.evaluation_interval,
            train.checkpoint_interval,
            train.early_stopping_patience_evaluations,
            train.log_interval,
        )
        if any(float(value) <= 0 for value in positive_train):
            raise ValueError("training counts/rates must be positive")
        if train.inner_steps < 0 or train.validation_adaptation_steps < 0:
            raise ValueError("adaptation step counts cannot be negative")
        if train.validation_cells_per_protocol < 2:
            raise ValueError("source validation needs at least two cells per protocol")
        if train.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        for probability in (
            self.augmentation.profile_channel_mask_probability,
            self.augmentation.cycle_mask_probability,
        ):
            if not 0 <= probability <= 1:
                raise ValueError("mask probabilities must lie in [0, 1]")
        if self.augmentation.waveform_gaussian_std < 0 or self.augmentation.scalar_gaussian_std < 0:
            raise ValueError("augmentation noise cannot be negative")
        steps = self.evaluation.adaptation_steps
        if not steps or any(step < 0 for step in steps) or len(set(steps)) != len(steps):
            raise ValueError("adaptation_steps must be unique non-negative integers")
        if 0 not in steps:
            raise ValueError("adaptation_steps must include zero")
        if self.evaluation.primary_adaptation_step not in steps:
            raise ValueError("primary_adaptation_step must be in adaptation_steps")
        if self.evaluation.target_support_cells <= 0:
            raise ValueError("target_support_cells must be positive")
        if self.evaluation.residual_ratio_warning <= 0:
            raise ValueError("residual_ratio_warning must be positive")
        held_out = self.evaluation.held_out_protocols
        if held_out != "all" and (not isinstance(held_out, list) or not held_out):
            raise ValueError("held_out_protocols must be 'all' or a non-empty list")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(cls: type[Any], raw: Any, name: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown keys in {name}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    sections = {
        "seeds", "device", "method", "data", "model", "loss", "ablation",
        "augmentation", "train", "evaluation", "output",
    }
    unknown = set(raw) - sections
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    config = ExperimentConfig(
        seeds=list(raw.get("seeds", [42])),
        device=str(raw.get("device", "auto")),
        method=str(raw.get("method", "hybrid")),
        data=_section(DataConfig, raw.get("data", {}), "data"),
        model=_section(ModelConfig, raw.get("model", {}), "model"),
        loss=_section(LossConfig, raw.get("loss", {}), "loss"),
        ablation=_section(AblationConfig, raw.get("ablation", {}), "ablation"),
        augmentation=_section(AugmentationConfig, raw.get("augmentation", {}), "augmentation"),
        train=_section(TrainConfig, raw.get("train", {}), "train"),
        evaluation=_section(EvaluationConfig, raw.get("evaluation", {}), "evaluation"),
        output=_section(OutputConfig, raw.get("output", {}), "output"),
    )
    config.validate()
    return config


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

"""Strict configuration for the standalone direct-RUL BOIL project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DOMAIN_NAMES = ("CS2_0.5C", "CS2_1C", "CX2_0.5C", "CX2_1C")
FEATURE_NAMES = (
    "soh",
    "charge_duration_h",
    "discharge_duration_h",
    "charge_voltage_mean_v",
    "discharge_voltage_mean_v",
    "charge_c_rate",
    "discharge_c_rate",
)


@dataclass
class DataConfig:
    calce_dir: str = "data/CALCE"
    label_path: str = "data/Life labels/CALCE_labels.json"
    history_length: int = 100
    features: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    current_threshold_c_rate: float = 0.02
    domain_rate_tolerance: float = 0.2
    normalization_epsilon: float = 1.0e-6


@dataclass
class ModelConfig:
    d_model: int = 32
    nhead: int = 4
    attention_stages: int = 2
    transformer_layers_per_stage: int = 1
    dim_feedforward: int = 64
    dropout: float = 0.10
    cbam_reduction: int = 4
    cbam_kernel_size: int = 7
    embedding_dim: int = 32
    predictor_hidden: int = 32
    domain_hidden: int = 32


@dataclass
class LossConfig:
    huber_delta: float = 1.0
    domain_adversarial_weight: float = 0.10
    domain_fuzzy_weight: float = 0.05
    orthogonality_weight: float = 0.10
    meta_support_weight: float = 1.0
    meta_query_weight: float = 1.0
    grl_max_strength: float = 1.0


@dataclass
class AugmentationConfig:
    enabled: bool = True
    gaussian_std: float = 0.02
    feature_mask_probability: float = 0.03
    cycle_mask_probability: float = 0.03
    joint_views_per_cell: int = 2


@dataclass
class TrainConfig:
    iterations: int = 1000
    inner_steps: int = 1
    inner_lr: float = 0.05
    outer_lr: float = 0.001
    weight_decay: float = 1.0e-4
    second_order: bool = True
    gradient_clip_norm: float = 5.0
    evaluation_interval: int = 20
    checkpoint_interval: int = 100
    early_stopping_patience_evaluations: int = 15
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    held_out_domains: list[str] | str = "all"
    clip_negative_rul: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "outputs/calce_direct_rul_boil"


@dataclass
class ExperimentConfig:
    seeds: list[int] = field(default_factory=lambda: [42])
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if not self.seeds or any(not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must be a non-empty integer list")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds cannot contain duplicates")
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be auto, cpu, or cuda[:index]")
        if self.data.history_length != 100:
            raise ValueError("this experiment is intentionally fixed to the first 100 cycles")
        if tuple(self.data.features) != FEATURE_NAMES:
            raise ValueError(f"data.features must be exactly {list(FEATURE_NAMES)}")
        if self.data.current_threshold_c_rate <= 0:
            raise ValueError("current_threshold_c_rate must be positive")
        if self.data.domain_rate_tolerance <= 0:
            raise ValueError("domain_rate_tolerance must be positive")
        if self.data.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        model = self.model
        if model.nhead <= 0:
            raise ValueError("model.nhead must be positive")
        if model.d_model <= 0 or model.d_model % model.nhead != 0:
            raise ValueError("model.d_model must be positive and divisible by model.nhead")
        if model.attention_stages <= 0 or model.transformer_layers_per_stage <= 0:
            raise ValueError("attention stage/layer counts must be positive")
        if (
            model.dim_feedforward <= 0
            or model.embedding_dim <= 0
            or model.predictor_hidden <= 0
            or model.domain_hidden <= 0
        ):
            raise ValueError("model hidden dimensions must be positive")
        if model.cbam_reduction <= 0:
            raise ValueError("model.cbam_reduction must be positive")
        if model.cbam_kernel_size <= 0 or model.cbam_kernel_size % 2 == 0:
            raise ValueError("model.cbam_kernel_size must be a positive odd integer")
        if not 0.0 <= model.dropout < 1.0:
            raise ValueError("model.dropout must lie in [0, 1)")
        loss = self.loss
        if loss.huber_delta <= 0:
            raise ValueError("loss.huber_delta must be positive")
        if any(
            value < 0
            for value in (
                loss.domain_adversarial_weight,
                loss.domain_fuzzy_weight,
                loss.orthogonality_weight,
                loss.meta_support_weight,
                loss.meta_query_weight,
                loss.grl_max_strength,
            )
        ):
            raise ValueError("all loss weights must be non-negative")
        aug = self.augmentation
        if aug.gaussian_std < 0:
            raise ValueError("augmentation.gaussian_std cannot be negative")
        if not 0.0 <= aug.feature_mask_probability <= 1.0:
            raise ValueError("feature_mask_probability must lie in [0, 1]")
        if not 0.0 <= aug.cycle_mask_probability <= 1.0:
            raise ValueError("cycle_mask_probability must lie in [0, 1]")
        if aug.joint_views_per_cell <= 0:
            raise ValueError("joint_views_per_cell must be positive")
        train = self.train
        if train.iterations <= 0 or train.inner_steps <= 0:
            raise ValueError("training iteration counts must be positive")
        if train.inner_lr <= 0 or train.outer_lr <= 0:
            raise ValueError("learning rates must be positive")
        if train.weight_decay < 0 or train.gradient_clip_norm <= 0:
            raise ValueError("invalid weight decay or gradient clip norm")
        if train.evaluation_interval <= 0 or train.checkpoint_interval <= 0:
            raise ValueError("evaluation/checkpoint intervals must be positive")
        if train.early_stopping_patience_evaluations <= 0 or train.log_interval <= 0:
            raise ValueError("patience and log interval must be positive")
        held_out = self.evaluation.held_out_domains
        if held_out != "all":
            if not isinstance(held_out, list) or not held_out:
                raise ValueError("held_out_domains must be 'all' or a non-empty list")
            unknown = set(held_out) - set(DOMAIN_NAMES)
            if unknown:
                raise ValueError(f"unknown held-out domains: {sorted(unknown)}")
            if len(set(held_out)) != len(held_out):
                raise ValueError("held_out_domains cannot contain duplicates")
        if not self.output.output_dir.strip():
            raise ValueError("output.output_dir cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_dataclass(cls: type[Any], raw: dict[str, Any], section: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"config section {section!r} must be a mapping")
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    known = {
        "seeds", "device", "data", "model", "loss", "augmentation",
        "train", "evaluation", "output",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    config = ExperimentConfig(
        seeds=list(raw.get("seeds", [42])),
        device=str(raw.get("device", "auto")),
        data=_strict_dataclass(DataConfig, raw.get("data", {}), "data"),
        model=_strict_dataclass(ModelConfig, raw.get("model", {}), "model"),
        loss=_strict_dataclass(LossConfig, raw.get("loss", {}), "loss"),
        augmentation=_strict_dataclass(
            AugmentationConfig, raw.get("augmentation", {}), "augmentation"
        ),
        train=_strict_dataclass(TrainConfig, raw.get("train", {}), "train"),
        evaluation=_strict_dataclass(
            EvaluationConfig, raw.get("evaluation", {}), "evaluation"
        ),
        output=_strict_dataclass(OutputConfig, raw.get("output", {}), "output"),
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

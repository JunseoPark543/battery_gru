"""Strict configuration for the standalone HUST direct-RUL experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROFILE_CHANNELS = (
    "charge_voltage_v",
    "charge_current_c_rate",
    "charge_capacity_fraction",
    "charge_power_vc",
    "delta_voltage_from_cycle10",
    "delta_current_from_cycle10",
    "delta_capacity_from_cycle10",
    "delta_power_from_cycle10",
)

SCALAR_FEATURES = (
    "soh",
    "charge_duration_h",
    "discharge_duration_h",
    "charge_voltage_mean_v",
    "discharge_voltage_mean_v",
    "charge_c_rate",
    "discharge_c_rate",
    "coulombic_efficiency",
    "charge_energy_per_nominal_wh",
    "discharge_energy_per_nominal_wh",
    "delta_soh_from_cycle10",
    "delta_charge_duration_from_cycle10_h",
    "delta_discharge_duration_from_cycle10_h",
    "rolling_soh_slope_10",
)


@dataclass
class DataConfig:
    hust_dir: str = "data/HUST"
    label_path: str = "data/Life labels/HUST_labels.json"
    history_length: int = 100
    reference_cycle: int = 10
    waveform_points: int = 64
    profile_channels: list[str] = field(default_factory=lambda: list(PROFILE_CHANNELS))
    scalar_features: list[str] = field(default_factory=lambda: list(SCALAR_FEATURES))
    expected_protocol_count: int = 10
    current_threshold_c_rate: float = 0.02
    minimum_phase_points: int = 8
    normalization_epsilon: float = 1.0e-6


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
    rul_output_scale_cycles: float = 500.0


@dataclass
class LossConfig:
    raw_rul_huber_beta_cycles: float = 50.0
    raw_rul_loss_scale_cycles: float = 500.0
    domain_adversarial_weight: float = 0.10
    domain_fuzzy_weight: float = 0.05
    orthogonality_weight: float = 0.10
    meta_support_weight: float = 1.0
    meta_query_weight: float = 1.0
    grl_max_strength: float = 1.0


@dataclass
class AugmentationConfig:
    enabled: bool = True
    waveform_gaussian_std: float = 0.01
    scalar_gaussian_std: float = 0.02
    profile_channel_mask_probability: float = 0.03
    cycle_mask_probability: float = 0.03
    joint_views_per_cell: int = 2


@dataclass
class TrainConfig:
    iterations: int = 1500
    validation_cells_per_protocol: int = 1
    joint_cells_per_domain: int = 4
    meta_support_cells_per_domain: int = 2
    inner_steps: int = 1
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    weight_decay: float = 1.0e-4
    second_order: bool = True
    gradient_clip_norm: float = 5.0
    evaluation_interval: int = 25
    checkpoint_interval: int = 100
    early_stopping_patience_evaluations: int = 12
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    held_out_protocols: list[str] | str = "all"
    clip_negative_rul: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "outputs/hust_direct_rul_boil"


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
        data = self.data
        if data.history_length != 100:
            raise ValueError("this experiment is fixed to the first 100 cycles")
        if not 1 <= data.reference_cycle <= data.history_length:
            raise ValueError("reference_cycle must be inside the history window")
        if data.waveform_points < 16:
            raise ValueError("waveform_points must be at least 16")
        if tuple(data.profile_channels) != PROFILE_CHANNELS:
            raise ValueError(f"profile_channels must be exactly {list(PROFILE_CHANNELS)}")
        if tuple(data.scalar_features) != SCALAR_FEATURES:
            raise ValueError(f"scalar_features must be exactly {list(SCALAR_FEATURES)}")
        if data.expected_protocol_count < 2:
            raise ValueError("expected_protocol_count must be at least two")
        if data.current_threshold_c_rate <= 0 or data.minimum_phase_points < 2:
            raise ValueError("invalid phase extraction settings")
        if data.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        model = self.model
        positive_model = (
            model.waveform_hidden,
            model.waveform_embedding,
            model.scalar_embedding,
            model.d_model,
            model.nhead,
            model.attention_stages,
            model.dim_feedforward,
            model.cbam_reduction,
            model.cbam_kernel_size,
            model.embedding_dim,
            model.predictor_hidden,
            model.domain_hidden,
            model.rul_output_scale_cycles,
        )
        if any(value <= 0 for value in positive_model):
            raise ValueError("all model dimensions/scales must be positive")
        if model.d_model % model.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if model.cbam_kernel_size % 2 == 0:
            raise ValueError("cbam_kernel_size must be odd")
        if not 0 <= model.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        loss = self.loss
        positive_loss = (
            loss.raw_rul_huber_beta_cycles,
            loss.raw_rul_loss_scale_cycles,
        )
        if any(value <= 0 for value in positive_loss):
            raise ValueError("raw-RUL loss beta/scale must be positive")
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
            raise ValueError("loss weights must be non-negative")
        aug = self.augmentation
        if aug.waveform_gaussian_std < 0 or aug.scalar_gaussian_std < 0:
            raise ValueError("augmentation noise cannot be negative")
        if not 0 <= aug.profile_channel_mask_probability <= 1:
            raise ValueError("profile_channel_mask_probability must lie in [0, 1]")
        if not 0 <= aug.cycle_mask_probability <= 1:
            raise ValueError("cycle_mask_probability must lie in [0, 1]")
        if aug.joint_views_per_cell <= 0:
            raise ValueError("joint_views_per_cell must be positive")
        train = self.train
        if any(
            value <= 0
            for value in (
                train.iterations,
                train.validation_cells_per_protocol,
                train.joint_cells_per_domain,
                train.meta_support_cells_per_domain,
                train.inner_steps,
                train.inner_lr,
                train.outer_lr,
                train.gradient_clip_norm,
                train.evaluation_interval,
                train.checkpoint_interval,
                train.early_stopping_patience_evaluations,
                train.log_interval,
            )
        ):
            raise ValueError("training counts/rates must be positive")
        if train.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        held_out = self.evaluation.held_out_protocols
        if held_out != "all" and (not isinstance(held_out, list) or not held_out):
            raise ValueError("held_out_protocols must be 'all' or a non-empty list")
        if isinstance(held_out, list) and len(set(held_out)) != len(held_out):
            raise ValueError("held_out_protocols cannot contain duplicates")
        if not self.output.output_dir.strip():
            raise ValueError("output_dir cannot be empty")

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
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
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

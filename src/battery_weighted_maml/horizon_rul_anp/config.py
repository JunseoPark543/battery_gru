"""Typed configuration for horizon-conditioned MATR RUL ANP experiments."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from battery_weighted_maml.matr_anp.config import (
    DataConfig as SharedDataConfig,
    SplitConfig,
)


@dataclass
class PathsConfig:
    data_root: str | None = "data/MATR"
    output_root: str = "outputs/horizon_rul_anp"


@dataclass
class DataConfig:
    dataset: str = "MATR"
    file_globs: list[str] = field(default_factory=lambda: ["**/*.pkl"])
    minimum_valid_cycles: int = 30
    minimum_discharge_points: int = 16
    short_signal_threshold: int = 32
    reference_cycles: list[int] = field(
        default_factory=lambda: [5, 6, 7, 8, 9, 10]
    )
    minimum_reference_cycles: int = 3
    lifetime_source: str = "label_file"
    label_path: str | None = "data/Life labels/MATR_labels.json"

    def shared(self) -> SharedDataConfig:
        return SharedDataConfig(
            dataset=self.dataset,
            file_globs=list(self.file_globs),
            minimum_valid_cycles=self.minimum_valid_cycles,
            minimum_discharge_points=self.minimum_discharge_points,
            short_signal_threshold=self.short_signal_threshold,
            reference_cycles=list(self.reference_cycles),
            minimum_reference_cycles=self.minimum_reference_cycles,
        )


@dataclass
class TaskConfig:
    min_horizon: int = 20
    max_horizon: int = 500
    context_size_min: int = 8
    context_size_max: int = 16
    query_size: int = 8
    min_cells_per_task: int = 16
    max_resample_attempts: int = 100


@dataclass
class ModelConfig:
    prefix_feature_dim: int = 3
    d_model: int = 128
    attention_heads: int = 4
    prefix_layers: int = 2
    dropout: float = 0.1
    latent_dim: int = 64
    mlp_layers: int = 3
    minimum_std: float = 1.0e-3


@dataclass
class TrainingConfig:
    learning_rate: float = 1.0e-4
    max_steps: int = 20_000
    task_batch_size: int = 1
    gradient_clip_norm: float = 1.0
    beta_kl: float = 1.0
    kl_warmup_steps: int = 5_000
    validation_interval: int = 500
    validation_horizons: list[int] = field(
        default_factory=lambda: [20, 40, 60, 80, 100]
    )
    early_stopping_patience: int = 10
    checkpoint_interval: int = 500
    log_interval: int = 10
    use_amp: bool = True
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    horizons: list[int] = field(default_factory=lambda: [20, 40, 60, 80, 100])
    context_size: int = 16
    mc_samples: int = 50
    interval_level: float = 0.95
    mape_epsilon_cycles: float = 1.0
    plot_cell: str | None = None


@dataclass
class HorizonRULConfig:
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.data.dataset.upper() != "MATR":
            raise ValueError("the first horizon-RUL prototype supports MATR only")
        if self.data.lifetime_source not in {"last_observed_cycle", "label_file"}:
            raise ValueError(
                "data.lifetime_source must be last_observed_cycle or label_file"
            )
        if self.data.lifetime_source == "label_file" and not self.data.label_path:
            raise ValueError("data.label_path is required for lifetime_source=label_file")
        # Reuse validation of the established BatteryLife loader configuration.
        self.data.shared()
        if self.split.strategy not in {"kfold", "loocv"}:
            raise ValueError("split.strategy must be kfold or loocv")
        if self.split.strategy == "kfold" and self.split.num_folds < 2:
            raise ValueError("split.num_folds must be at least two")
        if not 0.0 < self.split.validation_fraction < 1.0:
            raise ValueError("split.validation_fraction must lie in (0,1)")
        task = self.task
        if not 2 <= task.min_horizon <= task.max_horizon:
            raise ValueError("task horizons must satisfy 2 <= min <= max")
        if not 1 <= task.context_size_min <= task.context_size_max:
            raise ValueError("invalid context size range")
        if task.query_size <= 0:
            raise ValueError("query_size must be positive")
        required = task.context_size_min + task.query_size
        if task.min_cells_per_task < required:
            raise ValueError(
                "min_cells_per_task must cover minimum context plus query cells"
            )
        if task.max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts must be positive")
        model = self.model
        if model.prefix_feature_dim != 3:
            raise ValueError("prototype prefix_feature_dim must be 3")
        if model.d_model <= 0 or model.d_model % model.attention_heads:
            raise ValueError("d_model must be positive and divisible by attention_heads")
        if model.prefix_layers <= 0 or model.latent_dim <= 0 or model.mlp_layers < 2:
            raise ValueError("model layer/dimension settings are invalid")
        if not 0.0 <= model.dropout < 1.0 or model.minimum_std <= 0:
            raise ValueError("dropout/minimum_std settings are invalid")
        training = self.training
        positive = {
            "learning_rate": training.learning_rate,
            "max_steps": training.max_steps,
            "task_batch_size": training.task_batch_size,
            "gradient_clip_norm": training.gradient_clip_norm,
            "validation_interval": training.validation_interval,
            "early_stopping_patience": training.early_stopping_patience,
            "checkpoint_interval": training.checkpoint_interval,
            "log_interval": training.log_interval,
            "context_size": self.evaluation.context_size,
            "mc_samples": self.evaluation.mc_samples,
            "mape_epsilon_cycles": self.evaluation.mape_epsilon_cycles,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"positive configuration values required: {positive}")
        if training.beta_kl < 0 or training.kl_warmup_steps < 0:
            raise ValueError("KL beta/warmup cannot be negative")
        all_horizons = training.validation_horizons + self.evaluation.horizons
        if not all_horizons or len(self.evaluation.horizons) != len(
            set(self.evaluation.horizons)
        ):
            raise ValueError("evaluation horizons must be non-empty and unique")
        if any(not task.min_horizon <= k <= task.max_horizon for k in all_horizons):
            raise ValueError("validation/evaluation horizons must lie in task range")
        if not 0.0 < self.evaluation.interval_level < 1.0:
            raise ValueError("evaluation.interval_level must lie in (0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> HorizonRULConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"horizon RUL ANP config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {
        "seed", "device", "paths", "data", "split", "task", "model",
        "training", "evaluation",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    try:
        config = HorizonRULConfig(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            paths=PathsConfig(**raw.get("paths", {})),
            data=DataConfig(**raw.get("data", {})),
            split=SplitConfig(**raw.get("split", {})),
            task=TaskConfig(**raw.get("task", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
    except TypeError as exc:
        raise ValueError(f"invalid horizon RUL configuration field: {exc}") from exc
    config.validate()
    return config


def save_config(config: HorizonRULConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def resolve_data_root(config: HorizonRULConfig, cli_root: str | None) -> Path:
    raw = cli_root or os.environ.get("BATTERYLIFE_DATA_ROOT") or config.paths.data_root
    if not raw:
        raise ValueError("MATR data root is unset")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MATR data root is not a directory: {root}")
    return root

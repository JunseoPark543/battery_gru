"""Typed configuration for the MATR lifetime I-V ANP."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from battery_weighted_maml.matr_anp.config import (
    DataConfig as SharedDataConfig,
    QGridConfig,
    SplitConfig,
)


@dataclass
class PathsConfig:
    data_root: str | None = "data/MATR"
    output_root: str = "outputs/horizon_lifetime_iv_anp"


@dataclass
class DataConfig:
    dataset: str = "MATR"
    file_globs: list[str] = field(default_factory=lambda: ["**/*.pkl"])
    minimum_valid_cycles: int = 30
    minimum_discharge_points: int = 16
    short_signal_threshold: int = 32
    reference_cycles: list[int] = field(default_factory=lambda: [5, 6, 7, 8, 9, 10])
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
    horizons: list[int] = field(
        default_factory=lambda: list(range(100, 301, 20))
    )
    context_size_min: int = 8
    context_size_max: int = 16
    query_size: int = 8
    min_cells_per_task: int = 16
    max_resample_attempts: int = 100


@dataclass
class ModelConfig:
    curve_input_dim: int = 3  # normalized q, voltage, current
    curve_d_model: int = 48
    curve_attention_heads: int = 4
    curve_layers: int = 2
    curve_patch_size: int = 8
    temporal_d_model: int = 128
    temporal_attention_heads: int = 4
    temporal_layers: int = 2
    dropout: float = 0.1
    latent_dim: int = 64
    anp_mlp_layers: int = 3
    minimum_std: float = 1.0e-3
    gradient_checkpoint_curves: bool = True


@dataclass
class TrainingConfig:
    learning_rate: float = 1.0e-4
    max_steps: int = 20_000
    task_batch_size: int = 1
    gradient_clip_norm: float = 1.0
    beta_kl: float = 1.0
    kl_warmup_steps: int = 5_000
    validation_interval: int = 500
    early_stopping_patience: int = 10
    checkpoint_interval: int = 500
    log_interval: int = 10
    use_amp: bool = True
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    horizons: list[int] = field(
        default_factory=lambda: list(range(100, 301, 20))
    )
    context_size: int = 16
    mc_samples: int = 50
    interval_level: float = 0.95
    mape_epsilon_cycles: float = 1.0
    plot_cell: str | None = None


@dataclass
class LifetimeIVConfig:
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    q_grid: QGridConfig = field(
        default_factory=lambda: QGridConfig(minimum=0.0, maximum=1.2, num_points=256)
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.data.dataset.upper() != "MATR":
            raise ValueError("horizon lifetime I-V ANP currently supports MATR only")
        if self.data.lifetime_source not in {"label_file", "last_observed_cycle"}:
            raise ValueError("invalid lifetime_source")
        if self.data.lifetime_source == "label_file" and not self.data.label_path:
            raise ValueError("label_path is required for label_file lifetime source")
        self.data.shared()
        if not self.q_grid.minimum < self.q_grid.maximum:
            raise ValueError("q_grid minimum must be smaller than maximum")
        if self.q_grid.num_points != 256:
            raise ValueError("this experiment requires exactly 256 interpolated q points")
        horizons = [int(value) for value in self.task.horizons]
        if not horizons or horizons != sorted(set(horizons)) or horizons[0] < 2:
            raise ValueError("task horizons must be sorted, unique, and >=2")
        if not set(self.evaluation.horizons).issubset(horizons):
            raise ValueError("evaluation horizons must be included in task horizons")
        task = self.task
        if not 1 <= task.context_size_min <= task.context_size_max:
            raise ValueError("invalid context size range")
        if task.query_size <= 0:
            raise ValueError("query_size must be positive")
        if task.min_cells_per_task < task.context_size_min + task.query_size:
            raise ValueError("min_cells_per_task must cover context plus query")
        model = self.model
        if model.curve_input_dim != 3:
            raise ValueError("curve_input_dim must represent [q, voltage, current]")
        if self.q_grid.num_points % model.curve_patch_size:
            raise ValueError("curve_patch_size must divide 256")
        if model.curve_d_model % model.curve_attention_heads:
            raise ValueError("curve_d_model must be divisible by curve heads")
        if model.temporal_d_model % model.temporal_attention_heads:
            raise ValueError("temporal_d_model must be divisible by temporal heads")
        if min(model.curve_layers, model.temporal_layers, model.anp_mlp_layers - 1) <= 0:
            raise ValueError("encoder/MLP layer counts are invalid")
        if not 0 <= model.dropout < 1 or model.minimum_std <= 0:
            raise ValueError("invalid dropout/minimum_std")
        positive = (
            self.training.learning_rate,
            self.training.max_steps,
            self.training.task_batch_size,
            self.training.gradient_clip_norm,
            self.training.validation_interval,
            self.training.early_stopping_patience,
            self.training.checkpoint_interval,
            self.training.log_interval,
            self.evaluation.context_size,
            self.evaluation.mc_samples,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("training/evaluation positive values must exceed zero")
        if not 0 < self.evaluation.interval_level < 1:
            raise ValueError("interval_level must lie in (0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> LifetimeIVConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"lifetime I-V ANP config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {"seed", "device", "paths", "data", "q_grid", "split", "task", "model", "training", "evaluation"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    try:
        config = LifetimeIVConfig(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            paths=PathsConfig(**raw.get("paths", {})),
            data=DataConfig(**raw.get("data", {})),
            q_grid=QGridConfig(**raw.get("q_grid", {})),
            split=SplitConfig(**raw.get("split", {})),
            task=TaskConfig(**raw.get("task", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
    except TypeError as exc:
        raise ValueError(f"invalid lifetime I-V configuration: {exc}") from exc
    config.validate()
    return config


def save_config(config: LifetimeIVConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def resolve_data_root(config: LifetimeIVConfig, cli_root: str | None) -> Path:
    raw = cli_root or os.environ.get("BATTERYLIFE_DATA_ROOT") or config.paths.data_root
    if not raw:
        raise ValueError("MATR data root is unset")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MATR data root is not a directory: {root}")
    return root

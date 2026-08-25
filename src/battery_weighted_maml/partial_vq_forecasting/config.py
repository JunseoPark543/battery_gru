"""Typed configuration for within-cycle partial V-Q forecasting."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from battery_weighted_maml.matr_anp.config import DataConfig, QGridConfig, SplitConfig


@dataclass
class PathsConfig:
    data_root: str | None = "data/MATR"
    output_root: str = "outputs/partial_vq_forecasting"


@dataclass
class EpisodeConfig:
    minimum_cycle_position: int = 10
    training_beta_range: list[float] = field(default_factory=lambda: [0.1, 0.8])
    evaluation_betas: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    evaluation_cycle_alphas: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    minimum_observed_points: int = 8
    minimum_future_points: int = 8


@dataclass
class ModelConfig:
    convolution_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    kernel_size: int = 5
    hidden_dim: int = 64
    attention_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 128
    decoder_hidden_dim: int = 128
    dropout: float = 0.1
    use_attention: bool = True


@dataclass
class TrainingConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    max_steps: int = 20_000
    batch_size: int = 16
    voltage_huber_delta: float = 0.5
    endpoint_huber_delta: float = 0.05
    endpoint_weight: float = 1.0
    observed_reconstruction_weight: float = 0.1
    monotonic_weight: float = 0.0
    gradient_clip_norm: float = 1.0
    validation_interval: int = 500
    checkpoint_interval: int = 500
    log_interval: int = 10
    early_stopping_patience: int = 12
    # This transformer is small enough that fp32 is the safer default. AMP can
    # be enabled explicitly; the trainer then skips/reduces-scale on overflow.
    use_amp: bool = False
    amp_initial_scale: float = 1024.0
    max_consecutive_amp_overflows: int = 8
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    plot_cells: int = 6
    dpi: int = 180


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    q_grid: QGridConfig = field(
        default_factory=lambda: QGridConfig(minimum=0.0, maximum=1.2, num_points=256)
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.data.dataset.upper() not in {"MATR", "CALCE", "HUST"}:
            raise ValueError("data.dataset must be MATR, CALCE, or HUST")
        if not self.data.file_globs:
            raise ValueError("data.file_globs cannot be empty")
        if self.data.minimum_discharge_points < 2:
            raise ValueError("minimum_discharge_points must be at least two")
        if not self.q_grid.minimum < self.q_grid.maximum or self.q_grid.num_points < 16:
            raise ValueError("q_grid requires minimum < maximum and at least 16 points")
        if self.split.strategy not in {"kfold", "loocv"}:
            raise ValueError("split.strategy must be kfold or loocv")
        if self.split.strategy == "kfold" and self.split.num_folds < 2:
            raise ValueError("split.num_folds must be at least two")
        if not 0.0 < self.split.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0,1)")
        episode = self.episode
        if episode.minimum_cycle_position < 1:
            raise ValueError("minimum_cycle_position must be positive")
        if (
            len(episode.training_beta_range) != 2
            or not 0.0 < episode.training_beta_range[0] < episode.training_beta_range[1] < 1.0
        ):
            raise ValueError("training_beta_range must contain 0 < low < high < 1")
        if any(not 0.0 < value < 1.0 for value in episode.evaluation_betas):
            raise ValueError("evaluation_betas must lie in (0,1)")
        if any(not 0.0 <= value <= 1.0 for value in episode.evaluation_cycle_alphas):
            raise ValueError("evaluation_cycle_alphas must lie in [0,1]")
        if episode.minimum_observed_points < 2 or episode.minimum_future_points < 2:
            raise ValueError("minimum observed/future points must be at least two")
        model = self.model
        if not model.convolution_channels or any(value <= 0 for value in model.convolution_channels):
            raise ValueError("convolution_channels must be positive")
        if model.kernel_size < 3 or model.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if model.hidden_dim <= 0 or model.hidden_dim % model.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if model.attention_layers < 1 or model.feedforward_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        if not 0.0 <= model.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        training = self.training
        positive = {
            "learning_rate": training.learning_rate,
            "max_steps": training.max_steps,
            "batch_size": training.batch_size,
            "voltage_huber_delta": training.voltage_huber_delta,
            "endpoint_huber_delta": training.endpoint_huber_delta,
            "gradient_clip_norm": training.gradient_clip_norm,
            "validation_interval": training.validation_interval,
            "checkpoint_interval": training.checkpoint_interval,
            "log_interval": training.log_interval,
            "early_stopping_patience": training.early_stopping_patience,
            "amp_initial_scale": training.amp_initial_scale,
            "max_consecutive_amp_overflows": training.max_consecutive_amp_overflows,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"training values must be positive: {positive}")
        if any(
            value < 0
            for value in (
                training.weight_decay,
                training.endpoint_weight,
                training.observed_reconstruction_weight,
                training.monotonic_weight,
            )
        ):
            raise ValueError("loss and optimizer weights cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    allowed = {
        "seed", "device", "paths", "data", "q_grid", "split", "episode",
        "model", "training", "evaluation",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    grid = dict(raw.get("q_grid", {}))
    for alias, canonical in (("min", "minimum"), ("max", "maximum")):
        if alias in grid:
            if canonical in grid:
                raise ValueError(f"q_grid cannot contain both {alias} and {canonical}")
            grid[canonical] = grid.pop(alias)
    config = ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        paths=PathsConfig(**raw.get("paths", {})),
        data=DataConfig(**raw.get("data", {})),
        q_grid=QGridConfig(**grid),
        split=SplitConfig(**raw.get("split", {})),
        episode=EpisodeConfig(**raw.get("episode", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    return config_from_dict(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def resolve_data_root(config: ExperimentConfig, cli_data_root: str | None) -> Path:
    raw = cli_data_root or os.environ.get("BATTERYLIFE_DATA_ROOT") or config.paths.data_root
    if not raw:
        raise ValueError("data root is unset; pass --data-root or set paths.data_root")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root is not a directory: {root}")
    return root

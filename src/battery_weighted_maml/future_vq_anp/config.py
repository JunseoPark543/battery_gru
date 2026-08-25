"""Typed configuration for completed-cycle V-Q surface forecasting."""

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
    output_root: str = "outputs/future_vq_anp"


@dataclass
class EpisodeConfig:
    history_cycles: int = 100
    minimum_future_cycles: int = 20
    maximum_training_future_cycles: int = 64
    training_cut_alpha_range: list[float] = field(default_factory=lambda: [0.2, 0.8])
    evaluation_cut_cycles: list[int] = field(default_factory=lambda: [100, 130, 200])
    minimum_q_points: int = 12
    cycle_scale: float = 1000.0


@dataclass
class ModelConfig:
    convolution_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    kernel_size: int = 5
    curve_embedding_dim: int = 64
    cycle_feature_dim: int = 96
    gru_hidden_dim: int = 128
    gru_layers: int = 2
    attention_heads: int = 4
    q_embedding_dim: int = 32
    latent_dim: int = 32
    latent_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    dropout: float = 0.1
    minimum_latent_std: float = 0.02
    minimum_voltage_std: float = 0.003
    minimum_endpoint_std: float = 0.002


@dataclass
class TrainingConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    max_steps: int = 25_000
    batch_size: int = 4
    voltage_huber_delta: float = 0.5
    voltage_huber_weight: float = 0.2
    endpoint_huber_delta: float = 0.03
    endpoint_weight: float = 0.5
    kl_weight: float = 0.01
    kl_warmup_steps: int = 5_000
    kl_free_bits: float = 0.02
    q_monotonic_weight: float = 0.01
    endpoint_monotonic_weight: float = 0.01
    temporal_smoothness_weight: float = 0.001
    gradient_clip_norm: float = 1.0
    validation_interval: int = 500
    validation_latent_samples: int = 10
    checkpoint_interval: int = 500
    log_interval: int = 10
    early_stopping_patience: int = 12
    selection_metric: str = "crps"
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    latent_samples: int = 50
    query_chunk_size: int = 16
    plot_cells: int = 4
    interval_level: float = 0.95
    dpi: int = 180


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    q_grid: QGridConfig = field(
        default_factory=lambda: QGridConfig(minimum=0.0, maximum=1.2, num_points=128)
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.data.dataset.upper() not in {"MATR", "CALCE", "HUST"}:
            raise ValueError("data.dataset must be MATR, CALCE, or HUST")
        if not self.data.file_globs or self.data.minimum_discharge_points < 2:
            raise ValueError("data file_globs and minimum_discharge_points are invalid")
        if not self.q_grid.minimum < self.q_grid.maximum or self.q_grid.num_points < 32:
            raise ValueError("q_grid requires minimum < maximum and at least 32 points")
        if self.split.strategy not in {"kfold", "loocv"}:
            raise ValueError("split.strategy must be kfold or loocv")
        if self.split.strategy == "kfold" and self.split.num_folds < 2:
            raise ValueError("split.num_folds must be at least two")
        if not 0.0 < self.split.validation_fraction < 1.0:
            raise ValueError("split.validation_fraction must lie in (0,1)")
        episode = self.episode
        if any(value <= 0 for value in (
            episode.history_cycles, episode.minimum_future_cycles,
            episode.maximum_training_future_cycles, episode.minimum_q_points,
            episode.cycle_scale,
        )):
            raise ValueError("episode counts and cycle_scale must be positive")
        if episode.maximum_training_future_cycles < 2:
            raise ValueError("maximum_training_future_cycles must be at least two")
        if (
            len(episode.training_cut_alpha_range) != 2
            or not 0.0 <= episode.training_cut_alpha_range[0]
            < episode.training_cut_alpha_range[1] <= 1.0
        ):
            raise ValueError("training_cut_alpha_range must contain 0 <= low < high <= 1")
        if any(cycle < episode.history_cycles for cycle in episode.evaluation_cut_cycles):
            raise ValueError("evaluation cut cycles cannot precede the required history")
        model = self.model
        dimensions = (
            model.curve_embedding_dim, model.cycle_feature_dim, model.gru_hidden_dim,
            model.gru_layers, model.attention_heads, model.q_embedding_dim,
            model.latent_dim, model.latent_hidden_dim, model.decoder_hidden_dim,
        )
        if not model.convolution_channels or any(value <= 0 for value in dimensions):
            raise ValueError("model dimensions must be positive")
        if any(value <= 0 for value in model.convolution_channels):
            raise ValueError("convolution channels must be positive")
        if model.kernel_size < 3 or model.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if model.gru_hidden_dim % model.attention_heads:
            raise ValueError("gru_hidden_dim must be divisible by attention_heads")
        if not 0.0 <= model.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if min(model.minimum_latent_std, model.minimum_voltage_std,
               model.minimum_endpoint_std) <= 0:
            raise ValueError("minimum standard deviations must be positive")
        training = self.training
        positive = (
            training.learning_rate, training.max_steps, training.batch_size,
            training.voltage_huber_delta, training.endpoint_huber_delta,
            training.kl_warmup_steps, training.gradient_clip_norm,
            training.validation_interval, training.validation_latent_samples,
            training.checkpoint_interval, training.log_interval,
            training.early_stopping_patience,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("positive training settings must be greater than zero")
        nonnegative = (
            training.weight_decay, training.voltage_huber_weight,
            training.endpoint_weight, training.kl_weight, training.kl_free_bits,
            training.q_monotonic_weight, training.endpoint_monotonic_weight,
            training.temporal_smoothness_weight,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("loss weights cannot be negative")
        if training.selection_metric not in {"crps", "rmse"}:
            raise ValueError("training.selection_metric must be crps or rmse")
        if self.evaluation.latent_samples <= 1 or self.evaluation.query_chunk_size <= 0:
            raise ValueError("evaluation latent samples/chunk size are invalid")
        if not 0.0 < self.evaluation.interval_level < 1.0:
            raise ValueError("evaluation interval_level must lie in (0,1)")

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
    try:
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
    except TypeError as exc:
        raise ValueError(f"invalid configuration field: {exc}") from exc
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"future V-Q config not found: {source}")
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

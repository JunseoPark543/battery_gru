"""Configuration for the latent ANP streaming SOH model."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from battery_weighted_maml.matr_anp.config import DataConfig, QGridConfig, SplitConfig
from battery_weighted_maml.streaming_soh.config import EpisodeConfig


@dataclass
class PathsConfig:
    data_root: str | None = "data/MATR"
    output_root: str = "outputs/streaming_soh_anp"


@dataclass
class ModelConfig:
    convolution_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    kernel_size: int = 5
    curve_embedding_dim: int = 64
    cycle_feature_dim: int = 96
    gru_hidden_dim: int = 128
    gru_layers: int = 2
    attention_heads: int = 4
    latent_dim: int = 32
    latent_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    dropout: float = 0.1
    minimum_latent_std: float = 0.02
    minimum_observation_std: float = 0.003


@dataclass
class TrainingConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    max_steps: int = 25_000
    batch_size: int = 8
    soh_huber_delta: float = 0.02
    soh_huber_weight: float = 0.2
    voltage_huber_delta: float = 0.5
    endpoint_huber_delta: float = 0.05
    kl_weight: float = 0.01
    kl_warmup_steps: int = 5_000
    kl_free_bits: float = 0.02
    voltage_completion_weight: float = 0.2
    endpoint_weight: float = 0.1
    monotonic_weight: float = 0.01
    gradient_clip_norm: float = 1.0
    validation_interval: int = 500
    validation_latent_samples: int = 20
    selection_metric: str = "crps"
    checkpoint_interval: int = 500
    log_interval: int = 10
    early_stopping_patience: int = 12
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    latent_samples: int = 50
    plot_cells: int = 6
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
        if not self.data.file_globs:
            raise ValueError("data.file_globs cannot be empty")
        if not self.q_grid.minimum < self.q_grid.maximum or self.q_grid.num_points < 32:
            raise ValueError("q_grid requires minimum < maximum and at least 32 points")
        if self.split.strategy not in {"kfold", "loocv"}:
            raise ValueError("split.strategy must be kfold or loocv")
        episode = self.episode
        if any(
            value <= 0
            for value in (
                episode.minimum_current_cycle,
                episode.minimum_history_cycles,
                episode.maximum_history_cycles,
                episode.minimum_future_cycles,
                episode.maximum_training_future_points,
                episode.minimum_observed_q_points,
                episode.minimum_future_q_points,
                episode.cycle_scale,
            )
        ):
            raise ValueError("episode counts and cycle_scale must be positive")
        if episode.maximum_history_cycles < episode.minimum_history_cycles:
            raise ValueError("maximum_history_cycles must be >= minimum_history_cycles")
        if (
            len(episode.training_cycle_alpha_range) != 2
            or not 0.0 <= episode.training_cycle_alpha_range[0]
            < episode.training_cycle_alpha_range[1] <= 1.0
        ):
            raise ValueError("training_cycle_alpha_range must contain 0 <= low < high <= 1")
        if (
            len(episode.training_beta_range) != 2
            or not 0.0 < episode.training_beta_range[0]
            < episode.training_beta_range[1] < 1.0
        ):
            raise ValueError("training_beta_range must contain 0 < low < high < 1")
        if any(not 0.0 <= value <= 1.0 for value in episode.evaluation_cycle_alphas):
            raise ValueError("evaluation_cycle_alphas must lie in [0,1]")
        if any(not 0.0 < value < 1.0 for value in episode.evaluation_betas):
            raise ValueError("evaluation_betas must lie in (0,1)")
        model = self.model
        if not model.convolution_channels or any(value <= 0 for value in model.convolution_channels):
            raise ValueError("convolution_channels must be positive")
        if model.kernel_size < 3 or model.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        dimensions = (
            model.curve_embedding_dim,
            model.cycle_feature_dim,
            model.gru_hidden_dim,
            model.gru_layers,
            model.attention_heads,
            model.latent_dim,
            model.latent_hidden_dim,
            model.decoder_hidden_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("model dimensions must be positive")
        if model.gru_hidden_dim % model.attention_heads:
            raise ValueError("gru_hidden_dim must be divisible by attention_heads")
        if model.minimum_latent_std <= 0 or model.minimum_observation_std <= 0:
            raise ValueError("minimum standard deviations must be positive")
        if not 0.0 <= model.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        training = self.training
        positive = (
            training.learning_rate,
            training.max_steps,
            training.batch_size,
            training.soh_huber_delta,
            training.voltage_huber_delta,
            training.endpoint_huber_delta,
            training.kl_warmup_steps,
            training.gradient_clip_norm,
            training.validation_interval,
            training.validation_latent_samples,
            training.checkpoint_interval,
            training.log_interval,
            training.early_stopping_patience,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("positive training values must be > 0")
        nonnegative = (
            training.weight_decay,
            training.soh_huber_weight,
            training.kl_weight,
            training.kl_free_bits,
            training.voltage_completion_weight,
            training.endpoint_weight,
            training.monotonic_weight,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("optimizer and loss weights cannot be negative")
        if training.selection_metric not in {"crps", "rmse"}:
            raise ValueError("training.selection_metric must be crps or rmse")
        if self.evaluation.latent_samples <= 1:
            raise ValueError("evaluation.latent_samples must be greater than one")
        if not 0.0 < self.evaluation.interval_level < 1.0:
            raise ValueError("interval_level must lie in (0,1)")

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

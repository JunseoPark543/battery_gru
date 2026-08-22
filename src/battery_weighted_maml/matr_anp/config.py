"""Typed configuration for BatteryLife-style partial I-V ANP experiments."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    data_root: str | None = "data/MATR"
    output_root: str = "outputs/matr_partial_iv_anp"


@dataclass
class DataConfig:
    dataset: str = "MATR"
    file_globs: list[str] = field(default_factory=lambda: ["**/*.pkl"])
    minimum_valid_cycles: int = 30
    minimum_discharge_points: int = 16
    short_signal_threshold: int = 32
    reference_cycles: list[int] = field(default_factory=lambda: [5, 6, 7, 8, 9, 10])
    minimum_reference_cycles: int = 3


@dataclass
class QGridConfig:
    minimum: float = 0.0
    maximum: float = 0.8
    num_points: int = 256


@dataclass
class SplitConfig:
    strategy: str = "kfold"
    num_folds: int = 5
    validation_fraction: float = 0.2
    seed: int = 42


@dataclass
class EpisodeConfig:
    minimum_current_cycle_position: int = 20
    training_alpha_range: list[float] = field(default_factory=lambda: [0.2, 0.8])
    evaluation_alphas: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    beta_values: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    min_context_points: int = 8
    max_context_points: int = 128
    max_target_points: int = 128


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    wide_hidden_min: int = 68
    wide_hidden_max: int = 192
    latent_dim: int = 32
    attention_heads: int = 4
    mlp_layers: int = 3
    iv_channels: list[int] = field(default_factory=lambda: [16, 32, 32])
    iv_embedding_dim: int = 32
    minimum_std: float = 1.0e-3


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 1.0e-4
    max_steps: int = 25_000
    batch_size: int = 16
    gradient_clip_norm: float = 1.0
    kl_warmup_steps: int = 5_000
    validation_interval: int = 500
    validation_episodes_per_cell: int = 3
    early_stopping_patience: int = 10
    checkpoint_interval: int = 500
    log_interval: int = 10
    use_amp: bool = True
    amp_initial_scale: float = 1024.0
    max_consecutive_amp_overflows: int = 20
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    mc_samples: int = 50
    interval_level: float = 0.95
    inference_repeats: int = 20
    inference_warmup: int = 3
    trajectory_plot_cell: str | None = None


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    q_grid: QGridConfig = field(default_factory=QGridConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.data.dataset.upper() not in {"MATR", "CALCE"}:
            raise ValueError("data.dataset must be MATR or CALCE")
        if not self.data.file_globs:
            raise ValueError("data.file_globs cannot be empty")
        if self.data.minimum_valid_cycles < 3:
            raise ValueError("data.minimum_valid_cycles must be at least 3")
        if self.data.minimum_discharge_points < 2:
            raise ValueError("data.minimum_discharge_points must be at least 2")
        if self.data.minimum_reference_cycles <= 0:
            raise ValueError("minimum_reference_cycles must be positive")
        if len(set(self.data.reference_cycles)) != len(self.data.reference_cycles):
            raise ValueError("reference_cycles must be unique")
        if any(cycle <= 0 for cycle in self.data.reference_cycles):
            raise ValueError("reference_cycles must be positive")
        grid = self.q_grid
        if not grid.minimum < grid.maximum or grid.num_points < 2:
            raise ValueError("q_grid requires minimum < maximum and at least two points")
        split = self.split
        if split.strategy not in {"kfold", "loocv"}:
            raise ValueError("split.strategy must be kfold or loocv")
        if split.strategy == "kfold" and split.num_folds < 2:
            raise ValueError("split.num_folds must be at least two")
        if not 0.0 < split.validation_fraction < 1.0:
            raise ValueError("split.validation_fraction must be in (0,1)")
        episode = self.episode
        if episode.minimum_current_cycle_position < 2:
            raise ValueError("minimum_current_cycle_position must be at least two")
        if (
            len(episode.training_alpha_range) != 2
            or not 0.0 < episode.training_alpha_range[0] < episode.training_alpha_range[1] < 1.0
        ):
            raise ValueError("training_alpha_range must contain 0 < low < high < 1")
        if not episode.beta_values or 0.0 not in episode.beta_values:
            raise ValueError("beta_values must include beta=0")
        if any(not 0.0 <= beta <= 1.0 for beta in episode.beta_values):
            raise ValueError("beta_values must lie in [0,1]")
        if any(not 0.0 < alpha < 1.0 for alpha in episode.evaluation_alphas):
            raise ValueError("evaluation_alphas must lie in (0,1)")
        if not 1 <= episode.min_context_points <= episode.max_context_points:
            raise ValueError("context point bounds are invalid")
        if episode.max_target_points < 2:
            raise ValueError("max_target_points must be at least two")
        model = self.model
        if model.hidden_dim <= 0 or model.latent_dim <= 0 or model.iv_embedding_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if model.hidden_dim % model.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if model.wide_hidden_min > model.wide_hidden_max:
            raise ValueError("wide hidden search range is invalid")
        if any(channel <= 0 for channel in model.iv_channels):
            raise ValueError("iv_channels must be positive")
        training = self.training
        positive = {
            "learning_rate": training.learning_rate,
            "max_steps": training.max_steps,
            "batch_size": training.batch_size,
            "gradient_clip_norm": training.gradient_clip_norm,
            "validation_interval": training.validation_interval,
            "validation_episodes_per_cell": training.validation_episodes_per_cell,
            "early_stopping_patience": training.early_stopping_patience,
            "checkpoint_interval": training.checkpoint_interval,
            "log_interval": training.log_interval,
            "amp_initial_scale": training.amp_initial_scale,
            "max_consecutive_amp_overflows": training.max_consecutive_amp_overflows,
            "mc_samples": self.evaluation.mc_samples,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"training/evaluation values must be positive: {positive}")
        if training.optimizer != "adam":
            raise ValueError("only Adam is implemented")
        if training.kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps cannot be negative")
        if not 0.0 < self.evaluation.interval_level < 1.0:
            raise ValueError("interval_level must be in (0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ANP config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {
        "seed", "device", "paths", "data", "q_grid", "split", "episode",
        "model", "training", "evaluation",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    q_grid_raw = dict(raw.get("q_grid", {}))
    for alias, canonical in (("min", "minimum"), ("max", "maximum")):
        if alias in q_grid_raw:
            if canonical in q_grid_raw:
                raise ValueError(
                    f"q_grid cannot contain both '{alias}' and '{canonical}'"
                )
            q_grid_raw[canonical] = q_grid_raw.pop(alias)
    try:
        config = ExperimentConfig(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            paths=PathsConfig(**raw.get("paths", {})),
            data=DataConfig(**raw.get("data", {})),
            q_grid=QGridConfig(**q_grid_raw),
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


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def resolve_data_root(config: ExperimentConfig, cli_data_root: str | None) -> Path:
    """Resolve CLI -> environment -> config without inventing a server path."""
    raw = cli_data_root or os.environ.get("BATTERYLIFE_DATA_ROOT") or config.paths.data_root
    if not raw:
        raise ValueError(
            f"{config.data.dataset.upper()} data root is unset; pass --data-root, "
            "set BATTERYLIFE_DATA_ROOT, "
            "or set paths.data_root in the config"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BatteryLife data root is not a directory: {root}")
    return root

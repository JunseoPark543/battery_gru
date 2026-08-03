"""Validated YAML configuration loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    calce_dir: str = "data/CALCE"
    label_path: str = "data/Life labels/CALCE_labels.json"
    eol_threshold: float = 0.8
    history_lengths: list[int] = field(default_factory=lambda: [50, 100, 150])
    max_forecast_cycle: int = 1000


@dataclass
class ModelConfig:
    input_size: int = 1
    features: list[str] = field(default_factory=lambda: ["soh"])
    voltage_normalization: str = "support_zscore"
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    teacher_forcing_ratio: float = 0.5


@dataclass
class MAMLConfig:
    full_maml: bool = True
    inner_steps: int = 1
    inner_lr: float = 0.05
    inner_batch_size: int = 64
    outer_lr: float = 0.001
    meta_iterations: int = 10000
    gradient_clip_norm: float = 5.0


@dataclass
class WeightConfig:
    method: str = "mmd_qp"
    kernel: str = "rbf"
    sigma: str | float = "median"
    qp_solver_primary: str = "OSQP"
    qp_solver_fallback: str = "SCS"
    diagonal_jitter: float = 1.0e-8
    recompute_every_iteration: bool = True
    detach_alpha: bool = True


@dataclass
class AdaptationConfig:
    fast_steps: int = 1
    full_max_steps: int = 200
    full_patience: int = 20
    learning_rate: float = 0.05


@dataclass
class LoggingConfig:
    log_interval: int = 10
    checkpoint_interval: int = 100
    save_alpha_interval: int = 10


@dataclass
class EvaluationConfig:
    primary_adaptation: str = "full"
    save_per_cycle_predictions: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_mode: str | None = None

    def validate(self) -> None:
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device")
        if not 0.0 < self.data.eol_threshold < 2.0:
            raise ValueError("data.eol_threshold must be between 0 and 2")
        if not self.data.history_lengths or any(x < 2 for x in self.data.history_lengths):
            raise ValueError("all data.history_lengths must be >= 2")
        if self.data.max_forecast_cycle <= max(self.data.history_lengths):
            raise ValueError("max_forecast_cycle must exceed every history length")
        allowed_features = {"soh", "voltage_mean"}
        if not self.model.features or self.model.features[0] != "soh":
            raise ValueError("model.features must start with 'soh'")
        if len(set(self.model.features)) != len(self.model.features):
            raise ValueError("model.features cannot contain duplicates")
        unknown_features = set(self.model.features) - allowed_features
        if unknown_features:
            raise ValueError(f"unsupported model.features: {sorted(unknown_features)}")
        if self.model.input_size != len(self.model.features):
            raise ValueError("model.input_size must equal len(model.features)")
        if self.model.voltage_normalization != "support_zscore":
            raise ValueError("only support_zscore voltage normalization is supported")
        if self.model.hidden_size <= 0 or self.model.num_layers <= 0:
            raise ValueError("model sizes must be positive")
        if not 0.0 <= self.model.teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        if self.maml.inner_steps <= 0 or self.maml.inner_batch_size <= 0:
            raise ValueError("MAML inner settings must be positive")
        if self.maml.inner_lr <= 0 or self.maml.outer_lr <= 0:
            raise ValueError("MAML learning rates must be positive")
        if self.maml.meta_iterations <= 0 or self.maml.gradient_clip_norm <= 0:
            raise ValueError("MAML iteration and clipping settings must be positive")
        if self.weights.method != "mmd_qp" or self.weights.kernel != "rbf":
            raise ValueError("only mmd_qp with an rbf kernel is supported")
        if isinstance(self.weights.sigma, str) and self.weights.sigma != "median":
            raise ValueError("weights.sigma must be 'median' or a positive number")
        if isinstance(self.weights.sigma, (int, float)) and self.weights.sigma <= 0:
            raise ValueError("fixed weights.sigma must be positive")
        if self.weights.diagonal_jitter < 0:
            raise ValueError("weights.diagonal_jitter cannot be negative")
        if not self.weights.detach_alpha:
            raise ValueError("detach_alpha must remain true for the QP weighting design")
        if self.adaptation.fast_steps <= 0 or self.adaptation.full_max_steps <= 0:
            raise ValueError("adaptation steps must be positive")
        if self.adaptation.full_patience <= 0 or self.adaptation.learning_rate <= 0:
            raise ValueError("adaptation patience and learning rate must be positive")
        if self.evaluation.primary_adaptation not in {"fast", "full"}:
            raise ValueError("primary_adaptation must be 'fast' or 'full'")
        if self.source_mode not in {None, "same_family", "all_calce"}:
            raise ValueError("source_mode must be same_family or all_calce")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _construct(raw: dict[str, Any]) -> ExperimentConfig:
    known = {
        "seed", "device", "data", "model", "maml", "weights", "adaptation",
        "logging", "evaluation", "source_mode",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    try:
        config = ExperimentConfig(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            maml=MAMLConfig(**raw.get("maml", {})),
            weights=WeightConfig(**raw.get("weights", {})),
            adaptation=AdaptationConfig(**raw.get("adaptation", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            source_mode=raw.get("source_mode"),
        )
    except TypeError as exc:
        raise ValueError(f"invalid config field: {exc}") from exc
    config.validate()
    return config


def load_config(path: str | Path, overlay: str | Path | None = None) -> ExperimentConfig:
    """Load and validate a base YAML file and an optional overlay."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    if overlay is not None:
        overlay_path = Path(overlay)
        extra = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
        if not isinstance(extra, dict):
            raise ValueError(f"configuration overlay must be a mapping: {overlay_path}")
        raw = _deep_merge(raw, extra)
    return _construct(raw)


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

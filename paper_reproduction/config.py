"""Typed configuration for the paper reproduction.

Values explicitly stated by the paper/request are marked ``paper``. Values
that were not reported and therefore remain implementation choices are marked
``implementation choice``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    calce_dir: str = "data/CALCE"
    output_dir: str = "outputs/paper_reproduction"


@dataclass
class DataConfig:
    history_length: int = 500  # paper
    train_cells: list[str] = field(default_factory=lambda: [
        "CALCE_CX2_16.pkl", "CALCE_CX2_33.pkl", "CALCE_CX2_34.pkl",
        "CALCE_CX2_35.pkl", "CALCE_CX2_36.pkl",
    ])
    test_cells: list[str] = field(default_factory=lambda: [
        "CALCE_CX2_37.pkl", "CALCE_CX2_38.pkl",
    ])


@dataclass
class ModelConfig:
    input_size: int = 1  # paper
    hidden_size: int = 64  # paper
    num_layers: int = 1  # paper
    dropout: float = 0.0  # paper
    predicted_input_probability: float = 0.5  # paper
    weight_initialization: str = "pytorch_default"  # implementation choice


@dataclass
class LossConfig:
    kind: str = "mse"  # implementation choice: paper denotes only L
    recursive_reduction: str = "sample_balanced"  # stabilization default


@dataclass
class MAMLConfig:
    max_epochs: int = 500  # paper
    meta_batch_size: int = 5  # paper: all five training cells
    inner_learning_rate: float = 5.0e-2  # paper
    inner_steps: int = 1  # paper
    inner_steps_candidates: list[int] = field(default_factory=lambda: [1, 3, 5])
    multi_step_query_weights: dict[int, float] = field(
        default_factory=lambda: {1: 1.0}
    )
    experiment_label: str = "paper_default"
    inner_batch_size: int = 64  # paper
    outer_learning_rate: float = 1.0e-3  # implementation choice
    query_batch_size: int = 1  # implementation choice; one full query per task
    gradient_clip_norm: float = 5.0  # implementation choice
    early_stopping: bool = True  # paper
    early_stopping_patience: int = 30  # implementation choice
    early_stopping_min_delta: float = 1.0e-7  # implementation choice
    log_interval: int = 1
    checkpoint_interval: int = 1
    optuna_trials: int = 0  # 0 disables the optional search
    optuna_lr_low: float = 1.0e-5  # implementation choice
    optuna_lr_high: float = 1.0e-2  # implementation choice


@dataclass
class AdaptationConfig:
    # Legacy field: when present in an old YAML, it overrides both new rates.
    learning_rate: float | None = None
    fast_learning_rate: float = 5.0e-2  # paper one-step adaptation
    complete_learning_rate: float = 5.0e-3  # stabilization default
    complete_learning_rate_candidates: list[float] = field(
        default_factory=lambda: [0.05, 0.01, 0.005, 0.001, 0.0005]
    )
    batch_size: int = 64  # paper fast-step definition
    fast_steps: list[int] = field(default_factory=lambda: [0, 1, 3, 5])  # paper
    complete_max_steps: int = 500
    complete_patience: int = 30
    complete_min_delta: float = 1.0e-6
    recursive_loss_reduction: str = "sample_balanced"
    fast_sampling_mode: str = "random"
    sampling_mode: str = "length_stratified"
    length_stratified_bins: int = 5
    checkpoint_selection: str = "support_recursive_validation"
    validation_ratio: float = 0.2
    minimum_validation_length: int = 20
    maximum_validation_length: int = 100
    scheduler: str = "constant"
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5
    scheduler_patience: int = 10
    gradient_clip_norm: float | None = 1.0
    gradient_clip_candidates: list[float | None] = field(
        default_factory=lambda: [None, 5.0, 1.0, 0.5]
    )
    relative_update_warning_threshold: float = 0.5
    diagnostics: bool = True
    oracle_diagnostics: bool = True

    def resolved_fast_learning_rate(self) -> float:
        return self.learning_rate if self.learning_rate is not None else self.fast_learning_rate

    def resolved_complete_learning_rate(self) -> float:
        return (
            self.learning_rate
            if self.learning_rate is not None
            else self.complete_learning_rate
        )


@dataclass
class EvaluationConfig:
    eol_threshold: float = 0.70  # paper
    forecast_mode: str = "paper"  # paper or deployment
    max_prediction_length: int | None = None  # deployment implementation choice
    # Legacy deployment boundary retained for old configs. Paper mode ignores it.
    max_forecast_cycle: int = 2000


@dataclass
class ExperimentConfig:
    seed: int = 42  # paper
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be auto, cpu, or a CUDA device")
        if self.data.history_length < 2:
            raise ValueError("history_length must be at least 2")
        if len(self.data.train_cells) != 5 or self.maml.meta_batch_size != 5:
            raise ValueError("the paper split requires five training tasks per meta-batch")
        if set(self.data.train_cells) & set(self.data.test_cells):
            raise ValueError("meta-training and meta-testing cells must be disjoint")
        if self.model.input_size != 1:
            raise ValueError("the paper model uses scalar SOH input_size=1")
        if self.model.hidden_size <= 0 or self.model.num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if self.model.dropout != 0.0:
            raise ValueError("the reproduced one-layer paper model uses dropout=0")
        if self.model.weight_initialization != "pytorch_default":
            raise ValueError("only pytorch_default initialization is implemented")
        if not 0.0 <= self.model.predicted_input_probability <= 1.0:
            raise ValueError("predicted_input_probability must be in [0,1]")
        if self.loss.kind not in {"mse", "mae"}:
            raise ValueError("loss.kind must be mse or mae")
        if self.loss.recursive_reduction not in {"point_balanced", "sample_balanced"}:
            raise ValueError("loss.recursive_reduction is invalid")
        positive = {
            "max_epochs": self.maml.max_epochs,
            "inner_learning_rate": self.maml.inner_learning_rate,
            "inner_steps": self.maml.inner_steps,
            "inner_batch_size": self.maml.inner_batch_size,
            "outer_learning_rate": self.maml.outer_learning_rate,
            "query_batch_size": self.maml.query_batch_size,
            "gradient_clip_norm": self.maml.gradient_clip_norm,
            "early_stopping_patience": self.maml.early_stopping_patience,
            "fast_learning_rate": self.adaptation.resolved_fast_learning_rate(),
            "complete_learning_rate": self.adaptation.resolved_complete_learning_rate(),
            "adaptation_batch_size": self.adaptation.batch_size,
            "complete_max_steps": self.adaptation.complete_max_steps,
            "complete_patience": self.adaptation.complete_patience,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"these values must be positive: {positive}")
        if self.maml.early_stopping_min_delta < 0 or self.adaptation.complete_min_delta < 0:
            raise ValueError("early-stopping min_delta values cannot be negative")
        if sorted(set(self.adaptation.fast_steps)) != sorted(self.adaptation.fast_steps):
            raise ValueError("fast_steps must be unique")
        if any(step < 0 for step in self.adaptation.fast_steps):
            raise ValueError("fast_steps cannot contain negative values")
        if self.maml.inner_steps not in self.maml.inner_steps_candidates:
            raise ValueError("maml.inner_steps must be listed in inner_steps_candidates")
        weights = {int(step): float(weight) for step, weight in self.maml.multi_step_query_weights.items()}
        if not weights or any(step <= 0 or step > self.maml.inner_steps for step in weights):
            raise ValueError("multi_step_query_weights steps must be within inner_steps")
        if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("multi-step query weights must be nonnegative with positive sum")
        self.maml.multi_step_query_weights = weights
        adaptation = self.adaptation
        if adaptation.recursive_loss_reduction not in {"point_balanced", "sample_balanced"}:
            raise ValueError("adaptation recursive_loss_reduction is invalid")
        if adaptation.fast_sampling_mode not in {"random", "length_stratified", "full_support"}:
            raise ValueError("adaptation.fast_sampling_mode is invalid")
        if adaptation.sampling_mode not in {"random", "length_stratified", "full_support"}:
            raise ValueError("adaptation.sampling_mode is invalid")
        if adaptation.checkpoint_selection != "support_recursive_validation":
            raise ValueError("only support_recursive_validation checkpoint selection is safe")
        if adaptation.scheduler not in {"constant", "step", "plateau"}:
            raise ValueError("adaptation.scheduler must be constant, step, or plateau")
        if not 0.0 < adaptation.validation_ratio < 1.0:
            raise ValueError("adaptation.validation_ratio must be in (0,1)")
        if adaptation.minimum_validation_length <= 0:
            raise ValueError("minimum_validation_length must be positive")
        if adaptation.maximum_validation_length < adaptation.minimum_validation_length:
            raise ValueError("maximum_validation_length must be >= minimum")
        if adaptation.length_stratified_bins <= 0:
            raise ValueError("length_stratified_bins must be positive")
        if adaptation.gradient_clip_norm is not None and adaptation.gradient_clip_norm <= 0:
            raise ValueError("adaptation.gradient_clip_norm must be positive or null")
        if not 0.0 < self.evaluation.eol_threshold < 1.0:
            raise ValueError("eol_threshold must be between 0 and 1")
        if self.evaluation.forecast_mode not in {"paper", "deployment"}:
            raise ValueError("evaluation.forecast_mode must be paper or deployment")
        if (
            self.evaluation.max_prediction_length is not None
            and self.evaluation.max_prediction_length <= 0
        ):
            raise ValueError("max_prediction_length must be positive when provided")
        if (
            self.evaluation.forecast_mode == "deployment"
            and self.evaluation.max_prediction_length is None
            and self.evaluation.max_forecast_cycle <= self.data.history_length
        ):
            raise ValueError("legacy deployment max_forecast_cycle must exceed history_length")
        if self.maml.optuna_trials < 0:
            raise ValueError("optuna_trials cannot be negative")
        if not 0 < self.maml.optuna_lr_low < self.maml.optuna_lr_high:
            raise ValueError("Optuna learning-rate bounds are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"configuration file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {"seed", "device", "paths", "data", "model", "loss", "maml", "adaptation", "evaluation"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    try:
        config = ExperimentConfig(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            paths=PathsConfig(**raw.get("paths", {})),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            loss=LossConfig(**raw.get("loss", {})),
            maml=MAMLConfig(**raw.get("maml", {})),
            adaptation=AdaptationConfig(**raw.get("adaptation", {})),
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

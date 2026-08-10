"""Configuration for the isolated HUST weighted-GRU experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from battery_weighted_maml.config import (
    AdaptationConfig,
    DataConfig as LegacyDataConfig,
    EvaluationConfig,
    ExperimentConfig as LegacyExperimentConfig,
    LoggingConfig,
    MAMLConfig,
    ModelConfig,
    WeightConfig,
)


SOURCE_MODES = ("same_protocol", "all_hust", "leave_protocol_out")


@dataclass
class HUSTDataConfig:
    hust_dir: str = "data/HUST"
    label_path: str = "data/Life labels/HUST_labels.json"
    eol_threshold: float = 0.8
    history_lengths: list[int] = field(default_factory=lambda: [100])
    # None means cycle 101 through the cell's final observed cycle.
    max_forecast_cycle: int | None = None
    expected_protocol_count: int = 10


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    data: HUSTDataConfig = field(default_factory=HUSTDataConfig)
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            input_size=3,
            features=["soh", "voltage_mean", "current_mean"],
        )
    )
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_mode: str | None = "same_protocol"

    def validate(self) -> None:
        """Reuse the proven GRU/MAML validation, then validate HUST fields."""
        proxy = LegacyExperimentConfig(
            seed=self.seed,
            device=self.device,
            data=LegacyDataConfig(
                calce_dir=self.data.hust_dir,
                label_path=self.data.label_path,
                eol_threshold=self.data.eol_threshold,
                history_lengths=list(self.data.history_lengths),
                max_forecast_cycle=self.data.max_forecast_cycle,
            ),
            model=self.model,
            maml=self.maml,
            weights=self.weights,
            adaptation=self.adaptation,
            logging=self.logging,
            evaluation=self.evaluation,
            source_mode=None,
        )
        proxy.validate()
        # Legacy validation normalizes scalar fast_steps to a list.
        self.adaptation.fast_steps = proxy.adaptation.fast_steps
        if self.data.history_lengths != [100]:
            raise ValueError(
                "this experiment is intentionally fixed to data.history_lengths: [100]"
            )
        if self.data.expected_protocol_count <= 0:
            raise ValueError("data.expected_protocol_count must be positive")
        if self.source_mode not in (None, *SOURCE_MODES):
            raise ValueError(f"source_mode must be one of {SOURCE_MODES}")

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
        "seed",
        "device",
        "data",
        "model",
        "maml",
        "weights",
        "adaptation",
        "logging",
        "evaluation",
        "source_mode",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    try:
        config = ExperimentConfig(
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            data=HUSTDataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            maml=MAMLConfig(**raw.get("maml", {})),
            weights=WeightConfig(**raw.get("weights", {})),
            adaptation=AdaptationConfig(**raw.get("adaptation", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            source_mode=raw.get("source_mode", "same_protocol"),
        )
    except TypeError as exc:
        raise ValueError(f"invalid config field: {exc}") from exc
    config.validate()
    return config


def load_config(path: str | Path, overlay: str | Path | None = None) -> ExperimentConfig:
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


"""Independent CALCE loader and leakage-safe first-100-cycle features."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import DOMAIN_NAMES, FEATURE_NAMES, DataConfig


@dataclass(frozen=True)
class CellSample:
    file_name: str
    cell_id: str
    family: str
    domain: str
    discharge_rate_c: float
    nominal_capacity_ah: float
    features: np.ndarray  # [100, 7], unnormalized
    eol_cycle: int
    rul_cycles: float

    def summary(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "cell_id": self.cell_id,
            "family": self.family,
            "domain": self.domain,
            "discharge_rate_c": self.discharge_rate_c,
            "nominal_capacity_ah": self.nominal_capacity_ah,
            "history_length": int(self.features.shape[0]),
            "feature_count": int(self.features.shape[1]),
            "eol_cycle": self.eol_cycle,
            "rul_at_history_end": self.rul_cycles,
        }


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit(cls, samples: Sequence[CellSample], epsilon: float) -> "FeatureNormalizer":
        if not samples:
            raise ValueError("cannot fit feature normalization without source samples")
        stacked = np.concatenate([sample.features for sample in samples], axis=0)
        mean = stacked.mean(axis=0, dtype=np.float64)
        std = stacked.std(axis=0, dtype=np.float64)
        std = np.where(std < epsilon, 1.0, std)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise FloatingPointError("non-finite source feature normalization statistics")
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError("feature dimension does not match the fitted normalizer")
        result = (array - self.mean) / self.std
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("normalized features contain non-finite values")
        return result.astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureNormalizer":
        names = tuple(str(name) for name in state["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError("checkpoint feature names do not match this project")
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            feature_names=names,
        )


@dataclass(frozen=True)
class RULNormalizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, samples: Sequence[CellSample], epsilon: float) -> "RULNormalizer":
        if not samples:
            raise ValueError("cannot fit RUL normalization without source samples")
        labels = np.asarray([sample.rul_cycles for sample in samples], dtype=np.float64)
        mean = float(labels.mean())
        std = float(labels.std())
        if std < epsilon:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray | float) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RULNormalizer":
        return cls(mean=float(state["mean"]), std=float(state["std"]))


def _flatten_numeric(value: Any) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, (pd.Series, pd.Index)):
        raw = value.to_numpy().reshape(-1)
    else:
        raw = np.asarray(value if value is not None else [], dtype=object).reshape(-1)
    return pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)


def _last_finite(value: Any) -> float:
    array = _flatten_numeric(value)
    finite = array[np.isfinite(array)]
    return float(finite[-1]) if finite.size else float("nan")


def _stage_duration_hours(time_s: np.ndarray, mask: np.ndarray) -> float:
    selected = time_s[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size < 2:
        return float("nan")
    duration = float(selected.max() - selected.min()) / 3600.0
    return duration if duration >= 0.0 else float("nan")


def _stage_mean(values: np.ndarray, mask: np.ndarray, absolute: bool = False) -> float:
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return float("nan")
    if absolute:
        selected = np.abs(selected)
    return float(selected.mean())


def _cycle_features(
    record: Mapping[str, Any], nominal_capacity_ah: float, threshold_c_rate: float
) -> np.ndarray:
    capacity = _last_finite(record.get("discharge_capacity_in_Ah", []))
    soh = capacity / nominal_capacity_ah
    current = _flatten_numeric(record.get("current_in_A", []))
    voltage = _flatten_numeric(record.get("voltage_in_V", []))
    time_s = _flatten_numeric(record.get("time_in_s", []))
    length = min(len(current), len(voltage), len(time_s))
    if length == 0:
        return np.asarray([soh] + [float("nan")] * 6, dtype=np.float64)
    current = current[:length]
    voltage = voltage[:length]
    time_s = time_s[:length]
    aligned = np.isfinite(current) & np.isfinite(voltage) & np.isfinite(time_s)
    threshold_a = nominal_capacity_ah * threshold_c_rate
    charge = aligned & (current > threshold_a)
    discharge = aligned & (current < -threshold_a)
    return np.asarray(
        [
            soh,
            _stage_duration_hours(time_s, charge),
            _stage_duration_hours(time_s, discharge),
            _stage_mean(voltage, charge),
            _stage_mean(voltage, discharge),
            _stage_mean(current, charge) / nominal_capacity_ah,
            _stage_mean(current, discharge, absolute=True) / nominal_capacity_ah,
        ],
        dtype=np.float64,
    )


def _records(value: Any, file_name: str) -> list[Mapping[str, Any]]:
    if isinstance(value, pd.DataFrame):
        raw = value.to_dict(orient="records")
    elif isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple, pd.Series)):
        raw = list(value)
    else:
        raise ValueError(f"{file_name}: cycle_data has unsupported type {type(value).__name__}")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, pd.Series):
            item = item.to_dict()
        if not isinstance(item, Mapping):
            raise ValueError(f"{file_name}: cycle_data[{index}] is not a mapping")
        records.append(item)
    return records


def _family(file_name: str) -> str:
    match = re.fullmatch(r"CALCE_(CS2|CX2)_\d+\.pkl", file_name)
    if match is None:
        raise ValueError(f"cannot infer CALCE family from filename: {file_name}")
    return match.group(1)


def _infer_domain(
    family: str, features: np.ndarray, tolerance: float
) -> tuple[str, float]:
    measured = float(np.median(features[:, FEATURE_NAMES.index("discharge_c_rate")]))
    candidates = np.asarray([0.5, 1.0], dtype=np.float64)
    nearest = float(candidates[np.argmin(np.abs(candidates - measured))])
    if abs(measured - nearest) > tolerance:
        raise ValueError(
            f"discharge rate {measured:.4f}C is not within {tolerance}C of 0.5C or 1C"
        )
    domain = f"{family}_{nearest:.1f}C"
    if domain not in DOMAIN_NAMES:
        raise ValueError(f"unsupported inferred domain: {domain}")
    return domain, measured


def _load_labels(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"CALCE life-label file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("CALCE life-label JSON must contain an object")
    labels: dict[str, int] = {}
    for name, value in payload.items():
        label = int(value)
        if label <= 0:
            raise ValueError(f"invalid EOL label for {name}: {label}")
        labels[str(name)] = label
    return labels


def load_cell(path: Path, eol_cycle: int, config: DataConfig) -> CellSample:
    # CALCE pickle files are trusted local experiment inputs. Loading arbitrary
    # third-party pickle files is unsafe and intentionally unsupported.
    try:
        with path.open("rb") as handle:
            root = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"could not load trusted CALCE pickle {path.name}: {exc}") from exc
    if not isinstance(root, Mapping):
        raise ValueError(f"{path.name}: pickle root must be a mapping")
    required = {"cell_id", "nominal_capacity_in_Ah", "cycle_data"}
    missing = required - set(root)
    if missing:
        raise ValueError(f"{path.name}: missing root keys {sorted(missing)}")
    nominal = float(root["nominal_capacity_in_Ah"])
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(f"{path.name}: invalid nominal capacity {nominal}")
    by_cycle: dict[int, np.ndarray] = {}
    for index, record in enumerate(_records(root["cycle_data"], path.name)):
        try:
            cycle_number = float(record["cycle_number"])
            cycle = int(cycle_number)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: invalid cycle number at record {index}") from exc
        if cycle <= 0 or cycle_number != cycle:
            raise ValueError(f"{path.name}: cycle number must be a positive integer")
        if cycle <= config.history_length:
            by_cycle[cycle] = _cycle_features(
                record, nominal, config.current_threshold_c_rate
            )
    grid = np.arange(1, config.history_length + 1, dtype=np.int64)
    frame = pd.DataFrame.from_dict(
        {cycle: values for cycle, values in by_cycle.items()},
        orient="index",
        columns=FEATURE_NAMES,
    ).reindex(grid)
    if frame.empty or frame.notna().sum().min() == 0:
        raise ValueError(f"{path.name}: first 100 cycles do not contain every feature")
    frame = frame.interpolate(method="linear", limit_direction="both")
    if frame.isna().any().any():
        missing_columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"{path.name}: unresolved feature gaps in {missing_columns}")
    features = frame.to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError(f"{path.name}: non-finite first-100-cycle features")
    family = _family(path.name)
    domain, measured_rate = _infer_domain(
        family, features, config.domain_rate_tolerance
    )
    rul = float(eol_cycle - config.history_length)
    if rul <= 0:
        raise ValueError(
            f"{path.name}: EOL={eol_cycle} must exceed history={config.history_length}"
        )
    return CellSample(
        file_name=path.name,
        cell_id=str(root["cell_id"]),
        family=family,
        domain=domain,
        discharge_rate_c=measured_rate,
        nominal_capacity_ah=nominal,
        features=features,
        eol_cycle=int(eol_cycle),
        rul_cycles=rul,
    )


def load_calce_samples(
    calce_dir: str | Path, label_path: str | Path, config: DataConfig
) -> list[CellSample]:
    root = Path(calce_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"CALCE directory not found: {root}")
    labels = _load_labels(Path(label_path))
    samples: list[CellSample] = []
    for name in sorted(labels):
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f"labelled CALCE cell is missing: {source}")
        samples.append(load_cell(source, labels[name], config))
    actual_names = {sample.file_name for sample in samples}
    if actual_names != set(labels):
        raise RuntimeError("loaded CALCE sample names differ from life-label names")
    domains = {sample.domain for sample in samples}
    if domains != set(DOMAIN_NAMES):
        raise ValueError(
            f"expected four CALCE family/rate domains {DOMAIN_NAMES}, found {sorted(domains)}"
        )
    for domain in DOMAIN_NAMES:
        count = sum(sample.domain == domain for sample in samples)
        if count < 2:
            raise ValueError(f"domain {domain} needs at least two cells, found {count}")
    return samples


def split_domain(
    samples: Sequence[CellSample], held_out_domain: str
) -> tuple[list[CellSample], list[CellSample]]:
    if held_out_domain not in DOMAIN_NAMES:
        raise ValueError(f"unknown held-out domain: {held_out_domain}")
    source = [sample for sample in samples if sample.domain != held_out_domain]
    target = [sample for sample in samples if sample.domain == held_out_domain]
    if not source or not target:
        raise ValueError("domain split produced an empty source or target partition")
    if any(sample.domain == held_out_domain for sample in source):
        raise RuntimeError("held-out target domain leaked into source samples")
    return source, target

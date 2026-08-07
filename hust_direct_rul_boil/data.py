"""HUST loader with cycle-10-referenced charge profiles and scalar features."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import DataConfig, PROFILE_CHANNELS, SCALAR_FEATURES


@dataclass(frozen=True)
class CellSample:
    file_name: str
    cell_id: str
    protocol: str
    replicate: int
    nominal_capacity_ah: float
    waveforms: np.ndarray  # [100, points, 8]
    scalars: np.ndarray  # [100, 14]
    eol_cycle: int
    rul_cycles: float

    def summary(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "cell_id": self.cell_id,
            "protocol": self.protocol,
            "replicate": self.replicate,
            "nominal_capacity_ah": self.nominal_capacity_ah,
            "history_length": int(self.waveforms.shape[0]),
            "waveform_points": int(self.waveforms.shape[1]),
            "waveform_channels": int(self.waveforms.shape[2]),
            "scalar_features": int(self.scalars.shape[1]),
            "eol_cycle": self.eol_cycle,
            "rul_at_cycle_100": self.rul_cycles,
        }


@dataclass(frozen=True)
class InputNormalizer:
    waveform_mean: np.ndarray
    waveform_std: np.ndarray
    scalar_mean: np.ndarray
    scalar_std: np.ndarray

    @classmethod
    def fit(
        cls, samples: Sequence[CellSample], epsilon: float
    ) -> "InputNormalizer":
        if not samples:
            raise ValueError("cannot fit normalization without source cells")
        waveforms = np.concatenate([sample.waveforms for sample in samples], axis=0)
        scalars = np.concatenate([sample.scalars for sample in samples], axis=0)
        waveform_mean = waveforms.mean(axis=(0, 1), dtype=np.float64)
        waveform_std = waveforms.std(axis=(0, 1), dtype=np.float64)
        scalar_mean = scalars.mean(axis=0, dtype=np.float64)
        scalar_std = scalars.std(axis=0, dtype=np.float64)
        waveform_std = np.where(waveform_std < epsilon, 1.0, waveform_std)
        scalar_std = np.where(scalar_std < epsilon, 1.0, scalar_std)
        values = (waveform_mean, waveform_std, scalar_mean, scalar_std)
        if any(not np.all(np.isfinite(value)) for value in values):
            raise FloatingPointError("source normalization contains non-finite values")
        return cls(*(value.astype(np.float32) for value in values))

    def transform_waveforms(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(PROFILE_CHANNELS):
            raise ValueError("waveform channel count does not match normalizer")
        output = (array - self.waveform_mean) / self.waveform_std
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("normalized waveform contains non-finite values")
        return output.astype(np.float32, copy=False)

    def transform_scalars(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(SCALAR_FEATURES):
            raise ValueError("scalar feature count does not match normalizer")
        output = (array - self.scalar_mean) / self.scalar_std
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("normalized scalar features contain non-finite values")
        return output.astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "profile_channels": list(PROFILE_CHANNELS),
            "scalar_features": list(SCALAR_FEATURES),
            "waveform_mean": self.waveform_mean.tolist(),
            "waveform_std": self.waveform_std.tolist(),
            "scalar_mean": self.scalar_mean.tolist(),
            "scalar_std": self.scalar_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "InputNormalizer":
        if tuple(state["profile_channels"]) != PROFILE_CHANNELS:
            raise ValueError("checkpoint waveform channels do not match")
        if tuple(state["scalar_features"]) != SCALAR_FEATURES:
            raise ValueError("checkpoint scalar features do not match")
        return cls(
            waveform_mean=np.asarray(state["waveform_mean"], dtype=np.float32),
            waveform_std=np.asarray(state["waveform_std"], dtype=np.float32),
            scalar_mean=np.asarray(state["scalar_mean"], dtype=np.float32),
            scalar_std=np.asarray(state["scalar_std"], dtype=np.float32),
        )


def _numeric(value: Any) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, (pd.Series, pd.Index)):
        raw = value.to_numpy().reshape(-1)
    else:
        raw = np.asarray(value if value is not None else [], dtype=object).reshape(-1)
    return pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)


def _resize(values: Any, length: int) -> np.ndarray:
    raw = _numeric(values)
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    if raw.size == length:
        return raw
    finite = np.isfinite(raw)
    if not finite.any():
        return np.full(length, np.nan, dtype=np.float64)
    if raw.size == 1:
        return np.full(length, raw[0], dtype=np.float64)
    source_x = np.linspace(0.0, 1.0, raw.size)
    target_x = np.linspace(0.0, 1.0, length)
    return np.interp(target_x, source_x[finite], raw[finite])


def _last_finite(values: Any) -> float:
    raw = _numeric(values)
    finite = raw[np.isfinite(raw)]
    return float(finite[-1]) if finite.size else float("nan")


def _phase_variation(capacity: np.ndarray, mask: np.ndarray) -> float:
    selected = capacity[mask & np.isfinite(capacity)]
    if selected.size < 2:
        return -1.0
    return float(np.ptp(selected))


def _phase_masks(
    current: np.ndarray,
    charge_capacity: np.ndarray,
    discharge_capacity: np.ndarray,
    threshold_a: float,
    minimum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(current)
    positive = finite & (current > threshold_a)
    negative = finite & (current < -threshold_a)
    if positive.sum() < minimum_points and negative.sum() < minimum_points:
        return positive, negative
    charge_positive = _phase_variation(charge_capacity, positive)
    charge_negative = _phase_variation(charge_capacity, negative)
    discharge_positive = _phase_variation(discharge_capacity, positive)
    discharge_negative = _phase_variation(discharge_capacity, negative)
    charge = positive if charge_positive >= charge_negative else negative
    discharge = negative if discharge_negative >= discharge_positive else positive
    if np.array_equal(charge, discharge):
        discharge = negative if np.array_equal(charge, positive) else positive
    return charge, discharge


def _duration_hours(time_s: np.ndarray, mask: np.ndarray) -> float:
    selected = time_s[mask & np.isfinite(time_s)]
    if selected.size < 2:
        return float("nan")
    return max(0.0, float(selected.max() - selected.min()) / 3600.0)


def _mean(values: np.ndarray, mask: np.ndarray, absolute: bool = False) -> float:
    selected = values[mask & np.isfinite(values)]
    if not selected.size:
        return float("nan")
    if absolute:
        selected = np.abs(selected)
    return float(selected.mean())


def _energy_per_nominal(
    time_s: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    mask: np.ndarray,
    nominal_capacity_ah: float,
) -> float:
    valid = mask & np.isfinite(time_s) & np.isfinite(current) & np.isfinite(voltage)
    if valid.sum() < 2:
        return float("nan")
    selected_time = time_s[valid]
    order = np.argsort(selected_time)
    selected_time = selected_time[order]
    power = np.abs(current[valid][order]) * voltage[valid][order]
    energy_wh = float(np.trapezoid(power, selected_time) / 3600.0)
    return energy_wh / nominal_capacity_ah


def _resample_profile(
    time_s: np.ndarray,
    values: Sequence[np.ndarray],
    mask: np.ndarray,
    points: int,
) -> np.ndarray:
    valid = mask & np.isfinite(time_s)
    for value in values:
        valid &= np.isfinite(value)
    if valid.sum() < 2:
        return np.full((points, len(values)), np.nan, dtype=np.float64)
    time = time_s[valid]
    order = np.argsort(time)
    time = time[order]
    unique_time, unique_index = np.unique(time, return_index=True)
    if unique_time.size < 2:
        return np.full((points, len(values)), np.nan, dtype=np.float64)
    relative = (unique_time - unique_time[0]) / (unique_time[-1] - unique_time[0])
    grid = np.linspace(0.0, 1.0, points)
    columns = []
    for value in values:
        selected = value[valid][order][unique_index]
        columns.append(np.interp(grid, relative, selected))
    return np.stack(columns, axis=-1)


def _integrated_capacity_fraction(
    time_s: np.ndarray,
    current: np.ndarray,
    nominal_capacity_ah: float,
) -> np.ndarray:
    if time_s.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(time_s)
    sorted_time = time_s[order]
    sorted_current = np.abs(current[order])
    dt = np.diff(sorted_time, prepend=sorted_time[0])
    increment = sorted_current * np.maximum(dt, 0.0) / 3600.0
    cumulative = np.cumsum(increment) / nominal_capacity_ah
    output = np.empty_like(cumulative)
    output[order] = cumulative
    return output


def _cycle_base_features(
    record: Mapping[str, Any], nominal: float, config: DataConfig
) -> tuple[np.ndarray, np.ndarray]:
    current_raw = _numeric(record.get("current_in_A", []))
    voltage_raw = _numeric(record.get("voltage_in_V", []))
    time_raw = _numeric(record.get("time_in_s", []))
    length = min(current_raw.size, voltage_raw.size, time_raw.size)
    if length < 2:
        return (
            np.full((config.waveform_points, 4), np.nan, dtype=np.float64),
            np.full(10, np.nan, dtype=np.float64),
        )
    current = current_raw[:length]
    voltage = voltage_raw[:length]
    time_s = time_raw[:length]
    charge_capacity = _resize(record.get("charge_capacity_in_Ah", []), length)
    discharge_capacity = _resize(record.get("discharge_capacity_in_Ah", []), length)
    charge, discharge = _phase_masks(
        current,
        charge_capacity,
        discharge_capacity,
        nominal * config.current_threshold_c_rate,
        config.minimum_phase_points,
    )
    if charge.sum() < config.minimum_phase_points:
        waveform = np.full((config.waveform_points, 4), np.nan, dtype=np.float64)
    else:
        charge_fraction = charge_capacity / nominal
        observed_charge_fraction = charge_fraction[charge]
        finite_charge_fraction = observed_charge_fraction[
            np.isfinite(observed_charge_fraction)
        ]
        if (
            finite_charge_fraction.size < config.minimum_phase_points
            or np.ptp(finite_charge_fraction) < config.normalization_epsilon
        ):
            charge_fraction = _integrated_capacity_fraction(time_s, current, nominal)
        current_c = np.abs(current) / nominal
        power_vc = voltage * current_c
        waveform = _resample_profile(
            time_s,
            (voltage, current_c, charge_fraction, power_vc),
            charge,
            config.waveform_points,
        )
    final_discharge = _last_finite(record.get("discharge_capacity_in_Ah", []))
    final_charge = _last_finite(record.get("charge_capacity_in_Ah", []))
    soh = final_discharge / nominal
    efficiency = final_discharge / final_charge if final_charge > 0 else float("nan")
    scalars = np.asarray(
        [
            soh,
            _duration_hours(time_s, charge),
            _duration_hours(time_s, discharge),
            _mean(voltage, charge),
            _mean(voltage, discharge),
            _mean(current, charge, absolute=True) / nominal,
            _mean(current, discharge, absolute=True) / nominal,
            efficiency,
            _energy_per_nominal(time_s, current, voltage, charge, nominal),
            _energy_per_nominal(time_s, current, voltage, discharge, nominal),
        ],
        dtype=np.float64,
    )
    return waveform, scalars


def _interpolate_cycles(values: np.ndarray, file_name: str, kind: str) -> np.ndarray:
    original_shape = values.shape
    frame = pd.DataFrame(values.reshape(values.shape[0], -1))
    if frame.notna().sum().min() == 0:
        missing = np.flatnonzero(frame.notna().sum().to_numpy() == 0).tolist()
        raise ValueError(f"{file_name}: {kind} columns never observed: {missing[:10]}")
    frame = frame.interpolate(method="linear", limit_direction="both")
    if frame.isna().any().any():
        raise ValueError(f"{file_name}: unresolved missing values in {kind}")
    output = frame.to_numpy(dtype=np.float64).reshape(original_shape)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError(f"{file_name}: non-finite interpolated {kind}")
    return output


def _rolling_slope(values: np.ndarray, window: int = 10) -> np.ndarray:
    slopes = np.zeros_like(values, dtype=np.float64)
    for end in range(1, len(values)):
        start = max(0, end - window + 1)
        y = values[start : end + 1]
        x = np.arange(start, end + 1, dtype=np.float64)
        centered = x - x.mean()
        denominator = float(np.sum(np.square(centered)))
        slopes[end] = 0.0 if denominator == 0 else float(
            np.sum(centered * (y - y.mean())) / denominator
        )
    return slopes


def _records(value: Any, file_name: str) -> list[Mapping[str, Any]]:
    if isinstance(value, pd.DataFrame):
        raw = value.to_dict(orient="records")
    elif isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple, pd.Series)):
        raw = list(value)
    else:
        raise ValueError(f"{file_name}: unsupported cycle_data type {type(value).__name__}")
    output: list[Mapping[str, Any]] = []
    for index, record in enumerate(raw):
        if isinstance(record, pd.Series):
            record = record.to_dict()
        if not isinstance(record, Mapping):
            raise ValueError(f"{file_name}: cycle_data[{index}] is not a mapping")
        output.append(record)
    return output


def parse_protocol(file_name: str) -> tuple[str, int]:
    match = re.fullmatch(r"HUST_(\d+)-(\d+)\.pkl", file_name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"HUST filename must match HUST_<protocol>-<replicate>.pkl: {file_name}"
        )
    return f"protocol_{int(match.group(1))}", int(match.group(2))


def protocol_sort_key(protocol: str) -> int:
    match = re.fullmatch(r"protocol_(\d+)", protocol)
    if match is None:
        raise ValueError(f"invalid protocol ID: {protocol}")
    return int(match.group(1))


def load_cell(path: Path, eol_cycle: int, config: DataConfig) -> CellSample:
    # Only trusted BatteryLife/BatteryML pickle inputs should be loaded.
    try:
        with path.open("rb") as handle:
            root = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"could not load trusted HUST pickle {path.name}: {exc}") from exc
    if not isinstance(root, Mapping):
        raise ValueError(f"{path.name}: pickle root must be a mapping")
    required = {"cell_id", "nominal_capacity_in_Ah", "cycle_data"}
    missing = required - set(root)
    if missing:
        raise ValueError(f"{path.name}: missing root keys {sorted(missing)}")
    nominal = float(root["nominal_capacity_in_Ah"])
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(f"{path.name}: invalid nominal capacity {nominal}")
    by_cycle: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, record in enumerate(_records(root["cycle_data"], path.name)):
        try:
            cycle_float = float(record["cycle_number"])
            cycle = int(cycle_float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: invalid cycle number at record {index}") from exc
        if cycle <= 0 or cycle_float != cycle:
            raise ValueError(f"{path.name}: cycle number must be a positive integer")
        if cycle <= config.history_length:
            by_cycle[cycle] = _cycle_base_features(record, nominal, config)
    base_waveforms = np.full(
        (config.history_length, config.waveform_points, 4), np.nan, dtype=np.float64
    )
    base_scalars = np.full((config.history_length, 10), np.nan, dtype=np.float64)
    for cycle, (waveform, scalars) in by_cycle.items():
        base_waveforms[cycle - 1] = waveform
        base_scalars[cycle - 1] = scalars
    base_waveforms = _interpolate_cycles(base_waveforms, path.name, "waveform")
    base_scalars = _interpolate_cycles(base_scalars, path.name, "scalar feature")
    reference_index = config.reference_cycle - 1
    reference_waveform = base_waveforms[reference_index]
    waveform_delta = base_waveforms - reference_waveform[None, :, :]
    waveforms = np.concatenate((base_waveforms, waveform_delta), axis=-1)
    soh = base_scalars[:, 0]
    reference_scalars = base_scalars[reference_index]
    derived = np.stack(
        (
            soh - reference_scalars[0],
            base_scalars[:, 1] - reference_scalars[1],
            base_scalars[:, 2] - reference_scalars[2],
            _rolling_slope(soh, window=10),
        ),
        axis=-1,
    )
    scalars = np.concatenate((base_scalars, derived), axis=-1)
    if waveforms.shape[-1] != len(PROFILE_CHANNELS):
        raise RuntimeError("internal waveform channel definition mismatch")
    if scalars.shape[-1] != len(SCALAR_FEATURES):
        raise RuntimeError("internal scalar feature definition mismatch")
    protocol, replicate = parse_protocol(path.name)
    rul = float(eol_cycle - config.history_length)
    if rul <= 0:
        raise ValueError(f"{path.name}: EOL={eol_cycle} must exceed cycle 100")
    return CellSample(
        file_name=path.name,
        cell_id=str(root["cell_id"]),
        protocol=protocol,
        replicate=replicate,
        nominal_capacity_ah=nominal,
        waveforms=waveforms.astype(np.float32),
        scalars=scalars.astype(np.float32),
        eol_cycle=int(eol_cycle),
        rul_cycles=rul,
    )


def _load_labels(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"HUST life-label file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("HUST life-label JSON must contain an object")
    labels: dict[str, int] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name)
        if not name.lower().endswith(".pkl"):
            name += ".pkl"
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"invalid HUST EOL label for {name}: {value}")
        labels[name] = value
    return labels


def load_hust_samples(
    hust_dir: str | Path, label_path: str | Path, config: DataConfig
) -> list[CellSample]:
    root = Path(hust_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"HUST directory not found: {root}")
    labels = _load_labels(Path(label_path))
    samples: list[CellSample] = []
    for name in sorted(labels, key=lambda item: (*map(int, re.findall(r"\d+", item)), item)):
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f"labelled HUST cell is missing: {source}")
        samples.append(load_cell(source, labels[name], config))
    protocols = sorted({sample.protocol for sample in samples}, key=protocol_sort_key)
    if len(protocols) != config.expected_protocol_count:
        raise ValueError(
            f"expected {config.expected_protocol_count} HUST protocols, found "
            f"{len(protocols)}: {protocols}"
        )
    for protocol in protocols:
        count = sum(sample.protocol == protocol for sample in samples)
        if count < 2:
            raise ValueError(f"{protocol} requires at least two replicate cells, found {count}")
    return samples


def split_protocol(
    samples: Sequence[CellSample], held_out_protocol: str
) -> tuple[list[CellSample], list[CellSample]]:
    source = [sample for sample in samples if sample.protocol != held_out_protocol]
    target = [sample for sample in samples if sample.protocol == held_out_protocol]
    if not source or not target:
        raise ValueError(f"protocol split is empty for {held_out_protocol}")
    if any(sample.protocol == held_out_protocol for sample in source):
        raise RuntimeError("held-out HUST protocol leaked into source samples")
    return source, target

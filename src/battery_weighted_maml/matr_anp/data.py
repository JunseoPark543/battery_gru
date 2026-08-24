"""BatteryLife-style MATR/CALCE loading, discharge extraction, and auditing."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass(frozen=True)
class DischargeCurve:
    """One discharge signal in real-time-available nominal-capacity coordinates."""

    q: np.ndarray
    voltage_v: np.ndarray
    current_a_magnitude: np.ndarray
    original_current_sign: int
    monotonic_before_cleanup: bool
    duplicate_q_count: int


@dataclass(frozen=True)
class CycleData:
    cycle_number: int
    discharge_capacity_ah: float
    soh: float
    discharge: DischargeCurve | None
    raw_signal_length: int
    issue: str | None = None


@dataclass(frozen=True)
class CellData:
    cell_id: str
    source_file: str
    nominal_capacity_ah: float
    cycles: tuple[CycleData, ...]

    @property
    def cycle_numbers(self) -> np.ndarray:
        return np.asarray([cycle.cycle_number for cycle in self.cycles], dtype=np.int64)

    @property
    def soh(self) -> np.ndarray:
        return np.asarray([cycle.soh for cycle in self.cycles], dtype=np.float64)

    def cycle_by_number(self, cycle_number: int) -> CycleData:
        for cycle in self.cycles:
            if cycle.cycle_number == cycle_number:
                return cycle
        raise KeyError(f"{self.cell_id}: cycle {cycle_number} is unavailable")


@dataclass
class CellAudit:
    source_file: str
    cell_id: str | None = None
    status: str = "invalid"
    nominal_capacity_ah: float | None = None
    raw_cycle_count: int = 0
    valid_cycle_count: int = 0
    curve_cycle_count: int = 0
    missing_curve_count: int = 0
    short_curve_count: int = 0
    long_curve_count: int = 0
    minimum_signal_length: int = 0
    median_signal_length: float = 0.0
    maximum_signal_length: int = 0
    duplicate_cycle_count: int = 0
    nonfinite_capacity_count: int = 0
    nonmonotonic_curve_count: int = 0
    duplicate_q_count: int = 0
    current_signs: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["current_signs"] = ",".join(map(str, sorted(set(self.current_signs))))
        payload["issues"] = " | ".join(self.issues)
        return payload


def _records(value: Any, file_name: str) -> list[Mapping[str, Any]]:
    if isinstance(value, pd.DataFrame):
        raw: Sequence[Any] = value.to_dict(orient="records")
    elif isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple, pd.Series)):
        raw = list(value)
    else:
        raise ValueError(f"{file_name}: unsupported cycle_data type {type(value).__name__}")
    output: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, pd.Series):
            item = item.to_dict()
        if not isinstance(item, Mapping):
            raise ValueError(f"{file_name}: cycle_data[{index}] is not a mapping")
        output.append(item)
    return output


def _numeric_array(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64)
    return array


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _cycle_number(record: Mapping[str, Any], index: int, file_name: str) -> int:
    raw = _first_present(record, ("cycle_number", "cycle_index", "cycle"))
    if raw is None:
        raise ValueError(f"{file_name}: cycle_data[{index}] has no cycle number")
    try:
        numeric = float(raw)
        cycle = int(numeric)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{file_name}: invalid cycle number at record {index}") from exc
    if cycle <= 0 or numeric != cycle:
        raise ValueError(f"{file_name}: cycle number must be a positive integer")
    return cycle


def _final_capacity(record: Mapping[str, Any]) -> float:
    raw = _first_present(
        record,
        (
            "discharge_capacity_in_Ah",
            "discharge_capacity_ah",
            "discharge_capacity",
            "QD",
        ),
    )
    values = _numeric_array(raw)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    # BatteryLife stores cumulative discharge capacity. The largest observed
    # value is robust to a trailing reset while never entering an input feature.
    return float(np.max(values))


def _stage_discharge_mask(record: Mapping[str, Any], size: int) -> np.ndarray | None:
    raw = _first_present(record, ("step_type", "stage", "state", "operation"))
    if raw is None:
        return None
    values = np.asarray(raw).reshape(-1)
    if values.size != size:
        return None
    labels = np.char.lower(values.astype(str))
    mask = np.asarray(["discharg" in label for label in labels], dtype=bool)
    return mask if np.any(mask) else None


def extract_discharge_curve(
    record: Mapping[str, Any],
    nominal_capacity_ah: float,
    minimum_points: int,
) -> DischargeCurve:
    """Extract discharge using capacity growth, stage, and inferred current sign.

    Current is converted to magnitude. The sign reported in the result records
    the dataset convention inferred specifically from capacity-increasing points.
    No final-cycle capacity normalization is used: q=Qd(t)/Q_nominal.
    """
    voltage = _numeric_array(
        _first_present(record, ("voltage_in_V", "voltage_v", "voltage"))
    )
    current = _numeric_array(
        _first_present(record, ("current_in_A", "current_a", "current"))
    )
    capacity = _numeric_array(
        _first_present(
            record,
            ("discharge_capacity_in_Ah", "discharge_capacity_ah", "discharge_capacity", "QD"),
        )
    )
    size = min(voltage.size, current.size, capacity.size)
    if size < minimum_points:
        raise ValueError(f"discharge signals have only {size} aligned points")
    voltage, current, capacity = voltage[:size], current[:size], capacity[:size]
    finite = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(capacity)
    if np.count_nonzero(finite) < minimum_points:
        raise ValueError("too few finite aligned discharge samples")

    delta = np.diff(capacity, prepend=capacity[0])
    tolerance = max(1.0e-10, 1.0e-7 * nominal_capacity_ah)
    growing = finite & (delta > tolerance)
    if np.count_nonzero(growing) < 2:
        raise ValueError("discharge capacity does not increase")
    grow_indices = np.flatnonzero(growing)
    segment = np.zeros(size, dtype=bool)
    segment[max(0, grow_indices[0] - 1) : grow_indices[-1] + 1] = True

    stage_mask = _stage_discharge_mask(record, size)
    if stage_mask is not None and np.count_nonzero(segment & stage_mask) >= minimum_points:
        segment &= stage_mask

    signed_samples = current[growing & (np.abs(current) > 1.0e-8)]
    sign = int(np.sign(np.median(signed_samples))) if signed_samples.size else 0
    if sign:
        polarity = np.sign(current) == sign
        if np.count_nonzero(segment & polarity & finite) >= minimum_points:
            segment &= polarity
    selected = segment & finite
    if np.count_nonzero(selected) < minimum_points:
        raise ValueError("validated discharge segment is too short")

    q_ordered = capacity[selected] / nominal_capacity_ah
    v_ordered = voltage[selected]
    i_ordered = np.abs(current[selected])
    monotonic = bool(np.all(np.diff(q_ordered) >= -1.0e-10))
    order = np.argsort(q_ordered, kind="stable")
    q_sorted, v_sorted, i_sorted = q_ordered[order], v_ordered[order], i_ordered[order]
    unique_q, inverse, counts = np.unique(q_sorted, return_inverse=True, return_counts=True)
    duplicate_count = int(np.sum(counts - 1))
    voltage_sum = np.zeros_like(unique_q)
    current_sum = np.zeros_like(unique_q)
    np.add.at(voltage_sum, inverse, v_sorted)
    np.add.at(current_sum, inverse, i_sorted)
    v_unique = voltage_sum / counts
    i_unique = current_sum / counts
    if unique_q.size < minimum_points or not np.all(np.diff(unique_q) > 0):
        raise ValueError("too few unique monotonic q coordinates")
    return DischargeCurve(
        q=unique_q.astype(np.float64),
        voltage_v=v_unique.astype(np.float64),
        current_a_magnitude=i_unique.astype(np.float64),
        original_current_sign=sign,
        monotonic_before_cleanup=monotonic,
        duplicate_q_count=duplicate_count,
    )


def _dataset_marker(root: Mapping[str, Any], path: Path) -> str | None:
    for key in ("dataset", "dataset_name", "source_dataset"):
        if key in root:
            return str(root[key])
    candidates = [str(root.get("cell_id", "")), path.stem, *map(str, path.parts)]
    for dataset in ("MATR", "CALCE", "HUST"):
        if any(dataset.lower() in candidate.lower() for candidate in candidates):
            return dataset
    return None


def load_cell(path: str | Path, config: DataConfig) -> tuple[CellData, CellAudit]:
    """Load one trusted BatteryLife-style MATR or CALCE pickle."""
    source = Path(path)
    dataset = config.dataset.upper()
    audit = CellAudit(source_file=str(source))
    try:
        with source.open("rb") as handle:
            root = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"could not load trusted BatteryLife pickle {source}: {exc}") from exc
    if not isinstance(root, Mapping):
        raise ValueError(f"{source.name}: pickle root must be a mapping")
    marker = _dataset_marker(root, source)
    if marker is None or marker.upper() != dataset:
        raise ValueError(
            f"{source.name}: expected {dataset}, but metadata/path identifies "
            f"{marker or 'no supported dataset'}"
        )
    required = {"cell_id", "nominal_capacity_in_Ah", "cycle_data"}
    missing = required - set(root)
    if missing:
        raise ValueError(f"{source.name}: missing root keys {sorted(missing)}")
    nominal = float(root["nominal_capacity_in_Ah"])
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(f"{source.name}: invalid nominal capacity {nominal}")
    cell_id = str(root["cell_id"])
    records = _records(root["cycle_data"], source.name)
    audit.cell_id = cell_id
    audit.nominal_capacity_ah = nominal
    audit.raw_cycle_count = len(records)
    cycles: list[CycleData] = []
    seen: set[int] = set()
    for index, record in enumerate(records):
        cycle_number = _cycle_number(record, index, source.name)
        if cycle_number in seen:
            audit.duplicate_cycle_count += 1
            audit.issues.append(f"duplicate cycle {cycle_number}")
            continue
        seen.add(cycle_number)
        capacity = _final_capacity(record)
        if not np.isfinite(capacity) or capacity <= 0:
            audit.nonfinite_capacity_count += 1
            audit.issues.append(f"cycle {cycle_number}: invalid discharge capacity")
            continue
        raw_length = min(
            _numeric_array(_first_present(record, ("voltage_in_V", "voltage_v", "voltage"))).size,
            _numeric_array(_first_present(record, ("current_in_A", "current_a", "current"))).size,
        )
        curve: DischargeCurve | None
        issue: str | None = None
        try:
            curve = extract_discharge_curve(record, nominal, config.minimum_discharge_points)
            audit.curve_cycle_count += 1
            audit.current_signs.append(curve.original_current_sign)
            audit.duplicate_q_count += curve.duplicate_q_count
            if not curve.monotonic_before_cleanup:
                audit.nonmonotonic_curve_count += 1
            if raw_length < config.short_signal_threshold:
                audit.short_curve_count += 1
        except ValueError as exc:
            curve = None
            issue = str(exc)
            audit.missing_curve_count += 1
            audit.issues.append(f"cycle {cycle_number}: {issue}")
        cycles.append(
            CycleData(
                cycle_number=cycle_number,
                discharge_capacity_ah=capacity,
                soh=capacity / nominal,
                discharge=curve,
                raw_signal_length=raw_length,
                issue=issue,
            )
        )
    cycles.sort(key=lambda item: item.cycle_number)
    audit.valid_cycle_count = len(cycles)
    signal_lengths = np.asarray(
        [cycle.raw_signal_length for cycle in cycles if cycle.raw_signal_length > 0],
        dtype=np.float64,
    )
    if signal_lengths.size:
        audit.minimum_signal_length = int(np.min(signal_lengths))
        audit.median_signal_length = float(np.median(signal_lengths))
        audit.maximum_signal_length = int(np.max(signal_lengths))
        first_quartile, third_quartile = np.quantile(signal_lengths, [0.25, 0.75])
        long_boundary = third_quartile + 3.0 * (third_quartile - first_quartile)
        audit.long_curve_count = int(np.count_nonzero(signal_lengths > long_boundary))
    if len(cycles) < config.minimum_valid_cycles:
        raise ValueError(
            f"{source.name}: only {len(cycles)} valid cycles; "
            f"minimum is {config.minimum_valid_cycles}"
        )
    if not np.all(np.diff([cycle.cycle_number for cycle in cycles]) > 0):
        raise RuntimeError(f"{source.name}: internal cycle ordering failure")
    audit.status = "valid"
    return CellData(cell_id, str(source), nominal, tuple(cycles)), audit


def discover_dataset_files(root: str | Path, config: DataConfig) -> list[Path]:
    data_root = Path(root)
    dataset = config.dataset.upper()
    candidates: set[Path] = set()
    for pattern in config.file_globs:
        candidates.update(path for path in data_root.glob(pattern) if path.is_file())
    # Path-level filtering prevents adjacent datasets from reaching pickle.load.
    # Metadata/cell ID is checked again after loading.
    selected = sorted(
        path
        for path in candidates
        if any(dataset.lower() in part.lower() for part in path.parts)
    )
    if not selected:
        raise FileNotFoundError(
            f"no {dataset} pickle files found below {data_root}; searched "
            f"{config.file_globs}. The directory or file names must contain '{dataset}'."
        )
    return selected


def load_dataset(
    root: str | Path,
    config: DataConfig,
    *,
    tolerate_invalid_cells: bool = False,
) -> tuple[list[CellData], pd.DataFrame]:
    dataset = config.dataset.upper()
    cells: list[CellData] = []
    audits: list[dict[str, Any]] = []
    cell_ids: set[str] = set()
    for path in discover_dataset_files(root, config):
        try:
            cell, audit = load_cell(path, config)
            if cell.cell_id in cell_ids:
                raise ValueError(f"duplicate {dataset} cell_id across files: {cell.cell_id}")
            cell_ids.add(cell.cell_id)
            cells.append(cell)
            audits.append(audit.to_dict())
        except Exception as exc:
            audit = CellAudit(source_file=str(path), issues=[str(exc)])
            audits.append(audit.to_dict())
            if not tolerate_invalid_cells:
                raise
    cells.sort(key=lambda cell: cell.cell_id)
    return cells, pd.DataFrame(audits)


# Backward-compatible names retained for the existing MATR scripts/tests.
load_matr_cell = load_cell
discover_matr_files = discover_dataset_files
load_matr_dataset = load_dataset

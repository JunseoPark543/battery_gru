"""HUST loader for cycle-level SOH, mean voltage, and mean current."""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from battery_weighted_maml.data.task_views import FullCellTrajectory


_HUST_FILE = re.compile(r"HUST_(\d+)-(\d+)\.pkl", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RawHUSTCell:
    file_name: str
    cell_id: str
    protocol: str
    replicate: int
    nominal_capacity_ah: float
    cycle_records: tuple[Mapping[str, Any], ...]


def parse_protocol(file_name: str) -> tuple[str, int]:
    match = _HUST_FILE.fullmatch(file_name)
    if match is None:
        raise ValueError(
            f"HUST filename must match HUST_<protocol>-<replicate>.pkl: {file_name}"
        )
    return f"protocol_{int(match.group(1))}", int(match.group(2))


def _protocol_number(protocol: str) -> int:
    match = re.fullmatch(r"protocol_(\d+)", protocol)
    if match is None:
        raise ValueError(f"invalid HUST protocol: {protocol}")
    return int(match.group(1))


def hust_sort_key(file_name: str) -> tuple[int, int, str]:
    protocol, replicate = parse_protocol(file_name)
    return _protocol_number(protocol), replicate, file_name.lower()


def _records(value: Any, file_name: str) -> list[Mapping[str, Any]]:
    if isinstance(value, pd.DataFrame):
        raw = value.to_dict(orient="records")
    elif isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple, pd.Series)):
        raw = list(value)
    else:
        raise ValueError(
            f"{file_name}: cycle_data must be a list, numpy array, or pandas object"
        )
    records: list[Mapping[str, Any]] = []
    for index, record in enumerate(raw):
        if isinstance(record, pd.Series):
            record = record.to_dict()
        if not isinstance(record, Mapping):
            raise ValueError(f"{file_name}: cycle_data[{index}] is not a mapping")
        records.append(record)
    if not records:
        raise ValueError(f"{file_name}: cycle_data is empty")
    return records


def _numeric(value: Any) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, (pd.Series, pd.Index)):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, np.ndarray):
        raw = value.reshape(-1)
    elif isinstance(value, (list, tuple)):
        raw = np.asarray(value, dtype=object).reshape(-1)
    else:
        raw = np.asarray([value], dtype=object)
    return pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)


def _last_finite(value: Any) -> float:
    values = _numeric(value)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float("nan")


def _finite_mean(value: Any) -> float:
    values = _numeric(value)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def load_labels(path: str | Path) -> dict[str, int]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"HUST life-label file not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid HUST life-label JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("HUST life-label JSON must contain an object")
    labels: dict[str, int] = {}
    for raw_name, raw_cycle in payload.items():
        name = str(raw_name)
        if not name.lower().endswith(".pkl"):
            name += ".pkl"
        parse_protocol(name)
        try:
            cycle = int(raw_cycle)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid HUST EOL label for {name}: {raw_cycle!r}") from exc
        if cycle <= 100:
            raise ValueError(f"HUST EOL for {name} must exceed cycle 100, got {cycle}")
        labels[name] = cycle
    if not labels:
        raise ValueError(f"HUST life-label JSON is empty: {source}")
    return labels


def load_hust_pickle(path: str | Path) -> RawHUSTCell:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"HUST pickle not found: {source}")
    # These pickle files must come from the trusted local HUST dataset.
    try:
        with source.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"{source.name}: could not load trusted pickle: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source.name}: pickle root must be a mapping")
    required = {"cell_id", "nominal_capacity_in_Ah", "cycle_data"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{source.name}: missing root keys {sorted(missing)}")
    try:
        nominal = float(payload["nominal_capacity_in_Ah"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source.name}: nominal capacity must be numeric") from exc
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(f"{source.name}: invalid nominal capacity {nominal}")
    protocol, replicate = parse_protocol(source.name)
    return RawHUSTCell(
        file_name=source.name,
        cell_id=str(payload["cell_id"]),
        protocol=protocol,
        replicate=replicate,
        nominal_capacity_ah=nominal,
        cycle_records=tuple(_records(payload["cycle_data"], source.name)),
    )


def _interpolate_signal(
    values: Mapping[int, float], grid: np.ndarray, file_name: str, signal: str
) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.Series(values, dtype=float).reindex(grid)
    missing = raw.isna().to_numpy(dtype=bool)
    if not raw.notna().any():
        raise ValueError(f"{file_name}: {signal} is missing for every cycle")
    filled = raw.interpolate(method="linear", limit_direction="both")
    if filled.isna().any():
        raise ValueError(f"{file_name}: unresolved missing {signal} after interpolation")
    output = filled.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError(f"{file_name}: non-finite interpolated {signal}")
    return output, missing


def preprocess_cell(
    raw: RawHUSTCell,
    true_eol_cycle: int,
    logger: logging.Logger | None = None,
) -> FullCellTrajectory:
    """Convert one HUST cell to a continuous cycle-1 trajectory."""
    log = logger or logging.getLogger("battery_weighted_maml")
    capacities: dict[int, float] = {}
    voltages: dict[int, float] = {}
    currents: dict[int, float] = {}
    observed_cycles: set[int] = set()
    duplicates: set[int] = set()
    for index, record in enumerate(raw.cycle_records):
        try:
            cycle_float = float(record["cycle_number"])
            cycle = int(cycle_float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{raw.file_name}: invalid cycle number at record {index}") from exc
        if not np.isfinite(cycle_float) or cycle_float != cycle or cycle <= 0:
            raise ValueError(
                f"{raw.file_name}: cycle_data[{index}] cycle_number must be a positive integer"
            )
        if cycle in observed_cycles:
            duplicates.add(cycle)
        observed_cycles.add(cycle)
        capacity = _last_finite(record.get("discharge_capacity_in_Ah", []))
        voltage = _finite_mean(record.get("voltage_in_V", []))
        current = _finite_mean(record.get("current_in_A", []))
        # Last valid duplicate wins; invalid duplicates never erase valid data.
        if np.isfinite(capacity) or cycle not in capacities:
            capacities[cycle] = capacity
        if np.isfinite(voltage) or cycle not in voltages:
            voltages[cycle] = voltage
        if np.isfinite(current) or cycle not in currents:
            currents[cycle] = current
    if not observed_cycles:
        raise ValueError(f"{raw.file_name}: no cycle records could be parsed")
    first_observed_cycle = min(observed_cycles)
    if first_observed_cycle > 1:
        # A few HUST cells omit one or more leading records even though their
        # life labels and the rest of the dataset use absolute cycle numbers.
        # Keep the common 1..N axis: pandas' bidirectional interpolation below
        # fills this boundary gap with the earliest finite observation.  The
        # filled positions remain explicitly marked in ``is_interpolated``.
        log.warning(
            "%s: leading cycle(s) 1..%d are absent; filling them from the "
            "earliest finite per-signal observation",
            raw.file_name,
            first_observed_cycle - 1,
        )
    last_cycle = max(observed_cycles)
    if last_cycle <= 100:
        raise ValueError(f"{raw.file_name}: final observed cycle must exceed 100")
    grid = np.arange(1, last_cycle + 1, dtype=np.int64)
    capacity, capacity_missing = _interpolate_signal(
        capacities, grid, raw.file_name, "discharge capacity"
    )
    voltage, voltage_missing = _interpolate_signal(
        voltages, grid, raw.file_name, "mean voltage"
    )
    current, current_missing = _interpolate_signal(
        currents, grid, raw.file_name, "mean current"
    )
    interpolated = capacity_missing | voltage_missing | current_missing
    soh = capacity / raw.nominal_capacity_ah
    if not np.all(np.isfinite(soh)):
        raise FloatingPointError(f"{raw.file_name}: SOH contains non-finite values")
    if duplicates:
        log.warning(
            "%s: duplicate cycles %s; retained the last valid value per signal",
            raw.file_name,
            sorted(duplicates),
        )
    log.info(
        "%s: protocol=%s raw=%d processed=%d interpolated_cycles=%d",
        raw.file_name,
        raw.protocol,
        len(raw.cycle_records),
        len(grid),
        int(interpolated.sum()),
    )
    return FullCellTrajectory(
        file_name=raw.file_name,
        cell_id=raw.cell_id,
        family=raw.protocol,
        nominal_capacity_ah=raw.nominal_capacity_ah,
        cycles=grid,
        capacities_ah=capacity,
        soh=soh,
        is_interpolated=interpolated,
        true_eol_cycle=int(true_eol_cycle),
        raw_cycle_count=len(raw.cycle_records),
        missing_count_before=int(interpolated.sum()),
        missing_count_after=0,
        mean_voltage_v=voltage,
        mean_current_a=current,
    )


def trajectory_frame(cell: FullCellTrajectory) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_name": cell.file_name,
            "cell_id": cell.cell_id,
            "protocol": cell.family,
            "cycle": cell.cycles,
            "discharge_capacity_Ah": cell.capacities_ah,
            "soh": cell.soh,
            "mean_voltage_V": cell.mean_voltage_v,
            "mean_current_A": cell.mean_current_a,
            "is_interpolated": cell.is_interpolated,
            "true_eol_cycle": cell.true_eol_cycle,
        }
    )


def summary_record(cell: FullCellTrajectory) -> dict[str, Any]:
    protocol, replicate = parse_protocol(cell.file_name)
    eol_soh = float("nan")
    if cell.true_eol_cycle is not None:
        match = np.flatnonzero(cell.cycles == cell.true_eol_cycle)
        if match.size:
            eol_soh = float(cell.soh[match[0]])
    return {
        "file_name": cell.file_name,
        "cell_id": cell.cell_id,
        "protocol": protocol,
        "replicate": replicate,
        "nominal_capacity_Ah": cell.nominal_capacity_ah,
        "raw_cycle_count": cell.raw_cycle_count,
        "processed_cycle_count": len(cell.cycles),
        "interpolated_cycle_count": cell.missing_count_before,
        "first_cycle": int(cell.cycles[0]),
        "last_cycle": int(cell.cycles[-1]),
        "true_eol_cycle": cell.true_eol_cycle,
        "soh_at_true_eol": eol_soh,
    }


def preprocess_dataset(
    hust_dir: str | Path,
    label_path: str | Path,
    output_dir: str | Path,
    expected_protocol_count: int = 10,
    logger: logging.Logger | None = None,
) -> dict[str, FullCellTrajectory]:
    log = logger or logging.getLogger("battery_weighted_maml")
    root = Path(hust_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"HUST data directory not found: {root}")
    labels = load_labels(label_path)
    disk_names = {path.name for path in root.glob("*.pkl")}
    missing_files = sorted(set(labels) - disk_names, key=hust_sort_key)
    if missing_files:
        raise FileNotFoundError(
            f"labelled HUST pickle(s) are missing: {', '.join(missing_files)}"
        )
    missing_labels = sorted(disk_names - set(labels), key=hust_sort_key)
    if missing_labels:
        raise ValueError(f"missing HUST EOL label(s) for: {', '.join(missing_labels)}")
    trajectories: dict[str, FullCellTrajectory] = {}
    for name in sorted(labels, key=hust_sort_key):
        trajectories[name] = preprocess_cell(
            load_hust_pickle(root / name), labels[name], logger=log
        )
    protocols = sorted({cell.family for cell in trajectories.values()}, key=_protocol_number)
    if len(protocols) != expected_protocol_count:
        raise ValueError(
            f"expected {expected_protocol_count} HUST protocols, found {len(protocols)}: "
            f"{protocols}"
        )
    for protocol in protocols:
        count = sum(cell.family == protocol for cell in trajectories.values())
        if count < 2:
            raise ValueError(f"{protocol} needs at least two replicate cells, found {count}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [trajectory_frame(cell) for cell in trajectories.values()], ignore_index=True
    ).to_csv(destination / "all_cells_soh_voltage_current.csv", index=False)
    pd.DataFrame([summary_record(cell) for cell in trajectories.values()]).to_csv(
        destination / "cell_summary.csv", index=False
    )
    return trajectories


def select_source_names(
    trajectories: Mapping[str, FullCellTrajectory],
    target_name: str,
    source_mode: str,
) -> list[str]:
    if target_name not in trajectories:
        raise FileNotFoundError(f"HUST target was not found: {target_name}")
    target_protocol = trajectories[target_name].family
    if source_mode == "same_protocol":
        names = [
            name
            for name, cell in trajectories.items()
            if name != target_name and cell.family == target_protocol
        ]
    elif source_mode == "all_hust":
        names = [name for name in trajectories if name != target_name]
    elif source_mode == "leave_protocol_out":
        names = [
            name
            for name, cell in trajectories.items()
            if cell.family != target_protocol
        ]
    else:
        raise ValueError(
            "source_mode must be same_protocol, all_hust, or leave_protocol_out"
        )
    names = sorted(names, key=hust_sort_key)
    if not names:
        raise ValueError(f"no HUST sources for {target_name} in mode {source_mode}")
    return names


def protocol_counts(cells: Sequence[FullCellTrajectory]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.family] = counts.get(cell.family, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: _protocol_number(item[0])))

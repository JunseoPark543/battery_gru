"""Schema-aware CALCE pickle loading."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RawCell:
    """Validated but otherwise unmodified raw cell data."""

    file_name: str
    cell_id: str
    nominal_capacity_ah: float
    cycle_records: tuple[Mapping[str, Any], ...]


def _as_mapping(value: Any, file_name: str, context: str) -> Mapping[str, Any]:
    if isinstance(value, pd.Series):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError(f"{file_name}: {context} must be a mapping, got {type(value).__name__}")
    return value


def _records(value: Any, file_name: str) -> list[Mapping[str, Any]]:
    if isinstance(value, pd.DataFrame):
        raw_records: list[Any] = value.to_dict(orient="records")
    elif isinstance(value, np.ndarray):
        raw_records = value.tolist()
    elif isinstance(value, (list, tuple, pd.Series)):
        raw_records = list(value)
    else:
        raise ValueError(
            f"{file_name}: key 'cycle_data' must be a list, numpy array, or pandas object"
        )
    return [
        _as_mapping(record, file_name, f"cycle_data[{index}]")
        for index, record in enumerate(raw_records)
    ]


def load_calce_pickle(path: str | Path) -> RawCell:
    """Load a CALCE pickle and validate all fields needed for SOH extraction."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CALCE pickle not found: {source}")
    try:
        with source.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"{source.name}: could not unpickle file: {exc}") from exc
    root = _as_mapping(payload, source.name, "pickle root")
    required = {"cell_id", "nominal_capacity_in_Ah", "cycle_data"}
    missing = sorted(required - set(root))
    if missing:
        raise ValueError(f"{source.name}: missing required key(s): {', '.join(missing)}")
    try:
        nominal = float(root["nominal_capacity_in_Ah"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source.name}: key 'nominal_capacity_in_Ah' must be numeric"
        ) from exc
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(
            f"{source.name}: key 'nominal_capacity_in_Ah' must be finite and > 0, got {nominal}"
        )
    records = _records(root["cycle_data"], source.name)
    if not records:
        raise ValueError(f"{source.name}: key 'cycle_data' is empty")
    for index, record in enumerate(records):
        missing_cycle = {"cycle_number", "discharge_capacity_in_Ah"} - set(record)
        if missing_cycle:
            keys = ", ".join(sorted(missing_cycle))
            raise ValueError(f"{source.name}: cycle_data[{index}] missing required key(s): {keys}")
    return RawCell(
        file_name=source.name,
        cell_id=str(root["cell_id"]),
        nominal_capacity_ah=nominal,
        cycle_records=tuple(records),
    )


def load_eol_labels(path: str | Path) -> dict[str, int]:
    """Load the filename-to-EOL-cycle JSON mapping."""
    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(f"CALCE EOL label file not found: {label_path}")
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid EOL label JSON {label_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"EOL label JSON must contain a dictionary: {label_path}")
    labels: dict[str, int] = {}
    for name, cycle in payload.items():
        try:
            parsed = int(cycle)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid EOL cycle for {name!r}: {cycle!r}") from exc
        if parsed <= 0:
            raise ValueError(f"EOL cycle for {name!r} must be positive, got {parsed}")
        labels[str(name)] = parsed
    return labels


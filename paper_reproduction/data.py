"""CALCE cell tasks and variable-length recursive support data."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if isinstance(value, pd.Series):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _last_finite(value: Any) -> float:
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, np.ndarray):
        raw = value.reshape(-1)
    elif isinstance(value, (list, tuple)):
        raw = np.asarray(value, dtype=object).reshape(-1)
    else:
        raw = np.asarray([value], dtype=object)
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return float(finite[-1]) if finite.size else float("nan")


@dataclass(frozen=True)
class CellTask:
    """One battery cell as one MAML task."""

    name: str
    cycles: np.ndarray
    soh: np.ndarray
    raw_cycle_count: int | None = None
    cleaned_cycle_count: int | None = None
    interpolated_cycle_count: int = 0
    missing_cycle_count: int = 0
    removed_outlier_count: int = 0
    nominal_capacity_ah: float | None = None

    def __post_init__(self) -> None:
        cycles = np.asarray(self.cycles, dtype=np.int64).copy()
        soh = np.asarray(self.soh, dtype=np.float64).copy()
        if cycles.ndim != 1 or soh.ndim != 1 or len(cycles) != len(soh):
            raise ValueError(f"{self.name}: cycles and SOH must be aligned 1-D arrays")
        if len(soh) < 2 or not np.all(np.isfinite(soh)):
            raise ValueError(f"{self.name}: SOH must contain at least two finite values")
        if np.any(np.diff(cycles) <= 0):
            raise ValueError(f"{self.name}: cycles must be strictly increasing")
        cycles.setflags(write=False)
        soh.setflags(write=False)
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "soh", soh)

    def split(self, history_length: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self.soh) <= history_length:
            raise ValueError(
                f"{self.name}: needs more than L={history_length} cycles, found {len(self.soh)}"
            )
        return self.soh[:history_length], self.soh[history_length:]


def load_cell_task(path: str | Path) -> CellTask:
    """Load a CALCE pickle and compute unnormalized, unclipped scalar SOH."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CALCE pickle not found: {source}")
    with source.open("rb") as handle:
        root = _mapping(pickle.load(handle), source.name)
    required = {"nominal_capacity_in_Ah", "cycle_data"}
    missing = required - set(root)
    if missing:
        raise ValueError(f"{source.name}: missing keys {sorted(missing)}")
    nominal = float(root["nominal_capacity_in_Ah"])
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError(f"{source.name}: nominal capacity must be finite and positive")
    raw_records = root["cycle_data"]
    if isinstance(raw_records, pd.DataFrame):
        records: Sequence[Any] = raw_records.to_dict(orient="records")
    elif isinstance(raw_records, np.ndarray):
        records = raw_records.tolist()
    elif isinstance(raw_records, (list, tuple, pd.Series)):
        records = list(raw_records)
    else:
        raise ValueError(f"{source.name}: cycle_data has unsupported type")
    by_cycle: dict[int, float] = {}
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"{source.name} cycle_data[{index}]")
        if "cycle_number" not in record or "discharge_capacity_in_Ah" not in record:
            raise ValueError(f"{source.name}: cycle record {index} lacks cycle/capacity")
        cycle_number = float(record["cycle_number"])
        cycle = int(cycle_number)
        if not np.isfinite(cycle_number) or cycle_number != cycle or cycle <= 0:
            raise ValueError(f"{source.name}: invalid cycle number at record {index}")
        capacity = _last_finite(record["discharge_capacity_in_Ah"])
        if np.isfinite(capacity) or cycle not in by_cycle:
            by_cycle[cycle] = capacity
    if not by_cycle:
        raise ValueError(f"{source.name}: no cycle data")
    cycles = np.arange(min(by_cycle), max(by_cycle) + 1, dtype=np.int64)
    original_capacity = pd.Series(by_cycle, dtype=float).reindex(cycles)
    interpolated_count = int(original_capacity.isna().sum())
    missing_cycle_count = len(cycles) - len(by_cycle)
    capacity_series = original_capacity.interpolate(method="linear", limit_direction="both")
    capacity_series = capacity_series.interpolate(method="linear", limit_direction="both")
    if capacity_series.isna().any():
        raise ValueError(f"{source.name}: capacity remains missing after interpolation")
    soh = capacity_series.to_numpy(dtype=np.float64) / nominal
    # Paper assumption: SOH is already scaled. Reproduction choice: do not clip
    # or normalize even if a measured value is marginally outside [0, 1].
    return CellTask(
        source.name,
        cycles,
        soh,
        raw_cycle_count=len(records),
        cleaned_cycle_count=len(cycles),
        interpolated_cycle_count=interpolated_count,
        missing_cycle_count=missing_cycle_count,
        removed_outlier_count=0,
        nominal_capacity_ah=nominal,
    )


def load_tasks(calce_dir: str | Path, names: Sequence[str]) -> list[CellTask]:
    root = Path(calce_dir)
    tasks = [load_cell_task(root / name) for name in names]
    if [task.name for task in tasks] != list(names):
        raise RuntimeError("loaded task order changed unexpectedly")
    return tasks


def build_recursive_pairs(
    sequence: np.ndarray | Sequence[float] | torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return L-1 ``([1..j], [j+1..L])`` scalar SOH pairs."""
    values = (
        sequence.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
        if isinstance(sequence, torch.Tensor)
        else torch.tensor(np.asarray(sequence, dtype=np.float32).copy()).flatten()
    )
    if len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("recursive pairs require at least two finite SOH values")
    return [
        (values[:split].unsqueeze(-1), values[split:].unsqueeze(-1))
        for split in range(1, len(values))
    ]


class RecursivePairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, sequence: np.ndarray | Sequence[float] | torch.Tensor) -> None:
        self.pairs = build_recursive_pairs(sequence)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[index]

    def split_index(self, index: int) -> int:
        return len(self.pairs[index][0])


class QuerySequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """One exact task query: initial support history -> complete later SOH."""

    def __init__(self, support: np.ndarray, query: np.ndarray) -> None:
        history = torch.tensor(np.asarray(support, dtype=np.float32).copy()).view(-1, 1)
        future = torch.tensor(np.asarray(query, dtype=np.float32).copy()).view(-1, 1)
        if len(history) < 1 or len(future) < 1:
            raise ValueError("query history/future cannot be empty")
        self.sample = (history, future)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index not in {0, -1}:
            raise IndexError(index)
        return self.sample


def variable_length_collate(
    samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad histories/targets and return lengths plus a boolean target mask."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    histories, futures = zip(*samples)
    input_lengths = torch.tensor([len(item) for item in histories], dtype=torch.long)
    target_lengths = torch.tensor([len(item) for item in futures], dtype=torch.long)
    history = pad_sequence(histories, batch_first=True, padding_value=0.0)
    target = pad_sequence(futures, batch_first=True, padding_value=0.0)
    positions = torch.arange(target.shape[1]).unsqueeze(0)
    target_mask = positions < target_lengths.unsqueeze(1)
    return {
        "history": history,
        "target": target,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "target_mask": target_mask,
    }


def sample_support_batch(
    dataset: RecursivePairDataset,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    mode: str = "random",
    length_bins: int = 5,
) -> dict[str, torch.Tensor]:
    count = min(batch_size, len(dataset))
    if mode == "full_support":
        indices = list(range(len(dataset)))
    elif mode == "random":
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    elif mode == "length_stratified":
        bins = [chunk for chunk in torch.tensor_split(torch.arange(len(dataset)), length_bins) if len(chunk)]
        base, remainder = divmod(count, len(bins))
        selected: list[int] = []
        for bin_index, candidates in enumerate(bins):
            requested = base + (1 if bin_index < remainder else 0)
            order = torch.randperm(len(candidates), generator=generator)
            selected.extend(candidates[order[:min(requested, len(candidates))]].tolist())
        if len(selected) < count:
            remaining = torch.tensor(
                [index for index in range(len(dataset)) if index not in set(selected)],
                dtype=torch.long,
            )
            if len(remaining):
                order = torch.randperm(len(remaining), generator=generator)
                selected.extend(remaining[order[:count - len(selected)]].tolist())
        shuffle = torch.randperm(len(selected), generator=generator)
        indices = torch.tensor(selected, dtype=torch.long)[shuffle].tolist()
    else:
        raise ValueError("sampling mode must be random, length_stratified, or full_support")
    batch = variable_length_collate([dataset[index] for index in indices])
    batch["sample_indices"] = torch.tensor(indices, dtype=torch.long)
    batch["split_indices"] = torch.tensor(
        [dataset.split_index(index) for index in indices], dtype=torch.long
    )
    return {key: value.to(device) for key, value in batch.items()}


def preprocessing_summary(
    tasks: Sequence[CellTask],
    history_length: int,
    eol_threshold: float,
) -> pd.DataFrame:
    """Describe the exact preprocessing and usable query length of every cell."""
    from .metrics import last_hitting_eol

    records: list[dict[str, Any]] = []
    for task in tasks:
        cycle_500 = np.flatnonzero(task.cycles == 500)
        records.append(
            {
                "cell": task.name,
                "raw_cycle_count": task.raw_cycle_count,
                "cleaned_cycle_count": task.cleaned_cycle_count or len(task.soh),
                "interpolated_cycle_count": task.interpolated_cycle_count,
                "first_cycle": int(task.cycles[0]),
                "last_cycle": int(task.cycles[-1]),
                "missing_cycle_count": task.missing_cycle_count,
                "removed_outlier_count": task.removed_outlier_count,
                "nominal_capacity_ah": task.nominal_capacity_ah,
                "first_soh": float(task.soh[0]),
                "soh_at_cycle_500": (
                    float(task.soh[cycle_500[0]]) if cycle_500.size else float("nan")
                ),
                "final_soh": float(task.soh[-1]),
                "actual_eol_cycle": last_hitting_eol(
                    task.cycles, task.soh, eol_threshold
                ),
                "history_length": history_length,
                "query_length": max(0, len(task.soh) - history_length),
                "capacity_extraction": "last_finite_discharge_capacity",
                "soh_formula": "discharge_capacity_ah/nominal_capacity_ah",
                "interpolation": "linear_both_directions",
                "outlier_removal": "none",
                "sequence_truncated": False,
            }
        )
    return pd.DataFrame(records)

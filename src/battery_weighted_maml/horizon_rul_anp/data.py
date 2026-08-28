"""Labeled MATR cells and train-only scaling for direct RUL tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from battery_weighted_maml.matr_anp.data import CellData, load_dataset

from .config import DataConfig, TaskConfig


@dataclass(frozen=True)
class LabeledCell:
    """One established BatteryLife cell paired with its EOL-cycle label."""

    cell: CellData
    lifetime: int

    @property
    def cell_id(self) -> str:
        return self.cell.cell_id


def load_lifetime_labels(path: str | Path) -> dict[str, int]:
    """Load the same filename/cell-ID to EOL-cycle mapping used by CALCE."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"MATR lifetime label file not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MATR lifetime label JSON {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("MATR lifetime label JSON must contain a dictionary")
    labels: dict[str, int] = {}
    for key, value in raw.items():
        try:
            lifetime = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid lifetime for {key!r}: {value!r}") from exc
        if lifetime <= 1:
            raise ValueError(f"lifetime for {key!r} must exceed one cycle")
        labels[str(key)] = lifetime
    return labels


def _lookup_lifetime(cell: CellData, labels: dict[str, int]) -> int:
    source = Path(cell.source_file)
    candidates = (
        cell.cell_id,
        source.name,
        source.stem,
        f"{cell.cell_id}.pkl",
    )
    matches = {labels[key] for key in candidates if key in labels}
    if not matches:
        raise KeyError(
            f"no MATR lifetime label for {cell.cell_id}; tried {list(candidates)}"
        )
    if len(matches) != 1:
        raise ValueError(f"conflicting MATR lifetime labels for {cell.cell_id}: {matches}")
    return int(matches.pop())


def load_labeled_cells(
    data_root: str | Path,
    config: DataConfig,
) -> tuple[list[LabeledCell], pd.DataFrame]:
    """Reuse the shared MATR loader, then attach explicit EOL labels."""
    cells, audit = load_dataset(
        data_root,
        config.shared(),
        tolerate_invalid_cells=True,
    )
    labels = (
        load_lifetime_labels(config.label_path)
        if config.lifetime_source == "label_file" and config.label_path
        else {}
    )
    labeled: list[LabeledCell] = []
    label_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for cell in cells:
        try:
            lifetime = (
                _lookup_lifetime(cell, labels)
                if config.lifetime_source == "label_file"
                else int(cell.cycle_numbers[-1])
            )
            if lifetime <= int(cell.cycle_numbers[0]):
                raise ValueError("lifetime does not follow the first observed cycle")
            labeled.append(LabeledCell(cell, lifetime))
            label_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "source_file": cell.source_file,
                    "lifetime_cycle": lifetime,
                    "last_observed_cycle": int(cell.cycle_numbers[-1]),
                    "lifetime_minus_last_observed": (
                        lifetime - int(cell.cycle_numbers[-1])
                    ),
                    "lifetime_source": config.lifetime_source,
                    "label_status": "valid",
                    "label_issue": "",
                }
            )
        except (KeyError, ValueError) as exc:
            failures.append(str(exc))
            label_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "source_file": cell.source_file,
                    "lifetime_cycle": np.nan,
                    "last_observed_cycle": int(cell.cycle_numbers[-1]),
                    "lifetime_minus_last_observed": np.nan,
                    "lifetime_source": config.lifetime_source,
                    "label_status": "invalid",
                    "label_issue": str(exc),
                }
            )
    label_audit = pd.DataFrame(label_rows)
    if failures:
        preview = "; ".join(failures[:5])
        raise ValueError(
            f"{len(failures)} valid MATR cells lack usable lifetime labels: {preview}"
        )
    if not labeled:
        raise ValueError("MATR loader produced no labeled cells")
    # Preserve the established loader audit and make label checks explicit.
    combined = audit.merge(
        label_audit,
        how="left",
        on=["cell_id", "source_file"],
    )
    return sorted(labeled, key=lambda item: item.cell_id), combined


@dataclass
class RULScalers:
    """Prefix-feature and RUL statistics fitted on training cells only."""

    fit_cell_ids: list[str]
    cycle_scale: float
    soh_mean: float
    soh_std: float
    delta_soh_mean: float
    delta_soh_std: float
    rul_mean: float
    rul_std: float

    @classmethod
    def fit(
        cls,
        cells: Sequence[LabeledCell],
        task: TaskConfig,
    ) -> "RULScalers":
        if not cells:
            raise ValueError("cannot fit RUL scalers without training cells")
        ids = [item.cell_id for item in cells]
        if len(ids) != len(set(ids)):
            raise ValueError("training cell IDs must be unique")
        soh_values: list[np.ndarray] = []
        delta_values: list[np.ndarray] = []
        rul_values: list[np.ndarray] = []
        maximum_cycle = 1
        for item in cells:
            cycle_numbers = item.cell.cycle_numbers
            usable = cycle_numbers <= item.lifetime
            cycles = cycle_numbers[usable]
            soh = item.cell.soh[usable]
            if cycles.size < 2:
                continue
            maximum_cycle = max(maximum_cycle, int(cycles[-1]))
            soh_values.append(soh)
            delta_values.append(np.diff(soh, prepend=soh[0]))
            upper = min(task.max_horizon, item.lifetime - 1, int(cycles[-1]))
            if upper >= task.min_horizon:
                horizons = np.arange(task.min_horizon, upper + 1, dtype=np.float64)
                rul_values.append(item.lifetime - horizons)
        if not soh_values or not rul_values:
            raise ValueError("training cells cannot form configured horizon/RUL tasks")
        soh_all = np.concatenate(soh_values)
        delta_all = np.concatenate(delta_values)
        rul_all = np.concatenate(rul_values)
        return cls(
            fit_cell_ids=sorted(ids),
            cycle_scale=float(maximum_cycle),
            soh_mean=float(np.mean(soh_all)),
            soh_std=_safe_std(soh_all),
            delta_soh_mean=float(np.mean(delta_all)),
            delta_soh_std=_safe_std(delta_all),
            rul_mean=float(np.mean(rul_all)),
            rul_std=_safe_std(rul_all),
        )

    def prefix(self, item: LabeledCell, horizon: int) -> np.ndarray:
        """Return causal features [cycle, SOH, delta-SOH] through exactly k."""
        cycles = item.cell.cycle_numbers
        selected = cycles <= int(horizon)
        prefix_cycles = cycles[selected]
        prefix_soh = item.cell.soh[selected]
        if prefix_cycles.size == 0 or int(prefix_cycles[-1]) != int(horizon):
            raise ValueError(f"{item.cell_id} has no complete prefix through cycle {horizon}")
        delta = np.diff(prefix_soh, prepend=prefix_soh[0])
        return np.stack(
            [
                prefix_cycles / self.cycle_scale,
                (prefix_soh - self.soh_mean) / self.soh_std,
                (delta - self.delta_soh_mean) / self.delta_soh_std,
            ],
            axis=-1,
        ).astype(np.float32)

    def transform_rul(self, values: np.ndarray | float) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.rul_mean) / self.rul_std

    def inverse_rul(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.rul_std + self.rul_mean

    def std_to_cycles(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.rul_std

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RULScalers":
        return cls(**payload)  # type: ignore[arg-type]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values))
    return value if np.isfinite(value) and value > 1.0e-8 else 1.0

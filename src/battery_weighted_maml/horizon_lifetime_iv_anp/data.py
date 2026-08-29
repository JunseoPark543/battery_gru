"""Leakage-safe lifetime labels and 256-point historical I-V prefixes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from battery_weighted_maml.horizon_rul_anp.data import (
    LabeledCell,
    load_labeled_cells,
)
from battery_weighted_maml.matr_anp.config import QGridConfig


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values))
    return value if np.isfinite(value) and value > 1.0e-8 else 1.0


@dataclass
class LifetimeIVScalers:
    """All statistics are fitted on train cells only."""

    fit_cell_ids: list[str]
    cycle_scale: float
    soh_mean: float
    soh_std: float
    voltage_mean: float
    voltage_std: float
    current_mean: float
    current_std: float
    lifetime_mean: float
    lifetime_std: float

    @classmethod
    def fit(
        cls,
        cells: Sequence[LabeledCell],
        maximum_horizon: int,
    ) -> "LifetimeIVScalers":
        if not cells:
            raise ValueError("cannot fit lifetime I-V scalers without train cells")
        ids = [item.cell_id for item in cells]
        if len(ids) != len(set(ids)):
            raise ValueError("train cell IDs must be unique")
        soh_parts: list[np.ndarray] = []
        voltage_parts: list[np.ndarray] = []
        current_parts: list[np.ndarray] = []
        maximum_cycle = 1
        for item in cells:
            for cycle in item.cell.cycles:
                if cycle.cycle_number > min(maximum_horizon, item.lifetime - 1):
                    break
                maximum_cycle = max(maximum_cycle, cycle.cycle_number)
                soh_parts.append(np.asarray([cycle.soh], dtype=np.float64))
                if cycle.discharge is not None:
                    voltage_parts.append(cycle.discharge.voltage_v)
                    current_parts.append(cycle.discharge.current_a_magnitude)
        if not soh_parts or not voltage_parts or not current_parts:
            raise ValueError("train cells contain no usable SOH/I-V observations")
        soh = np.concatenate(soh_parts)
        voltage = np.concatenate(voltage_parts)
        current = np.concatenate(current_parts)
        lifetime = np.asarray([item.lifetime for item in cells], dtype=np.float64)
        return cls(
            fit_cell_ids=sorted(ids),
            cycle_scale=float(maximum_cycle),
            soh_mean=float(np.mean(soh)),
            soh_std=_safe_std(soh),
            voltage_mean=float(np.mean(voltage)),
            voltage_std=_safe_std(voltage),
            current_mean=float(np.mean(current)),
            current_std=_safe_std(current),
            lifetime_mean=float(np.mean(lifetime)),
            lifetime_std=_safe_std(lifetime),
        )

    def transform_lifetime(self, value: float | np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float64) - self.lifetime_mean) / self.lifetime_std

    def inverse_lifetime(self, value: float | np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64) * self.lifetime_std + self.lifetime_mean

    def std_to_cycles(self, value: float | np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64) * self.lifetime_std

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LifetimeIVScalers":
        return cls(**payload)  # type: ignore[arg-type]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class CellPrefixArrays:
    cycle_features: np.ndarray  # [K,2]: normalized cycle and SOH
    curves: np.ndarray  # [K,Q,3]: normalized q, voltage, current
    curve_masks: np.ndarray  # [K,Q]
    cycle_numbers: np.ndarray  # [K]


class LifetimeIVPrefixStore:
    """Lazily cache causal, train-normalized cell prefixes in host memory."""

    def __init__(
        self,
        scalers: LifetimeIVScalers,
        q_grid: QGridConfig,
        maximum_horizon: int,
    ) -> None:
        self.scalers = scalers
        self.grid = np.linspace(
            q_grid.minimum,
            q_grid.maximum,
            q_grid.num_points,
            dtype=np.float64,
        )
        self.q_min = float(q_grid.minimum)
        self.q_max = float(q_grid.maximum)
        self.maximum_horizon = int(maximum_horizon)
        self._cache: dict[str, CellPrefixArrays] = {}

    def _curve(self, item: LabeledCell, cycle_index: int) -> tuple[np.ndarray, np.ndarray]:
        cycle = item.cell.cycles[cycle_index]
        values = np.zeros((self.grid.size, 3), dtype=np.float32)
        mask = np.zeros(self.grid.size, dtype=bool)
        curve = cycle.discharge
        if curve is None:
            return values, mask
        q = np.asarray(curve.q, dtype=np.float64)
        voltage = np.asarray(curve.voltage_v, dtype=np.float64)
        current = np.asarray(curve.current_a_magnitude, dtype=np.float64)
        finite = np.isfinite(q) & np.isfinite(voltage) & np.isfinite(current)
        q, voltage, current = q[finite], voltage[finite], current[finite]
        inside = (q >= self.q_min) & (q <= self.q_max)
        q, voltage, current = q[inside], voltage[inside], current[inside]
        if q.size < 2:
            return values, mask
        mask = (self.grid >= q[0]) & (self.grid <= q[-1])
        if np.count_nonzero(mask) < 2:
            return values, np.zeros_like(mask)
        q_scale = max(self.q_max - self.q_min, 1.0e-8)
        values[mask, 0] = (2.0 * (self.grid[mask] - self.q_min) / q_scale - 1.0).astype(np.float32)
        values[mask, 1] = (
            (np.interp(self.grid[mask], q, voltage) - self.scalers.voltage_mean)
            / self.scalers.voltage_std
        ).astype(np.float32)
        values[mask, 2] = (
            (np.interp(self.grid[mask], q, current) - self.scalers.current_mean)
            / self.scalers.current_std
        ).astype(np.float32)
        return values, mask

    def build(self, item: LabeledCell) -> CellPrefixArrays:
        cached = self._cache.get(item.cell_id)
        if cached is not None:
            return cached
        cycles = [
            cycle
            for cycle in item.cell.cycles
            if cycle.cycle_number <= min(self.maximum_horizon, item.lifetime - 1)
        ]
        if not cycles:
            raise ValueError(f"{item.cell_id}: no cycles are usable before EOL")
        cycle_numbers = np.asarray([cycle.cycle_number for cycle in cycles], dtype=np.int64)
        cycle_features = np.stack(
            [
                cycle_numbers / self.scalers.cycle_scale,
                (np.asarray([cycle.soh for cycle in cycles]) - self.scalers.soh_mean)
                / self.scalers.soh_std,
            ],
            axis=-1,
        ).astype(np.float32)
        curves = np.zeros((len(cycles), self.grid.size, 3), dtype=np.float32)
        masks = np.zeros((len(cycles), self.grid.size), dtype=bool)
        by_number = {cycle.cycle_number: index for index, cycle in enumerate(item.cell.cycles)}
        for output_index, cycle in enumerate(cycles):
            curves[output_index], masks[output_index] = self._curve(
                item, by_number[cycle.cycle_number]
            )
        result = CellPrefixArrays(cycle_features, curves, masks, cycle_numbers)
        self._cache[item.cell_id] = result
        return result

    def prefix(self, item: LabeledCell, horizon: int) -> CellPrefixArrays:
        arrays = self.build(item)
        end = int(np.searchsorted(arrays.cycle_numbers, int(horizon), side="right"))
        if end == 0 or int(arrays.cycle_numbers[end - 1]) != int(horizon):
            raise ValueError(f"{item.cell_id}: cycle {horizon} is unavailable")
        return CellPrefixArrays(
            arrays.cycle_features[:end],
            arrays.curves[:end],
            arrays.curve_masks[:end],
            arrays.cycle_numbers[:end],
        )

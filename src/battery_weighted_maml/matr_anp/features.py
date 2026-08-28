"""Leakage-safe fixed-q discharge and partial intra-cell I-V features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DataConfig, QGridConfig
from .data import CellData, CycleData, DischargeCurve


class EpisodeUnavailable(ValueError):
    """Raised when a current cycle cannot form a leakage-safe I-V episode."""


@dataclass(frozen=True)
class GridCurve:
    voltage_v: np.ndarray
    current_a_magnitude: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class FastFeature:
    values: np.ndarray  # [3, Q]: delta V, |I|, observation mask
    mask: np.ndarray
    beta: float
    current_cycle: int
    reference_cycles: tuple[int, ...]


@dataclass(frozen=True)
class ContextSignal:
    """One historical cycle on the existing fixed-q grid.

    ``values`` has shape ``[Q,2]`` with training-fold-normalized discharge
    voltage and current magnitude. ``mask`` marks coordinates observed in the
    historical cycle; values outside the mask are exactly zero.
    """

    values: np.ndarray
    mask: np.ndarray


@dataclass
class FoldScalers:
    fit_cell_ids: list[str]
    soh_mean: float
    soh_std: float
    max_cycle_train: int
    delta_voltage_mean: float
    delta_voltage_std: float
    current_mean: float
    current_std: float
    voltage_mean: float = 0.0
    voltage_std: float = 1.0

    @classmethod
    def fit(
        cls,
        train_cells: Sequence[CellData],
        processor: "PartialIVProcessor",
        minimum_current_position: int,
    ) -> "FoldScalers":
        if not train_cells:
            raise ValueError("cannot fit scalers without training cells")
        cell_ids = [cell.cell_id for cell in train_cells]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("training scaler cell IDs must be unique")
        soh = np.concatenate([cell.soh for cell in train_cells])
        max_cycle = max(int(cell.cycle_numbers.max()) for cell in train_cells)
        delta_values: list[np.ndarray] = []
        current_values: list[np.ndarray] = []
        voltage_count = 0
        voltage_sum = 0.0
        voltage_square_sum = 0.0
        for cell in train_cells:
            for cycle in cell.cycles:
                if cycle.discharge is None:
                    continue
                grid_curve = processor._interpolate_cycle(cell, cycle)
                values = grid_curve.voltage_v[grid_curve.mask]
                voltage_count += int(values.size)
                voltage_sum += float(np.sum(values, dtype=np.float64))
                voltage_square_sum += float(
                    np.sum(np.square(values, dtype=np.float64), dtype=np.float64)
                )
            for position in range(minimum_current_position, len(cell.cycles)):
                current_cycle = cell.cycles[position]
                if current_cycle.discharge is None:
                    continue
                try:
                    delta, current, mask, _ = processor.raw_feature(
                        cell, current_cycle.cycle_number
                    )
                except EpisodeUnavailable:
                    continue
                delta_values.append(delta[mask])
                current_values.append(current[mask])
        if not delta_values or not current_values:
            raise ValueError("training cells contain no valid partial I-V features")
        if voltage_count == 0:
            raise ValueError("training cells contain no valid historical voltage signals")
        delta = np.concatenate(delta_values)
        current = np.concatenate(current_values)
        voltage_mean = voltage_sum / voltage_count
        voltage_variance = max(
            0.0,
            voltage_square_sum / voltage_count - voltage_mean * voltage_mean,
        )
        return cls(
            fit_cell_ids=sorted(cell_ids),
            soh_mean=float(np.mean(soh)),
            soh_std=_safe_std(soh),
            max_cycle_train=max_cycle,
            delta_voltage_mean=float(np.mean(delta)),
            delta_voltage_std=_safe_std(delta),
            current_mean=float(np.mean(current)),
            current_std=_safe_std(current),
            voltage_mean=float(voltage_mean),
            voltage_std=max(float(np.sqrt(voltage_variance)), 1.0e-8),
        )

    def transform_soh(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.soh_mean) / self.soh_std

    def inverse_soh(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.soh_std + self.soh_mean

    def transform_cycles(self, cycles: np.ndarray) -> np.ndarray:
        # Do not clip test cycles that exceed the training maximum.
        return np.asarray(cycles, dtype=np.float64) / float(self.max_cycle_train)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FoldScalers":
        # Checkpoints created before HS-ANP have no raw-voltage scaler. They
        # remain valid for the unchanged SOH-only/partial-I-V baselines.
        values = dict(payload)
        values.setdefault("voltage_mean", 0.0)
        values.setdefault("voltage_std", 1.0)
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> "FoldScalers":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _safe_std(values: np.ndarray) -> float:
    value = float(np.std(values))
    return value if np.isfinite(value) and value > 1.0e-8 else 1.0


class PartialIVProcessor:
    def __init__(self, grid_config: QGridConfig, data_config: DataConfig):
        self.grid = np.linspace(
            grid_config.minimum,
            grid_config.maximum,
            grid_config.num_points,
            dtype=np.float64,
        )
        self.grid_config = grid_config
        self.data_config = data_config
        # Feature construction used to repeat interpolation and reference-curve
        # aggregation for every sampled episode.  A training fold revisits the
        # same cell/cycle thousands of times, so retain these immutable results.
        self._grid_curve_cache: dict[tuple[str, str, int], GridCurve] = {}
        self._cycle_lookup_cache: dict[
            tuple[str, str], tuple[dict[int, CycleData], tuple[CycleData, ...]]
        ] = {}
        self._reference_cache: dict[
            tuple[str, str, tuple[int, ...]],
            tuple[np.ndarray, np.ndarray, tuple[int, ...]],
        ] = {}
        self._raw_feature_cache: dict[
            tuple[str, str, int],
            tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]],
        ] = {}
        self._context_signal_cache: dict[
            tuple[str, str, int, float, float, float, float], ContextSignal
        ] = {}

    @staticmethod
    def _cell_key(cell: CellData) -> tuple[str, str]:
        return cell.source_file, cell.cell_id

    @staticmethod
    def _readonly(array: np.ndarray) -> np.ndarray:
        array.setflags(write=False)
        return array

    def cache_info(self) -> dict[str, int]:
        """Return cache sizes for startup/training diagnostics."""
        return {
            "grid_curves": len(self._grid_curve_cache),
            "references": len(self._reference_cache),
            "raw_features": len(self._raw_feature_cache),
            "context_signals": len(self._context_signal_cache),
        }

    def _cycle_lookup(
        self, cell: CellData
    ) -> tuple[dict[int, CycleData], tuple[CycleData, ...]]:
        key = self._cell_key(cell)
        cached = self._cycle_lookup_cache.get(key)
        if cached is not None:
            return cached
        by_number = {cycle.cycle_number: cycle for cycle in cell.cycles}
        curves = tuple(cycle for cycle in cell.cycles if cycle.discharge is not None)
        result = by_number, curves
        self._cycle_lookup_cache[key] = result
        return result

    def interpolate(self, curve: DischargeCurve) -> GridCurve:
        """Interpolate only inside the observed q range; never extrapolate."""
        if not np.all(np.diff(curve.q) > 0):
            raise ValueError("discharge q must be strictly increasing before interpolation")
        mask = (self.grid >= curve.q[0]) & (self.grid <= curve.q[-1])
        voltage = np.zeros_like(self.grid)
        current = np.zeros_like(self.grid)
        if np.any(mask):
            voltage[mask] = np.interp(self.grid[mask], curve.q, curve.voltage_v)
            current[mask] = np.interp(
                self.grid[mask], curve.q, curve.current_a_magnitude
            )
        return GridCurve(voltage, current, mask)

    def _interpolate_cycle(self, cell: CellData, cycle: CycleData) -> GridCurve:
        if cycle.discharge is None:
            raise EpisodeUnavailable(
                f"{cell.cell_id} cycle {cycle.cycle_number}: discharge unavailable"
            )
        key = (*self._cell_key(cell), cycle.cycle_number)
        cached = self._grid_curve_cache.get(key)
        if cached is not None:
            return cached
        curve = self.interpolate(cycle.discharge)
        frozen = GridCurve(
            self._readonly(curve.voltage_v),
            self._readonly(curve.current_a_magnitude),
            self._readonly(curve.mask),
        )
        self._grid_curve_cache[key] = frozen
        return frozen

    def _reference(
        self, cell: CellData, current_cycle_number: int
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        by_number, curve_cycles = self._cycle_lookup(cell)
        selected: list[CycleData] = [
            by_number[number]
            for number in self.data_config.reference_cycles
            if number < current_cycle_number
            and number in by_number
            and by_number[number].discharge is not None
        ]
        selected_numbers = {cycle.cycle_number for cycle in selected}
        for cycle in curve_cycles:
            if len(selected) >= self.data_config.minimum_reference_cycles:
                break
            if cycle.cycle_number >= current_cycle_number:
                break
            if cycle.cycle_number not in selected_numbers:
                selected.append(cycle)
                selected_numbers.add(cycle.cycle_number)
        if len(selected) < self.data_config.minimum_reference_cycles:
            raise EpisodeUnavailable(
                f"{cell.cell_id} cycle {current_cycle_number}: only {len(selected)} "
                "valid earlier reference cycles"
            )
        reference_cycles = tuple(cycle.cycle_number for cycle in selected)
        cache_key = (*self._cell_key(cell), reference_cycles)
        cached = self._reference_cache.get(cache_key)
        if cached is not None:
            return cached
        curves = [self._interpolate_cycle(cell, cycle) for cycle in selected]
        voltage_stack = np.stack([curve.voltage_v for curve in curves])
        mask_stack = np.stack([curve.mask for curve in curves])
        counts = mask_stack.sum(axis=0)
        reference_mask = counts >= self.data_config.minimum_reference_cycles
        reference = np.zeros_like(self.grid)
        masked_voltage = np.where(mask_stack, voltage_stack, np.nan)
        # Every selected column has at least minimum_reference_cycles finite
        # observations, so compute only there and avoid all-NaN warnings outside it.
        reference[reference_mask] = np.nanmedian(
            masked_voltage[:, reference_mask], axis=0
        )
        result = (
            self._readonly(reference),
            self._readonly(reference_mask),
            reference_cycles,
        )
        self._reference_cache[cache_key] = result
        return result

    def raw_feature(
        self, cell: CellData, current_cycle_number: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
        cache_key = (*self._cell_key(cell), int(current_cycle_number))
        cached = self._raw_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        by_number, _ = self._cycle_lookup(cell)
        try:
            current_cycle = by_number[int(current_cycle_number)]
        except KeyError as exc:
            raise EpisodeUnavailable(
                f"{cell.cell_id}: cycle {current_cycle_number} is unavailable"
            ) from exc
        if current_cycle.discharge is None:
            raise EpisodeUnavailable(
                f"{cell.cell_id} cycle {current_cycle_number}: current discharge unavailable"
            )
        current_curve = self._interpolate_cycle(cell, current_cycle)
        reference, reference_mask, reference_cycles = self._reference(
            cell, current_cycle_number
        )
        valid = current_curve.mask & reference_mask
        if not np.any(valid):
            raise EpisodeUnavailable(
                f"{cell.cell_id} cycle {current_cycle_number}: no current/reference q overlap"
            )
        delta = np.zeros_like(self.grid)
        delta[valid] = current_curve.voltage_v[valid] - reference[valid]
        current = np.zeros_like(self.grid)
        current[valid] = current_curve.current_a_magnitude[valid]
        result = (
            self._readonly(delta),
            self._readonly(current),
            self._readonly(valid),
            reference_cycles,
        )
        self._raw_feature_cache[cache_key] = result
        return result

    def build_context_signal(
        self,
        cell: CellData,
        cycle_number: int,
        scalers: FoldScalers,
    ) -> ContextSignal:
        """Build leakage-safe V/I input for one historical context cycle."""
        cache_key = (
            *self._cell_key(cell),
            int(cycle_number),
            float(scalers.voltage_mean),
            float(scalers.voltage_std),
            float(scalers.current_mean),
            float(scalers.current_std),
        )
        cached = self._context_signal_cache.get(cache_key)
        if cached is not None:
            return cached
        by_number, _ = self._cycle_lookup(cell)
        cycle = by_number.get(int(cycle_number))
        if cycle is None or cycle.discharge is None:
            raise EpisodeUnavailable(
                f"{cell.cell_id} cycle {cycle_number}: historical V/I unavailable"
            )
        curve = self._interpolate_cycle(cell, cycle)
        mask = np.asarray(curve.mask, dtype=bool)
        voltage = np.zeros_like(curve.voltage_v, dtype=np.float64)
        current = np.zeros_like(curve.current_a_magnitude, dtype=np.float64)
        voltage[mask] = (
            curve.voltage_v[mask] - scalers.voltage_mean
        ) / scalers.voltage_std
        current[mask] = (
            curve.current_a_magnitude[mask] - scalers.current_mean
        ) / scalers.current_std
        values = np.stack([voltage, current], axis=-1).astype(np.float32)
        result = ContextSignal(
            values=self._readonly(values),
            mask=self._readonly(mask.copy()),
        )
        self._context_signal_cache[cache_key] = result
        return result

    def build(
        self,
        cell: CellData,
        current_cycle_number: int,
        beta: float,
        scalers: FoldScalers,
    ) -> FastFeature:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0,1]")
        delta, current, valid, reference_cycles = self.raw_feature(
            cell, current_cycle_number
        )
        if beta == 0.0:
            observed = np.zeros_like(valid)
        else:
            q_beta = self.grid_config.minimum + beta * (
                self.grid_config.maximum - self.grid_config.minimum
            )
            observed = valid & (self.grid <= q_beta + 1.0e-12)
        normalized_delta = np.zeros_like(delta)
        normalized_current = np.zeros_like(current)
        normalized_delta[observed] = (
            delta[observed] - scalers.delta_voltage_mean
        ) / scalers.delta_voltage_std
        normalized_current[observed] = (
            current[observed] - scalers.current_mean
        ) / scalers.current_std
        values = np.stack(
            [normalized_delta, normalized_current, observed.astype(np.float64)], axis=0
        ).astype(np.float32)
        return FastFeature(
            values=values,
            mask=observed,
            beta=float(beta),
            current_cycle=current_cycle_number,
            reference_cycles=reference_cycles,
        )

"""Leakage-safe fixed-grid representations of complete discharge V-Q curves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from battery_weighted_maml.matr_anp.config import QGridConfig
from battery_weighted_maml.matr_anp.data import CellData, DischargeCurve


class EpisodeUnavailable(ValueError):
    """A cell/cut cannot form a valid future-curve episode."""


@dataclass(frozen=True)
class VoltageScaler:
    mean: float
    std: float
    fit_cell_ids: tuple[str, ...]

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (value - self.mean) / self.std

    def inverse(self, value: np.ndarray) -> np.ndarray:
        return value * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fit_cell_ids"] = list(self.fit_cell_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VoltageScaler":
        raw = dict(payload)
        raw["fit_cell_ids"] = tuple(raw["fit_cell_ids"])
        return cls(**raw)


@dataclass(frozen=True)
class FullCurveGrid:
    feature: np.ndarray
    target_voltage: np.ndarray
    valid_mask: np.ndarray
    endpoint_fraction: float
    q_end: float


class CurveGridProcessor:
    """Interpolate complete curves on an absolute Q/Q_nominal grid.

    The fixed grid is configured before splitting. A curve is never divided by
    its own final capacity, so the endpoint remains an unknown prediction target.
    """

    def __init__(self, grid: QGridConfig, minimum_q_points: int):
        self.grid = np.linspace(grid.minimum, grid.maximum, grid.num_points, dtype=np.float64)
        self.q_min = float(grid.minimum)
        self.q_max = float(grid.maximum)
        self.minimum_q_points = int(minimum_q_points)
        self.q_coordinate = ((self.grid - self.q_min) / (self.q_max - self.q_min)).astype(
            np.float32
        )

    def build(self, curve: DischargeCurve, scaler: VoltageScaler) -> FullCurveGrid:
        q = np.asarray(curve.q, dtype=np.float64)
        voltage = np.asarray(curve.voltage_v, dtype=np.float64)
        finite = np.isfinite(q) & np.isfinite(voltage)
        q, voltage = q[finite], voltage[finite]
        inside = (q >= self.q_min) & (q <= self.q_max)
        q, voltage = q[inside], voltage[inside]
        if len(q) < self.minimum_q_points:
            raise EpisodeUnavailable("curve has too few points inside the q grid")
        q_end = float(curve.q[-1])
        if q_end > self.q_max + 1.0e-8:
            raise EpisodeUnavailable(
                f"q_end={q_end:.5g} exceeds q_grid.maximum={self.q_max:.5g}"
            )
        valid = (self.grid >= q[0]) & (self.grid <= q[-1])
        if np.count_nonzero(valid) < self.minimum_q_points:
            raise EpisodeUnavailable("resampled curve has too few valid grid points")
        target = np.zeros(len(self.grid), dtype=np.float32)
        target[valid] = scaler.normalize(np.interp(self.grid[valid], q, voltage)).astype(
            np.float32
        )
        feature = np.stack([target, valid.astype(np.float32)], axis=-1)
        return FullCurveGrid(feature, target, valid, q_end / self.q_max, q_end)


def fit_voltage_scaler(cells: Sequence[CellData]) -> VoltageScaler:
    values: list[np.ndarray] = []
    used: list[str] = []
    for cell in cells:
        curves = [cycle.discharge.voltage_v for cycle in cell.cycles if cycle.discharge is not None]
        if not curves:
            raise EpisodeUnavailable(f"{cell.cell_id}: no usable discharge curves")
        values.extend(curves)
        used.append(cell.cell_id)
    combined = np.concatenate(values).astype(np.float64)
    combined = combined[np.isfinite(combined)]
    if not combined.size:
        raise EpisodeUnavailable("training cells contain no finite voltage")
    return VoltageScaler(
        mean=float(np.mean(combined)),
        std=max(float(np.std(combined)), 1.0e-6),
        fit_cell_ids=tuple(sorted(used)),
    )

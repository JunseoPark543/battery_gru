"""Leakage-safe fixed-grid V/I features for completed and streaming cycles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from battery_weighted_maml.matr_anp.config import QGridConfig
from battery_weighted_maml.matr_anp.data import CellData, DischargeCurve


class EpisodeUnavailable(ValueError):
    """A requested cell/cycle cannot form a valid streaming episode."""


@dataclass(frozen=True)
class SignalScaler:
    voltage_mean: float
    voltage_std: float
    current_mean: float
    current_std: float
    fit_cell_ids: tuple[str, ...]

    def normalize_voltage(self, value: np.ndarray) -> np.ndarray:
        return (value - self.voltage_mean) / self.voltage_std

    def inverse_voltage(self, value: np.ndarray) -> np.ndarray:
        return value * self.voltage_std + self.voltage_mean

    def normalize_current(self, value: np.ndarray) -> np.ndarray:
        return (value - self.current_mean) / self.current_std

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["fit_cell_ids"] = list(self.fit_cell_ids)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SignalScaler":
        payload = dict(raw)
        payload["fit_cell_ids"] = tuple(payload["fit_cell_ids"])
        return cls(**payload)


@dataclass(frozen=True)
class GridCurve:
    feature: np.ndarray
    target_voltage: np.ndarray
    observed_mask: np.ndarray
    future_mask: np.ndarray
    valid_mask: np.ndarray
    q_cut: float
    q_end: float
    endpoint_fraction: float


class CycleGridProcessor:
    """Build grid features using only samples available at the current q cut."""

    def __init__(
        self,
        grid: QGridConfig,
        minimum_observed_points: int,
        minimum_future_points: int,
    ):
        self.grid = np.linspace(grid.minimum, grid.maximum, grid.num_points, dtype=np.float64)
        self.coordinate = ((self.grid - grid.minimum) / (grid.maximum - grid.minimum)).astype(
            np.float32
        )
        self.q_min = float(grid.minimum)
        self.q_max = float(grid.maximum)
        self.minimum_observed_points = int(minimum_observed_points)
        self.minimum_future_points = int(minimum_future_points)

    def _bounded(self, curve: DischargeCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(curve.q, dtype=np.float64)
        voltage = np.asarray(curve.voltage_v, dtype=np.float64)
        current = np.asarray(curve.current_a_magnitude, dtype=np.float64)
        within = (q >= self.q_min) & (q <= self.q_max)
        q, voltage, current = q[within], voltage[within], current[within]
        if len(q) < self.minimum_observed_points + self.minimum_future_points:
            raise EpisodeUnavailable("curve has too few points inside the q grid")
        if float(curve.q[-1]) > self.q_max + 1.0e-8:
            raise EpisodeUnavailable(
                f"q_end={float(curve.q[-1]):.5g} exceeds q_grid.maximum={self.q_max:.5g}"
            )
        return q, voltage, current

    def full(self, curve: DischargeCurve, scaler: SignalScaler) -> GridCurve:
        return self.build_prefix(curve, 1.0, scaler, require_future=False)

    def build_prefix(
        self,
        curve: DischargeCurve,
        beta: float,
        scaler: SignalScaler,
        *,
        require_future: bool = True,
    ) -> GridCurve:
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must lie in (0,1]")
        q, voltage, current = self._bounded(curve)
        if beta == 1.0:
            cut_index = len(q) - 1
        else:
            cut_index = int(round(beta * (len(q) - 1)))
            cut_index = min(max(cut_index, self.minimum_observed_points - 1), len(q) - 2)
        prefix_q = q[: cut_index + 1]
        prefix_v = voltage[: cut_index + 1]
        prefix_i = current[: cut_index + 1]
        q_cut = float(prefix_q[-1])
        valid = (self.grid >= q[0]) & (self.grid <= q[-1])
        observed = valid & (self.grid <= q_cut)
        future = valid & (self.grid > q_cut)
        if np.count_nonzero(observed) < self.minimum_observed_points:
            raise EpisodeUnavailable("prefix has too few resampled observed points")
        if require_future and np.count_nonzero(future) < self.minimum_future_points:
            raise EpisodeUnavailable("prefix has too few resampled future points")

        feature = np.zeros((len(self.grid), 3), dtype=np.float32)
        feature[observed, 0] = scaler.normalize_voltage(
            np.interp(self.grid[observed], prefix_q, prefix_v)
        ).astype(np.float32)
        feature[observed, 1] = scaler.normalize_current(
            np.interp(self.grid[observed], prefix_q, prefix_i)
        ).astype(np.float32)
        feature[:, 2] = observed.astype(np.float32)
        target_voltage = np.zeros(len(self.grid), dtype=np.float32)
        target_voltage[valid] = scaler.normalize_voltage(
            np.interp(self.grid[valid], q, voltage)
        ).astype(np.float32)
        return GridCurve(
            feature=feature,
            target_voltage=target_voltage,
            observed_mask=observed,
            future_mask=future,
            valid_mask=valid,
            q_cut=q_cut,
            q_end=float(curve.q[-1]),
            endpoint_fraction=float(curve.q[-1]) / self.q_max,
        )

    def empty_feature(self) -> np.ndarray:
        return np.zeros((len(self.grid), 3), dtype=np.float32)

    def observed_samples(
        self,
        q: np.ndarray,
        voltage_v: np.ndarray,
        current_a_magnitude: np.ndarray,
        scaler: SignalScaler,
    ) -> tuple[np.ndarray, float]:
        """Create an inference feature from samples observed so far only.

        Unlike ``build_prefix``, this method neither accepts nor requires a
        completed curve, a beta value, the eventual q_end, or future labels.
        """
        q_array = np.asarray(q, dtype=np.float64).reshape(-1)
        voltage = np.asarray(voltage_v, dtype=np.float64).reshape(-1)
        current = np.abs(np.asarray(current_a_magnitude, dtype=np.float64).reshape(-1))
        size = min(len(q_array), len(voltage), len(current))
        q_array, voltage, current = q_array[:size], voltage[:size], current[:size]
        finite = np.isfinite(q_array) & np.isfinite(voltage) & np.isfinite(current)
        within = finite & (q_array >= self.q_min) & (q_array <= self.q_max)
        q_array, voltage, current = q_array[within], voltage[within], current[within]
        if len(q_array) < 2:
            raise EpisodeUnavailable("live prefix needs at least two finite aligned samples")
        order = np.argsort(q_array, kind="stable")
        q_array, voltage, current = q_array[order], voltage[order], current[order]
        unique_q, inverse, counts = np.unique(q_array, return_inverse=True, return_counts=True)
        voltage_sum = np.zeros_like(unique_q)
        current_sum = np.zeros_like(unique_q)
        np.add.at(voltage_sum, inverse, voltage)
        np.add.at(current_sum, inverse, current)
        voltage = voltage_sum / counts
        current = current_sum / counts
        q_array = unique_q
        q_cut = float(q_array[-1])
        observed = (self.grid >= q_array[0]) & (self.grid <= q_cut)
        if np.count_nonzero(observed) < self.minimum_observed_points:
            raise EpisodeUnavailable(
                "live prefix has not reached enough fixed-grid q points for inference"
            )
        feature = self.empty_feature()
        feature[observed, 0] = scaler.normalize_voltage(
            np.interp(self.grid[observed], q_array, voltage)
        ).astype(np.float32)
        feature[observed, 1] = scaler.normalize_current(
            np.interp(self.grid[observed], q_array, current)
        ).astype(np.float32)
        feature[:, 2] = observed.astype(np.float32)
        return feature, q_cut / self.q_max


def integrate_discharge_q(
    time_s: np.ndarray,
    current_a: np.ndarray,
    nominal_capacity_ah: float,
) -> np.ndarray:
    """Causally integrate current into q=discharged_Ah/nominal_Ah."""
    if nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be positive")
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    current = np.abs(np.asarray(current_a, dtype=np.float64).reshape(-1))
    size = min(len(time), len(current))
    time, current = time[:size], current[:size]
    if size < 2 or not np.all(np.isfinite(time)) or not np.all(np.isfinite(current)):
        raise ValueError("time/current need at least two finite aligned samples")
    delta_t = np.diff(time, prepend=time[0])
    if np.any(delta_t < 0):
        raise ValueError("time_s must be nondecreasing")
    trapezoids = 0.5 * (current + np.roll(current, 1)) * delta_t / 3600.0
    trapezoids[0] = 0.0
    return np.cumsum(trapezoids) / float(nominal_capacity_ah)


def fit_signal_scaler(cells: Sequence[CellData]) -> SignalScaler:
    voltage: list[np.ndarray] = []
    current: list[np.ndarray] = []
    used: list[str] = []
    for cell in cells:
        curves = [cycle.discharge for cycle in cell.cycles if cycle.discharge is not None]
        if not curves:
            raise EpisodeUnavailable(f"{cell.cell_id}: no discharge curves")
        voltage.extend(curve.voltage_v for curve in curves)
        current.extend(curve.current_a_magnitude for curve in curves)
        used.append(cell.cell_id)
    all_voltage = np.concatenate(voltage).astype(np.float64)
    all_current = np.concatenate(current).astype(np.float64)
    all_voltage = all_voltage[np.isfinite(all_voltage)]
    all_current = all_current[np.isfinite(all_current)]
    if not all_voltage.size or not all_current.size:
        raise EpisodeUnavailable("training cells contain no finite V/I samples")
    return SignalScaler(
        voltage_mean=float(np.mean(all_voltage)),
        voltage_std=max(float(np.std(all_voltage)), 1.0e-6),
        current_mean=float(np.mean(all_current)),
        current_std=max(float(np.std(all_current)), 1.0e-6),
        fit_cell_ids=tuple(sorted(used)),
    )

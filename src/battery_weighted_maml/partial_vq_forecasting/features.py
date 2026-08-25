"""Real-time-safe partial V-Q samples and training-fold voltage scaling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from battery_weighted_maml.matr_anp.config import QGridConfig
from battery_weighted_maml.matr_anp.data import CellData, DischargeCurve


class EpisodeUnavailable(ValueError):
    """The selected curve cannot form a valid prefix/future episode."""


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
        raw = asdict(self)
        raw["fit_cell_ids"] = list(self.fit_cell_ids)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VoltageScaler":
        payload = dict(raw)
        payload["fit_cell_ids"] = tuple(payload["fit_cell_ids"])
        return cls(**payload)


@dataclass(frozen=True)
class PartialCurve:
    input_feature: np.ndarray
    q_coordinate: np.ndarray
    target_voltage: np.ndarray
    observed_mask: np.ndarray
    future_mask: np.ndarray
    valid_mask: np.ndarray
    q_cut: float
    q_end: float
    endpoint_fraction: float
    observed_points: int
    future_points: int


class PartialVQProcessor:
    """Resample only the observed prefix while retaining full-curve labels."""

    def __init__(
        self,
        grid: QGridConfig,
        minimum_observed_points: int,
        minimum_future_points: int,
    ):
        self.grid = np.linspace(grid.minimum, grid.maximum, grid.num_points, dtype=np.float64)
        self.q_min = float(grid.minimum)
        self.q_max = float(grid.maximum)
        self.minimum_observed_points = int(minimum_observed_points)
        self.minimum_future_points = int(minimum_future_points)

    def build(
        self,
        curve: DischargeCurve,
        beta: float,
        scaler: VoltageScaler,
    ) -> PartialCurve:
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0,1)")
        q = np.asarray(curve.q, dtype=np.float64)
        voltage = np.asarray(curve.voltage_v, dtype=np.float64)
        within = (q >= self.q_min) & (q <= self.q_max)
        q, voltage = q[within], voltage[within]
        if len(q) < self.minimum_observed_points + self.minimum_future_points:
            raise EpisodeUnavailable("curve has too few points inside the configured q range")
        actual_q_end = float(curve.q[-1])
        if actual_q_end > self.q_max + 1.0e-8:
            raise EpisodeUnavailable(
                f"q_end={actual_q_end:.5g} exceeds q_grid.maximum={self.q_max:.5g}; "
                "increase q_grid.maximum to forecast the full curve"
            )
        cut_index = int(round(beta * (len(q) - 1)))
        cut_index = min(max(cut_index, self.minimum_observed_points - 1), len(q) - 2)
        observed_q = q[: cut_index + 1]
        observed_voltage = voltage[: cut_index + 1]
        q_cut = float(observed_q[-1])
        valid_mask = (self.grid >= q[0]) & (self.grid <= q[-1])
        observed_mask = valid_mask & (self.grid <= q_cut)
        future_mask = valid_mask & (self.grid > q_cut)
        if np.count_nonzero(observed_mask) < self.minimum_observed_points:
            raise EpisodeUnavailable("resampled prefix has too few observed q points")
        if np.count_nonzero(future_mask) < self.minimum_future_points:
            raise EpisodeUnavailable("resampled remainder has too few future q points")

        # Input interpolation uses only raw samples available at q_cut. It never
        # touches the labels after q_cut.
        input_voltage = np.zeros(len(self.grid), dtype=np.float32)
        input_voltage[observed_mask] = scaler.normalize(
            np.interp(self.grid[observed_mask], observed_q, observed_voltage)
        ).astype(np.float32)
        target_voltage = np.zeros(len(self.grid), dtype=np.float32)
        target_voltage[valid_mask] = scaler.normalize(
            np.interp(self.grid[valid_mask], q, voltage)
        ).astype(np.float32)
        mask_channel = observed_mask.astype(np.float32)
        return PartialCurve(
            input_feature=np.stack([input_voltage, mask_channel], axis=-1),
            q_coordinate=((self.grid - self.q_min) / (self.q_max - self.q_min)).astype(np.float32),
            target_voltage=target_voltage,
            observed_mask=observed_mask,
            future_mask=future_mask,
            valid_mask=valid_mask,
            q_cut=q_cut,
            q_end=actual_q_end,
            endpoint_fraction=actual_q_end / self.q_max,
            observed_points=int(np.count_nonzero(observed_mask)),
            future_points=int(np.count_nonzero(future_mask)),
        )


def eligible_cycle_indices(
    cell: CellData,
    processor: PartialVQProcessor,
    scaler: VoltageScaler,
    minimum_position: int,
    beta: float = 0.5,
) -> list[int]:
    indices: list[int] = []
    for index in range(max(0, minimum_position - 1), len(cell.cycles)):
        curve = cell.cycles[index].discharge
        if curve is None:
            continue
        try:
            processor.build(curve, beta, scaler)
        except EpisodeUnavailable:
            continue
        indices.append(index)
    return indices


def fit_voltage_scaler(cells: Sequence[CellData], minimum_position: int) -> VoltageScaler:
    values: list[np.ndarray] = []
    used: list[str] = []
    for cell in cells:
        cell_values = [
            cycle.discharge.voltage_v
            for cycle in cell.cycles[max(0, minimum_position - 1) :]
            if cycle.discharge is not None
        ]
        if not cell_values:
            raise EpisodeUnavailable(f"{cell.cell_id}: no usable discharge voltage curves")
        values.extend(cell_values)
        used.append(cell.cell_id)
    combined = np.concatenate(values).astype(np.float64)
    combined = combined[np.isfinite(combined)]
    if not combined.size:
        raise EpisodeUnavailable("training cells contain no finite voltage samples")
    return VoltageScaler(
        mean=float(np.mean(combined)),
        std=max(float(np.std(combined)), 1.0e-6),
        fit_cell_ids=tuple(sorted(used)),
    )

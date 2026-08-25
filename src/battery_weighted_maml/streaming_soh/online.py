"""State-safe online inference for progressively arriving current-cycle samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from battery_weighted_maml.matr_anp.data import CellData

from .features import CycleGridProcessor, SignalScaler
from .model import StreamingSOHForecaster


@dataclass(frozen=True)
class OnlineForecast:
    cycles: np.ndarray
    soh_mean: np.ndarray
    soh_std: np.ndarray
    prefix_fraction_q_grid: float
    candidate_state_change_l2: float


class OnlineSOHSession:
    """Cache completed history and recompute the current-cycle candidate state.

    ``observe`` expects the entire prefix accumulated so far. Calling it many
    times for one cycle never commits that cycle to the completed GRU state.
    Model parameters are not changed.
    """

    def __init__(
        self,
        model: StreamingSOHForecaster,
        processor: CycleGridProcessor,
        scaler: SignalScaler,
        cell: CellData,
        *,
        current_cycle: int,
        forecast_cycles: Sequence[int],
        maximum_history_cycles: int,
        cycle_scale: float,
        device: torch.device,
    ):
        if maximum_history_cycles <= 0 or cycle_scale <= 0:
            raise ValueError("maximum_history_cycles and cycle_scale must be positive")
        forecast = np.asarray(forecast_cycles, dtype=np.int64)
        if not len(forecast) or np.any(forecast < current_cycle) or np.any(np.diff(forecast) < 0):
            raise ValueError("forecast_cycles must be sorted and start at/after current_cycle")
        history = [cycle for cycle in cell.cycles if cycle.cycle_number < current_cycle]
        if not history:
            raise ValueError(f"{cell.cell_id}: no completed cycle before {current_cycle}")
        history = history[-maximum_history_cycles:]
        curve_features: list[np.ndarray] = []
        for cycle in history:
            if cycle.discharge is None:
                curve_features.append(processor.empty_feature())
            else:
                try:
                    curve_features.append(processor.full(cycle.discharge, scaler).feature)
                except ValueError:
                    curve_features.append(processor.empty_feature())
        cycle_numbers = np.asarray([cycle.cycle_number for cycle in history], dtype=np.float32)
        soh = np.asarray([cycle.soh for cycle in history], dtype=np.float32)
        gaps = np.diff(cycle_numbers, prepend=cycle_numbers[0] - 1.0) / cycle_scale
        self.model = model.eval()
        self.processor = processor
        self.scaler = scaler
        self.device = device
        self.current_cycle = int(current_cycle)
        self.forecast_cycles = forecast
        self.cycle_scale = float(cycle_scale)
        self._last_cycle = int(cycle_numbers[-1])
        q_coordinate = torch.from_numpy(processor.coordinate[None]).to(device)
        with torch.no_grad():
            self._completed_state, self._last_soh = model.encode_history(
                torch.from_numpy(np.stack(curve_features)[None]).to(device),
                torch.from_numpy(soh[None]).to(device),
                torch.from_numpy(gaps.astype(np.float32)[None]).to(device),
                torch.from_numpy((cycle_numbers / cycle_scale).astype(np.float32)[None]).to(device),
                torch.ones((1, len(history)), dtype=torch.bool, device=device),
                q_coordinate,
            )
        self._q_coordinate = q_coordinate

    def observe(
        self,
        q: np.ndarray,
        voltage_v: np.ndarray,
        current_a: np.ndarray,
    ) -> OnlineForecast:
        feature, prefix_fraction = self.processor.observed_samples(
            q, voltage_v, current_a, self.scaler
        )
        current_curve = torch.from_numpy(feature[None]).to(self.device)
        gap = torch.tensor(
            [(self.current_cycle - self._last_cycle) / self.cycle_scale],
            dtype=torch.float32,
            device=self.device,
        )
        current_scaled = torch.tensor(
            [self.current_cycle / self.cycle_scale], dtype=torch.float32, device=self.device
        )
        prefix = torch.tensor([prefix_fraction], dtype=torch.float32, device=self.device)
        query = torch.from_numpy(
            (self.forecast_cycles.astype(np.float32) / self.cycle_scale)[None]
        ).to(self.device)
        with torch.no_grad():
            candidate = self.model.condition_current(
                self._completed_state,
                self._last_soh,
                current_curve,
                self._q_coordinate,
                gap,
                current_scaled,
                prefix,
            )
            mean, std = self.model.decode_trajectory(
                candidate, self._last_soh, current_scaled, query, prefix
            )
        state_change = torch.linalg.vector_norm(
            candidate[-1] - self._completed_state[-1]
        ).item()
        return OnlineForecast(
            cycles=self.forecast_cycles.copy(),
            soh_mean=mean[0].float().cpu().numpy(),
            soh_std=std[0].float().cpu().numpy(),
            prefix_fraction_q_grid=float(prefix_fraction),
            candidate_state_change_l2=float(state_change),
        )

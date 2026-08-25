"""Online context-prior updates for the streaming latent ANP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from battery_weighted_maml.matr_anp.data import CellData
from battery_weighted_maml.streaming_soh.features import CycleGridProcessor, SignalScaler

from .model import StreamingSOHLatentANP


@dataclass(frozen=True)
class OnlineANPForecast:
    cycles: np.ndarray
    soh_mean: np.ndarray
    predictive_std: np.ndarray
    epistemic_std: np.ndarray
    aleatoric_std: np.ndarray
    prior_mean: np.ndarray
    prior_std: np.ndarray
    prefix_fraction_q_grid: float
    candidate_state_change_l2: float


class OnlineLatentANPSession:
    """Keep completed context immutable while current-cycle samples arrive."""

    def __init__(
        self,
        model: StreamingSOHLatentANP,
        processor: CycleGridProcessor,
        scaler: SignalScaler,
        cell: CellData,
        *,
        current_cycle: int,
        forecast_cycles: Sequence[int],
        maximum_history_cycles: int,
        cycle_scale: float,
        latent_samples: int,
        device: torch.device,
    ):
        if maximum_history_cycles <= 0 or cycle_scale <= 0 or latent_samples <= 1:
            raise ValueError("history, cycle scale, and latent sample count are invalid")
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
        self.latent_samples = int(latent_samples)
        self._last_cycle = int(cycle_numbers[-1])
        self._history_mask = torch.ones(
            (1, len(history)), dtype=torch.bool, device=device
        )
        self._q_coordinate = torch.from_numpy(processor.coordinate[None]).to(device)
        with torch.no_grad():
            (
                self._completed_state,
                self._last_soh,
                self._completed_tokens,
            ) = model.encode_completed_context(
                torch.from_numpy(np.stack(curve_features)[None]).to(device),
                torch.from_numpy(soh[None]).to(device),
                torch.from_numpy(gaps.astype(np.float32)[None]).to(device),
                torch.from_numpy((cycle_numbers / cycle_scale).astype(np.float32)[None]).to(device),
                self._history_mask,
                self._q_coordinate,
            )

    def observe(
        self,
        q: np.ndarray,
        voltage_v: np.ndarray,
        current_a: np.ndarray,
    ) -> OnlineANPForecast:
        feature, prefix_fraction = self.processor.observed_samples(
            q, voltage_v, current_a, self.scaler
        )
        current_curve = torch.from_numpy(feature[None]).to(self.device)
        current_gap = torch.tensor(
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
                current_gap,
                current_scaled,
                prefix,
            )
            memory, mask = self.model.build_context(
                self._completed_tokens, self._history_mask, candidate
            )
            prior_mean, prior_std = self.model.prior_parameters(memory, mask)
            deterministic = self.model.deterministic_path(
                memory, mask, query, current_scaled
            )
            noise = torch.randn(
                self.latent_samples,
                *prior_mean.shape,
                dtype=prior_mean.dtype,
                device=prior_mean.device,
            )
            latent = prior_mean.unsqueeze(0) + prior_std.unsqueeze(0) * noise
            sample_mean, observation_std = self.model.decode_trajectory(
                deterministic,
                candidate,
                self._last_soh,
                current_scaled,
                query,
                prefix,
                latent,
            )
            mean = sample_mean.mean(dim=0)
            epistemic_variance = sample_mean.var(dim=0, unbiased=False)
            aleatoric_variance = observation_std.square().mean(dim=0)
            total_std = torch.sqrt(
                (epistemic_variance + aleatoric_variance).clamp_min(1.0e-12)
            )
        state_change = torch.linalg.vector_norm(
            candidate[-1] - self._completed_state[-1]
        ).item()
        return OnlineANPForecast(
            cycles=self.forecast_cycles.copy(),
            soh_mean=mean[0].float().cpu().numpy(),
            predictive_std=total_std[0].float().cpu().numpy(),
            epistemic_std=torch.sqrt(epistemic_variance[0]).float().cpu().numpy(),
            aleatoric_std=torch.sqrt(aleatoric_variance[0]).float().cpu().numpy(),
            prior_mean=prior_mean[0].float().cpu().numpy(),
            prior_std=prior_std[0].float().cpu().numpy(),
            prefix_fraction_q_grid=float(prefix_fraction),
            candidate_state_change_l2=float(state_change),
        )

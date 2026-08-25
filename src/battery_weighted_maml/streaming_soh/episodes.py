"""Cell-balanced causal episodes for streaming SOH trajectory learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from battery_weighted_maml.matr_anp.data import CellData

from .config import EpisodeConfig
from .features import CycleGridProcessor, EpisodeUnavailable, GridCurve, SignalScaler


@dataclass(frozen=True)
class StreamingEpisode:
    cell_id: str
    current_cycle: int
    beta: float
    history_curve: np.ndarray
    history_soh: np.ndarray
    history_gap: np.ndarray
    history_cycle_scaled: np.ndarray
    history_mask: np.ndarray
    current_curve: np.ndarray
    q_coordinate: np.ndarray
    current_gap: float
    current_cycle_scaled: float
    prefix_fraction: float
    query_cycle_scaled: np.ndarray
    target_soh: np.ndarray
    target_voltage: np.ndarray
    observed_q_mask: np.ndarray
    future_q_mask: np.ndarray
    valid_q_mask: np.ndarray
    endpoint_fraction: float
    q_cut: float
    q_end: float


@dataclass
class StreamingBatch:
    history_curve: torch.Tensor
    history_soh: torch.Tensor
    history_gap: torch.Tensor
    history_cycle_scaled: torch.Tensor
    history_mask: torch.Tensor
    current_curve: torch.Tensor
    q_coordinate: torch.Tensor
    current_gap: torch.Tensor
    current_cycle_scaled: torch.Tensor
    prefix_fraction: torch.Tensor
    query_cycle_scaled: torch.Tensor
    query_mask: torch.Tensor
    target_soh: torch.Tensor
    target_voltage: torch.Tensor
    observed_q_mask: torch.Tensor
    future_q_mask: torch.Tensor
    valid_q_mask: torch.Tensor
    endpoint_fraction: torch.Tensor

    def to(self, device: torch.device) -> "StreamingBatch":
        return StreamingBatch(
            **{
                name: value.to(device, non_blocking=True)
                for name, value in vars(self).items()
            }
        )


class EpisodeSampler:
    def __init__(
        self,
        config: EpisodeConfig,
        processor: CycleGridProcessor,
        scaler: SignalScaler,
    ):
        self.config = config
        self.processor = processor
        self.scaler = scaler
        self._candidate_cache: dict[str, list[int]] = {}
        self._full_curve_cache: dict[tuple[str, int], GridCurve] = {}

    def _full_curve(self, cell: CellData, index: int) -> GridCurve | None:
        cycle = cell.cycles[index]
        if cycle.discharge is None:
            return None
        key = (cell.cell_id, cycle.cycle_number)
        if key not in self._full_curve_cache:
            try:
                self._full_curve_cache[key] = self.processor.full(cycle.discharge, self.scaler)
            except EpisodeUnavailable:
                return None
        return self._full_curve_cache[key]

    def candidate_indices(self, cell: CellData) -> list[int]:
        cached = self._candidate_cache.get(cell.cell_id)
        if cached is not None:
            return cached
        candidates: list[int] = []
        for index, cycle in enumerate(cell.cycles):
            if cycle.cycle_number < self.config.minimum_current_cycle:
                continue
            if index < self.config.minimum_history_cycles:
                continue
            if len(cell.cycles) - index - 1 < self.config.minimum_future_cycles:
                continue
            if cycle.discharge is None:
                continue
            try:
                self.processor.build_prefix(cycle.discharge, 0.5, self.scaler)
            except EpisodeUnavailable:
                continue
            candidates.append(index)
        self._candidate_cache[cell.cell_id] = candidates
        return candidates

    def _select_index(self, cell: CellData, alpha: float) -> int:
        candidates = self.candidate_indices(cell)
        if not candidates:
            raise EpisodeUnavailable(f"{cell.cell_id}: no eligible streaming cycles")
        desired = int(round(alpha * (len(cell.cycles) - 1)))
        return min(candidates, key=lambda value: abs(value - desired))

    def _build(
        self,
        cell: CellData,
        index: int,
        beta: float,
        *,
        training: bool,
    ) -> StreamingEpisode:
        current = cell.cycles[index]
        if current.discharge is None:
            raise EpisodeUnavailable(f"{cell.cell_id} cycle {current.cycle_number}: no curve")
        prefix = self.processor.build_prefix(current.discharge, beta, self.scaler)

        history_indices = list(range(max(0, index - self.config.maximum_history_cycles), index))
        if len(history_indices) < self.config.minimum_history_cycles:
            raise EpisodeUnavailable(f"{cell.cell_id}: insufficient history")
        history_curve: list[np.ndarray] = []
        history_soh: list[float] = []
        history_cycles: list[int] = []
        for history_index in history_indices:
            cycle = cell.cycles[history_index]
            full = self._full_curve(cell, history_index)
            history_curve.append(
                full.feature if full is not None else self.processor.empty_feature()
            )
            history_soh.append(float(cycle.soh))
            history_cycles.append(int(cycle.cycle_number))
        cycles_array = np.asarray(history_cycles, dtype=np.float32)
        gaps = np.diff(cycles_array, prepend=cycles_array[0] - 1.0)

        future_indices = np.arange(index, len(cell.cycles), dtype=np.int64)
        if training and len(future_indices) > self.config.maximum_training_future_points:
            selected = np.linspace(
                0,
                len(future_indices) - 1,
                self.config.maximum_training_future_points,
                dtype=np.int64,
            )
            future_indices = future_indices[np.unique(selected)]
        query_cycles = np.asarray(
            [cell.cycles[item].cycle_number for item in future_indices], dtype=np.float32
        )
        target_soh = np.asarray(
            [cell.cycles[item].soh for item in future_indices], dtype=np.float32
        )
        scale = float(self.config.cycle_scale)
        return StreamingEpisode(
            cell_id=cell.cell_id,
            current_cycle=int(current.cycle_number),
            beta=float(beta),
            history_curve=np.stack(history_curve).astype(np.float32),
            history_soh=np.asarray(history_soh, dtype=np.float32),
            history_gap=(gaps / scale).astype(np.float32),
            history_cycle_scaled=(cycles_array / scale).astype(np.float32),
            history_mask=np.ones(len(history_indices), dtype=bool),
            current_curve=prefix.feature,
            q_coordinate=self.processor.coordinate,
            current_gap=float((current.cycle_number - history_cycles[-1]) / scale),
            current_cycle_scaled=float(current.cycle_number / scale),
            prefix_fraction=float(prefix.q_cut / self.processor.q_max),
            query_cycle_scaled=(query_cycles / scale).astype(np.float32),
            target_soh=target_soh,
            target_voltage=prefix.target_voltage,
            observed_q_mask=prefix.observed_mask,
            future_q_mask=prefix.future_mask,
            valid_q_mask=prefix.valid_mask,
            endpoint_fraction=prefix.endpoint_fraction,
            q_cut=prefix.q_cut,
            q_end=prefix.q_end,
        )

    def training(self, cell: CellData, rng: np.random.Generator) -> StreamingEpisode:
        low_alpha, high_alpha = self.config.training_cycle_alpha_range
        low_beta, high_beta = self.config.training_beta_range
        last_error: Exception | None = None
        for _ in range(12):
            alpha = float(rng.uniform(low_alpha, high_alpha))
            beta = float(rng.uniform(low_beta, high_beta))
            try:
                return self._build(cell, self._select_index(cell, alpha), beta, training=True)
            except EpisodeUnavailable as exc:
                last_error = exc
        raise EpisodeUnavailable(f"{cell.cell_id}: could not sample episode: {last_error}")

    def evaluation(
        self,
        cell: CellData,
        cycle_alpha: float,
        beta: float,
        *,
        current_cycle: int | None = None,
    ) -> StreamingEpisode:
        if current_cycle is None:
            index = self._select_index(cell, cycle_alpha)
        else:
            candidates = self.candidate_indices(cell)
            exact = [item for item in candidates if cell.cycles[item].cycle_number == current_cycle]
            if not exact:
                raise EpisodeUnavailable(
                    f"{cell.cell_id}: cycle {current_cycle} is not an eligible streaming cut"
                )
            index = exact[0]
        return self._build(cell, index, beta, training=False)


def collate_episodes(episodes: Sequence[StreamingEpisode]) -> StreamingBatch:
    if not episodes:
        raise ValueError("cannot collate an empty episode list")
    batch_size = len(episodes)
    history_length = max(len(item.history_soh) for item in episodes)
    query_length = max(len(item.target_soh) for item in episodes)
    q_points = episodes[0].current_curve.shape[0]
    channels = episodes[0].current_curve.shape[1]
    history_curve = np.zeros(
        (batch_size, history_length, q_points, channels), dtype=np.float32
    )
    history_soh = np.zeros((batch_size, history_length), dtype=np.float32)
    history_gap = np.zeros_like(history_soh)
    history_cycle_scaled = np.zeros_like(history_soh)
    history_mask = np.zeros((batch_size, history_length), dtype=bool)
    query_cycle_scaled = np.zeros((batch_size, query_length), dtype=np.float32)
    target_soh = np.zeros_like(query_cycle_scaled)
    query_mask = np.zeros((batch_size, query_length), dtype=bool)
    for row, episode in enumerate(episodes):
        history_count = len(episode.history_soh)
        offset = history_length - history_count
        history_curve[row, offset:] = episode.history_curve
        history_soh[row, offset:] = episode.history_soh
        history_gap[row, offset:] = episode.history_gap
        history_cycle_scaled[row, offset:] = episode.history_cycle_scaled
        history_mask[row, offset:] = episode.history_mask
        query_count = len(episode.target_soh)
        query_cycle_scaled[row, :query_count] = episode.query_cycle_scaled
        target_soh[row, :query_count] = episode.target_soh
        query_mask[row, :query_count] = True
    return StreamingBatch(
        history_curve=torch.from_numpy(history_curve),
        history_soh=torch.from_numpy(history_soh),
        history_gap=torch.from_numpy(history_gap),
        history_cycle_scaled=torch.from_numpy(history_cycle_scaled),
        history_mask=torch.from_numpy(history_mask),
        current_curve=torch.from_numpy(np.stack([item.current_curve for item in episodes])),
        q_coordinate=torch.from_numpy(np.stack([item.q_coordinate for item in episodes])),
        current_gap=torch.tensor([item.current_gap for item in episodes], dtype=torch.float32),
        current_cycle_scaled=torch.tensor(
            [item.current_cycle_scaled for item in episodes], dtype=torch.float32
        ),
        prefix_fraction=torch.tensor(
            [item.prefix_fraction for item in episodes], dtype=torch.float32
        ),
        query_cycle_scaled=torch.from_numpy(query_cycle_scaled),
        query_mask=torch.from_numpy(query_mask),
        target_soh=torch.from_numpy(target_soh),
        target_voltage=torch.from_numpy(
            np.stack([item.target_voltage for item in episodes])
        ),
        observed_q_mask=torch.from_numpy(
            np.stack([item.observed_q_mask for item in episodes])
        ),
        future_q_mask=torch.from_numpy(
            np.stack([item.future_q_mask for item in episodes])
        ),
        valid_q_mask=torch.from_numpy(
            np.stack([item.valid_q_mask for item in episodes])
        ),
        endpoint_fraction=torch.tensor(
            [item.endpoint_fraction for item in episodes], dtype=torch.float32
        ),
    )

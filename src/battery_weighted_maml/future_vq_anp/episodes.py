"""Cell-balanced episodes for forecasting all curves after an observed cut."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from battery_weighted_maml.matr_anp.data import CellData

from .config import EpisodeConfig
from .features import CurveGridProcessor, EpisodeUnavailable, FullCurveGrid, VoltageScaler


@dataclass(frozen=True)
class FutureVQEpisode:
    cell_id: str
    cut_cycle: int
    history_curve: np.ndarray
    history_endpoint_fraction: np.ndarray
    history_cycle_scaled: np.ndarray
    history_gap_scaled: np.ndarray
    history_mask: np.ndarray
    q_coordinate: np.ndarray
    query_cycle_numbers: np.ndarray
    query_cycle_scaled: np.ndarray
    target_voltage: np.ndarray
    target_q_mask: np.ndarray
    target_endpoint_fraction: np.ndarray


@dataclass
class FutureVQBatch:
    history_curve: torch.Tensor
    history_endpoint_fraction: torch.Tensor
    history_cycle_scaled: torch.Tensor
    history_gap_scaled: torch.Tensor
    history_mask: torch.Tensor
    q_coordinate: torch.Tensor
    query_cycle_scaled: torch.Tensor
    query_mask: torch.Tensor
    target_voltage: torch.Tensor
    target_q_mask: torch.Tensor
    target_endpoint_fraction: torch.Tensor

    def to(self, device: torch.device) -> "FutureVQBatch":
        return FutureVQBatch(
            **{name: value.to(device, non_blocking=True) for name, value in vars(self).items()}
        )


class EpisodeSampler:
    def __init__(
        self,
        config: EpisodeConfig,
        processor: CurveGridProcessor,
        scaler: VoltageScaler,
    ):
        self.config = config
        self.processor = processor
        self.scaler = scaler
        self._curve_cache: dict[str, list[tuple[int, FullCurveGrid]]] = {}

    def usable_curves(self, cell: CellData) -> list[tuple[int, FullCurveGrid]]:
        cached = self._curve_cache.get(cell.cell_id)
        if cached is not None:
            return cached
        output: list[tuple[int, FullCurveGrid]] = []
        for cycle in cell.cycles:
            if cycle.discharge is None:
                continue
            try:
                grid = self.processor.build(cycle.discharge, self.scaler)
            except EpisodeUnavailable:
                continue
            output.append((cycle.cycle_number, grid))
        self._curve_cache[cell.cell_id] = output
        return output

    def eligible_cut_indices(self, cell: CellData) -> list[int]:
        count = len(self.usable_curves(cell))
        first = self.config.history_cycles - 1
        last = count - self.config.minimum_future_cycles - 1
        return list(range(first, last + 1)) if last >= first else []

    @staticmethod
    def _even_subsample(indices: np.ndarray, maximum: int) -> np.ndarray:
        if len(indices) <= maximum:
            return indices
        positions = np.linspace(0, len(indices) - 1, maximum).round().astype(np.int64)
        return indices[np.unique(positions)]

    def _make(
        self,
        cell: CellData,
        cut_index: int,
        *,
        maximum_targets: int | None,
    ) -> FutureVQEpisode:
        curves = self.usable_curves(cell)
        if cut_index not in self.eligible_cut_indices(cell):
            raise EpisodeUnavailable(f"{cell.cell_id}: cut index {cut_index} is not eligible")
        history = curves[cut_index - self.config.history_cycles + 1 : cut_index + 1]
        future_indices = np.arange(cut_index + 1, len(curves), dtype=np.int64)
        if maximum_targets is not None:
            future_indices = self._even_subsample(future_indices, maximum_targets)
        future = [curves[int(index)] for index in future_indices]
        history_numbers = np.asarray([number for number, _ in history], dtype=np.float32)
        query_numbers = np.asarray([number for number, _ in future], dtype=np.int64)
        gaps = np.diff(history_numbers, prepend=history_numbers[0] - 1.0)
        return FutureVQEpisode(
            cell_id=cell.cell_id,
            cut_cycle=int(curves[cut_index][0]),
            history_curve=np.stack([grid.feature for _, grid in history]),
            history_endpoint_fraction=np.asarray(
                [grid.endpoint_fraction for _, grid in history], dtype=np.float32
            ),
            history_cycle_scaled=(history_numbers / self.config.cycle_scale).astype(np.float32),
            history_gap_scaled=(gaps / self.config.cycle_scale).astype(np.float32),
            history_mask=np.ones(len(history), dtype=bool),
            q_coordinate=self.processor.q_coordinate.copy(),
            query_cycle_numbers=query_numbers,
            query_cycle_scaled=(query_numbers / self.config.cycle_scale).astype(np.float32),
            target_voltage=np.stack([grid.target_voltage for _, grid in future]),
            target_q_mask=np.stack([grid.valid_mask for _, grid in future]),
            target_endpoint_fraction=np.asarray(
                [grid.endpoint_fraction for _, grid in future], dtype=np.float32
            ),
        )

    def training(self, cell: CellData, rng: np.random.Generator) -> FutureVQEpisode:
        eligible = self.eligible_cut_indices(cell)
        if not eligible:
            raise EpisodeUnavailable(f"{cell.cell_id}: no eligible history/future cut")
        low, high = self.config.training_cut_alpha_range
        low_pos = int(np.floor(low * (len(eligible) - 1)))
        high_pos = int(np.ceil(high * (len(eligible) - 1)))
        chosen = eligible[int(rng.integers(low_pos, high_pos + 1))]
        return self._make(
            cell,
            chosen,
            maximum_targets=self.config.maximum_training_future_cycles,
        )

    def evaluation(self, cell: CellData, cut_cycle: int) -> FutureVQEpisode:
        curves = self.usable_curves(cell)
        matches = [index for index, (number, _) in enumerate(curves) if number == cut_cycle]
        if not matches:
            raise EpisodeUnavailable(f"{cell.cell_id}: cycle {cut_cycle} has no usable curve")
        return self._make(cell, matches[0], maximum_targets=None)


def collate_episodes(episodes: Sequence[FutureVQEpisode]) -> FutureVQBatch:
    if not episodes:
        raise ValueError("cannot collate an empty episode list")
    maximum_future = max(len(item.query_cycle_scaled) for item in episodes)
    batch_size = len(episodes)
    q_points = len(episodes[0].q_coordinate)
    target = np.zeros((batch_size, maximum_future, q_points), dtype=np.float32)
    target_mask = np.zeros((batch_size, maximum_future, q_points), dtype=bool)
    endpoint = np.zeros((batch_size, maximum_future), dtype=np.float32)
    query = np.zeros((batch_size, maximum_future), dtype=np.float32)
    query_mask = np.zeros((batch_size, maximum_future), dtype=bool)
    for row, item in enumerate(episodes):
        count = len(item.query_cycle_scaled)
        target[row, :count] = item.target_voltage
        target_mask[row, :count] = item.target_q_mask
        endpoint[row, :count] = item.target_endpoint_fraction
        query[row, :count] = item.query_cycle_scaled
        query_mask[row, :count] = True
    return FutureVQBatch(
        history_curve=torch.from_numpy(np.stack([item.history_curve for item in episodes])),
        history_endpoint_fraction=torch.from_numpy(
            np.stack([item.history_endpoint_fraction for item in episodes])
        ),
        history_cycle_scaled=torch.from_numpy(
            np.stack([item.history_cycle_scaled for item in episodes])
        ),
        history_gap_scaled=torch.from_numpy(
            np.stack([item.history_gap_scaled for item in episodes])
        ),
        history_mask=torch.from_numpy(np.stack([item.history_mask for item in episodes])),
        q_coordinate=torch.from_numpy(np.stack([item.q_coordinate for item in episodes])),
        query_cycle_scaled=torch.from_numpy(query),
        query_mask=torch.from_numpy(query_mask),
        target_voltage=torch.from_numpy(target),
        target_q_mask=torch.from_numpy(target_mask),
        target_endpoint_fraction=torch.from_numpy(endpoint),
    )

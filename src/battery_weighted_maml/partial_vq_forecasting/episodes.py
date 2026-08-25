"""Cell-balanced within-cycle V-Q prefix episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from battery_weighted_maml.matr_anp.data import CellData

from .config import EpisodeConfig
from .features import (
    EpisodeUnavailable,
    PartialVQProcessor,
    VoltageScaler,
    eligible_cycle_indices,
)


@dataclass(frozen=True)
class VQEpisode:
    cell_id: str
    cycle_number: int
    beta: float
    input_feature: np.ndarray
    q_coordinate: np.ndarray
    target_voltage: np.ndarray
    observed_mask: np.ndarray
    future_mask: np.ndarray
    valid_mask: np.ndarray
    endpoint_fraction: float
    q_cut: float
    q_end: float
    observed_points: int
    future_points: int


@dataclass
class VQBatch:
    input_feature: torch.Tensor
    q_coordinate: torch.Tensor
    target_voltage: torch.Tensor
    observed_mask: torch.Tensor
    future_mask: torch.Tensor
    valid_mask: torch.Tensor
    endpoint_fraction: torch.Tensor

    def to(self, device: torch.device) -> "VQBatch":
        return VQBatch(**{
            name: value.to(device, non_blocking=True)
            for name, value in vars(self).items()
        })


class EpisodeSampler:
    def __init__(
        self,
        config: EpisodeConfig,
        processor: PartialVQProcessor,
        scaler: VoltageScaler,
    ):
        self.config = config
        self.processor = processor
        self.scaler = scaler
        self._training_index_cache: dict[str, list[int]] = {}
        self._evaluation_index_cache: dict[tuple[str, float], list[int]] = {}

    def _episode(self, cell: CellData, index: int, beta: float) -> VQEpisode:
        cycle = cell.cycles[index]
        if cycle.discharge is None:
            raise EpisodeUnavailable(f"{cell.cell_id} cycle {cycle.cycle_number}: no curve")
        partial = self.processor.build(cycle.discharge, beta, self.scaler)
        return VQEpisode(
            cell_id=cell.cell_id,
            cycle_number=cycle.cycle_number,
            beta=float(beta),
            input_feature=partial.input_feature,
            q_coordinate=partial.q_coordinate,
            target_voltage=partial.target_voltage,
            observed_mask=partial.observed_mask,
            future_mask=partial.future_mask,
            valid_mask=partial.valid_mask,
            endpoint_fraction=partial.endpoint_fraction,
            q_cut=partial.q_cut,
            q_end=partial.q_end,
            observed_points=partial.observed_points,
            future_points=partial.future_points,
        )

    def training(self, cell: CellData, rng: np.random.Generator) -> VQEpisode:
        indices = self._training_index_cache.get(cell.cell_id)
        if indices is None:
            indices = eligible_cycle_indices(
                cell,
                self.processor,
                self.scaler,
                self.config.minimum_cycle_position,
            )
            self._training_index_cache[cell.cell_id] = indices
        if not indices:
            raise EpisodeUnavailable(f"{cell.cell_id}: no eligible cycles")
        low, high = self.config.training_beta_range
        last_error: EpisodeUnavailable | None = None
        for _ in range(12):
            index = int(rng.choice(indices))
            beta = float(rng.uniform(low, high))
            try:
                return self._episode(cell, index, beta)
            except EpisodeUnavailable as exc:
                last_error = exc
        raise EpisodeUnavailable(f"{cell.cell_id}: sampled prefixes were invalid: {last_error}")

    def evaluation(self, cell: CellData, cycle_alpha: float, beta: float) -> VQEpisode:
        key = (cell.cell_id, float(beta))
        indices = self._evaluation_index_cache.get(key)
        if indices is None:
            indices = eligible_cycle_indices(
                cell,
                self.processor,
                self.scaler,
                self.config.minimum_cycle_position,
                beta=beta,
            )
            self._evaluation_index_cache[key] = indices
        if not indices:
            raise EpisodeUnavailable(f"{cell.cell_id}: no eligible evaluation cycles")
        desired = int(round(cycle_alpha * (len(cell.cycles) - 1)))
        index = min(indices, key=lambda candidate: abs(candidate - desired))
        return self._episode(cell, index, beta)


def collate_episodes(episodes: Sequence[VQEpisode]) -> VQBatch:
    if not episodes:
        raise ValueError("cannot collate an empty episode list")
    return VQBatch(
        input_feature=torch.from_numpy(np.stack([item.input_feature for item in episodes])),
        q_coordinate=torch.from_numpy(np.stack([item.q_coordinate for item in episodes])),
        target_voltage=torch.from_numpy(np.stack([item.target_voltage for item in episodes])),
        observed_mask=torch.from_numpy(np.stack([item.observed_mask for item in episodes])),
        future_mask=torch.from_numpy(np.stack([item.future_mask for item in episodes])),
        valid_mask=torch.from_numpy(np.stack([item.valid_mask for item in episodes])),
        endpoint_fraction=torch.tensor(
            [item.endpoint_fraction for item in episodes], dtype=torch.float32
        ),
    )

"""Online trajectory episodes with variable-length context and targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .config import EpisodeConfig
from .data import CellData
from .features import EpisodeUnavailable, FastFeature, FoldScalers, PartialIVProcessor


@dataclass(frozen=True)
class Episode:
    cell_id: str
    current_cycle: int
    alpha: float | None
    beta: float
    context_x: np.ndarray
    context_y: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    target_cycles: np.ndarray
    target_soh_raw: np.ndarray
    iv_feature: np.ndarray
    reference_cycles: tuple[int, ...]


@dataclass
class EpisodeBatch:
    context_x: torch.Tensor
    context_y: torch.Tensor
    context_mask: torch.Tensor
    target_x: torch.Tensor
    target_y: torch.Tensor
    target_mask: torch.Tensor
    iv_feature: torch.Tensor
    cell_ids: list[str]
    current_cycles: list[int]
    betas: list[float]

    def to(self, device: torch.device) -> "EpisodeBatch":
        for name in (
            "context_x", "context_y", "context_mask", "target_x", "target_y",
            "target_mask", "iv_feature",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self


class EpisodeSampler:
    def __init__(
        self,
        config: EpisodeConfig,
        processor: PartialIVProcessor,
        scalers: FoldScalers,
    ) -> None:
        self.config = config
        self.processor = processor
        self.scalers = scalers

    def _eligible_positions(self, cell: CellData) -> list[int]:
        count = len(cell.cycles)
        low = max(
            self.config.minimum_current_cycle_position - 1,
            int(np.floor(self.config.training_alpha_range[0] * count)) - 1,
            self.config.min_context_points,
        )
        high = min(count - 1, int(np.floor(self.config.training_alpha_range[1] * count)) - 1)
        positions: list[int] = []
        for position in range(low, high + 1):
            cycle = cell.cycles[position]
            if cycle.discharge is None:
                continue
            try:
                self.processor.raw_feature(cell, cycle.cycle_number)
            except EpisodeUnavailable:
                continue
            positions.append(position)
        return positions

    def sample_training(self, cell: CellData, rng: np.random.Generator) -> Episode:
        eligible = self._eligible_positions(cell)
        if not eligible:
            raise EpisodeUnavailable(f"{cell.cell_id}: no eligible training current cycle")
        position = int(rng.choice(eligible))
        beta = float(rng.choice(self.config.beta_values))
        return self._build(cell, position, beta, rng=rng, training=True, alpha=None)

    def evaluation(self, cell: CellData, alpha: float, beta: float) -> Episode:
        count = len(cell.cycles)
        # alpha selects a one-based position k=floor(alpha*N); array index is k-1.
        position = max(1, min(count - 1, int(np.floor(alpha * count)) - 1))
        return self._build(cell, position, beta, rng=None, training=False, alpha=alpha)

    def _build(
        self,
        cell: CellData,
        position: int,
        beta: float,
        *,
        rng: np.random.Generator | None,
        training: bool,
        alpha: float | None,
    ) -> Episode:
        if position <= 0 or position >= len(cell.cycles):
            raise EpisodeUnavailable(f"{cell.cell_id}: invalid current position {position}")
        prefix_indices = np.arange(position, dtype=np.int64)
        if prefix_indices.size < self.config.min_context_points:
            raise EpisodeUnavailable(f"{cell.cell_id}: too few SOH context cycles")
        if training:
            assert rng is not None
            maximum = min(self.config.max_context_points, prefix_indices.size)
            minimum = min(self.config.min_context_points, maximum)
            context_count = int(rng.integers(minimum, maximum + 1))
            latest = prefix_indices[-1]
            if context_count == 1:
                selected_context = np.asarray([latest])
            else:
                candidates = prefix_indices[:-1]
                chosen = rng.choice(candidates, size=context_count - 1, replace=False)
                selected_context = np.sort(np.concatenate([chosen, [latest]]))
        else:
            selected_context = _uniform_indices(
                prefix_indices, min(prefix_indices.size, self.config.max_context_points)
            )

        target_indices = np.arange(position, len(cell.cycles), dtype=np.int64)
        if training and target_indices.size > self.config.max_target_points:
            assert rng is not None
            endpoints = np.asarray([target_indices[0], target_indices[-1]], dtype=np.int64)
            middle = target_indices[1:-1]
            remaining = self.config.max_target_points - len(endpoints)
            selected_middle = rng.choice(middle, size=remaining, replace=False)
            selected_target = np.sort(np.concatenate([endpoints, selected_middle]))
        else:
            selected_target = target_indices

        cycles = cell.cycle_numbers
        soh = cell.soh
        current_cycle = int(cycles[position])
        fast: FastFeature = self.processor.build(
            cell, current_cycle, beta, self.scalers
        )
        return Episode(
            cell_id=cell.cell_id,
            current_cycle=current_cycle,
            alpha=alpha,
            beta=beta,
            context_x=self.scalers.transform_cycles(cycles[selected_context])[:, None].astype(np.float32),
            context_y=self.scalers.transform_soh(soh[selected_context])[:, None].astype(np.float32),
            target_x=self.scalers.transform_cycles(cycles[selected_target])[:, None].astype(np.float32),
            target_y=self.scalers.transform_soh(soh[selected_target])[:, None].astype(np.float32),
            target_cycles=cycles[selected_target].copy(),
            target_soh_raw=soh[selected_target].copy(),
            iv_feature=fast.values,
            reference_cycles=fast.reference_cycles,
        )


def _uniform_indices(indices: np.ndarray, count: int) -> np.ndarray:
    if count >= len(indices):
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    positions[-1] = len(indices) - 1
    return np.unique(indices[positions])


def collate_episodes(episodes: Sequence[Episode]) -> EpisodeBatch:
    if not episodes:
        raise ValueError("cannot collate an empty episode list")
    batch = len(episodes)
    max_context = max(len(episode.context_x) for episode in episodes)
    max_target = max(len(episode.target_x) for episode in episodes)
    q_length = episodes[0].iv_feature.shape[-1]
    context_x = torch.zeros(batch, max_context, 1, dtype=torch.float32)
    context_y = torch.zeros_like(context_x)
    context_mask = torch.zeros(batch, max_context, dtype=torch.bool)
    target_x = torch.zeros(batch, max_target, 1, dtype=torch.float32)
    target_y = torch.zeros_like(target_x)
    target_mask = torch.zeros(batch, max_target, dtype=torch.bool)
    iv_feature = torch.zeros(batch, 3, q_length, dtype=torch.float32)
    for index, episode in enumerate(episodes):
        context_count = len(episode.context_x)
        target_count = len(episode.target_x)
        context_x[index, :context_count] = torch.from_numpy(episode.context_x)
        context_y[index, :context_count] = torch.from_numpy(episode.context_y)
        context_mask[index, :context_count] = True
        target_x[index, :target_count] = torch.from_numpy(episode.target_x)
        target_y[index, :target_count] = torch.from_numpy(episode.target_y)
        target_mask[index, :target_count] = True
        if episode.iv_feature.shape != (3, q_length):
            raise ValueError("every I-V feature must share shape [3,Q]")
        iv_feature[index] = torch.from_numpy(episode.iv_feature)
    return EpisodeBatch(
        context_x=context_x,
        context_y=context_y,
        context_mask=context_mask,
        target_x=target_x,
        target_y=target_y,
        target_mask=target_mask,
        iv_feature=iv_feature,
        cell_ids=[episode.cell_id for episode in episodes],
        current_cycles=[episode.current_cycle for episode in episodes],
        betas=[episode.beta for episode in episodes],
    )

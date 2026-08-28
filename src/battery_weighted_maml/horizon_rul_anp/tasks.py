"""Leakage-safe horizon tasks where each ANP point is one battery cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .config import TaskConfig
from .data import LabeledCell, RULScalers


class TaskUnavailable(ValueError):
    """Raised when a horizon cannot form the requested inter-cell task."""


@dataclass(frozen=True)
class HorizonPoint:
    cell_id: str
    horizon: int
    prefix: np.ndarray  # [K,F], contains observations through cycle k only
    rul_normalized: float
    rul_cycles: float
    lifetime: int


@dataclass(frozen=True)
class HorizonTask:
    """One tau_k task; every context/query point uses exactly the same k."""

    horizon: int
    context: tuple[HorizonPoint, ...]
    query: tuple[HorizonPoint, ...]

    def validate(self) -> None:
        if not self.context or not self.query:
            raise ValueError("a horizon task needs non-empty context and query sets")
        points = (*self.context, *self.query)
        if any(point.horizon != self.horizon for point in points):
            raise ValueError("all points in tau_k must use the same horizon k")
        context_ids = {point.cell_id for point in self.context}
        query_ids = {point.cell_id for point in self.query}
        if context_ids & query_ids:
            raise ValueError("context/query battery cells must be disjoint")
        if any(point.lifetime <= self.horizon for point in points):
            raise ValueError("task includes a battery with lifetime <= k")
        if any(not np.isclose(point.rul_cycles, point.lifetime - self.horizon) for point in points):
            raise ValueError("RUL label does not equal lifetime minus horizon")


class HorizonTaskSampler:
    """Sample tau_k episodes without ever mixing a cell across data splits."""

    def __init__(self, config: TaskConfig, scalers: RULScalers):
        self.config = config
        self.scalers = scalers

    @staticmethod
    def eligible(cells: Sequence[LabeledCell], horizon: int) -> list[LabeledCell]:
        selected = []
        for item in cells:
            if item.lifetime <= horizon:
                continue
            cycles = item.cell.cycle_numbers
            index = int(np.searchsorted(cycles, horizon))
            if index < len(cycles) and int(cycles[index]) == int(horizon):
                selected.append(item)
        return selected

    def point(self, item: LabeledCell, horizon: int) -> HorizonPoint:
        if item.lifetime <= horizon:
            raise TaskUnavailable(
                f"{item.cell_id}: lifetime {item.lifetime} does not exceed k={horizon}"
            )
        try:
            prefix = self.scalers.prefix(item, horizon)
        except ValueError as exc:
            raise TaskUnavailable(str(exc)) from exc
        rul = float(item.lifetime - horizon)
        normalized = float(self.scalers.transform_rul(rul))
        return HorizonPoint(
            item.cell_id,
            int(horizon),
            prefix,
            normalized,
            rul,
            item.lifetime,
        )

    def sample_training(
        self,
        train_cells: Sequence[LabeledCell],
        rng: np.random.Generator,
    ) -> HorizonTask:
        """Sample one task; context and query are distinct training cells."""
        required = max(
            self.config.min_cells_per_task,
            self.config.context_size_min + self.config.query_size,
        )
        for _ in range(self.config.max_resample_attempts):
            horizon = int(
                rng.integers(
                    self.config.min_horizon,
                    self.config.max_horizon + 1,
                )
            )
            eligible = self.eligible(train_cells, horizon)
            if len(eligible) < required:
                continue
            maximum_context = min(
                self.config.context_size_max,
                len(eligible) - self.config.query_size,
            )
            if maximum_context < self.config.context_size_min:
                continue
            context_count = int(
                rng.integers(self.config.context_size_min, maximum_context + 1)
            )
            count = context_count + self.config.query_size
            indices = rng.choice(len(eligible), size=count, replace=False)
            context = tuple(
                self.point(eligible[int(index)], horizon)
                for index in indices[:context_count]
            )
            query = tuple(
                self.point(eligible[int(index)], horizon)
                for index in indices[context_count:]
            )
            task = HorizonTask(horizon, context, query)
            task.validate()
            return task
        raise TaskUnavailable(
            "could not sample a horizon with enough eligible training cells; "
            "reduce max_horizon/min_cells_per_task or inspect lifetime labels"
        )

    def evaluation(
        self,
        horizon: int,
        reference_cells: Sequence[LabeledCell],
        query_cells: Sequence[LabeledCell],
        *,
        context_size: int,
        seed: int,
    ) -> HorizonTask:
        """Use train references as context and unseen split cells as queries."""
        references = self.eligible(reference_cells, horizon)
        queries = self.eligible(query_cells, horizon)
        reference_ids = {item.cell_id for item in references}
        query_ids = {item.cell_id for item in queries}
        if reference_ids & query_ids:
            raise ValueError("reference/query split leakage in evaluation task")
        if not queries:
            raise TaskUnavailable(f"no query cells are valid at horizon {horizon}")
        if len(references) < self.config.context_size_min:
            raise TaskUnavailable(
                f"only {len(references)} reference cells are valid at horizon {horizon}"
            )
        count = min(int(context_size), len(references))
        if count < self.config.context_size_min:
            raise TaskUnavailable("evaluation context is smaller than configured minimum")
        rng = np.random.default_rng(int(seed) + 97_003 * int(horizon))
        indices = rng.choice(len(references), size=count, replace=False)
        context = tuple(
            self.point(references[int(index)], horizon) for index in indices
        )
        query = tuple(self.point(item, horizon) for item in queries)
        task = HorizonTask(int(horizon), context, query)
        task.validate()
        return task


@dataclass
class HorizonBatch:
    """Padded task tensors.

    Prefix tensors are ``[B_task,N_cell,K,F]``. ANP point masks are
    ``[B_task,N_cell]`` and RUL labels are ``[B_task,N_cell,1]``.
    """

    horizons: torch.Tensor
    context_prefix: torch.Tensor
    context_prefix_mask: torch.Tensor
    context_mask: torch.Tensor
    context_y: torch.Tensor
    query_prefix: torch.Tensor
    query_prefix_mask: torch.Tensor
    query_mask: torch.Tensor
    query_y: torch.Tensor
    query_rul_cycles: torch.Tensor
    query_lifetimes: torch.Tensor
    context_cell_ids: list[list[str]]
    query_cell_ids: list[list[str]]

    def to(self, device: torch.device) -> "HorizonBatch":
        for name in (
            "horizons",
            "context_prefix",
            "context_prefix_mask",
            "context_mask",
            "context_y",
            "query_prefix",
            "query_prefix_mask",
            "query_mask",
            "query_y",
            "query_rul_cycles",
            "query_lifetimes",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self


def collate_tasks(tasks: Sequence[HorizonTask]) -> HorizonBatch:
    if not tasks:
        raise ValueError("cannot collate an empty task list")
    for task in tasks:
        task.validate()
    batch_size = len(tasks)
    max_context = max(len(task.context) for task in tasks)
    max_query = max(len(task.query) for task in tasks)
    max_prefix = max(
        point.prefix.shape[0]
        for task in tasks
        for point in (*task.context, *task.query)
    )
    feature_dim = tasks[0].context[0].prefix.shape[1]
    for task in tasks:
        for point in (*task.context, *task.query):
            if point.prefix.ndim != 2 or point.prefix.shape[1] != feature_dim:
                raise ValueError("all prefix tensors must share feature dimension")

    context_prefix = torch.zeros(
        batch_size, max_context, max_prefix, feature_dim, dtype=torch.float32
    )
    context_prefix_mask = torch.zeros(
        batch_size, max_context, max_prefix, dtype=torch.bool
    )
    context_mask = torch.zeros(batch_size, max_context, dtype=torch.bool)
    context_y = torch.zeros(batch_size, max_context, 1, dtype=torch.float32)
    query_prefix = torch.zeros(
        batch_size, max_query, max_prefix, feature_dim, dtype=torch.float32
    )
    query_prefix_mask = torch.zeros(
        batch_size, max_query, max_prefix, dtype=torch.bool
    )
    query_mask = torch.zeros(batch_size, max_query, dtype=torch.bool)
    query_y = torch.zeros(batch_size, max_query, 1, dtype=torch.float32)
    query_rul = torch.zeros(batch_size, max_query, dtype=torch.float32)
    query_lifetimes = torch.zeros(batch_size, max_query, dtype=torch.int64)
    context_ids: list[list[str]] = []
    query_ids: list[list[str]] = []

    for batch_index, task in enumerate(tasks):
        context_ids.append([point.cell_id for point in task.context])
        query_ids.append([point.cell_id for point in task.query])
        for point_index, point in enumerate(task.context):
            length = point.prefix.shape[0]
            context_prefix[batch_index, point_index, :length] = torch.from_numpy(
                point.prefix
            )
            context_prefix_mask[batch_index, point_index, :length] = True
            context_mask[batch_index, point_index] = True
            context_y[batch_index, point_index, 0] = point.rul_normalized
        for point_index, point in enumerate(task.query):
            length = point.prefix.shape[0]
            query_prefix[batch_index, point_index, :length] = torch.from_numpy(
                point.prefix
            )
            query_prefix_mask[batch_index, point_index, :length] = True
            query_mask[batch_index, point_index] = True
            query_y[batch_index, point_index, 0] = point.rul_normalized
            query_rul[batch_index, point_index] = point.rul_cycles
            query_lifetimes[batch_index, point_index] = point.lifetime

    return HorizonBatch(
        horizons=torch.tensor([task.horizon for task in tasks], dtype=torch.int64),
        context_prefix=context_prefix,
        context_prefix_mask=context_prefix_mask,
        context_mask=context_mask,
        context_y=context_y,
        query_prefix=query_prefix,
        query_prefix_mask=query_prefix_mask,
        query_mask=query_mask,
        query_y=query_y,
        query_rul_cycles=query_rul,
        query_lifetimes=query_lifetimes,
        context_cell_ids=context_ids,
        query_cell_ids=query_ids,
    )

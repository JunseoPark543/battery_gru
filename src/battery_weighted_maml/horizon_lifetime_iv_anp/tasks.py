"""Same-horizon inter-cell lifetime tasks and padded tensor collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .config import TaskConfig
from .data import LabeledCell, LifetimeIVPrefixStore, LifetimeIVScalers


class TaskUnavailable(ValueError):
    """Raised when a requested horizon cannot form a valid task."""


@dataclass(frozen=True)
class LifetimePoint:
    cell_id: str
    horizon: int
    cycle_features: np.ndarray
    curves: np.ndarray
    curve_masks: np.ndarray
    lifetime_normalized: float
    lifetime_cycles: float


@dataclass(frozen=True)
class LifetimeTask:
    horizon: int
    context: tuple[LifetimePoint, ...]
    query: tuple[LifetimePoint, ...]

    def validate(self) -> None:
        if not self.context or not self.query:
            raise ValueError("lifetime task needs non-empty context and query")
        points = (*self.context, *self.query)
        if any(point.horizon != self.horizon for point in points):
            raise ValueError("all task points must share horizon k")
        context_ids = {point.cell_id for point in self.context}
        query_ids = {point.cell_id for point in self.query}
        if context_ids & query_ids:
            raise ValueError("context/query cells overlap")
        if any(point.lifetime_cycles <= self.horizon for point in points):
            raise ValueError("task contains a cell at or beyond EOL")


class LifetimeTaskSampler:
    def __init__(
        self,
        config: TaskConfig,
        scalers: LifetimeIVScalers,
        store: LifetimeIVPrefixStore,
    ) -> None:
        self.config = config
        self.scalers = scalers
        self.store = store

    @staticmethod
    def eligible(cells: Sequence[LabeledCell], horizon: int) -> list[LabeledCell]:
        output: list[LabeledCell] = []
        for item in cells:
            if item.lifetime <= horizon:
                continue
            cycles = item.cell.cycle_numbers
            position = int(np.searchsorted(cycles, int(horizon)))
            if position < len(cycles) and int(cycles[position]) == int(horizon):
                output.append(item)
        return output

    def point(self, item: LabeledCell, horizon: int) -> LifetimePoint:
        if item.lifetime <= horizon:
            raise TaskUnavailable(f"{item.cell_id}: EOL does not exceed k={horizon}")
        try:
            prefix = self.store.prefix(item, horizon)
        except ValueError as exc:
            raise TaskUnavailable(str(exc)) from exc
        return LifetimePoint(
            cell_id=item.cell_id,
            horizon=int(horizon),
            cycle_features=prefix.cycle_features,
            curves=prefix.curves,
            curve_masks=prefix.curve_masks,
            lifetime_normalized=float(self.scalers.transform_lifetime(item.lifetime)),
            lifetime_cycles=float(item.lifetime),
        )

    def sample_training(
        self,
        train_cells: Sequence[LabeledCell],
        rng: np.random.Generator,
    ) -> LifetimeTask:
        required = max(
            self.config.min_cells_per_task,
            self.config.context_size_min + self.config.query_size,
        )
        for _ in range(self.config.max_resample_attempts):
            horizon = int(rng.choice(self.config.horizons))
            eligible = self.eligible(train_cells, horizon)
            if len(eligible) < required:
                continue
            maximum_context = min(
                self.config.context_size_max,
                len(eligible) - self.config.query_size,
            )
            if maximum_context < self.config.context_size_min:
                continue
            context_size = int(
                rng.integers(self.config.context_size_min, maximum_context + 1)
            )
            selected = rng.choice(
                len(eligible),
                size=context_size + self.config.query_size,
                replace=False,
            )
            context = tuple(
                self.point(eligible[int(index)], horizon)
                for index in selected[:context_size]
            )
            query = tuple(
                self.point(eligible[int(index)], horizon)
                for index in selected[context_size:]
            )
            task = LifetimeTask(horizon, context, query)
            task.validate()
            return task
        raise TaskUnavailable(
            "could not sample a horizon with enough eligible cells; inspect EOL labels"
        )

    def sample_training_pair(
        self,
        train_cells: Sequence[LabeledCell],
        rng: np.random.Generator,
        horizon_gap: int,
    ) -> tuple[LifetimeTask, LifetimeTask]:
        """Sample two horizons with identical context/query cell identities."""
        horizon_set = set(self.config.horizons)
        pairs = [
            (int(early), int(early + horizon_gap))
            for early in self.config.horizons
            if early + horizon_gap in horizon_set
        ]
        if not pairs:
            raise TaskUnavailable(
                f"no configured horizon pair has gap={horizon_gap}"
            )
        required = max(
            self.config.min_cells_per_task,
            self.config.context_size_min + self.config.query_size,
        )
        for _ in range(self.config.max_resample_attempts):
            early, late = pairs[int(rng.integers(0, len(pairs)))]
            late_eligible = self.eligible(train_cells, late)
            eligible: list[LabeledCell] = []
            for item in late_eligible:
                cycles = item.cell.cycle_numbers
                position = int(np.searchsorted(cycles, early))
                if position < len(cycles) and int(cycles[position]) == early:
                    eligible.append(item)
            if len(eligible) < required:
                continue
            maximum_context = min(
                self.config.context_size_max,
                len(eligible) - self.config.query_size,
            )
            if maximum_context < self.config.context_size_min:
                continue
            context_size = int(
                rng.integers(self.config.context_size_min, maximum_context + 1)
            )
            selected = rng.choice(
                len(eligible),
                size=context_size + self.config.query_size,
                replace=False,
            )
            context_cells = [eligible[int(index)] for index in selected[:context_size]]
            query_cells = [eligible[int(index)] for index in selected[context_size:]]

            def build(horizon: int) -> LifetimeTask:
                task = LifetimeTask(
                    horizon,
                    tuple(self.point(item, horizon) for item in context_cells),
                    tuple(self.point(item, horizon) for item in query_cells),
                )
                task.validate()
                return task

            early_task, late_task = build(early), build(late)
            if [point.cell_id for point in early_task.context] != [
                point.cell_id for point in late_task.context
            ]:
                raise RuntimeError("paired context identities differ")
            if [point.cell_id for point in early_task.query] != [
                point.cell_id for point in late_task.query
            ]:
                raise RuntimeError("paired query identities differ")
            return early_task, late_task
        raise TaskUnavailable(
            "could not sample paired horizons with enough common train cells"
        )

    def evaluation(
        self,
        horizon: int,
        reference_cells: Sequence[LabeledCell],
        query_cells: Sequence[LabeledCell],
        *,
        context_size: int,
        seed: int,
    ) -> LifetimeTask:
        references = self.eligible(reference_cells, int(horizon))
        queries = self.eligible(query_cells, int(horizon))
        if {item.cell_id for item in references} & {item.cell_id for item in queries}:
            raise ValueError("reference/query split leakage")
        if not queries:
            raise TaskUnavailable(f"no query cells are valid at k={horizon}")
        count = min(int(context_size), len(references))
        if count < self.config.context_size_min:
            raise TaskUnavailable(f"only {len(references)} reference cells are valid")
        rng = np.random.default_rng(int(seed) + 97_003 * int(horizon))
        indices = rng.choice(len(references), size=count, replace=False)
        task = LifetimeTask(
            int(horizon),
            tuple(self.point(references[int(index)], horizon) for index in indices),
            tuple(self.point(item, horizon) for item in queries),
        )
        task.validate()
        return task


@dataclass
class LifetimeBatch:
    horizons: torch.Tensor
    context_cycles: torch.Tensor
    context_cycle_mask: torch.Tensor
    context_curves: torch.Tensor
    context_curve_mask: torch.Tensor
    context_point_mask: torch.Tensor
    context_y: torch.Tensor
    query_cycles: torch.Tensor
    query_cycle_mask: torch.Tensor
    query_curves: torch.Tensor
    query_curve_mask: torch.Tensor
    query_point_mask: torch.Tensor
    query_y: torch.Tensor
    query_lifetime_cycles: torch.Tensor
    context_cell_ids: list[list[str]]
    query_cell_ids: list[list[str]]

    def to(self, device: torch.device | str) -> "LifetimeBatch":
        for name in (
            "horizons", "context_cycles", "context_cycle_mask", "context_curves",
            "context_curve_mask", "context_point_mask", "context_y", "query_cycles",
            "query_cycle_mask", "query_curves", "query_curve_mask", "query_point_mask",
            "query_y", "query_lifetime_cycles",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self


def collate_tasks(tasks: Sequence[LifetimeTask]) -> LifetimeBatch:
    if not tasks:
        raise ValueError("cannot collate an empty task list")
    for task in tasks:
        task.validate()
    batch_size = len(tasks)
    max_context = max(len(task.context) for task in tasks)
    max_query = max(len(task.query) for task in tasks)
    max_cycles = max(
        point.cycle_features.shape[0]
        for task in tasks
        for point in (*task.context, *task.query)
    )
    q_points = tasks[0].context[0].curves.shape[1]

    def allocate(points: int):
        return (
            torch.zeros(batch_size, points, max_cycles, 2, dtype=torch.float32),
            torch.zeros(batch_size, points, max_cycles, dtype=torch.bool),
            torch.zeros(batch_size, points, max_cycles, q_points, 3, dtype=torch.float32),
            torch.zeros(batch_size, points, max_cycles, q_points, dtype=torch.bool),
            torch.zeros(batch_size, points, dtype=torch.bool),
            torch.zeros(batch_size, points, 1, dtype=torch.float32),
            torch.zeros(batch_size, points, dtype=torch.float32),
        )

    cc, ccm, cv, cvm, cpm, cy, cl = allocate(max_context)
    qc, qcm, qv, qvm, qpm, qy, ql = allocate(max_query)
    context_ids: list[list[str]] = []
    query_ids: list[list[str]] = []

    def fill(batch_index: int, points, tensors) -> None:
        cycles, cycle_mask, curves, curve_mask, point_mask, labels, lifetimes = tensors
        for point_index, point in enumerate(points):
            length = point.cycle_features.shape[0]
            cycles[batch_index, point_index, :length] = torch.from_numpy(point.cycle_features)
            cycle_mask[batch_index, point_index, :length] = True
            curves[batch_index, point_index, :length] = torch.from_numpy(point.curves)
            curve_mask[batch_index, point_index, :length] = torch.from_numpy(point.curve_masks)
            point_mask[batch_index, point_index] = True
            labels[batch_index, point_index, 0] = point.lifetime_normalized
            lifetimes[batch_index, point_index] = point.lifetime_cycles

    for batch_index, task in enumerate(tasks):
        context_ids.append([point.cell_id for point in task.context])
        query_ids.append([point.cell_id for point in task.query])
        fill(batch_index, task.context, (cc, ccm, cv, cvm, cpm, cy, cl))
        fill(batch_index, task.query, (qc, qcm, qv, qvm, qpm, qy, ql))
    return LifetimeBatch(
        horizons=torch.tensor([task.horizon for task in tasks], dtype=torch.int64),
        context_cycles=cc,
        context_cycle_mask=ccm,
        context_curves=cv,
        context_curve_mask=cvm,
        context_point_mask=cpm,
        context_y=cy,
        query_cycles=qc,
        query_cycle_mask=qcm,
        query_curves=qv,
        query_curve_mask=qvm,
        query_point_mask=qpm,
        query_y=qy,
        query_lifetime_cycles=ql,
        context_cell_ids=context_ids,
        query_cell_ids=query_ids,
    )

"""Leakage-safe, deliberately narrow views of battery trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def infer_family(file_name: str) -> str:
    upper = file_name.upper()
    if "_CX2_" in upper:
        return "CX2"
    if "_CS2_" in upper:
        return "CS2"
    raise ValueError(f"cannot infer CALCE family (CX2/CS2) from filename: {file_name}")


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FullCellTrajectory:
    """Complete preprocessed trajectory; keep outside all training APIs."""

    file_name: str
    cell_id: str
    family: str
    nominal_capacity_ah: float
    cycles: np.ndarray
    capacities_ah: np.ndarray
    soh: np.ndarray
    is_interpolated: np.ndarray
    true_eol_cycle: int | None
    raw_cycle_count: int
    missing_count_before: int
    missing_count_after: int

    def __post_init__(self) -> None:
        sizes = {len(self.cycles), len(self.capacities_ah), len(self.soh), len(self.is_interpolated)}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError(f"{self.file_name}: trajectory arrays must have one equal, nonzero length")
        object.__setattr__(self, "cycles", _readonly(self.cycles, np.int64))
        object.__setattr__(self, "capacities_ah", _readonly(self.capacities_ah, np.float64))
        object.__setattr__(self, "soh", _readonly(self.soh, np.float64))
        object.__setattr__(self, "is_interpolated", _readonly(self.is_interpolated, np.bool_))

    def source_task(self, history_length: int) -> "SourceTaskView":
        if len(self.soh) < history_length + 1:
            raise ValueError(
                f"{self.file_name}: needs at least L+1={history_length + 1} processed cycles, "
                f"found {len(self.soh)}"
            )
        return SourceTaskView(
            file_name=self.file_name,
            support_cycles=self.cycles[:history_length],
            support_soh=self.soh[:history_length],
            query_cycles=self.cycles[history_length:],
            query_soh=self.soh[history_length:],
        )

    def target_support(self, history_length: int) -> "TargetSupportView":
        if len(self.soh) < history_length + 1:
            raise ValueError(
                f"{self.file_name}: needs at least L+1={history_length + 1} processed cycles, "
                f"found {len(self.soh)}"
            )
        return TargetSupportView(
            file_name=self.file_name,
            cycles=self.cycles[:history_length],
            soh=self.soh[:history_length],
        )


@dataclass(frozen=True, slots=True)
class SourceTaskView:
    """A source task with its initial support and complete later query."""

    file_name: str
    support_cycles: np.ndarray
    support_soh: np.ndarray
    query_cycles: np.ndarray
    query_soh: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_cycles", _readonly(self.support_cycles, np.int64))
        object.__setattr__(self, "support_soh", _readonly(self.support_soh, np.float64))
        object.__setattr__(self, "query_cycles", _readonly(self.query_cycles, np.int64))
        object.__setattr__(self, "query_soh", _readonly(self.query_soh, np.float64))
        if len(self.support_soh) < 2 or len(self.query_soh) < 1:
            raise ValueError(f"{self.file_name}: source support/query cannot be empty")


@dataclass(frozen=True, slots=True)
class TargetSupportView:
    """Only the target's first L observations; it has no future or EOL field."""

    file_name: str
    cycles: np.ndarray
    soh: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycles", _readonly(self.cycles, np.int64))
        object.__setattr__(self, "soh", _readonly(self.soh, np.float64))
        if len(self.soh) < 2 or len(self.cycles) != len(self.soh):
            raise ValueError(f"{self.file_name}: target support must contain at least two aligned points")

    @property
    def history_length(self) -> int:
        return len(self.soh)


@dataclass(frozen=True, slots=True)
class TargetEvaluationView:
    """Target future and label, created only after training and adaptation."""

    file_name: str
    support_cycles: np.ndarray
    support_soh: np.ndarray
    future_cycles: np.ndarray
    future_soh: np.ndarray
    true_eol_cycle: int

    @classmethod
    def after_training(
        cls, full: FullCellTrajectory, history_length: int
    ) -> "TargetEvaluationView":
        if full.true_eol_cycle is None:
            raise ValueError(f"{full.file_name}: true EOL label is required for final evaluation")
        if len(full.soh) < history_length + 1:
            raise ValueError(f"{full.file_name}: insufficient target data for L={history_length}")
        return cls(
            file_name=full.file_name,
            support_cycles=_readonly(full.cycles[:history_length], np.int64),
            support_soh=_readonly(full.soh[:history_length], np.float64),
            future_cycles=_readonly(full.cycles[history_length:], np.int64),
            future_soh=_readonly(full.soh[history_length:], np.float64),
            true_eol_cycle=int(full.true_eol_cycle),
        )


"""Leakage-safe, deliberately narrow views of battery trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def build_support_features(
    soh: np.ndarray,
    mean_voltage_v: np.ndarray,
    feature_names: Sequence[str] = ("soh",),
) -> np.ndarray:
    """Build features using statistics from this support prefix only."""
    names = tuple(feature_names)
    if not names or names[0] != "soh":
        raise ValueError("feature_names must start with 'soh'")
    if len(names) != len(set(names)):
        raise ValueError("feature_names must not contain duplicates")

    support_soh = np.asarray(soh, dtype=np.float64).reshape(-1)
    voltage = np.asarray(mean_voltage_v, dtype=np.float64).reshape(-1)
    if len(support_soh) != len(voltage):
        raise ValueError("SOH and mean-voltage support arrays must be aligned")
    if not np.all(np.isfinite(support_soh)):
        raise ValueError("support SOH contains non-finite values")

    columns: list[np.ndarray] = []
    for name in names:
        if name == "soh":
            columns.append(support_soh)
        elif name == "voltage_mean":
            if not np.all(np.isfinite(voltage)):
                raise ValueError(
                    "voltage_mean was requested, but the support prefix contains missing voltage"
                )
            mean = float(voltage.mean())
            std = float(voltage.std(ddof=0))
            columns.append(np.zeros_like(voltage) if std < 1e-8 else (voltage - mean) / std)
        else:
            raise ValueError(f"unsupported model feature: {name}")
    return _readonly(np.column_stack(columns), np.float64)


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
    mean_voltage_v: np.ndarray | None = None

    def __post_init__(self) -> None:
        voltage = (
            np.full(len(self.soh), np.nan, dtype=np.float64)
            if self.mean_voltage_v is None
            else np.asarray(self.mean_voltage_v, dtype=np.float64)
        )
        sizes = {
            len(self.cycles), len(self.capacities_ah), len(self.soh),
            len(self.is_interpolated), len(voltage),
        }
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError(f"{self.file_name}: trajectory arrays must have one equal, nonzero length")
        object.__setattr__(self, "cycles", _readonly(self.cycles, np.int64))
        object.__setattr__(self, "capacities_ah", _readonly(self.capacities_ah, np.float64))
        object.__setattr__(self, "soh", _readonly(self.soh, np.float64))
        object.__setattr__(self, "is_interpolated", _readonly(self.is_interpolated, np.bool_))
        object.__setattr__(self, "mean_voltage_v", _readonly(voltage, np.float64))

    def source_task(
        self, history_length: int, feature_names: Sequence[str] = ("soh",)
    ) -> "SourceTaskView":
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
            support_features=build_support_features(
                self.soh[:history_length],
                self.mean_voltage_v[:history_length],
                feature_names,
            ),
            feature_names=tuple(feature_names),
        )

    def target_support(
        self, history_length: int, feature_names: Sequence[str] = ("soh",)
    ) -> "TargetSupportView":
        if len(self.soh) < history_length + 1:
            raise ValueError(
                f"{self.file_name}: needs at least L+1={history_length + 1} processed cycles, "
                f"found {len(self.soh)}"
            )
        return TargetSupportView(
            file_name=self.file_name,
            cycles=self.cycles[:history_length],
            soh=self.soh[:history_length],
            features=build_support_features(
                self.soh[:history_length],
                self.mean_voltage_v[:history_length],
                feature_names,
            ),
            feature_names=tuple(feature_names),
        )


@dataclass(frozen=True, slots=True)
class SourceTaskView:
    """A source task with its initial support and complete later query."""

    file_name: str
    support_cycles: np.ndarray
    support_soh: np.ndarray
    query_cycles: np.ndarray
    query_soh: np.ndarray
    support_features: np.ndarray | None = None
    feature_names: tuple[str, ...] = ("soh",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_cycles", _readonly(self.support_cycles, np.int64))
        object.__setattr__(self, "support_soh", _readonly(self.support_soh, np.float64))
        object.__setattr__(self, "query_cycles", _readonly(self.query_cycles, np.int64))
        object.__setattr__(self, "query_soh", _readonly(self.query_soh, np.float64))
        if len(self.support_soh) < 2 or len(self.query_soh) < 1:
            raise ValueError(f"{self.file_name}: source support/query cannot be empty")
        features = (
            self.support_soh[:, None]
            if self.support_features is None
            else np.asarray(self.support_features, dtype=np.float64)
        )
        names = tuple(self.feature_names)
        if features.shape != (len(self.support_soh), len(names)):
            raise ValueError(f"{self.file_name}: source support feature shape is invalid")
        if not np.all(np.isfinite(features)) or not np.allclose(features[:, 0], self.support_soh):
            raise ValueError(f"{self.file_name}: source features must be finite and start with SOH")
        object.__setattr__(self, "support_features", _readonly(features, np.float64))
        object.__setattr__(self, "feature_names", names)


@dataclass(frozen=True, slots=True)
class TargetSupportView:
    """Only the target's first L observations; it has no future or EOL field."""

    file_name: str
    cycles: np.ndarray
    soh: np.ndarray
    features: np.ndarray | None = None
    feature_names: tuple[str, ...] = ("soh",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycles", _readonly(self.cycles, np.int64))
        object.__setattr__(self, "soh", _readonly(self.soh, np.float64))
        if len(self.soh) < 2 or len(self.cycles) != len(self.soh):
            raise ValueError(f"{self.file_name}: target support must contain at least two aligned points")
        features = (
            self.soh[:, None]
            if self.features is None
            else np.asarray(self.features, dtype=np.float64)
        )
        names = tuple(self.feature_names)
        if features.shape != (len(self.soh), len(names)):
            raise ValueError(f"{self.file_name}: target support feature shape is invalid")
        if not np.all(np.isfinite(features)) or not np.allclose(features[:, 0], self.soh):
            raise ValueError(f"{self.file_name}: target features must be finite and start with SOH")
        object.__setattr__(self, "features", _readonly(features, np.float64))
        object.__setattr__(self, "feature_names", names)

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
    support_features: np.ndarray | None = None
    feature_names: tuple[str, ...] = ("soh",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_cycles", _readonly(self.support_cycles, np.int64))
        object.__setattr__(self, "support_soh", _readonly(self.support_soh, np.float64))
        object.__setattr__(self, "future_cycles", _readonly(self.future_cycles, np.int64))
        object.__setattr__(self, "future_soh", _readonly(self.future_soh, np.float64))
        features = (
            self.support_soh[:, None]
            if self.support_features is None
            else np.asarray(self.support_features, dtype=np.float64)
        )
        names = tuple(self.feature_names)
        if features.shape != (len(self.support_soh), len(names)):
            raise ValueError(f"{self.file_name}: evaluation support feature shape is invalid")
        if not np.all(np.isfinite(features)) or not np.allclose(features[:, 0], self.support_soh):
            raise ValueError(f"{self.file_name}: evaluation features must be finite and start with SOH")
        object.__setattr__(self, "support_features", _readonly(features, np.float64))
        object.__setattr__(self, "feature_names", names)

    @classmethod
    def after_training(
        cls,
        full: FullCellTrajectory,
        history_length: int,
        feature_names: Sequence[str] = ("soh",),
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
            support_features=build_support_features(
                full.soh[:history_length],
                full.mean_voltage_v[:history_length],
                feature_names,
            ),
            feature_names=tuple(feature_names),
        )

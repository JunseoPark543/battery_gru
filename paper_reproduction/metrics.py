"""SOH metrics plus paper-style last-hitting EOL/RUL."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def soh_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted SOH shapes differ")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(valid):
        raise ValueError("SOH metrics have no finite overlapping positions")
    y = actual[valid]
    yhat = predicted[valid]
    error = yhat - y
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if denominator == 0.0 else float(1.0 - np.sum(error ** 2) / denominator)
    return {
        # Backward-compatible raw SOH-fraction keys.
        "mae": mae,
        "rmse": rmse,
        # Paper tables report percentage points, e.g. 0.01156 -> 1.156%.
        "mae_percent": 100.0 * mae,
        "rmse_percent": 100.0 * rmse,
        "r2": r2,
        "point_count": int(len(y)),
    }


def last_hitting_eol(
    cycles: np.ndarray | Sequence[int],
    soh: np.ndarray | Sequence[float],
    threshold: float,
) -> int | None:
    """Return the final cycle whose finite SOH is strictly above threshold."""
    cycles = np.asarray(cycles).reshape(-1)
    soh = np.asarray(soh, dtype=float).reshape(-1)
    if cycles.size == 0 or soh.size == 0:
        raise ValueError("cycles and SOH must be nonempty")
    if cycles.shape != soh.shape:
        raise ValueError("cycles and SOH must have equal lengths")
    valid = np.isfinite(soh)
    cycles = cycles[valid]
    soh = soh[valid]
    above = np.flatnonzero(soh > threshold)
    if above.size == 0:
        return None
    return int(cycles[above[-1]])


def evaluate_prediction(
    actual_cycles: np.ndarray,
    actual_soh: np.ndarray,
    forecast_cycles: np.ndarray,
    forecast_soh: np.ndarray,
    current_cycle: int,
    current_soh: float,
    threshold: float,
) -> dict[str, Any]:
    actual_by_cycle = dict(zip(np.asarray(actual_cycles).tolist(), np.asarray(actual_soh).tolist()))
    overlap_actual = np.asarray(
        [actual_by_cycle.get(int(cycle), float("nan")) for cycle in forecast_cycles],
        dtype=float,
    )
    metrics: dict[str, Any] = soh_metrics(overlap_actual, forecast_soh)
    actual_eol = last_hitting_eol(actual_cycles, actual_soh, threshold)
    predicted_eol = last_hitting_eol(
        np.concatenate([[current_cycle], np.asarray(forecast_cycles, dtype=np.int64)]),
        np.concatenate([[current_soh], np.asarray(forecast_soh, dtype=float)]),
        threshold,
    )
    actual_rul = None if actual_eol is None else int(actual_eol - current_cycle)
    predicted_rul = None if predicted_eol is None else int(predicted_eol - current_cycle)
    rul_error = (
        None if actual_rul is None or predicted_rul is None
        else int(actual_rul - predicted_rul)
    )
    metrics.update(
        {
            "eol_threshold": threshold,
            "current_cycle": current_cycle,
            "actual_eol_cycle_last_hitting": actual_eol,
            "predicted_eol_cycle_last_hitting": predicted_eol,
            "actual_rul": actual_rul,
            "predicted_rul": predicted_rul,
            "rul_error_actual_minus_predicted": rul_error,
        }
    )
    return metrics

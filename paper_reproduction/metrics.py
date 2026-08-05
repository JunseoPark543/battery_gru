"""SOH metrics plus paper-style last-hitting EOL/RUL."""

from __future__ import annotations

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
    return {"mae": mae, "rmse": rmse, "r2": r2, "point_count": int(len(y))}


def last_hitting_eol(
    cycles: np.ndarray,
    soh: np.ndarray,
    threshold: float,
) -> int | None:
    """Return the cycle after the final SOH>threshold observation.

    A crossing is confirmed only when a later available point exists. This
    prevents declaring EOL at ``max_forecast_cycle + 1`` when every forecast
    remains above the threshold.
    """
    cycles = np.asarray(cycles, dtype=np.int64).reshape(-1)
    soh = np.asarray(soh, dtype=float).reshape(-1)
    if cycles.shape != soh.shape or len(cycles) == 0:
        raise ValueError("cycles and SOH must be nonempty aligned arrays")
    valid = np.isfinite(soh)
    cycles = cycles[valid]
    soh = soh[valid]
    if len(cycles) == 0:
        return None
    above = np.flatnonzero(soh > threshold)
    if above.size == 0:
        return int(cycles[0])
    last_above = int(above[-1])
    if last_above == len(cycles) - 1:
        return None
    return int(cycles[last_above + 1])


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

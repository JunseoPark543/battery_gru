"""Cycle-unit direct RUL metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    mape_epsilon_cycles: float,
) -> dict[str, float]:
    true = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if true.shape != predicted.shape or true.size == 0:
        raise ValueError("truth/prediction must be matching non-empty arrays")
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("RUL metrics require finite arrays")
    error = predicted - true
    denominator = np.maximum(np.abs(true), float(mape_epsilon_cycles))
    return {
        "rmse_cycles": float(np.sqrt(np.mean(np.square(error)))),
        "mae_cycles": float(np.mean(np.abs(error))),
        "mape_percent": float(100.0 * np.mean(np.abs(error) / denominator)),
        "bias_cycles": float(np.mean(error)),
    }


def interval_metrics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    true = np.asarray(truth, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if not (true.shape == low.shape == high.shape) or true.size == 0:
        raise ValueError("interval arrays must have matching non-empty shapes")
    return {
        "coverage": float(np.mean((true >= low) & (true <= high))),
        "mean_interval_width_cycles": float(np.mean(high - low)),
    }

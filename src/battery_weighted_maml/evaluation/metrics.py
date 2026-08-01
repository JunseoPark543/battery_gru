"""Per-target SOH curve metrics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import r2_score


def _mae(errors: np.ndarray) -> float:
    return float(np.mean(np.abs(errors))) if errors.size else float("nan")


def curve_metrics(
    cycles: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    true_eol_cycle: int,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or logging.getLogger("battery_weighted_maml")
    x = np.asarray(cycles, dtype=float)
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    valid = np.isfinite(x) & np.isfinite(truth) & np.isfinite(estimate)
    x, truth, estimate = x[valid], truth[valid], estimate[valid]
    if truth.size == 0:
        raise ValueError("no aligned finite future observations are available for evaluation")
    errors = estimate - truth
    reason: str | None = None
    if truth.size < 2:
        r2 = float("nan")
        reason = "fewer than two prediction points"
    elif np.allclose(truth, truth[0]):
        r2 = float("nan")
        reason = "observed future is constant"
    else:
        r2 = float(r2_score(truth, estimate))
    if reason:
        log.warning("R2 is NaN: %s", reason)
    before = errors[x <= true_eol_cycle]
    after = errors[x > true_eol_cycle]
    return {
        "mae": _mae(errors),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": r2,
        "r2_nan_reason": reason,
        "max_absolute_error": float(np.max(np.abs(errors))),
        "final_cycle_absolute_error": float(abs(errors[-1])),
        "pre_eol_mae": _mae(before),
        "post_eol_mae": _mae(after),
        "prediction_point_count": int(truth.size),
    }


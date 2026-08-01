"""EOL crossing and RUL metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def predicted_eol(
    cycles: np.ndarray,
    predicted_soh: np.ndarray,
    threshold: float,
    previous_cycle: float | None = None,
    previous_soh: float | None = None,
) -> tuple[float, float]:
    values = np.asarray(predicted_soh, dtype=float)
    x = np.asarray(cycles, dtype=float)
    valid = np.isfinite(values) & np.isfinite(x)
    crossing = np.flatnonzero(valid & (values <= threshold))
    if crossing.size == 0:
        return float("nan"), float("nan")
    index = int(crossing[0])
    discrete = float(x[index])
    if index > 0:
        x0, y0 = x[index - 1], values[index - 1]
    elif previous_cycle is not None and previous_soh is not None:
        x0, y0 = float(previous_cycle), float(previous_soh)
    else:
        return discrete, discrete
    x1, y1 = x[index], values[index]
    if not all(map(math.isfinite, [x0, y0, x1, y1])) or y1 == y0:
        interpolated = discrete
    else:
        fraction = (threshold - y0) / (y1 - y0)
        interpolated = float(x0 + fraction * (x1 - x0)) if 0 <= fraction <= 1 else discrete
    return discrete, interpolated


def rul_metrics(
    cycles: np.ndarray,
    predicted_soh_values: np.ndarray,
    true_eol_cycle: int,
    history_length: int,
    threshold: float,
    last_support_soh: float,
) -> dict[str, Any]:
    discrete, interpolated = predicted_eol(
        cycles,
        predicted_soh_values,
        threshold,
        previous_cycle=history_length,
        previous_soh=last_support_soh,
    )
    true_rul = float(true_eol_cycle - history_length)
    crossing_found = math.isfinite(discrete)
    predicted_rul = discrete - history_length if crossing_found else float("nan")
    signed = predicted_rul - true_rul if crossing_found else float("nan")
    absolute = abs(signed) if crossing_found else float("nan")
    relative = absolute / abs(true_rul) if crossing_found and true_rul != 0 else float("nan")
    return {
        "true_eol_cycle": int(true_eol_cycle),
        "predicted_eol_cycle_discrete": discrete,
        "predicted_eol_cycle_interpolated": interpolated,
        "true_rul": true_rul,
        "predicted_rul": predicted_rul,
        "signed_rul_error": signed,
        "absolute_rul_error": absolute,
        "relative_absolute_rul_error": relative,
        "crossing_found": crossing_found,
    }


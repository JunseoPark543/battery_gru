"""Recursive target forecast, alignment, and local result serialization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.task_views import TargetEvaluationView
from ..models.gru_seq2seq import GRUSeq2Seq
from .metrics import curve_metrics
from .rul import rul_metrics


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def evaluate_target(
    model: GRUSeq2Seq,
    target: TargetEvaluationView,
    history_length: int,
    max_forecast_cycle: int,
    eol_threshold: float,
    adaptation_mode: str,
    output_dir: str | Path,
    logger: logging.Logger | None = None,
) -> EvaluationResult:
    """Read target future only here, after adaptation, and save predictions/metrics."""
    log = logger or logging.getLogger("battery_weighted_maml")
    if max_forecast_cycle <= history_length:
        raise ValueError("max_forecast_cycle must be greater than history_length")
    horizon = max_forecast_cycle - history_length
    model.eval()
    forecast_tensor = model.recursive_forecast(target.support_features, horizon)
    forecast = forecast_tensor[0, :, 0].detach().cpu().numpy().astype(float)
    finite = np.isfinite(forecast)
    if not finite.all():
        first_bad = int(np.flatnonzero(~finite)[0])
        log.error("numerical failure in target forecast at zero-based horizon index %d", first_bad)
        forecast = forecast[:first_bad]
    forecast_cycles = np.arange(
        history_length + 1, history_length + 1 + len(forecast), dtype=np.int64
    )
    rul = rul_metrics(
        forecast_cycles,
        forecast,
        target.true_eol_cycle,
        history_length,
        eol_threshold,
        float(target.support_soh[-1]),
    )
    actual_by_cycle = dict(zip(target.future_cycles.tolist(), target.future_soh.tolist()))
    observed_future = np.asarray(
        [actual_by_cycle.get(int(cycle), float("nan")) for cycle in forecast_cycles], dtype=float
    )
    overlap = np.isfinite(observed_future)
    curve = curve_metrics(
        forecast_cycles[overlap],
        observed_future[overlap],
        forecast[overlap],
        target.true_eol_cycle,
        logger=log,
    )
    metrics = {**curve, **rul, "adaptation_mode": adaptation_mode}
    predicted_eol_value = rul["predicted_eol_cycle_discrete"]
    support_frame = pd.DataFrame(
        {
            "cycle": target.support_cycles,
            "observed_soh": target.support_soh,
            "predicted_soh": target.support_soh,
            "split": "support",
            "eol_threshold": eol_threshold,
            "true_eol_cycle": target.true_eol_cycle,
            "predicted_eol_cycle": predicted_eol_value,
            "adaptation_mode": adaptation_mode,
        }
    )
    future_frame = pd.DataFrame(
        {
            "cycle": forecast_cycles,
            "observed_soh": observed_future,
            "predicted_soh": forecast,
            "split": "future",
            "eol_threshold": eol_threshold,
            "true_eol_cycle": target.true_eol_cycle,
            "predicted_eol_cycle": predicted_eol_value,
            "adaptation_mode": adaptation_mode,
        }
    )
    predictions = pd.concat([support_frame, future_frame], ignore_index=True)
    root = Path(output_dir)
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(root / f"predictions/target_{adaptation_mode}_prediction.csv", index=False)
    (root / f"metrics/{adaptation_mode}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default, allow_nan=True), encoding="utf-8"
    )
    return EvaluationResult(predictions=predictions, metrics=metrics)

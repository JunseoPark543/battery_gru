"""Monte-Carlo ANP inference and trajectory metrics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .episodes import Episode, collate_episodes
from .features import FoldScalers


@dataclass(frozen=True)
class PredictionResult:
    mean: np.ndarray
    standard_deviation: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nll: np.ndarray


def _cuda_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def predict_episode(
    model: torch.nn.Module,
    episode: Episode,
    scalers: FoldScalers,
    device: torch.device,
    *,
    mc_samples: int,
    interval_level: float,
    seed: int,
) -> PredictionResult:
    """Predict from the latent prior only; target SOH is never passed to the model."""
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    batch = collate_episodes([episode]).to(device)
    target_count = len(episode.target_x)
    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []
    draws: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=_cuda_devices(device)):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for _ in range(mc_samples):
            output = model(
                batch.context_x,
                batch.context_y,
                batch.context_mask,
                batch.target_x,
                iv_feature=batch.iv_feature,
                sample_latent=True,
            )
            normalized_mean = output["mean"][0, :target_count, 0]
            normalized_std = output["std"][0, :target_count, 0]
            normalized_draw = normalized_mean + normalized_std * torch.randn_like(normalized_std)
            means.append(scalers.inverse_soh(normalized_mean.cpu().numpy()))
            standard_deviations.append(
                np.asarray(normalized_std.cpu().numpy(), dtype=np.float64) * scalers.soh_std
            )
            draws.append(scalers.inverse_soh(normalized_draw.cpu().numpy()))
    if was_training:
        model.train()
    component_means = np.stack(means)
    component_stds = np.stack(standard_deviations)
    predictive_draws = np.stack(draws)
    mean = component_means.mean(axis=0)
    total_variance = np.mean(component_stds**2 + component_means**2, axis=0) - mean**2
    standard_deviation = np.sqrt(np.maximum(total_variance, 1.0e-12))
    tail = (1.0 - interval_level) / 2.0
    lower, upper = np.quantile(predictive_draws, [tail, 1.0 - tail], axis=0)

    actual = episode.target_soh_raw[None, :]
    variance = np.maximum(component_stds**2, 1.0e-12)
    log_density = -0.5 * (
        np.square(actual - component_means) / variance
        + np.log(2.0 * math.pi * variance)
    )
    maximum = np.max(log_density, axis=0)
    log_mixture = maximum + np.log(np.mean(np.exp(log_density - maximum), axis=0))
    return PredictionResult(mean, standard_deviation, lower, upper, -log_mixture)


def trajectory_metrics(episode: Episode, result: PredictionResult) -> dict[str, float]:
    actual = episode.target_soh_raw
    # The benchmark target trajectory T begins at the current cycle n*.
    future = episode.target_cycles >= episode.current_cycle
    current = episode.target_cycles == episode.current_cycle
    if not np.any(current):
        raise RuntimeError("evaluation target must contain the current cycle")
    future_rmse = (
        float(np.sqrt(np.mean(np.square(result.mean[future] - actual[future]))))
        if np.any(future)
        else float("nan")
    )
    return {
        "future_rmse": future_rmse,
        "current_soh_abs_error": float(np.mean(np.abs(result.mean[current] - actual[current]))),
        "nll": float(np.mean(result.nll[future])) if np.any(future) else float("nan"),
        "coverage_95": (
            float(np.mean((actual[future] >= result.lower[future]) & (actual[future] <= result.upper[future])))
            if np.any(future)
            else float("nan")
        ),
        "interval_width_95": (
            float(np.mean(result.upper[future] - result.lower[future]))
            if np.any(future)
            else float("nan")
        ),
    }


def prediction_frame(
    episode: Episode,
    result: PredictionResult,
    scalers: FoldScalers,
    *,
    model_name: str,
    fold: int,
    seed: int,
    beta_label: float | None = None,
) -> pd.DataFrame:
    beta = episode.beta if beta_label is None else beta_label
    context_cycles = np.rint(
        episode.context_x[:, 0].astype(np.float64) * scalers.max_cycle_train
    ).astype(np.int64)
    context_actual = scalers.inverse_soh(episode.context_y[:, 0])
    context = pd.DataFrame(
        {
            "fold": fold,
            "seed": seed,
            "model": model_name,
            "cell_id": episode.cell_id,
            "alpha": episode.alpha,
            "beta": beta,
            "current_cycle": episode.current_cycle,
            "split": "context",
            "cycle": context_cycles,
            "actual_soh": context_actual,
            "predicted_mean": np.nan,
            "predicted_std": np.nan,
            "lower_95": np.nan,
            "upper_95": np.nan,
        }
    )
    target = pd.DataFrame(
        {
            "fold": fold,
            "seed": seed,
            "model": model_name,
            "cell_id": episode.cell_id,
            "alpha": episode.alpha,
            "beta": beta,
            "current_cycle": episode.current_cycle,
            "split": "target",
            "cycle": episode.target_cycles,
            "actual_soh": episode.target_soh_raw,
            "predicted_mean": result.mean,
            "predicted_std": result.standard_deviation,
            "lower_95": result.lower,
            "upper_95": result.upper,
        }
    )
    return pd.concat([context, target], ignore_index=True)


def measure_forward_latency(
    model: torch.nn.Module,
    episode: Episode,
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float]:
    """Measure pure prior forward time with a pre-collated device batch."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("latency warmup/repeats are invalid")
    batch = collate_episodes([episode]).to(device)
    was_training = model.training
    model.eval()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(
                batch.context_x, batch.context_y, batch.context_mask, batch.target_x,
                iv_feature=batch.iv_feature, sample_latent=False,
            )
        synchronize()
        values = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            model(
                batch.context_x, batch.context_y, batch.context_mask, batch.target_x,
                iv_feature=batch.iv_feature, sample_latent=False,
            )
            synchronize()
            values.append((time.perf_counter() - start) * 1_000.0)
    if was_training:
        model.train()
    return float(np.mean(values)), float(np.median(values)), float(np.std(values))

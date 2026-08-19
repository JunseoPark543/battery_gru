"""Streaming beta demo with immutable ANP parameters and latency accounting."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ExperimentConfig, load_config, resolve_data_root, save_config
from .data import load_matr_dataset
from .episodes import EpisodeSampler
from .evaluate import _validate_checkpoint_config
from .features import FoldScalers, PartialIVProcessor
from .inference import measure_forward_latency, predict_episode, prediction_frame
from .model import build_model
from .plotting import plot_trajectory_betas
from .runtime import parameter_checksum, resolve_device, seed_everything, write_json


def streaming_run(
    config: ExperimentConfig,
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    alpha: float,
    cell_id: str | None = None,
    output_dir: str | Path | None = None,
    mc_samples: int | None = None,
) -> Path:
    if not 0.0 < alpha < 1.0:
        raise ValueError("streaming alpha must lie in (0,1)")
    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MATR ANP checkpoint not found: {source}")
    payload: dict[str, Any] = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("dataset") != "MATR" or payload.get("algorithm") != "attentive_neural_process":
        raise ValueError("checkpoint is not a MATR Attentive Neural Process checkpoint")
    _validate_checkpoint_config(config, payload)
    seed_everything(config.seed, config.training.deterministic)
    device = resolve_device(config.device)
    spec = payload["model_spec"]
    model_name = str(spec["model_name"])
    model, rebuilt = build_model(
        model_name, config.model, resolved_hidden_dim=int(spec["hidden_dim"])
    )
    if rebuilt.parameter_count != int(spec["parameter_count"]):
        raise ValueError("checkpoint/model parameter count mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    state_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checksum_before = parameter_checksum(model)

    cells, audit = load_matr_dataset(data_root, config.data, tolerate_invalid_cells=True)
    by_id = {cell.cell_id: cell for cell in cells}
    test_cells = list(payload["fold_split"]["test_cells"])
    selected_id = cell_id or test_cells[0]
    if selected_id not in test_cells:
        raise ValueError(f"streaming cell must be held out in this fold; choices={test_cells}")
    if selected_id not in by_id:
        raise ValueError(f"streaming test cell is missing from data: {selected_id}")
    scalers = FoldScalers.from_dict(payload["scalers"])
    if set(scalers.fit_cell_ids) & set(test_cells):
        raise ValueError("test-cell leakage detected in checkpoint scalers")
    sampler = EpisodeSampler(
        config.episode,
        PartialIVProcessor(config.q_grid, config.data),
        scalers,
    )
    sample_count = mc_samples or config.evaluation.mc_samples
    destination = (
        Path(output_dir).resolve()
        if output_dir
        else source.parent.parent / "streaming" / f"{selected_id}_alpha{alpha:g}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    save_config(config, destination / "resolved_config.yaml")
    audit.to_csv(destination / "data_audit.csv", index=False)

    latency_rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    current_cycle: int | None = None
    for beta_index, beta in enumerate(config.episode.beta_values):
        preprocess_start = time.perf_counter()
        episode = sampler.evaluation(by_id[selected_id], alpha, beta)
        preprocessing_ms = (time.perf_counter() - preprocess_start) * 1_000.0
        inference_ms, inference_median_ms, inference_std_ms = measure_forward_latency(
            model,
            episode,
            device,
            warmup=config.evaluation.inference_warmup,
            repeats=config.evaluation.inference_repeats,
        )
        result = predict_episode(
            model,
            episode,
            scalers,
            device,
            mc_samples=sample_count,
            interval_level=config.evaluation.interval_level,
            seed=config.seed + 97_003,
        )
        frames.append(
            prediction_frame(
                episode,
                result,
                scalers,
                model_name=model_name,
                fold=int(payload["fold_split"]["fold"]),
                seed=config.seed,
            )
        )
        latency_rows.append(
            {
                "cell_id": selected_id,
                "alpha": alpha,
                "beta": beta,
                "current_cycle": episode.current_cycle,
                "preprocessing_latency_ms": preprocessing_ms,
                "inference_latency_mean_ms": inference_ms,
                "inference_latency_median_ms": inference_median_ms,
                "inference_latency_std_ms": inference_std_ms,
                "combined_latency_ms": preprocessing_ms + inference_ms,
                "warmup": config.evaluation.inference_warmup,
                "repeats": config.evaluation.inference_repeats,
                "device": str(device),
            }
        )
        current_cycle = episode.current_cycle
    predictions = pd.concat(frames, ignore_index=True)
    pd.DataFrame(latency_rows).to_csv(destination / "latency.csv", index=False)
    predictions.to_csv(destination / "trajectory_predictions.csv", index=False)
    assert current_cycle is not None
    plot_trajectory_betas(
        predictions,
        destination / "streaming_trajectory.png",
        cell_id=selected_id,
        alpha=alpha,
        current_cycle=current_cycle,
    )

    checksum_after = parameter_checksum(model)
    exactly_equal = all(
        torch.equal(state_before[name], value.detach().cpu())
        for name, value in model.state_dict().items()
    )
    if checksum_before != checksum_after or not exactly_equal:
        raise RuntimeError("streaming inference changed model parameters or buffers")
    write_json(
        destination / "streaming_manifest.json",
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(source),
            "dataset": "MATR",
            "model": model_name,
            "cell_id": selected_id,
            "alpha": alpha,
            "betas": config.episode.beta_values,
            "mc_samples": sample_count,
            "test_time_optimizer": False,
            "backward_called": False,
            "parameter_checksum_before": checksum_before,
            "parameter_checksum_after": checksum_after,
            "parameters_exactly_equal": exactly_equal,
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run immutable-parameter MATR streaming inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cell-id")
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    data_root = resolve_data_root(config, args.data_root)
    destination = streaming_run(
        config,
        args.checkpoint,
        data_root,
        alpha=args.alpha,
        cell_id=args.cell_id,
        output_dir=args.output_dir,
        mc_samples=args.mc_samples,
    )
    print(f"Streaming directory: {destination}")


if __name__ == "__main__":
    main()

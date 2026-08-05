"""CLI and end-to-end experiment orchestration."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, load_config, save_resolved_config
from .data.preprocess import preprocess_dataset, summary_record
from .data.task_views import FullCellTrajectory, TargetEvaluationView
from .evaluation.evaluator import evaluate_target
from .evaluation.plots import plot_target_prediction, plot_training_outputs
from .logging_utils import configure_logging, parameter_counts, resolve_device
from .meta.target_adaptation import adapt_target
from .models.gru_seq2seq import GRUSeq2Seq
from .seed import make_generator, seed_everything
from .training.checkpoint import load_checkpoint
from .training.trainer import WeightedMAMLTrainer


TARGETS = ("CALCE_CX2_37.pkl", "CALCE_CS2_37.pkl")


def _rooted(path: str | Path, project_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root / value


def select_source_names(
    trajectories: dict[str, FullCellTrajectory], target_name: str, source_mode: str
) -> list[str]:
    if target_name not in trajectories:
        raise FileNotFoundError(f"target cell was not found after preprocessing: {target_name}")
    if source_mode == "same_family":
        family = trajectories[target_name].family
        names = sorted(
            name for name, cell in trajectories.items()
            if name != target_name and cell.family == family
        )
    elif source_mode == "all_calce":
        names = sorted(name for name in trajectories if name != target_name)
    else:
        raise ValueError("source_mode must be 'same_family' or 'all_calce'")
    if not names:
        raise ValueError(f"no sources available for {target_name} in mode {source_mode}")
    return names


def _package_versions() -> dict[str, str | None]:
    packages = [
        "torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
        "PyYAML", "tqdm", "cvxpy", "osqp", "scs", "higher",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            check=True, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _manifest(
    config: ExperimentConfig,
    target: str,
    sources: Sequence[str],
    history_length: int,
    source_mode: str,
    project_root: Path,
) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "os": platform.platform(),
        "package_versions": _package_versions(),
        "seed": config.seed,
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(project_root),
        "target": target,
        "sources": list(sources),
        "history_length": history_length,
        "source_mode": source_mode,
        "resolved_config": config.to_dict(),
        "status": "running",
    }


def _make_run_tree(run_dir: Path) -> None:
    for child in (
        "logs", "checkpoints", "preprocessing", "training", "weights",
        "adaptation", "predictions", "metrics", "figures",
    ):
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def _create_model(config: ExperimentConfig) -> GRUSeq2Seq:
    return GRUSeq2Seq(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )


def _write_final_weights(
    run_dir: Path,
    source_names: Sequence[str],
    alpha: torch.Tensor,
    kernel: np.ndarray,
) -> None:
    values = alpha.detach().cpu().numpy()
    pd.DataFrame({"source": source_names, "alpha": values}).to_csv(
        run_dir / "weights/final_alpha.csv", index=False
    )
    pd.DataFrame(kernel, index=source_names, columns=source_names).to_csv(
        run_dir / "weights/kernel_matrix_final.csv", index_label="source"
    )


def run_experiment(
    config: ExperimentConfig,
    target_name: str,
    history_length: int,
    source_mode: str,
    project_root: str | Path = ".",
    smoke_test: bool = False,
    resume: str | Path | None = None,
    adapt_only: bool = False,
    trajectories: dict[str, FullCellTrajectory] | None = None,
) -> Path:
    """Run one target-specific meta-training, adaptation, and final evaluation."""
    if adapt_only and resume is None:
        raise ValueError("adapt_only requires a checkpoint through resume")
    root = Path(project_root).resolve()
    resolved = copy.deepcopy(config)
    resolved.source_mode = source_mode
    if history_length < 2:
        raise ValueError("history_length must be at least 2")
    if smoke_test:
        resolved.maml.meta_iterations = 2
        resolved.adaptation.fast_steps = [1, 2]
        resolved.adaptation.full_max_steps = 2
        resolved.adaptation.full_patience = 2
        resolved.data.history_lengths = [history_length]
        resolved.data.max_forecast_cycle = history_length + 10
        resolved.logging.log_interval = 1
        resolved.logging.checkpoint_interval = 1
        resolved.logging.save_alpha_interval = 1
    resolved.validate()
    seed_everything(resolved.seed)
    if trajectories is None:
        preprocessing_logger = configure_logging(None)
        trajectories = preprocess_dataset(
            _rooted(resolved.data.calce_dir, root),
            _rooted(resolved.data.label_path, root),
            root / "outputs/preprocessed",
            preprocessing_logger,
        )
    source_names = select_source_names(trajectories, target_name, source_mode)
    target_full = trajectories[target_name]
    # This is the only target object created before training; it physically contains just first L.
    target_support = target_full.target_support(history_length, resolved.model.features)
    source_tasks = [
        trajectories[name].source_task(history_length, resolved.model.features)
        for name in source_names
    ]
    if resume is not None:
        run_dir = Path(resume).resolve().parent.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        feature_tag = "-".join(
            feature.removesuffix("_mean") for feature in resolved.model.features
        )
        run_dir = root / "outputs/runs" / (
            f"{timestamp}_{source_mode}_{Path(target_name).stem}_"
            f"L{history_length}_{feature_tag}_seed{resolved.seed}"
        )
    _make_run_tree(run_dir)
    logger = configure_logging(run_dir / "logs/train.log")
    save_resolved_config(resolved, run_dir / "config_resolved.yaml")
    manifest = _manifest(
        resolved, target_name, source_names, history_length, source_mode, root
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.DataFrame([summary_record(trajectories[name]) for name in source_names]).to_csv(
        run_dir / "preprocessing/source_summary.csv", index=False
    )
    target_support_columns: dict[str, Any] = {
        "file_name": target_name,
        "cycle": target_support.cycles,
        "soh": target_support.soh,
        "mean_voltage_V": target_full.mean_voltage_v[:history_length],
        "mean_current_A": target_full.mean_current_a[:history_length],
    }
    for feature_index, feature_name in enumerate(target_support.feature_names):
        column_name = "input_soh" if feature_name == "soh" else f"input_{feature_name}_z"
        target_support_columns[column_name] = target_support.features[:, feature_index]
    pd.DataFrame(target_support_columns).to_csv(
        run_dir / "preprocessing/target_support.csv", index=False
    )
    device = resolve_device(resolved.device)
    model = _create_model(resolved)
    total, trainable = parameter_counts(model)
    logger.info(
        "Starting experiment target=%s source_mode=%s sources=%s L=%d device=%s "
        "seed=%d parameters=%d trainable=%d config=%s",
        target_name, source_mode, source_names, history_length, device, resolved.seed,
        total, trainable, resolved.to_dict(),
    )
    trainer = WeightedMAMLTrainer(
        model=model,
        source_tasks=source_tasks,
        target_support=target_support,
        config=resolved,
        device=device,
        run_dir=run_dir,
        source_mode=source_mode,
        logger=logger,
    )
    if adapt_only:
        adaptation_checkpoint = Path(resume).resolve()
        payload = load_checkpoint(adaptation_checkpoint, model, map_location=device)
        if payload["target_file_name"] != target_support.file_name:
            raise ValueError("adapt-only checkpoint target does not match requested target")
        if int(payload["L"]) != history_length:
            raise ValueError("adapt-only checkpoint L does not match requested history length")
        if payload["source_mode"] != source_mode:
            raise ValueError("adapt-only checkpoint source mode does not match")
        if list(payload["source_file_names"]) != source_names:
            raise ValueError("adapt-only checkpoint source list does not match")
        training_best_metric = float(payload["best_metric"])
        adaptation_checkpoint_iteration = int(payload["meta_iteration"])
        logger.info(
            "Skipping meta-training; adapting checkpoint=%s iteration=%d",
            adaptation_checkpoint,
            adaptation_checkpoint_iteration,
        )
    else:
        training_result = trainer.train(resume=resume)
        adaptation_checkpoint = run_dir / "checkpoints/best_source_meta_loss.pt"
        if not adaptation_checkpoint.is_file():
            raise RuntimeError("meta-training finished without a source-only best checkpoint")
        payload = load_checkpoint(adaptation_checkpoint, model, map_location=device)
        training_best_metric = training_result.best_metric
        adaptation_checkpoint_iteration = int(payload["meta_iteration"])
    model.to(device)
    selected_weights = trainer.compute_weights()
    fast_steps = list(resolved.adaptation.fast_steps)
    fast = adapt_target(
        model,
        target_support,
        max_steps=max(fast_steps),
        learning_rate=resolved.adaptation.learning_rate,
        batch_size=resolved.maml.inner_batch_size,
        teacher_forcing_ratio=resolved.model.teacher_forcing_ratio,
        device=device,
        generator=make_generator(resolved.seed + 2001, device),
        patience=None,
        capture_steps=fast_steps,
    )
    full = adapt_target(
        model,
        target_support,
        max_steps=resolved.adaptation.full_max_steps,
        learning_rate=resolved.adaptation.learning_rate,
        batch_size=resolved.maml.inner_batch_size,
        teacher_forcing_ratio=resolved.model.teacher_forcing_ratio,
        device=device,
        generator=make_generator(resolved.seed + 3001, device),
        patience=resolved.adaptation.full_patience,
    )
    fast.history["is_reported_step"] = fast.history["step"].isin(fast_steps)
    fast.history.to_csv(run_dir / "adaptation/fast_adaptation_history.csv", index=False)
    full.history.to_csv(run_dir / "adaptation/full_adaptation_history.csv", index=False)
    # Target future and true EOL become accessible only after both adaptations are complete.
    evaluation_view = TargetEvaluationView.after_training(
        target_full, history_length, resolved.model.features
    )
    fast_results: dict[int, Any] = {}
    for step in fast_steps:
        snapshot = fast.snapshots.get(step)
        if snapshot is None:
            raise RuntimeError(f"fast adaptation did not capture requested step {step}")
        mode = f"fast_{step}"
        result = evaluate_target(
            snapshot,
            evaluation_view,
            history_length,
            resolved.data.max_forecast_cycle,
            resolved.data.eol_threshold,
            mode,
            run_dir,
            logger,
        )
        plot_target_prediction(
            result.predictions,
            run_dir / f"figures/target_soh_{mode}.png",
            f"{target_name} fast adaptation ({step} steps)",
            metrics=result.metrics,
        )
        fast_results[step] = result
    pd.DataFrame(
        [
            {"fast_step": step, **result.metrics}
            for step, result in fast_results.items()
        ]
    ).to_csv(run_dir / "metrics/fast_metrics_by_step.csv", index=False)

    # Preserve the historical one-step filenames for downstream aggregation.
    legacy_fast_step = 1 if 1 in fast_results else min(fast_results)
    legacy_mode = f"fast_{legacy_fast_step}"
    shutil.copy2(
        run_dir / f"metrics/{legacy_mode}_metrics.json",
        run_dir / "metrics/fast_metrics.json",
    )
    shutil.copy2(
        run_dir / f"predictions/target_{legacy_mode}_prediction.csv",
        run_dir / "predictions/target_fast_prediction.csv",
    )
    shutil.copy2(
        run_dir / f"figures/target_soh_{legacy_mode}.png",
        run_dir / "figures/target_soh_fast.png",
    )
    full_result = evaluate_target(
        full.model, evaluation_view, history_length, resolved.data.max_forecast_cycle,
        resolved.data.eol_threshold, "full", run_dir, logger,
    )
    _write_final_weights(
        run_dir, source_names, selected_weights.alpha,
        selected_weights.kernel_ss,
    )
    plot_target_prediction(
        full_result.predictions, run_dir / "figures/target_soh_full.png",
        f"{target_name} full adaptation",
        metrics=full_result.metrics,
    )
    plot_training_outputs(run_dir)
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "fast_steps": fast_steps,
            "fast_metrics": fast_results[legacy_fast_step].metrics,
            "fast_metrics_legacy_step": legacy_fast_step,
            "fast_metrics_by_step": {
                str(step): result.metrics for step, result in fast_results.items()
            },
            "full_metrics": full_result.metrics,
            "best_source_meta_loss_ema": training_best_metric,
            "adapt_only": adapt_only,
            "adaptation_checkpoint": str(adaptation_checkpoint),
            "adaptation_checkpoint_iteration": adaptation_checkpoint_iteration,
        }
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )
    logger.info("Completed run: %s", run_dir)
    return run_dir


def _load_and_preprocess(config: ExperimentConfig, root: Path) -> dict[str, FullCellTrajectory]:
    logger = configure_logging(None)
    return preprocess_dataset(
        _rooted(config.data.calce_dir, root),
        _rooted(config.data.label_path, root),
        root / "outputs/preprocessed",
        logger,
    )


def run_batch(
    config: ExperimentConfig,
    modes: Sequence[str],
    project_root: str | Path = ".",
) -> list[Path]:
    root = Path(project_root).resolve()
    trajectories = _load_and_preprocess(config, root)
    results: list[Path] = []
    for mode in modes:
        for target in TARGETS:
            for history_length in config.data.history_lengths:
                results.append(
                    run_experiment(
                        config, target, history_length, mode, root,
                        trajectories=trajectories,
                    )
                )
    return results


def _read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_results(outputs_dir: str | Path) -> pd.DataFrame:
    runs_dir = Path(outputs_dir).resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")
    records: list[dict[str, Any]] = []
    weight_records: list[pd.DataFrame] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        fast_path = run_dir / "metrics/fast_metrics.json"
        full_path = run_dir / "metrics/full_metrics.json"
        if fast_path.is_file() and full_path.is_file():
            fast_metrics = _read_metrics(fast_path)
            full_metrics = _read_metrics(full_path)
            records.append(
                {
                    "run_dir": str(run_dir),
                    "target": manifest["target"],
                    "source_mode": manifest["source_mode"],
                    "history_length": manifest["history_length"],
                    "seed": manifest["seed"],
                    "primary_adaptation": "full",
                    **{f"fast_{key}": value for key, value in fast_metrics.items()},
                    **{f"full_{key}": value for key, value in full_metrics.items()},
                    **full_metrics,
                }
            )
        alpha_path = run_dir / "weights/final_alpha.csv"
        if alpha_path.is_file():
            alpha_frame = pd.read_csv(alpha_path)
            alpha_frame["target"] = manifest["target"]
            alpha_frame["source_mode"] = manifest["source_mode"]
            alpha_frame["history_length"] = manifest["history_length"]
            weight_records.append(alpha_frame)
    if not records:
        raise ValueError(f"no completed runs with metrics found in {runs_dir}")
    frame = pd.DataFrame(records)
    output_root = runs_dir.parent
    frame.to_csv(output_root / "experiment_summary.csv", index=False)
    (output_root / "experiment_summary.json").write_text(
        frame.to_json(orient="records", indent=2), encoding="utf-8"
    )
    _plot_comparisons(frame, pd.concat(weight_records, ignore_index=True) if weight_records else None, output_root)
    return frame


def _plot_comparisons(
    frame: pd.DataFrame, weights: pd.DataFrame | None, output_root: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_root / "comparison_figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = frame.copy()
    primary["label"] = (
        primary["target"].str.replace("CALCE_", "", regex=False).str.replace(".pkl", "", regex=False)
        + " L" + primary["history_length"].astype(str)
        + " " + primary["source_mode"]
    )
    for metric, name in (
        ("mae", "mae_by_target_L_source_mode.png"),
        ("rmse", "rmse_by_target_L_source_mode.png"),
        ("absolute_rul_error", "rul_error_by_target_L_source_mode.png"),
    ):
        fig, axis = plt.subplots(figsize=(max(8, len(primary) * 0.7), 5))
        axis.bar(primary["label"], primary[metric])
        axis.set(ylabel=metric, title=f"Full adaptation: {metric}")
        axis.tick_params(axis="x", rotation=55); axis.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(figures / name, dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5))
    if weights is not None and not weights.empty:
        grouped = weights.groupby("source", as_index=False)["alpha"].mean().sort_values("alpha")
        axis.barh(grouped["source"], grouped["alpha"])
        axis.set(xlabel="Mean final alpha", title="Source weight comparison")
    else:
        axis.text(0.5, 0.5, "No alpha data", ha="center", va="center")
    fig.tight_layout(); fig.savefig(figures / "source_weight_comparison.png", dpi=150); plt.close(fig)


def preprocess_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess CALCE battery pickles")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = load_config(args.config)
    _load_and_preprocess(config, root)


def single_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one target-aware weighted MAML experiment")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--target")
    parser.add_argument("--history-length", type=int)
    parser.add_argument("--source-mode", choices=["same_family", "all_calce"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument(
        "--adapt-only",
        action="store_true",
        help="load --resume checkpoint and skip all remaining meta-training",
    )
    args = parser.parse_args(argv)
    if args.adapt_only and not args.resume:
        parser.error("--adapt-only requires --resume CHECKPOINT")
    config = load_config(args.config)
    target, history_length, source_mode = args.target, args.history_length, args.source_mode
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        target = target or checkpoint.get("target_file_name")
        history_length = history_length or checkpoint.get("L")
        source_mode = source_mode or checkpoint.get("source_mode")
    if target is None or history_length is None or source_mode is None:
        parser.error("--target, --history-length, and --source-mode are required without --resume")
    run_experiment(
        config, str(target), int(history_length), str(source_mode), Path.cwd(),
        smoke_test=args.smoke_test, resume=args.resume, adapt_only=args.adapt_only,
    )


def batch_main(mode: str, argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Run {mode} CALCE experiments")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    modes = ["same_family", "all_calce"] if mode == "all" else [mode]
    run_batch(config, modes, Path.cwd())
    aggregate_results(Path.cwd() / "outputs/runs")


def aggregate_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed experiment runs")
    parser.add_argument("--outputs-dir", default="outputs/runs")
    args = parser.parse_args(argv)
    aggregate_results(args.outputs_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="battery-maml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preprocess")
    sub.add_parser("single")
    sub.add_parser("same-family")
    sub.add_parser("all-calce")
    sub.add_parser("all")
    sub.add_parser("aggregate")
    parsed, remaining = parser.parse_known_args(argv)
    dispatch = {
        "preprocess": lambda: preprocess_main(remaining),
        "single": lambda: single_main(remaining),
        "same-family": lambda: batch_main("same_family", remaining),
        "all-calce": lambda: batch_main("all_calce", remaining),
        "all": lambda: batch_main("all", remaining),
        "aggregate": lambda: aggregate_main(remaining),
    }
    dispatch[parsed.command]()


if __name__ == "__main__":
    main()

"""End-to-end HUST weighted-MAML GRU trajectory experiment."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
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

from battery_weighted_maml.data.task_views import FullCellTrajectory, TargetEvaluationView
from battery_weighted_maml.evaluation.evaluator import evaluate_target
from battery_weighted_maml.evaluation.plots import (
    plot_target_prediction,
    plot_training_outputs,
)
from battery_weighted_maml.logging_utils import (
    configure_logging,
    parameter_counts,
    resolve_device,
)
from battery_weighted_maml.meta.target_adaptation import adapt_target
from battery_weighted_maml.models.gru_seq2seq import GRUSeq2Seq
from battery_weighted_maml.seed import make_generator, seed_everything
from battery_weighted_maml.training.checkpoint import load_checkpoint
from battery_weighted_maml.training.trainer import WeightedMAMLTrainer

from .config import SOURCE_MODES, ExperimentConfig, load_config, save_resolved_config
from .data import (
    preprocess_dataset,
    protocol_counts,
    select_source_names,
    summary_record,
)
from .reporting import adaptation_comparison_frame, save_key_results_figure


HISTORY_LENGTH = 100


def _rooted(path: str | Path, project_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root / value


def _make_run_tree(run_dir: Path) -> None:
    for child in (
        "logs",
        "checkpoints",
        "preprocessing",
        "training",
        "weights",
        "adaptation",
        "predictions",
        "metrics",
        "figures",
    ):
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def _package_versions() -> dict[str, str | None]:
    packages = [
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
        "tqdm",
        "cvxpy",
        "osqp",
        "scs",
        "higher",
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
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _manifest(
    config: ExperimentConfig,
    target: str,
    sources: Sequence[str],
    source_mode: str,
    project_root: Path,
) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "experiment": "hust_soh_voltage_current_weighted_maml_gru",
        "dataset": "HUST",
        "prediction_target": "future_soh_trajectory",
        "input_definition": {
            "cycles": "1..100",
            "channels": ["soh", "cycle_mean_voltage", "cycle_mean_current"],
            "normalization": {
                "soh": "raw capacity / nominal capacity",
                "voltage": "per-cell support-only z-score",
                "current": "per-cell support-only z-score",
            },
        },
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
        "source_protocol_counts": protocol_counts_from_names(sources),
        "history_length": HISTORY_LENGTH,
        "source_mode": source_mode,
        "resolved_config": config.to_dict(),
        "status": "running",
    }


def protocol_counts_from_names(names: Sequence[str]) -> dict[str, int]:
    from .data import parse_protocol

    counts: dict[str, int] = {}
    for name in names:
        protocol, _ = parse_protocol(name)
        counts[protocol] = counts.get(protocol, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0].split("_")[-1])))


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
    pd.DataFrame(
        {"source": source_names, "alpha": alpha.detach().cpu().numpy()}
    ).to_csv(run_dir / "weights/final_alpha.csv", index=False)
    pd.DataFrame(kernel, index=source_names, columns=source_names).to_csv(
        run_dir / "weights/kernel_matrix_final.csv", index_label="source"
    )


def _run_dir_name(
    source_mode: str, target_name: str, config: ExperimentConfig
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return (
        f"{timestamp}_{source_mode}_{Path(target_name).stem}_L100_"
        f"soh-voltage-current_weighted-maml_seed{config.seed}"
    )


def run_experiment(
    config: ExperimentConfig,
    target_name: str,
    source_mode: str,
    project_root: str | Path = ".",
    *,
    smoke_test: bool = False,
    resume: str | Path | None = None,
    adapt_only: bool = False,
    trajectories: dict[str, FullCellTrajectory] | None = None,
) -> Path:
    if adapt_only and resume is None:
        raise ValueError("adapt_only requires a checkpoint through resume")
    root = Path(project_root).resolve()
    resolved = copy.deepcopy(config)
    resolved.source_mode = source_mode
    if smoke_test:
        resolved.maml.meta_iterations = 2
        resolved.adaptation.fast_steps = [1, 2]
        resolved.adaptation.full_max_steps = 2
        resolved.adaptation.full_patience = 2
        resolved.data.max_forecast_cycle = HISTORY_LENGTH + 10
        resolved.logging.log_interval = 1
        resolved.logging.checkpoint_interval = 1
        resolved.logging.save_alpha_interval = 1
    resolved.validate()
    seed_everything(resolved.seed)

    if trajectories is None:
        preprocessing_logger = configure_logging(None)
        trajectories = preprocess_dataset(
            _rooted(resolved.data.hust_dir, root),
            _rooted(resolved.data.label_path, root),
            root / "outputs/hust_weighted_gru/preprocessed",
            expected_protocol_count=resolved.data.expected_protocol_count,
            logger=preprocessing_logger,
        )
    source_names = select_source_names(trajectories, target_name, source_mode)
    target_full = trajectories[target_name]
    target_support = target_full.target_support(HISTORY_LENGTH, resolved.model.features)
    source_tasks = [
        trajectories[name].source_task(HISTORY_LENGTH, resolved.model.features)
        for name in source_names
    ]

    if resume is not None:
        run_dir = Path(resume).resolve().parent.parent
    else:
        run_dir = root / "outputs/hust_weighted_gru/runs" / _run_dir_name(
            source_mode, target_name, resolved
        )
    _make_run_tree(run_dir)
    logger = configure_logging(run_dir / "logs/train.log")
    if len(source_names) > 20:
        logger.warning(
            "source_mode=%s selected %d cells; full second-order MAML evaluates every "
            "source query at every iteration and may be very slow",
            source_mode,
            len(source_names),
        )
    save_resolved_config(resolved, run_dir / "config_resolved.yaml")
    manifest = _manifest(resolved, target_name, source_names, source_mode, root)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )

    pd.DataFrame([summary_record(trajectories[name]) for name in source_names]).to_csv(
        run_dir / "preprocessing/source_summary.csv", index=False
    )
    pd.DataFrame([summary_record(target_full)]).to_csv(
        run_dir / "preprocessing/target_summary.csv", index=False
    )
    target_support_columns: dict[str, Any] = {
        "file_name": target_name,
        "cycle": target_support.cycles,
        "soh": target_support.soh,
        "mean_voltage_V": target_full.mean_voltage_v[:HISTORY_LENGTH],
        "mean_current_A": target_full.mean_current_a[:HISTORY_LENGTH],
    }
    for index, feature in enumerate(target_support.feature_names):
        column = "input_soh" if feature == "soh" else f"input_{feature}_z"
        target_support_columns[column] = target_support.features[:, index]
    pd.DataFrame(target_support_columns).to_csv(
        run_dir / "preprocessing/target_support.csv", index=False
    )

    device = resolve_device(resolved.device)
    model = _create_model(resolved)
    total, trainable = parameter_counts(model)
    logger.info(
        "Starting HUST experiment target=%s target_protocol=%s source_mode=%s "
        "sources=%s source_protocol_counts=%s L=%d features=%s device=%s seed=%d "
        "parameters=%d trainable=%d config=%s",
        target_name,
        target_full.family,
        source_mode,
        source_names,
        protocol_counts([trajectories[name] for name in source_names]),
        HISTORY_LENGTH,
        resolved.model.features,
        device,
        resolved.seed,
        total,
        trainable,
        resolved.to_dict(),
    )
    trainer = WeightedMAMLTrainer(
        model=model,
        source_tasks=source_tasks,
        target_support=target_support,
        config=resolved,  # Same validated trainer interface as the CALCE experiment.
        device=device,
        run_dir=run_dir,
        source_mode=source_mode,
        logger=logger,
    )

    if adapt_only:
        adaptation_checkpoint = Path(resume).resolve()
        payload = load_checkpoint(adaptation_checkpoint, model, map_location=device)
        if payload["target_file_name"] != target_support.file_name:
            raise ValueError("adapt-only checkpoint target does not match")
        if int(payload["L"]) != HISTORY_LENGTH:
            raise ValueError("adapt-only checkpoint history length does not match L=100")
        if payload["source_mode"] != source_mode:
            raise ValueError("adapt-only checkpoint source mode does not match")
        if list(payload["source_file_names"]) != source_names:
            raise ValueError("adapt-only checkpoint source list does not match")
        trainer.validate_checkpoint_objective(payload)
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
            raise RuntimeError("meta-training produced no best source checkpoint")
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
        meta_algorithm=resolved.maml.algorithm,
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
        meta_algorithm=resolved.maml.algorithm,
    )
    fast.history["is_reported_step"] = fast.history["step"].isin(fast_steps)
    fast.history.to_csv(run_dir / "adaptation/fast_adaptation_history.csv", index=False)
    full.history.to_csv(run_dir / "adaptation/full_adaptation_history.csv", index=False)

    # Target future is made accessible only after training and both adaptations.
    evaluation_view = TargetEvaluationView.after_training(
        target_full, HISTORY_LENGTH, resolved.model.features
    )
    fast_results: dict[int, Any] = {}
    for step in fast_steps:
        snapshot = fast.snapshots.get(step)
        if snapshot is None:
            raise RuntimeError(f"fast adaptation did not capture step {step}")
        mode = f"fast_{step}"
        result = evaluate_target(
            snapshot,
            evaluation_view,
            HISTORY_LENGTH,
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

    full_result = evaluate_target(
        full.model,
        evaluation_view,
        HISTORY_LENGTH,
        resolved.data.max_forecast_cycle,
        resolved.data.eol_threshold,
        "full",
        run_dir,
        logger,
    )
    plot_target_prediction(
        full_result.predictions,
        run_dir / "figures/target_soh_full.png",
        f"{target_name} full adaptation",
        metrics=full_result.metrics,
    )
    comparison = adaptation_comparison_frame(fast_results, full_result)
    comparison.to_csv(run_dir / "metrics/adaptation_comparison.csv", index=False)
    save_key_results_figure(
        target_name,
        HISTORY_LENGTH,
        fast_results,
        full_result,
        run_dir / "figures/key_results_summary.png",
    )

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
    _write_final_weights(
        run_dir,
        source_names,
        selected_weights.alpha,
        selected_weights.kernel_ss,
    )
    training_files = (
        run_dir / "training/iteration_history.csv",
        run_dir / "training/source_loss_history.csv",
        run_dir / "weights/alpha_history.csv",
    )
    if all(path.is_file() for path in training_files):
        plot_training_outputs(run_dir)
    else:
        logger.warning("training-history plots skipped because history CSV files are missing")

    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "fast_steps": fast_steps,
            "fast_metrics_by_step": {
                str(step): result.metrics for step, result in fast_results.items()
            },
            "full_metrics": full_result.metrics,
            "best_source_meta_loss_ema": training_best_metric,
            "adapt_only": adapt_only,
            "adaptation_checkpoint": str(adaptation_checkpoint),
            "adaptation_checkpoint_iteration": adaptation_checkpoint_iteration,
            "key_results_figure": str(run_dir / "figures/key_results_summary.png"),
        }
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )
    logger.info("Completed HUST weighted-GRU run: %s", run_dir)
    return run_dir


def _checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {source}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HUST first-100-cycle SOH+voltage+current GRU with target-aware weighted MAML"
        )
    )
    parser.add_argument("--config", default="hust_weighted_gru/config.yaml")
    parser.add_argument("--target", help="for example HUST_1-1.pkl")
    parser.add_argument("--source-mode", choices=SOURCE_MODES)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--resume", help="resume meta-training from last.pt")
    parser.add_argument(
        "--adapt-only",
        action="store_true",
        help="skip meta-training and adapt/evaluate the supplied --resume checkpoint",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="two-iteration integration run for manual server validation",
    )
    args = parser.parse_args(argv)
    if args.adapt_only and not args.resume:
        parser.error("--adapt-only requires --resume CHECKPOINT")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    target = args.target
    source_mode = args.source_mode or config.source_mode
    if args.resume:
        checkpoint = _checkpoint_metadata(args.resume)
        target = target or checkpoint.get("target_file_name")
        source_mode = args.source_mode or checkpoint.get("source_mode")
        checkpoint_l = checkpoint.get("L")
        if checkpoint_l is not None and int(checkpoint_l) != HISTORY_LENGTH:
            raise ValueError(
                f"checkpoint L={checkpoint_l} is incompatible with fixed HUST L=100"
            )
    if target is None:
        raise SystemExit("--target HUST_<protocol>-<replicate>.pkl is required")
    if source_mode is None:
        raise SystemExit("--source-mode is required")
    if args.device:
        config.device = args.device
    run_experiment(
        config,
        str(target),
        str(source_mode),
        Path.cwd(),
        smoke_test=args.smoke_test,
        resume=args.resume,
        adapt_only=args.adapt_only,
    )


if __name__ == "__main__":
    main()


"""Run the paper-aligned CALCE recursive MAML experiments for L=500 and L=100."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import ExperimentConfig, load_config, save_config
from .main import run as run_experiment


DEFAULT_CONFIGS = {
    500: Path("paper_reproduction/configs/paper_recursive_l500.yaml"),
    100: Path("paper_reproduction/configs/paper_recursive_l100.yaml"),
}


def _main_args(config: Path, mode: str, device: str) -> argparse.Namespace:
    """Build the exact namespace accepted by ``paper_reproduction.main.run``."""
    return argparse.Namespace(
        config=str(config),
        mode=mode,
        device=device,
        checkpoint=None,
        resume=None,
        history_length=None,
        max_epochs=None,
        forecast_mode=None,
        max_prediction_length=None,
        test_cell=None,
        fast_learning_rate=None,
        complete_learning_rate=None,
        complete_max_steps=None,
        complete_patience=None,
        scheduler=None,
        loss_reduction=None,
        sampling_mode=None,
        fast_sampling_mode=None,
        inner_steps=None,
        multi_step_query_weights=None,
        experiment_label=None,
        oracle_diagnostics=None,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _select_outer_learning_rate(
    config: ExperimentConfig,
    device: str,
    trial_count: int,
    suite_dir: Path,
) -> tuple[float, Path | None]:
    """Run the paper's TPE step and return its selected outer learning rate."""
    search_config = suite_dir / "selected_configs" / (
        f"L{config.data.history_length}_optuna.yaml"
    )
    config.maml.optuna_trials = trial_count
    config.maml.experiment_label = (
        f"paper-rec-tpe{trial_count}"
    )
    save_config(config, search_config)
    optuna_run = run_experiment(_main_args(search_config, "optuna", device))
    manifest = json.loads((optuna_run / "run_manifest.json").read_text(encoding="utf-8"))
    learning_rate = float(manifest["optuna"]["best_params"]["outer_learning_rate"])
    return learning_rate, optuna_run


def _combine_results(
    run_dirs: dict[int, Path],
    suite_dir: Path,
    reference_path: Path,
) -> Path:
    frames: list[pd.DataFrame] = []
    for history_length, run_dir in run_dirs.items():
        source = run_dir / "meta_test/meta_test_summary.csv"
        frame = pd.read_csv(source)
        frame.insert(0, "history_length", history_length)
        frame.insert(1, "run_dir", str(run_dir))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined_path = suite_dir / "paper_recursive_results.csv"
    combined.to_csv(combined_path, index=False)

    reference = pd.read_csv(reference_path)
    keys = ["history_length", "cell", "mode"]
    observed = combined.merge(reference, on=keys, how="inner", suffixes=("", "_paper"))
    for metric in ["mae_percent", "rmse_percent", "r2", "rul_error_actual_minus_predicted"]:
        observed[f"{metric}_difference"] = observed[metric] - observed[f"{metric}_paper"]
    comparison_path = suite_dir / "paper_recursive_comparison.csv"
    observed.to_csv(comparison_path, index=False)
    _plot_comparison(observed, suite_dir / "paper_recursive_comparison.png")
    return comparison_path


def _plot_comparison(frame: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, history_length in zip(axes, [500, 100]):
        selected = frame[frame["history_length"] == history_length].copy()
        labels = selected["cell"].str.replace("CALCE_", "", regex=False).str.replace(
            ".pkl", "", regex=False
        )
        positions = range(len(selected))
        axis.bar([value - 0.18 for value in positions], selected["mae_percent_paper"],
                 width=0.36, label="paper")
        axis.bar([value + 0.18 for value in positions], selected["mae_percent"],
                 width=0.36, label="reproduction")
        axis.set_xticks(list(positions), labels)
        axis.set_title(f"Recursive L={history_length}")
        axis.set_xlabel("Meta-test cell")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("MAE (%)")
    axes[0].legend()
    figure.suptitle("Paper vs local CALCE recursive MAML")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    root = Path.cwd().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suite_dir = root / "outputs/paper_recursive_reproduction/suites" / timestamp
    suite_dir.mkdir(parents=True, exist_ok=False)
    run_dirs: dict[int, Path] = {}
    records: list[dict[str, Any]] = []

    for history_length in args.history_lengths:
        base_config_path = (root / DEFAULT_CONFIGS[history_length]).resolve()
        config = load_config(base_config_path)
        optuna_run: Path | None = None
        if args.outer_learning_rate is None:
            learning_rate, optuna_run = _select_outer_learning_rate(
                config,
                args.device,
                args.optuna_trials,
                suite_dir,
            )
        else:
            learning_rate = args.outer_learning_rate

        config.maml.outer_learning_rate = learning_rate
        config.maml.optuna_trials = 0
        config.maml.experiment_label = "paper-rec"
        selected_config = suite_dir / "selected_configs" / f"L{history_length}.yaml"
        save_config(config, selected_config)
        run_dir = run_experiment(_main_args(selected_config, "all", args.device))
        run_dirs[history_length] = run_dir
        records.append(
            {
                "history_length": history_length,
                "outer_learning_rate": learning_rate,
                "outer_learning_rate_source": (
                    "optuna_tpe" if optuna_run is not None else "cli_fixed"
                ),
                "optuna_trials": args.optuna_trials if optuna_run is not None else 0,
                "optuna_run": str(optuna_run) if optuna_run is not None else None,
                "experiment_run": str(run_dir),
                "selected_config": str(selected_config),
            }
        )

    reference_path = root / "paper_reproduction/paper_recursive_reference.csv"
    comparison_path = _combine_results(run_dirs, suite_dir, reference_path)
    manifest = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_protocol": {
            "prediction": "recursive_to_end_of_each_cell",
            "train_cells": load_config(root / DEFAULT_CONFIGS[500]).data.train_cells,
            "test_cells": load_config(root / DEFAULT_CONFIGS[500]).data.test_cells,
            "eol_threshold": 0.70,
            "inner_steps": 1,
            "inner_batch_size": 64,
            "inner_learning_rate": 0.05,
            "teacher_forcing_predicted_input_probability": 0.5,
            "maximum_epochs": 500,
        },
        "paper_unreported_reproduction_choices": {
            "loss": "masked_mse",
            "optuna_trial_count": (
                args.optuna_trials if args.outer_learning_rate is None else 0
            ),
            "optuna_search_range": [1.0e-5, 1.0e-2],
            "early_stopping_patience": 30,
            "early_stopping_min_delta": 1.0e-7,
            "complete_adaptation_min_delta": 1.0e-6,
        },
        "comparability_warning": (
            "The local CALCE preprocessing has 572 future points for CX2_37 at L=500, "
            "whereas the paper reports 744; exact numerical equality therefore requires "
            "the authors' BatteryML-cleaned sequences."
        ),
        "runs": records,
        "comparison_csv": str(comparison_path),
    }
    _write_json(suite_dir / "suite_manifest.json", manifest)
    print(f"Completed paper recursive suite: {suite_dir}")
    print(f"Comparison: {comparison_path}")
    return suite_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-aligned full second-order MAML for recursive L=500 and L=100"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--history-lengths",
        nargs="+",
        type=int,
        choices=sorted(DEFAULT_CONFIGS),
        default=[500, 100],
        help="default: 500 100",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--optuna-trials",
        type=int,
        default=20,
        help="TPE trials per L; paper does not report this count (default: 20)",
    )
    group.add_argument(
        "--outer-learning-rate",
        type=float,
        help="skip TPE and use this disclosed fixed outer learning rate",
    )
    args = parser.parse_args()
    if args.outer_learning_rate is not None and args.outer_learning_rate <= 0:
        parser.error("--outer-learning-rate must be positive")
    if len(set(args.history_lengths)) != len(args.history_lengths):
        parser.error("--history-lengths must not contain duplicates")
    return args


if __name__ == "__main__":
    run(parse_args())

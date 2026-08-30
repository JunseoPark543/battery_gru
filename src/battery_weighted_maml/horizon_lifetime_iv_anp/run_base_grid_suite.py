"""Resumable single-horizon seed/LR/fold sweep with context-size evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from battery_weighted_maml.matr_anp.runtime import write_json

from .config import LifetimeIVConfig, load_config, resolve_data_root
from .evaluate import evaluate_checkpoint
from .train import train_run


@dataclass(frozen=True)
class RunKey:
    horizon_scheme: str
    seed: int
    learning_rate: float
    fold: int

    @property
    def tag(self) -> str:
        value = f"{self.learning_rate:.2g}".replace(".", "p").replace("-", "m")
        return f"h{self.horizon_scheme}_s{self.seed}_lr{value}_f{self.fold}"


HORIZON_SCHEMES = {
    "original": list(range(100, 301, 20)),
    "expanded": list(range(60, 601, 20)),
}


def _horizon_scheme(values: Sequence[int]) -> str:
    normalized = list(map(int, values))
    for name, horizons in HORIZON_SCHEMES.items():
        if normalized == horizons:
            return name
    raise ValueError(f"unsupported reusable training horizons: {normalized}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_local_run(root: Path) -> tuple[Path | None, Path | None]:
    """Return a completed run, or an incomplete run's last checkpoint."""
    manifests = sorted(
        root.glob("*/run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    incomplete: tuple[Path | None, Path | None] = (None, None)
    for manifest_path in manifests:
        run_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status") == "completed"
            and (run_dir / "checkpoints/best.pt").is_file()
        ):
            return run_dir, None
        last = run_dir / "checkpoints/last.pt"
        if incomplete[0] is None and last.is_file():
            incomplete = run_dir, last
    return incomplete


def _index_reusable_runs(paths: Sequence[str]) -> dict[RunKey, Path]:
    indexed: dict[RunKey, Path] = {}
    for raw in paths:
        run_dir = Path(raw).expanduser().resolve()
        config_path = run_dir / "resolved_config.yaml"
        manifest_path = run_dir / "run_manifest.json"
        checkpoint = run_dir / "checkpoints/best.pt"
        if not (config_path.is_file() and manifest_path.is_file() and checkpoint.is_file()):
            raise FileNotFoundError(f"reusable run is incomplete: {run_dir}")
        config = load_config(config_path)
        manifest = _read_json(manifest_path)
        if config.training.paired_horizon_training or config.training.consistency_weight:
            raise ValueError(f"reusable run is not the single-horizon baseline: {run_dir}")
        key = RunKey(
            horizon_scheme=_horizon_scheme(config.task.horizons),
            seed=int(config.seed),
            learning_rate=float(config.training.learning_rate),
            fold=int(manifest["fold"]),
        )
        indexed[key] = run_dir
    return indexed


def _evaluation_is_complete(
    destination: Path,
    *,
    context_size: int,
    context_seed: int,
    horizons: Sequence[int],
) -> bool:
    manifest_path = destination / "evaluation_manifest.json"
    if not (
        manifest_path.is_file()
        and (destination / "aggregate_metrics.csv").is_file()
        and (destination / "per_cell_predictions.csv").is_file()
    ):
        return False
    manifest = _read_json(manifest_path)
    return bool(
        manifest.get("status") == "completed"
        and int(manifest.get("context_size", -1)) == int(context_size)
        and int(manifest.get("context_seed", -1)) == int(context_seed)
        and list(map(int, manifest.get("horizons", []))) == list(map(int, horizons))
        and manifest.get("nested_context_selection") is True
    )


def _run_context_row(
    key: RunKey,
    run_dir: Path,
    evaluation_dir: Path,
    context_size: int,
    trained_horizons: Sequence[int],
) -> dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json")
    aggregate = pd.read_csv(evaluation_dir / "aggregate_metrics.csv")
    predictions = pd.read_csv(evaluation_dir / "per_cell_predictions.csv")
    valid = aggregate[aggregate["status"] == "ok"]
    if valid.empty or predictions.empty:
        raise RuntimeError(f"{key.tag}/context{context_size}: empty evaluation")
    error = (
        predictions["predicted_lifetime_mean_cycles"]
        - predictions["true_lifetime_cycles"]
    ).to_numpy(dtype=float)
    covered = (
        (predictions["true_lifetime_cycles"] >= predictions["lifetime_lower_cycles"])
        & (predictions["true_lifetime_cycles"] <= predictions["lifetime_upper_cycles"])
    )
    trained_set = set(map(int, trained_horizons))
    below = valid[valid["horizon"] < min(trained_set)]
    within = valid[valid["horizon"].isin(trained_set)]
    above = valid[valid["horizon"] > max(trained_set)]

    def mean_rmse(frame: pd.DataFrame) -> float:
        return float(frame["lifetime_rmse_cycles"].mean()) if not frame.empty else math.nan

    return {
        "horizon_scheme": key.horizon_scheme,
        "seed": key.seed,
        "learning_rate": key.learning_rate,
        "fold": key.fold,
        "context_size": int(context_size),
        "run_tag": key.tag,
        "run_dir": str(run_dir),
        "evaluation_dir": str(evaluation_dir),
        "prediction_csv": str(evaluation_dir / "per_cell_predictions.csv"),
        "best_step": int(manifest["best_step"]),
        "best_validation_rmse_cycles": float(manifest["best_validation_rmse_cycles"]),
        "mean_horizon_lifetime_rmse_cycles": float(valid["lifetime_rmse_cycles"].mean()),
        "mean_horizon_lifetime_mae_cycles": float(valid["lifetime_mae_cycles"].mean()),
        "mean_rmse_below_training_range_cycles": mean_rmse(below),
        "mean_rmse_at_trained_horizons_cycles": mean_rmse(within),
        "mean_rmse_above_training_range_cycles": mean_rmse(above),
        "pooled_lifetime_rmse_cycles": float(np.sqrt(np.mean(np.square(error)))),
        "pooled_lifetime_mae_cycles": float(np.mean(np.abs(error))),
        "pooled_lifetime_bias_cycles": float(np.mean(error)),
        "pooled_coverage": float(covered.mean()),
        "pooled_interval_width_cycles": float(
            (predictions["lifetime_upper_cycles"] - predictions["lifetime_lower_cycles"]).mean()
        ),
        "num_predictions": int(len(predictions)),
        "num_test_cells": int(predictions["cell_id"].nunique()),
        "num_evaluation_horizons": int(len(aggregate)),
        "num_valid_horizons": int(len(valid)),
        "num_skipped_horizons": int((aggregate["status"] != "ok").sum()),
        "minimum_actual_context_cells": int(valid["num_context_cells"].min()),
        "mean_actual_context_cells": float(valid["num_context_cells"].mean()),
        "status": "completed",
        "error": "",
    }


def _pooled_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    error = (
        predictions["predicted_lifetime_mean_cycles"]
        - predictions["true_lifetime_cycles"]
    ).to_numpy(dtype=float)
    true_rul = predictions["true_rul_cycles"].to_numpy(dtype=float)
    covered = (
        (predictions["true_lifetime_cycles"] >= predictions["lifetime_lower_cycles"])
        & (predictions["true_lifetime_cycles"] <= predictions["lifetime_upper_cycles"])
    )
    return {
        "pooled_lifetime_rmse_cycles": float(np.sqrt(np.mean(np.square(error)))),
        "pooled_lifetime_mae_cycles": float(np.mean(np.abs(error))),
        "pooled_lifetime_bias_cycles": float(np.mean(error)),
        "pooled_rul_mape_percent": float(
            100.0 * np.mean(np.abs(error) / np.maximum(np.abs(true_rul), 1.0))
        ),
        "pooled_coverage": float(covered.mean()),
        "pooled_interval_width_cycles": float(
            (predictions["lifetime_upper_cycles"] - predictions["lifetime_lower_cycles"]).mean()
        ),
        "num_predictions": int(len(predictions)),
        "num_unique_test_cells": int(predictions["cell_id"].nunique()),
    }


def _build_cv_summaries(
    rows: pd.DataFrame,
    expected_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = rows[rows["status"] == "completed"]
    seed_rows: list[dict[str, Any]] = []
    keys = ["horizon_scheme", "seed", "learning_rate", "context_size"]
    for values, group in completed.groupby(keys, sort=True):
        predictions: list[pd.DataFrame] = []
        for path in group["prediction_csv"]:
            frame = pd.read_csv(path)
            frame["source_fold"] = int(
                group.loc[group["prediction_csv"] == path, "fold"].iloc[0]
            )
            predictions.append(frame)
        pooled = pd.concat(predictions, ignore_index=True)
        horizon_scheme, seed, learning_rate, context_size = values
        seed_rows.append({
            "horizon_scheme": str(horizon_scheme),
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "context_size": int(context_size),
            "folds_completed": int(group["fold"].nunique()),
            "expected_folds": int(expected_folds),
            "mean_best_validation_rmse_cycles": float(
                group["best_validation_rmse_cycles"].mean()
            ),
            **_pooled_metrics(pooled),
        })
    cv_seed = pd.DataFrame(seed_rows)
    aggregate_rows: list[dict[str, Any]] = []
    metric_columns = [
        "mean_best_validation_rmse_cycles",
        "pooled_lifetime_rmse_cycles",
        "pooled_lifetime_mae_cycles",
        "pooled_lifetime_bias_cycles",
        "pooled_rul_mape_percent",
        "pooled_coverage",
        "pooled_interval_width_cycles",
    ]
    if not cv_seed.empty:
        for values, group in cv_seed.groupby(
            ["horizon_scheme", "learning_rate", "context_size"], sort=True
        ):
            horizon_scheme, learning_rate, context_size = values
            record: dict[str, Any] = {
                "horizon_scheme": str(horizon_scheme),
                "learning_rate": float(learning_rate),
                "context_size": int(context_size),
                "seeds_completed": int(group["seed"].nunique()),
                "all_folds_complete": bool(
                    (group["folds_completed"] == expected_folds).all()
                ),
            }
            for column in metric_columns:
                record[f"{column}_mean"] = float(group[column].mean())
                record[f"{column}_std"] = float(group[column].std(ddof=1))
            aggregate_rows.append(record)
    return cv_seed, pd.DataFrame(aggregate_rows)


def _build_horizon_summaries(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    for record in rows.to_dict("records"):
        frame = pd.read_csv(record["prediction_csv"])
        frame["seed"] = int(record["seed"])
        frame["horizon_scheme"] = str(record["horizon_scheme"])
        frame["learning_rate"] = float(record["learning_rate"])
        frame["fold"] = int(record["fold"])
        frame["context_size"] = int(record["context_size"])
        prediction_frames.append(frame)
    if not prediction_frames:
        return pd.DataFrame(), pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True)
    seed_rows: list[dict[str, Any]] = []
    for values, group in predictions.groupby(
        ["horizon_scheme", "seed", "learning_rate", "context_size", "horizon"],
        sort=True,
    ):
        horizon_scheme, seed, learning_rate, context_size, horizon = values
        seed_rows.append({
            "horizon_scheme": str(horizon_scheme),
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "context_size": int(context_size),
            "horizon": int(horizon),
            "folds_represented": int(group["fold"].nunique()),
            **_pooled_metrics(group),
        })
    seed_horizon = pd.DataFrame(seed_rows)
    aggregate_rows: list[dict[str, Any]] = []
    metric_columns = [
        "pooled_lifetime_rmse_cycles",
        "pooled_lifetime_mae_cycles",
        "pooled_lifetime_bias_cycles",
        "pooled_rul_mape_percent",
        "pooled_coverage",
        "pooled_interval_width_cycles",
    ]
    for values, group in seed_horizon.groupby(
        ["horizon_scheme", "learning_rate", "context_size", "horizon"], sort=True
    ):
        horizon_scheme, learning_rate, context_size, horizon = values
        record: dict[str, Any] = {
            "horizon_scheme": str(horizon_scheme),
            "learning_rate": float(learning_rate),
            "context_size": int(context_size),
            "horizon": int(horizon),
            "seeds_completed": int(group["seed"].nunique()),
        }
        for column in metric_columns:
            record[f"{column}_mean"] = float(group[column].mean())
            record[f"{column}_std"] = float(group[column].std(ddof=1))
        aggregate_rows.append(record)
    return seed_horizon, pd.DataFrame(aggregate_rows)


def _plot_aggregate(frame: pd.DataFrame, destination: Path) -> None:
    if frame.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    specifications = (
        ("pooled_lifetime_rmse_cycles", "5-fold pooled lifetime RMSE", False),
        ("pooled_lifetime_mae_cycles", "5-fold pooled lifetime MAE", False),
        ("pooled_coverage", "95% interval coverage", True),
        ("pooled_interval_width_cycles", "Mean interval width", False),
    )
    for values, selected in frame.groupby(
        ["horizon_scheme", "learning_rate"], sort=True
    ):
        horizon_scheme, learning_rate = values
        selected = selected.sort_values("context_size")
        for axis, (column, title, coverage) in zip(axes.flat, specifications):
            axis.errorbar(
                selected["context_size"],
                selected[f"{column}_mean"],
                yerr=selected[f"{column}_std"].fillna(0.0),
                marker="o", capsize=4,
                label=f"{horizon_scheme}, lr={learning_rate:.1e}",
            )
            axis.set(xlabel="Test context cells", ylabel=title, title=title)
            axis.grid(alpha=0.25)
            if coverage:
                axis.axhline(0.95, color="black", ls="--", lw=1, label="target=0.95")
    for axis in axes.flat:
        axis.set_xticks(sorted(frame["context_size"].unique()))
        axis.legend(fontsize=8)
    figure.suptitle("Single-horizon lifetime I-V ANP: seed x LR x 5-fold")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_horizon_rmse(
    frame: pd.DataFrame,
    destination: Path,
    trained_horizons: Sequence[int],
) -> None:
    if frame.empty:
        return
    contexts = sorted(frame["context_size"].unique())
    figure, axes = plt.subplots(
        1, len(contexts), figsize=(6 * len(contexts), 5.5), sharey=True,
        squeeze=False,
    )
    for axis, context_size in zip(axes.flat, contexts):
        selected_context = frame[frame["context_size"] == context_size]
        for values, selected in selected_context.groupby(
            ["horizon_scheme", "learning_rate"], sort=True
        ):
            horizon_scheme, learning_rate = values
            selected = selected.sort_values("horizon")
            axis.plot(
                selected["horizon"],
                selected["pooled_lifetime_rmse_cycles_mean"],
                marker="o", ms=3,
                label=f"{horizon_scheme}, lr={learning_rate:.1e}",
            )
        axis.axvspan(
            min(trained_horizons), max(trained_horizons),
            color="green", alpha=0.08, label="trained horizon range",
        )
        axis.set(
            xlabel="Observation horizon",
            title=f"Context cells={context_size}",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes.flat[0].set_ylabel("5-fold pooled lifetime RMSE (cycles)")
    figure.suptitle("Evaluation from cycle 60 to 600 (20-cycle interval)")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-horizon seed/LR/5-fold/context-size suite"
    )
    parser.add_argument("--config", default="configs/matr_horizon_lifetime_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--suite-dir", help="Existing directory resumes/skips completed runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument(
        "--training-horizon-schemes", nargs="+",
        choices=sorted(HORIZON_SCHEMES), default=["original", "expanded"],
    )
    parser.add_argument(
        "--learning-rates", nargs="+", type=float,
        default=[2.5e-5, 5.0e-5, 1.0e-4],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument(
        "--evaluation-horizons", nargs="+", type=int,
        default=list(range(60, 601, 20)),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--reuse-run", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    if args.device:
        base.device = args.device
    base.training.paired_horizon_training = False
    base.training.consistency_weight = 0.0
    base.evaluation.context_seed = int(base.split.seed)
    data_root = resolve_data_root(base, args.data_root)
    horizon_schemes = list(dict.fromkeys(args.training_horizon_schemes))
    seeds = sorted(set(args.seeds))
    learning_rates = sorted(set(args.learning_rates))
    folds = sorted(set(args.folds))
    context_sizes = sorted(set(args.context_sizes))
    evaluation_horizons = sorted(set(args.evaluation_horizons))
    if not horizon_schemes or not seeds or not learning_rates or not folds or not context_sizes or not evaluation_horizons:
        raise ValueError("seed/LR/fold/context/horizon lists cannot be empty")
    if evaluation_horizons != args.evaluation_horizons or evaluation_horizons[0] < 2:
        raise ValueError("evaluation horizons must be sorted, unique, and >=2")
    if any(value <= 0 for value in learning_rates):
        raise ValueError("learning rates must be positive")
    if any(value < 0 or value >= base.split.num_folds for value in folds):
        raise ValueError(f"folds must lie in [0,{base.split.num_folds - 1}]")
    if any(value < base.task.context_size_min for value in context_sizes):
        raise ValueError(
            f"context sizes must be >= trained minimum {base.task.context_size_min}"
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suite_root = Path(
        args.suite_dir
        or Path("outputs/horizon_lifetime_iv_anp_grid") / "suites" / timestamp
    ).expanduser().resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    combinations = [
        RunKey(horizon_scheme, seed, learning_rate, fold)
        for horizon_scheme in horizon_schemes
        for seed in seeds
        for learning_rate in learning_rates
        for fold in folds
    ]
    write_json(suite_root / "suite_manifest.json", {
        "status": "running",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "sequential_single_gpu",
        "single_horizon_baseline": True,
        "split_seed": base.split.seed,
        "fixed_validation_test_context_seed": base.evaluation.context_seed,
        "nested_test_contexts": True,
        "training_horizon_schemes": {
            name: HORIZON_SCHEMES[name] for name in horizon_schemes
        },
        "seeds": seeds,
        "learning_rates": learning_rates,
        "folds": folds,
        "context_sizes": context_sizes,
        "evaluation_horizons": evaluation_horizons,
        "training_run_count": len(combinations),
    })

    result_path = suite_root / "run_context_summary.csv"
    rows: list[dict[str, Any]] = []
    if result_path.is_file():
        rows = pd.read_csv(result_path).to_dict("records")
    reusable = _index_reusable_runs(args.reuse_run)
    # Persist externally reused runs across suite restarts through the summary.
    for row in rows:
        if row.get("status") != "completed" or not row.get("run_dir"):
            continue
        key = RunKey(
            horizon_scheme=str(row.get("horizon_scheme", "original")),
            seed=int(row["seed"]),
            learning_rate=float(row["learning_rate"]),
            fold=int(row["fold"]),
        )
        run_dir = Path(str(row["run_dir"])).expanduser().resolve()
        if (run_dir / "checkpoints/best.pt").is_file():
            reusable.setdefault(key, run_dir)
    failures = 0
    for index, key in enumerate(combinations, start=1):
        print(f"[{index}/{len(combinations)}] {key.tag}", flush=True)
        config: LifetimeIVConfig = copy.deepcopy(base)
        config.task.horizons = list(HORIZON_SCHEMES[key.horizon_scheme])
        # Validation follows the training horizon scheme. Held-out testing is
        # independently fixed to evaluation_horizons below.
        config.evaluation.horizons = list(config.task.horizons)
        config.seed = key.seed
        config.training.learning_rate = key.learning_rate
        config.validate()
        run_root = suite_root / "runs" / key.tag
        try:
            if key in reusable:
                run_dir = reusable[key]
                print(f"  reuse completed run: {run_dir}", flush=True)
            else:
                existing, resume_checkpoint = _latest_local_run(run_root)
                if existing is not None and resume_checkpoint is None:
                    run_dir = existing
                    print(f"  skip completed training: {run_dir}", flush=True)
                else:
                    run_dir = train_run(
                        config, key.fold, data_root,
                        resume=resume_checkpoint,
                        max_steps=args.max_steps,
                        output_root=run_root,
                    )
            # Replace rows for this run so a resumed suite cannot duplicate results.
            rows = [
                row for row in rows
                if not (
                    str(row.get("horizon_scheme", "original")) == key.horizon_scheme
                    and int(row["seed"]) == key.seed
                    and math.isclose(float(row["learning_rate"]), key.learning_rate)
                    and int(row["fold"]) == key.fold
                )
            ]
            for context_size in context_sizes:
                evaluation_config = copy.deepcopy(config)
                evaluation_config.evaluation.context_size = context_size
                evaluation_config.evaluation.horizons = evaluation_horizons
                evaluation_config.validate()
                destination = (
                    suite_root / "evaluations" / key.tag / f"context_{context_size}"
                )
                if not _evaluation_is_complete(
                    destination,
                    context_size=context_size,
                    context_seed=int(base.evaluation.context_seed),
                    horizons=evaluation_horizons,
                ):
                    evaluate_checkpoint(
                        evaluation_config,
                        run_dir / "checkpoints/best.pt",
                        data_root,
                        output_dir=destination,
                        nested_context_selection=True,
                    )
                rows.append(_run_context_row(
                    key, run_dir, destination, context_size,
                    trained_horizons=config.task.horizons,
                ))
            print(f"  completed {key.tag}", flush=True)
        except Exception as exc:
            failures += 1
            traceback.print_exc()
            rows.append({
                "horizon_scheme": key.horizon_scheme,
                "seed": key.seed,
                "learning_rate": key.learning_rate,
                "fold": key.fold,
                "context_size": -1,
                "run_tag": key.tag,
                "run_dir": "",
                "evaluation_dir": "",
                "prediction_csv": "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
        result_frame = pd.DataFrame(rows)
        result_frame.to_csv(result_path, index=False)
        completed = result_frame[result_frame["status"] == "completed"]
        if not completed.empty:
            cv_seed, aggregate = _build_cv_summaries(completed, len(folds))
            cv_seed.to_csv(suite_root / "cv_seed_summary.csv", index=False)
            aggregate.to_csv(suite_root / "aggregate_summary.csv", index=False)
            _plot_aggregate(aggregate, suite_root / "grid_comparison.png")
            seed_horizon, horizon_aggregate = _build_horizon_summaries(completed)
            seed_horizon.to_csv(
                suite_root / "cv_seed_horizon_summary.csv", index=False
            )
            horizon_aggregate.to_csv(
                suite_root / "horizon_aggregate_summary.csv", index=False
            )
            _plot_horizon_rmse(
                horizon_aggregate,
                suite_root / "horizon_rmse_comparison.png",
                trained_horizons=base.task.horizons,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(suite_root / "suite_manifest.json", {
        "status": "completed" if failures == 0 else "completed_with_failures",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "sequential_single_gpu",
        "single_horizon_baseline": True,
        "split_seed": base.split.seed,
        "fixed_validation_test_context_seed": base.evaluation.context_seed,
        "nested_test_contexts": True,
        "training_horizon_schemes": {
            name: HORIZON_SCHEMES[name] for name in horizon_schemes
        },
        "seeds": seeds,
        "learning_rates": learning_rates,
        "folds": folds,
        "context_sizes": context_sizes,
        "evaluation_horizons": evaluation_horizons,
        "training_run_count": len(combinations),
        "failure_count": failures,
        "summary": str(result_path),
    })
    print(f"Base grid suite: {suite_root}")
    if failures:
        raise RuntimeError(f"{failures} base grid run(s) failed")


if __name__ == "__main__":
    main()

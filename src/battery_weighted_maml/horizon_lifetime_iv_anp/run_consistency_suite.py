"""Run compute-matched paired-horizon consistency ablations sequentially."""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from battery_weighted_maml.matr_anp.runtime import write_json

from .config import load_config, resolve_data_root
from .evaluate import evaluate_checkpoint
from .train import train_run


@dataclass(frozen=True)
class Variant:
    name: str
    weight: float
    gap: int


VARIANTS = {
    item.name: item
    for item in (
        Variant("pair_w0_g20", 0.0, 20),
        Variant("cons_w005_g20", 0.05, 20),
        Variant("cons_w010_g20", 0.10, 20),
        Variant("cons_w020_g20", 0.20, 20),
        Variant("cons_w010_g40", 0.10, 40),
    )
}


def _summarize(
    variant: Variant,
    run_dir: Path,
    evaluation_dir: Path,
) -> dict[str, object]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(evaluation_dir / "aggregate_metrics.csv")
    predictions = pd.read_csv(evaluation_dir / "per_cell_predictions.csv")
    valid = metrics[metrics["status"] == "ok"]
    if valid.empty:
        raise RuntimeError(f"{variant.name}: evaluation has no valid horizon")
    adjacent_updates = (
        predictions.sort_values(["cell_id", "horizon"])
        .groupby("cell_id")["predicted_lifetime_mean_cycles"]
        .diff()
        .abs()
        .dropna()
    )
    return {
        "variant": variant.name,
        "consistency_weight": variant.weight,
        "horizon_gap": variant.gap,
        "run_dir": str(run_dir),
        "best_step": int(manifest["best_step"]),
        "best_validation_rmse_cycles": float(manifest["best_validation_rmse_cycles"]),
        "mean_test_lifetime_rmse_cycles": float(valid["lifetime_rmse_cycles"].mean()),
        "mean_test_lifetime_mae_cycles": float(valid["lifetime_mae_cycles"].mean()),
        "mean_test_rul_mape_percent": float(valid["rul_mape_percent"].mean()),
        "mean_test_rul_coverage": float(valid["rul_coverage"].mean()),
        "mean_adjacent_lifetime_update_cycles": float(adjacent_updates.mean()),
        "status": "completed",
        "error": "",
    }


def _plot_summary(frame: pd.DataFrame, destination: Path) -> None:
    completed = frame[frame["status"] == "completed"].copy()
    if completed.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    specifications = (
        ("mean_test_lifetime_rmse_cycles", "Mean lifetime RMSE (cycles)"),
        ("mean_test_lifetime_mae_cycles", "Mean lifetime MAE (cycles)"),
        ("mean_test_rul_coverage", "Mean 95% interval coverage"),
        (
            "mean_adjacent_lifetime_update_cycles",
            "Mean adjacent-horizon lifetime update (cycles)",
        ),
    )
    for axis, (column, title) in zip(axes.flat, specifications):
        axis.bar(completed["variant"], completed[column])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(completed[column]):
            axis.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Paired-horizon lifetime consistency ablation")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lifetime consistency variants sequentially on one GPU"
    )
    parser.add_argument(
        "--config", default="configs/matr_horizon_lifetime_iv_anp_consistency.yaml"
    )
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--baseline-run",
        help="Optional completed single-horizon run to include without retraining",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--no-evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    if args.device:
        base.device = args.device
    data_root = resolve_data_root(base, args.data_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suite_root = Path(
        args.output_root
        or Path(base.paths.output_root) / "suites" / f"{timestamp}_f{args.fold}"
    ).resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    selected = [VARIANTS[name] for name in args.variants]
    write_json(suite_root / "suite_manifest.json", {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "sequential_single_gpu",
        "fold": args.fold,
        "variants": [item.__dict__ for item in selected],
    })

    rows: list[dict[str, object]] = []
    failures = 0
    if args.baseline_run:
        baseline_run = Path(args.baseline_run).resolve()
        baseline_config = load_config(baseline_run / "resolved_config.yaml")
        if args.device:
            baseline_config.device = args.device
        try:
            baseline_evaluation = evaluate_checkpoint(
                baseline_config,
                baseline_run / "checkpoints/best.pt",
                resolve_data_root(baseline_config, args.data_root),
            )
            rows.append(_summarize(
                Variant("single_horizon_baseline", 0.0, 0),
                baseline_run,
                baseline_evaluation,
            ))
        except Exception as exc:
            failures += 1
            traceback.print_exc()
            rows.append({
                "variant": "single_horizon_baseline",
                "consistency_weight": 0.0,
                "horizon_gap": 0,
                "run_dir": str(baseline_run),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
        pd.DataFrame(rows).to_csv(suite_root / "suite_summary.csv", index=False)

    for index, variant in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] starting {variant.name}", flush=True)
        config = copy.deepcopy(base)
        config.training.paired_horizon_training = True
        config.training.consistency_weight = variant.weight
        config.training.consistency_horizon_gap = variant.gap
        config.validate()
        try:
            run_dir = train_run(
                config, args.fold, data_root,
                max_steps=args.max_steps,
                output_root=suite_root / "runs" / variant.name,
            )
            if args.no_evaluate:
                row = {
                    "variant": variant.name,
                    "consistency_weight": variant.weight,
                    "horizon_gap": variant.gap,
                    "run_dir": str(run_dir),
                    "status": "trained_not_evaluated",
                    "error": "",
                }
            else:
                evaluation_dir = evaluate_checkpoint(
                    config, run_dir / "checkpoints/best.pt", data_root
                )
                row = _summarize(variant, run_dir, evaluation_dir)
            print(f"[{index}/{len(selected)}] completed {variant.name}", flush=True)
        except Exception as exc:  # Continue so one failed ablation does not waste the queue.
            failures += 1
            traceback.print_exc()
            row = {
                "variant": variant.name,
                "consistency_weight": variant.weight,
                "horizon_gap": variant.gap,
                "run_dir": "",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(suite_root / "suite_summary.csv", index=False)
        _plot_summary(frame, suite_root / "suite_comparison.png")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(suite_root / "suite_manifest.json", {
        "status": "completed" if failures == 0 else "completed_with_failures",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "sequential_single_gpu",
        "fold": args.fold,
        "variants": [item.__dict__ for item in selected],
        "failure_count": failures,
        "summary": str(suite_root / "suite_summary.csv"),
    })
    print(f"Consistency suite: {suite_root}")
    if failures:
        raise RuntimeError(f"{failures} consistency suite variant(s) failed")


if __name__ == "__main__":
    main()

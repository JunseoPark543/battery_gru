"""Combine fold/model evaluation artifacts and create comparison plots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .evaluate import _aggregate
from .plotting import plot_model_comparison
from .runtime import write_json


def compare_evaluations(
    evaluation_directories: list[str | Path], output_dir: str | Path
) -> Path:
    if not evaluation_directories:
        raise ValueError("at least one evaluation directory is required")
    frames = []
    sources = []
    for raw in evaluation_directories:
        directory = Path(raw).resolve()
        source = directory / "per_cell_metrics.csv"
        if not source.is_file():
            raise FileNotFoundError(f"per-cell metrics not found: {source}")
        frame = pd.read_csv(source)
        required = {
            "cell_id", "fold", "seed", "model", "alpha", "beta", "status",
            "future_rmse", "current_soh_abs_error", "interval_width_95",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{source} is missing columns {sorted(missing)}")
        frames.append(frame)
        sources.append(str(source))
    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.duplicated(
        ["cell_id", "fold", "seed", "model", "alpha", "beta"], keep=False
    )
    if duplicates.any():
        example = combined.loc[duplicates, ["cell_id", "fold", "model", "alpha", "beta"]].head()
        raise ValueError(f"duplicate evaluation combinations were supplied:\n{example}")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    valid = combined[combined["status"] == "ok"]
    if valid.empty:
        raise ValueError("combined evaluations contain no valid metric rows")
    combined.to_csv(destination / "combined_per_cell_metrics.csv", index=False)
    _aggregate(combined).to_csv(destination / "combined_aggregate_metrics.csv", index=False)
    plot_model_comparison(
        valid, "future_rmse", "Future SOH RMSE", destination / "rmse_model_comparison.png"
    )
    plot_model_comparison(
        valid,
        "current_soh_abs_error",
        "Current-cycle SOH absolute error",
        destination / "current_error_model_comparison.png",
    )
    plot_model_comparison(
        valid,
        "interval_width_95",
        "95% interval width",
        destination / "uncertainty_model_comparison.png",
    )
    write_json(
        destination / "comparison_manifest.json",
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "models": sorted(valid["model"].unique().tolist()),
            "folds": sorted(map(int, valid["fold"].unique().tolist())),
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MATR ANP model evaluations")
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = compare_evaluations(args.evaluations, args.output_dir)
    print(f"Comparison directory: {destination}")


if __name__ == "__main__":
    main()

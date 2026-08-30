"""Plot raw MATR SOH trajectories for all cells and file batches b1--b4."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import load_config, resolve_data_root
from .data import CellData, load_dataset
from .runtime import git_commit, write_json


BATCHES = ("b1", "b2", "b3", "b4")
BATCH_COLORS = {
    "b1": "#4C78A8",
    "b2": "#F28E2B",
    "b3": "#59A14F",
    "b4": "#E15759",
}
_CELL_ID_PATTERN = re.compile(r"^MATR_b([1-4])c\d+$", re.IGNORECASE)


def batch_from_cell_id(cell_id: str) -> str:
    """Return the MATR file batch encoded in IDs such as MATR_b2c17."""
    match = _CELL_ID_PATTERN.fullmatch(str(cell_id))
    if match is None:
        raise ValueError(
            f"unsupported MATR cell ID {cell_id!r}; expected MATR_b[1-4]c<number>"
        )
    return f"b{match.group(1)}"


def trajectory_frame(cells: list[CellData]) -> pd.DataFrame:
    """Convert cells to long-form raw-cycle SOH trajectories."""
    frames: list[pd.DataFrame] = []
    for cell in cells:
        batch = batch_from_cell_id(cell.cell_id)
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": cell.cell_id,
                    "batch": batch,
                    "cycle": cell.cycle_numbers.astype(np.int64),
                    "soh": cell.soh.astype(np.float64),
                }
            )
        )
    if not frames:
        raise ValueError("MATR contains no valid cells")
    result = pd.concat(frames, ignore_index=True)
    missing = set(BATCHES) - set(result["batch"].unique())
    if missing:
        raise ValueError(f"MATR is missing expected file batches: {sorted(missing)}")
    return result.sort_values(["batch", "cell_id", "cycle"]).reset_index(drop=True)


def batch_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Summarize cell counts and observed cycle ranges for each plot panel."""
    cells = (
        trajectories.groupby(["batch", "cell_id"], sort=True)
        .agg(
            first_cycle=("cycle", "min"),
            last_cycle=("cycle", "max"),
            initial_soh=("soh", "first"),
            final_soh=("soh", "last"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for batch in BATCHES:
        group = cells[cells["batch"] == batch]
        rows.append(
            {
                "batch": batch,
                "num_cells": int(len(group)),
                "minimum_last_cycle": int(group["last_cycle"].min()),
                "median_last_cycle": float(group["last_cycle"].median()),
                "maximum_last_cycle": int(group["last_cycle"].max()),
                "median_initial_soh": float(group["initial_soh"].median()),
                "median_final_soh": float(group["final_soh"].median()),
            }
        )
    return pd.DataFrame(rows)


def _axis_limits(
    trajectories: pd.DataFrame,
    y_min: float | None,
    y_max: float | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    maximum_cycle = float(trajectories["cycle"].max())
    x_limits = (0.0, maximum_cycle * 1.02)
    observed_min = float(trajectories["soh"].min())
    observed_max = float(trajectories["soh"].max())
    span = max(observed_max - observed_min, 0.05)
    lower = observed_min - 0.03 * span if y_min is None else float(y_min)
    upper = observed_max + 0.03 * span if y_max is None else float(y_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("SOH y-axis limits must be finite and increasing")
    return x_limits, (lower, upper)


def _draw_trajectories(
    axis: plt.Axes,
    trajectories: pd.DataFrame,
    *,
    batches: tuple[str, ...],
    title: str,
    eol_threshold: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    for (batch, _), group in trajectories.groupby(["batch", "cell_id"], sort=True):
        if batch not in batches:
            continue
        axis.plot(
            group["cycle"],
            group["soh"],
            color=BATCH_COLORS[str(batch)],
            linewidth=0.75,
            alpha=0.22 if len(batches) > 1 else 0.32,
        )
    axis.axhline(eol_threshold, color="black", linestyle="--", linewidth=1.1)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set(xlabel="Cycle", ylabel="SOH", title=title)
    axis.grid(alpha=0.2)


def plot_batch_trajectories(
    trajectories: pd.DataFrame,
    summary: pd.DataFrame,
    destination: str | Path,
    *,
    eol_threshold: float,
    y_min: float | None,
    y_max: float | None,
    dpi: int,
) -> Path:
    """Create exactly five panels: all MATR cells and one panel per batch."""
    if not 0.0 < eol_threshold < 1.5:
        raise ValueError("eol_threshold must lie in (0, 1.5)")
    x_limits, y_limits = _axis_limits(trajectories, y_min, y_max)
    figure = plt.figure(figsize=(18, 15), layout="constrained")
    grid = figure.add_gridspec(3, 2)
    axes = {
        "all": figure.add_subplot(grid[0, :]),
        "b1": figure.add_subplot(grid[1, 0]),
        "b2": figure.add_subplot(grid[1, 1]),
        "b3": figure.add_subplot(grid[2, 0]),
        "b4": figure.add_subplot(grid[2, 1]),
    }

    total_cells = int(summary["num_cells"].sum())
    _draw_trajectories(
        axes["all"], trajectories, batches=BATCHES,
        title=f"All MATR cells (n={total_cells})",
        eol_threshold=eol_threshold, x_limits=x_limits, y_limits=y_limits,
    )
    axes["all"].legend(
        handles=[
            Line2D([0], [0], color=BATCH_COLORS[batch], lw=2.2, label=batch)
            for batch in BATCHES
        ]
        + [
            Line2D(
                [0], [0], color="black", linestyle="--", lw=1.1,
                label=f"EOL SOH={eol_threshold:g}",
            )
        ],
        ncol=5,
    )

    for batch in BATCHES:
        row = summary[summary["batch"] == batch].iloc[0]
        title = (
            f"MATR {batch} (n={int(row['num_cells'])}; last cycle "
            f"min/median/max={int(row['minimum_last_cycle'])}/"
            f"{row['median_last_cycle']:.0f}/{int(row['maximum_last_cycle'])})"
        )
        _draw_trajectories(
            axes[batch], trajectories, batches=(batch,), title=title,
            eol_threshold=eol_threshold, x_limits=x_limits, y_limits=y_limits,
        )

    figure.suptitle(
        "MATR SOH trajectories by file-number batch (shared axes)", fontsize=16
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def run_analysis(
    config_path: str,
    *,
    data_root: str | None,
    output_dir: str | Path,
    eol_threshold: float,
    y_min: float | None,
    y_max: float | None,
    dpi: int,
) -> Path:
    config = load_config(config_path)
    if config.data.dataset.upper() != "MATR":
        raise ValueError("config must describe the MATR dataset")
    resolved_root = resolve_data_root(config, data_root)
    cells, audit = load_dataset(
        resolved_root, config.data, tolerate_invalid_cells=True
    )
    trajectories = trajectory_frame(cells)
    summary = batch_summary(trajectories)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plot_path = plot_batch_trajectories(
        trajectories,
        summary,
        destination / "matr_soh_trajectories_all_b1_b2_b3_b4.png",
        eol_threshold=eol_threshold,
        y_min=y_min,
        y_max=y_max,
        dpi=dpi,
    )
    trajectories.to_csv(destination / "trajectory_points.csv", index=False)
    summary.to_csv(destination / "batch_summary.csv", index=False)
    audit.to_csv(destination / "data_audit.csv", index=False)
    write_json(
        destination / "analysis_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "data_root": str(resolved_root),
            "cells_loaded": len(cells),
            "batches": summary.set_index("batch")["num_cells"].to_dict(),
            "eol_threshold": eol_threshold,
            "shared_y_limits": [y_min, y_max],
            "plot": str(plot_path),
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all MATR SOH trajectories and batches b1--b4"
    )
    parser.add_argument("--config", default="configs/matr_hs_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument(
        "--output-dir", default="outputs/data_analysis/matr_soh_by_batch"
    )
    parser.add_argument("--eol-threshold", type=float, default=0.8)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = run_analysis(
        args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        eol_threshold=args.eol_threshold,
        y_min=args.y_min,
        y_max=args.y_max,
        dpi=args.dpi,
    )
    summary = pd.read_csv(destination / "batch_summary.csv")
    print(summary.to_string(index=False))
    print(f"Output: {destination}")


if __name__ == "__main__":
    main()

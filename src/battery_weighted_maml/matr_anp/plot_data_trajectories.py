"""Plot observed MATR SOH trajectories without training or inference."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import load_config, resolve_data_root, save_config
from .data import CellData, load_matr_dataset
from .runtime import git_commit, write_json


_BATCH_PATTERN = re.compile(r"^MATR_b(?P<batch>\d+)c", re.IGNORECASE)


def matr_batch(cell_id: str) -> str:
    """Return the BatteryLife MATR batch label encoded in a cell ID."""
    match = _BATCH_PATTERN.match(cell_id)
    return f"b{match.group('batch')}" if match else "other"


def trajectory_frame(cells: Iterable[CellData]) -> pd.DataFrame:
    """Convert cell trajectories to a tidy plotting table."""
    frames: list[pd.DataFrame] = []
    for cell in cells:
        cycles = cell.cycle_numbers.astype(np.int64)
        if len(cycles) < 2:
            normalized_life = np.zeros(len(cycles), dtype=np.float64)
        else:
            span = float(cycles[-1] - cycles[0])
            normalized_life = (cycles - cycles[0]) / max(span, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": cell.cell_id,
                    "batch": matr_batch(cell.cell_id),
                    "cycle": cycles,
                    "normalized_life": normalized_life,
                    "soh": cell.soh,
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["cell_id", "batch", "cycle", "normalized_life", "soh"]
        )
    return pd.concat(frames, ignore_index=True)


def trajectory_summary(
    trajectories: pd.DataFrame,
    *,
    eol_threshold: float,
) -> pd.DataFrame:
    """Summarize lifespan and the first observed EOL-threshold crossing."""
    rows = []
    for (cell_id, batch), group in trajectories.groupby(
        ["cell_id", "batch"], sort=True
    ):
        ordered = group.sort_values("cycle")
        crossed = ordered[ordered["soh"] <= eol_threshold]
        rows.append(
            {
                "cell_id": cell_id,
                "batch": batch,
                "num_cycles": int(len(ordered)),
                "first_cycle": int(ordered["cycle"].iloc[0]),
                "last_cycle": int(ordered["cycle"].iloc[-1]),
                "initial_soh": float(ordered["soh"].iloc[0]),
                "final_soh": float(ordered["soh"].iloc[-1]),
                "eol_threshold": eol_threshold,
                "first_eol_crossing_cycle": (
                    int(crossed["cycle"].iloc[0]) if not crossed.empty else np.nan
                ),
                "reached_eol_threshold": not crossed.empty,
            }
        )
    return pd.DataFrame(rows)


def plot_trajectories(
    trajectories: pd.DataFrame,
    destination: str | Path,
    *,
    x_axis: str = "cycle",
    eol_threshold: float = 0.8,
    line_alpha: float = 0.28,
    highlight_cells: Iterable[str] = (),
    dpi: int = 200,
) -> Path:
    """Draw all selected cell trajectories on a single axis."""
    if trajectories.empty:
        raise ValueError("no MATR trajectories were selected")
    if x_axis not in {"cycle", "normalized-life"}:
        raise ValueError("x_axis must be 'cycle' or 'normalized-life'")
    if not 0.0 < eol_threshold <= 1.5:
        raise ValueError("eol_threshold must lie in (0,1.5]")
    if not 0.0 < line_alpha <= 1.0:
        raise ValueError("line_alpha must lie in (0,1]")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    x_column = "cycle" if x_axis == "cycle" else "normalized_life"
    x_label = "Cycle number" if x_axis == "cycle" else "Normalized lifetime"
    batches = sorted(trajectories["batch"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {batch: cmap(index % 10) for index, batch in enumerate(batches)}
    highlighted = set(highlight_cells)
    missing_highlights = sorted(highlighted - set(trajectories["cell_id"].unique()))
    if missing_highlights:
        raise ValueError(f"highlight cells are not selected: {missing_highlights}")

    figure, axis = plt.subplots(figsize=(14, 8))
    for (cell_id, batch), group in trajectories.groupby(
        ["cell_id", "batch"], sort=True
    ):
        ordered = group.sort_values(x_column)
        is_highlighted = cell_id in highlighted
        axis.plot(
            ordered[x_column],
            ordered["soh"],
            color=colors[batch],
            linewidth=2.4 if is_highlighted else 0.85,
            alpha=1.0 if is_highlighted else line_alpha,
            zorder=3 if is_highlighted else 1,
        )

    axis.axhline(
        eol_threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )
    cell_count = trajectories["cell_id"].nunique()
    axis.set(
        xlabel=x_label,
        ylabel="SOH (discharge capacity / nominal capacity)",
        title=f"MATR observed SOH trajectories ({cell_count} cells)",
    )
    axis.grid(alpha=0.22)
    handles = [
        Line2D([0], [0], color=colors[batch], linewidth=2, label=batch)
        for batch in batches
    ]
    handles.append(
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.2,
            label=f"EOL threshold = {eol_threshold:g}",
        )
    )
    for cell_id in sorted(highlighted):
        batch = str(
            trajectories.loc[trajectories["cell_id"] == cell_id, "batch"].iloc[0]
        )
        handles.append(
            Line2D([0], [0], color=colors[batch], linewidth=2.5, label=cell_id)
        )
    axis.legend(handles=handles, ncol=min(4, len(handles)), fontsize=9)
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def create_trajectory_plot(
    config_path: str,
    data_root_arg: str | None,
    output_dir: str | None,
    *,
    x_axis: str,
    batches: Iterable[str] | None,
    cell_ids: Iterable[str] | None,
    highlight_cells: Iterable[str],
    eol_threshold: float,
    line_alpha: float,
    dpi: int,
) -> Path:
    config = load_config(config_path)
    data_root = resolve_data_root(config, data_root_arg)
    destination = (
        Path(output_dir).resolve()
        if output_dir
        else Path(config.paths.output_root).resolve() / "data_trajectories"
    )
    destination.mkdir(parents=True, exist_ok=True)
    cells, audit = load_matr_dataset(
        data_root, config.data, tolerate_invalid_cells=True
    )
    selected_batches = set(batches or [])
    selected_ids = set(cell_ids or [])
    if selected_batches:
        cells = [cell for cell in cells if matr_batch(cell.cell_id) in selected_batches]
    if selected_ids:
        available = {cell.cell_id for cell in cells}
        missing = sorted(selected_ids - available)
        if missing:
            raise ValueError(f"requested MATR cells are unavailable: {missing}")
        cells = [cell for cell in cells if cell.cell_id in selected_ids]
    if not cells:
        raise ValueError("MATR trajectory selection contains no valid cells")

    trajectories = trajectory_frame(cells)
    summary = trajectory_summary(trajectories, eol_threshold=eol_threshold)
    plot_path = plot_trajectories(
        trajectories,
        destination / f"matr_soh_trajectories_{x_axis}.png",
        x_axis=x_axis,
        eol_threshold=eol_threshold,
        line_alpha=line_alpha,
        highlight_cells=highlight_cells,
        dpi=dpi,
    )
    trajectories.to_csv(destination / "trajectory_points.csv", index=False)
    summary.to_csv(destination / "trajectory_summary.csv", index=False)
    audit.to_csv(destination / "data_audit.csv", index=False)
    save_config(config, destination / "resolved_config.yaml")
    write_json(
        destination / "trajectory_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "MATR",
            "data_root": str(data_root),
            "git_commit": git_commit(),
            "plot": str(plot_path),
            "x_axis": x_axis,
            "eol_threshold": eol_threshold,
            "cell_count": len(cells),
            "batches": sorted(trajectories["batch"].unique()),
            "highlight_cells": sorted(set(highlight_cells)),
            "invalid_file_count": int((audit["status"] != "valid").sum()),
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all observed MATR SOH trajectories on one figure"
    )
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--x-axis", choices=("cycle", "normalized-life"), default="cycle"
    )
    parser.add_argument("--batch", dest="batches", nargs="+")
    parser.add_argument("--cell-id", dest="cell_ids", nargs="+")
    parser.add_argument("--highlight-cell", dest="highlight_cells", nargs="+", default=[])
    parser.add_argument("--eol-threshold", type=float, default=0.8)
    parser.add_argument("--line-alpha", type=float, default=0.28)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = create_trajectory_plot(
        args.config,
        args.data_root,
        args.output_dir,
        x_axis=args.x_axis,
        batches=args.batches,
        cell_ids=args.cell_ids,
        highlight_cells=args.highlight_cells,
        eol_threshold=args.eol_threshold,
        line_alpha=args.line_alpha,
        dpi=args.dpi,
    )
    print(f"MATR trajectory plot: {destination}")


if __name__ == "__main__":
    main()

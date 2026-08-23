"""Plot all discharge-voltage cycles for reproducibly sampled MATR cells."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .config import load_config, resolve_data_root, save_config
from .data import CellData, load_dataset
from .runtime import git_commit, write_json


def select_cells(
    cells: Sequence[CellData],
    *,
    count: int,
    seed: int,
    cell_ids: Iterable[str] | None = None,
) -> list[CellData]:
    """Select explicit cells or a deterministic sample without replacement."""
    if count <= 0:
        raise ValueError("count must be positive")
    by_id = {cell.cell_id: cell for cell in cells}
    requested = list(dict.fromkeys(cell_ids or []))
    if requested:
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(f"requested MATR cells are unavailable: {missing}")
        return [by_id[cell_id] for cell_id in requested]
    if len(cells) < count:
        raise ValueError(f"requested {count} cells, but only {len(cells)} are valid")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(index)] for index in indices]


def _cell_profiles(
    cell: CellData,
    *,
    q_min: float | None,
    q_max: float | None,
    max_points_per_curve: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    segments: list[np.ndarray] = []
    cycle_numbers: list[int] = []
    for cycle in cell.cycles:
        curve = cycle.discharge
        if curve is None:
            continue
        valid = np.isfinite(curve.q) & np.isfinite(curve.voltage_v)
        if q_min is not None:
            valid &= curve.q >= q_min
        if q_max is not None:
            valid &= curve.q <= q_max
        if np.count_nonzero(valid) < 2:
            continue
        q_values = curve.q[valid]
        voltage_values = curve.voltage_v[valid]
        if len(q_values) > max_points_per_curve:
            indices = np.linspace(
                0, len(q_values) - 1, max_points_per_curve, dtype=np.int64
            )
            q_values = q_values[indices]
            voltage_values = voltage_values[indices]
        segments.append(np.column_stack((q_values, voltage_values)))
        cycle_numbers.append(cycle.cycle_number)
    return segments, np.asarray(cycle_numbers, dtype=np.int64)


def _color_values(cycles: np.ndarray, color_by: str) -> np.ndarray:
    if color_by == "cycle":
        return cycles.astype(np.float64)
    if color_by != "normalized-life":
        raise ValueError("color_by must be 'cycle' or 'normalized-life'")
    if len(cycles) < 2 or cycles[-1] == cycles[0]:
        return np.zeros(len(cycles), dtype=np.float64)
    return (cycles - cycles[0]) / float(cycles[-1] - cycles[0])


def plot_voltage_grid(
    cells: Sequence[CellData],
    destination: str | Path,
    *,
    columns: int = 5,
    q_min: float | None = None,
    q_max: float | None = None,
    color_by: str = "normalized-life",
    cmap: str = "viridis",
    line_width: float = 0.38,
    line_alpha: float = 0.38,
    max_points_per_curve: int = 256,
    dpi: int = 200,
) -> pd.DataFrame:
    """Draw one all-cycle voltage subplot per cell and return its summary."""
    if not cells:
        raise ValueError("no MATR cells were selected")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if q_min is not None and q_max is not None and q_min >= q_max:
        raise ValueError("q_min must be smaller than q_max")
    if (
        line_width <= 0
        or not 0.0 < line_alpha <= 1.0
        or max_points_per_curve < 2
        or dpi <= 0
    ):
        raise ValueError("line_width, line_alpha, and dpi must be positive")
    if color_by not in {"cycle", "normalized-life"}:
        raise ValueError("color_by must be 'cycle' or 'normalized-life'")

    prepared: list[tuple[CellData, list[np.ndarray], np.ndarray, np.ndarray]] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for cell in cells:
        segments, cycles = _cell_profiles(
            cell,
            q_min=q_min,
            q_max=q_max,
            max_points_per_curve=max_points_per_curve,
        )
        if not segments:
            raise ValueError(f"{cell.cell_id}: no plottable discharge-voltage cycles")
        colors = _color_values(cycles, color_by)
        prepared.append((cell, segments, cycles, colors))
        all_x.extend(segment[:, 0] for segment in segments)
        all_y.extend(segment[:, 1] for segment in segments)
        all_colors.append(colors)

    color_values = np.concatenate(all_colors)
    color_min = float(np.min(color_values))
    color_max = float(np.max(color_values))
    if color_max <= color_min:
        color_max = color_min + 1.0
    norm = Normalize(color_min, color_max)

    x_min = float(min(np.min(values) for values in all_x))
    x_max = float(max(np.max(values) for values in all_x))
    y_min = float(min(np.min(values) for values in all_y))
    y_max = float(max(np.max(values) for values in all_y))
    x_margin = max(0.005, 0.015 * (x_max - x_min))
    y_margin = max(0.01, 0.035 * (y_max - y_min))

    rows = math.ceil(len(cells) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.45 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    flat_axes = axes.ravel()
    summary_rows: list[dict[str, object]] = []
    for axis, (cell, segments, cycles, colors) in zip(flat_axes, prepared):
        collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=line_width,
            alpha=line_alpha,
            rasterized=True,
        )
        collection.set_array(colors)
        axis.add_collection(collection)
        axis.set_xlim(x_min - x_margin, x_max + x_margin)
        axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.set_title(
            f"{cell.cell_id}\n{len(cycles):,} cycles ({cycles[0]}–{cycles[-1]})",
            fontsize=10,
        )
        summary_rows.append(
            {
                "cell_id": cell.cell_id,
                "source_file": cell.source_file,
                "total_valid_cycles": len(cell.cycles),
                "plotted_cycles": len(cycles),
                "first_plotted_cycle": int(cycles[0]),
                "last_plotted_cycle": int(cycles[-1]),
                "q_min": float(min(np.min(segment[:, 0]) for segment in segments)),
                "q_max": float(max(np.max(segment[:, 0]) for segment in segments)),
                "voltage_min_v": float(
                    min(np.min(segment[:, 1]) for segment in segments)
                ),
                "voltage_max_v": float(
                    max(np.max(segment[:, 1]) for segment in segments)
                ),
            }
        )
    for axis in flat_axes[len(prepared) :]:
        axis.set_visible(False)

    figure.supxlabel("Normalized discharged capacity q = Qd / Qnom")
    figure.supylabel("Discharge voltage (V)")
    figure.suptitle(
        f"MATR discharge-voltage profiles: every cycle for {len(cells)} cells",
        fontsize=15,
    )
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=[axis for axis in flat_axes[: len(prepared)]],
        shrink=0.88,
        pad=0.012,
    )
    colorbar.set_label(
        "Cycle number" if color_by == "cycle" else "Normalized cycle position"
    )

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return pd.DataFrame(summary_rows)


def create_voltage_plot(
    config_path: str,
    data_root_arg: str | None,
    output_dir: str | None,
    *,
    count: int,
    seed: int,
    cell_ids: Iterable[str] | None,
    columns: int,
    q_min: float | None,
    q_max: float | None,
    color_by: str,
    cmap: str,
    line_width: float,
    line_alpha: float,
    max_points_per_curve: int,
    dpi: int,
) -> Path:
    config = load_config(config_path)
    if config.data.dataset.upper() != "MATR":
        raise ValueError("this plot requires a MATR configuration")
    data_root = resolve_data_root(config, data_root_arg)
    destination = (
        Path(output_dir).resolve()
        if output_dir
        else Path(config.paths.output_root).resolve()
        / "data_voltage_cycles"
        / f"random{count}_seed{seed}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    cells, audit = load_dataset(data_root, config.data, tolerate_invalid_cells=True)
    selected = select_cells(cells, count=count, seed=seed, cell_ids=cell_ids)
    plot_path = destination / f"matr_{len(selected)}cells_all_cycle_voltage.png"
    summary = plot_voltage_grid(
        selected,
        plot_path,
        columns=columns,
        q_min=q_min,
        q_max=q_max,
        color_by=color_by,
        cmap=cmap,
        line_width=line_width,
        line_alpha=line_alpha,
        max_points_per_curve=max_points_per_curve,
        dpi=dpi,
    )
    summary.insert(0, "plot_order", np.arange(1, len(summary) + 1))
    summary.to_csv(destination / "selected_cells.csv", index=False)
    audit.to_csv(destination / "data_audit.csv", index=False)
    save_config(config, destination / "resolved_config.yaml")
    write_json(
        destination / "plot_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "MATR",
            "data_root": str(data_root),
            "git_commit": git_commit(),
            "plot": str(plot_path),
            "selection_seed": seed,
            "requested_random_count": count,
            "selected_cells": [cell.cell_id for cell in selected],
            "color_by": color_by,
            "q_min": q_min,
            "q_max": q_max,
            "max_points_per_curve": max_points_per_curve,
            "invalid_file_count": int((audit["status"] != "valid").sum()),
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot every discharge-voltage cycle for 10 reproducibly sampled MATR cells"
        )
    )
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--num-cells", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-id", dest="cell_ids", nargs="+")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--q-min", type=float)
    parser.add_argument("--q-max", type=float)
    parser.add_argument(
        "--color-by",
        choices=("cycle", "normalized-life"),
        default="normalized-life",
    )
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--line-width", type=float, default=0.38)
    parser.add_argument("--line-alpha", type=float, default=0.38)
    parser.add_argument("--max-points-per-curve", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = create_voltage_plot(
        args.config,
        args.data_root,
        args.output_dir,
        count=args.num_cells,
        seed=args.seed,
        cell_ids=args.cell_ids,
        columns=args.columns,
        q_min=args.q_min,
        q_max=args.q_max,
        color_by=args.color_by,
        cmap=args.cmap,
        line_width=args.line_width,
        line_alpha=args.line_alpha,
        max_points_per_curve=args.max_points_per_curve,
        dpi=args.dpi,
    )
    print(f"MATR voltage-cycle plot directory: {destination}")


if __name__ == "__main__":
    main()

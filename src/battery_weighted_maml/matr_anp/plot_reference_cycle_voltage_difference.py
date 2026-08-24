"""Plot per-cell voltage-Q changes relative to one reference cycle."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from .config import QGridConfig, load_config, resolve_data_root, save_config
from .data import CellData, load_dataset
from .features import PartialIVProcessor
from .runtime import git_commit, write_json


@dataclass(frozen=True)
class DifferenceCurve:
    cycle: int
    q: np.ndarray
    delta_voltage_v: np.ndarray
    normalized_cycle_position: float


def delta_voltage_axis_limits(summary: pd.DataFrame) -> tuple[float, float]:
    """Return the shared y range used for one difference-curve figure."""
    if summary.empty:
        raise ValueError("cannot determine axis limits from an empty summary")
    data_min = float(summary["delta_v_min"].min())
    data_max = float(summary["delta_v_max"].max())
    span = max(data_max - data_min, 1.0e-3)
    margin = max(0.005, 0.05 * span)
    return data_min - margin, data_max + margin


def load_axis_reference(path: str | Path) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read x/y limits from a prior run, including older manifests without y limits."""
    source = Path(path)
    directory = source if source.is_dir() else source.parent
    manifest_path = directory / "plot_manifest.json"
    summary_path = directory / "per_cycle_delta_voltage_summary.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"axis-reference manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    q_limits = float(manifest["q_min"]), float(manifest["q_max"])
    saved = manifest.get("axis_limits", {})
    if "delta_v_min" in saved and "delta_v_max" in saved:
        voltage_limits = float(saved["delta_v_min"]), float(saved["delta_v_max"])
    else:
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"older axis-reference needs its summary CSV: {summary_path}"
            )
        voltage_limits = delta_voltage_axis_limits(pd.read_csv(summary_path))
    if q_limits[0] >= q_limits[1] or voltage_limits[0] >= voltage_limits[1]:
        raise ValueError(f"invalid plot limits in axis reference: {directory}")
    return q_limits, voltage_limits


def _has_reference_and_future(cell: CellData, reference_cycle: int) -> bool:
    reference_available = any(
        cycle.cycle_number == reference_cycle and cycle.discharge is not None
        for cycle in cell.cycles
    )
    future_available = any(
        cycle.cycle_number > reference_cycle and cycle.discharge is not None
        for cycle in cell.cycles
    )
    return reference_available and future_available


def select_reference_cells(
    cells: Sequence[CellData],
    *,
    count: int,
    reference_cycle: int,
    seed: int,
    cell_ids: Iterable[str] | None = None,
) -> list[CellData]:
    """Select cells that contain the reference and at least one later curve."""
    if count <= 0:
        raise ValueError("count must be positive")
    eligible = [
        cell for cell in cells if _has_reference_and_future(cell, reference_cycle)
    ]
    by_id = {cell.cell_id: cell for cell in eligible}
    requested = list(dict.fromkeys(cell_ids or []))
    if requested:
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(
                f"cells missing cycle {reference_cycle} or later curves: {missing}"
            )
        return [by_id[cell_id] for cell_id in requested]
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} cells contain cycle {reference_cycle} and later curves"
        )
    indices = np.random.default_rng(seed).choice(len(eligible), size=count, replace=False)
    return [eligible[int(index)] for index in indices]


def voltage_difference_curves(
    cell: CellData,
    processor: PartialIVProcessor,
    *,
    reference_cycle: int = 10,
) -> list[DifferenceCurve]:
    """Calculate V_cycle(q)-V_reference(q) only over observed q overlap."""
    reference_record = cell.cycle_by_number(reference_cycle)
    if reference_record.discharge is None:
        raise ValueError(f"{cell.cell_id}: cycle {reference_cycle} has no discharge curve")
    reference = processor.interpolate(reference_record.discharge)
    future = [
        cycle
        for cycle in cell.cycles
        if cycle.cycle_number > reference_cycle and cycle.discharge is not None
    ]
    if not future:
        raise ValueError(f"{cell.cell_id}: no curves after cycle {reference_cycle}")
    first_cycle = future[0].cycle_number
    last_cycle = future[-1].cycle_number
    denominator = max(1, last_cycle - first_cycle)
    output: list[DifferenceCurve] = []
    for cycle in future:
        assert cycle.discharge is not None
        current = processor.interpolate(cycle.discharge)
        valid = reference.mask & current.mask
        if np.count_nonzero(valid) < 2:
            continue
        output.append(
            DifferenceCurve(
                cycle=cycle.cycle_number,
                q=processor.grid[valid].copy(),
                delta_voltage_v=(
                    current.voltage_v[valid] - reference.voltage_v[valid]
                ),
                normalized_cycle_position=(cycle.cycle_number - first_cycle)
                / denominator,
            )
        )
    if not output:
        raise ValueError(f"{cell.cell_id}: no q overlap with cycle {reference_cycle}")
    return output


def plot_reference_voltage_differences(
    cells: Sequence[CellData],
    processor: PartialIVProcessor,
    destination: str | Path,
    *,
    dataset_name: str = "MATR",
    reference_cycle: int = 10,
    delta_voltage_limits: tuple[float, float] | None = None,
    columns: int = 3,
    cmap: str = "viridis",
    line_width: float = 0.55,
    line_alpha: float = 0.42,
    dpi: int = 200,
) -> pd.DataFrame:
    """Create one subplot per cell and return per-cycle difference statistics."""
    if not cells:
        raise ValueError("no cells were selected")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if line_width <= 0 or not 0.0 < line_alpha <= 1.0 or dpi <= 0:
        raise ValueError("line width, alpha, and dpi must be positive")

    prepared: list[tuple[CellData, list[DifferenceCurve]]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for cell in cells:
        curves = voltage_difference_curves(
            cell, processor, reference_cycle=reference_cycle
        )
        prepared.append((cell, curves))
        for curve in curves:
            summary_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "reference_cycle": reference_cycle,
                    "cycle": curve.cycle,
                    "normalized_cycle_position": curve.normalized_cycle_position,
                    "q_min": float(curve.q[0]),
                    "q_max": float(curve.q[-1]),
                    "q_points": len(curve.q),
                    "delta_v_mean": float(np.mean(curve.delta_voltage_v)),
                    "delta_v_median": float(np.median(curve.delta_voltage_v)),
                    "delta_v_min": float(np.min(curve.delta_voltage_v)),
                    "delta_v_max": float(np.max(curve.delta_voltage_v)),
                    "delta_v_at_last_q": float(curve.delta_voltage_v[-1]),
                }
            )

    summary = pd.DataFrame(summary_rows)
    y_limits = (
        delta_voltage_axis_limits(summary)
        if delta_voltage_limits is None
        else tuple(map(float, delta_voltage_limits))
    )
    if y_limits[0] >= y_limits[1]:
        raise ValueError("delta_voltage_limits must be increasing")
    rows_count = math.ceil(len(prepared) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(5.2 * columns, 4.1 * rows_count),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    flat_axes = axes.ravel()
    norm = Normalize(0.0, 1.0)
    for axis, (cell, curves) in zip(flat_axes, prepared):
        segments = [
            np.column_stack((curve.q, curve.delta_voltage_v)) for curve in curves
        ]
        collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=line_width,
            alpha=line_alpha,
            rasterized=True,
        )
        collection.set_array(
            np.asarray([curve.normalized_cycle_position for curve in curves])
        )
        axis.add_collection(collection)
        axis.axhline(0.0, color="black", lw=0.9, alpha=0.65)
        axis.set_xlim(processor.grid[0], processor.grid[-1])
        axis.set_ylim(*y_limits)
        axis.set_title(
            f"{cell.cell_id}\n{len(curves):,} curves: "
            f"cycle {curves[0].cycle}–{curves[-1].cycle}",
            fontsize=10,
        )
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.set_xlabel("Normalized discharged capacity q = Qd / Qnom")
        axis.set_ylabel(f"ΔV = V(cycle) − V(cycle {reference_cycle})  [V]")
    for axis in flat_axes[len(prepared) :]:
        axis.set_visible(False)

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=[axis for axis in flat_axes[: len(prepared)]],
        shrink=0.88,
        pad=0.015,
    )
    colorbar.set_label(f"Normalized cycle position after cycle {reference_cycle}")
    figure.suptitle(
        f"{dataset_name.upper()} voltage–Q change relative to cycle {reference_cycle}",
        fontsize=15,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return summary


def parse_args(
    default_config: str = "configs/matr_partial_iv_anp.yaml",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot voltage-Q differences from cycle 10 for five cells"
    )
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--num-cells", type=int, default=5)
    parser.add_argument("--cell-id", dest="cell_ids", nargs="+")
    parser.add_argument("--reference-cycle", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-min", type=float, default=0.0)
    parser.add_argument("--q-max", type=float, default=1.0)
    parser.add_argument("--q-points", type=int, default=256)
    parser.add_argument("--delta-v-min", type=float)
    parser.add_argument("--delta-v-max", type=float)
    parser.add_argument(
        "--axis-reference",
        help=(
            "prior run directory whose q and delta-V axes must be reused; "
            "for CALCE, point this to the corresponding MATR run"
        ),
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--line-width", type=float, default=0.55)
    parser.add_argument("--line-alpha", type=float, default=0.42)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main(default_config: str = "configs/matr_partial_iv_anp.yaml") -> None:
    args = parse_args(default_config)
    if (args.delta_v_min is None) != (args.delta_v_max is None):
        raise ValueError("delta-v-min and delta-v-max must be supplied together")
    if args.q_min >= args.q_max or args.q_points < 2:
        raise ValueError("q range must increase and contain at least two points")
    config = load_config(args.config)
    dataset_name = config.data.dataset.upper()
    if dataset_name not in {"MATR", "CALCE"}:
        raise ValueError("this analysis requires a MATR or CALCE configuration")
    data_root = resolve_data_root(config, args.data_root)
    cells, audit = load_dataset(data_root, config.data, tolerate_invalid_cells=True)
    selected = select_reference_cells(
        cells,
        count=args.num_cells,
        reference_cycle=args.reference_cycle,
        seed=args.seed,
        cell_ids=args.cell_ids,
    )
    q_limits = (args.q_min, args.q_max)
    delta_voltage_limits = (
        None
        if args.delta_v_min is None
        else (args.delta_v_min, args.delta_v_max)
    )
    if args.axis_reference:
        if delta_voltage_limits is not None:
            raise ValueError("axis-reference cannot be combined with delta-v limits")
        q_limits, delta_voltage_limits = load_axis_reference(args.axis_reference)
    processor = PartialIVProcessor(
        QGridConfig(q_limits[0], q_limits[1], args.q_points), config.data
    )
    destination = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config.paths.output_root).resolve()
        / "data_cycle10_voltage_difference"
        / f"random{len(selected)}_seed{args.seed}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    plot_path = destination / (
        f"{dataset_name.lower()}_{len(selected)}cells_cycle"
        f"{args.reference_cycle}_delta_voltage_q.png"
    )
    summary = plot_reference_voltage_differences(
        selected,
        processor,
        plot_path,
        dataset_name=dataset_name,
        reference_cycle=args.reference_cycle,
        delta_voltage_limits=delta_voltage_limits,
        columns=args.columns,
        cmap=args.cmap,
        line_width=args.line_width,
        line_alpha=args.line_alpha,
        dpi=args.dpi,
    )
    summary.to_csv(destination / "per_cycle_delta_voltage_summary.csv", index=False)
    used_delta_limits = (
        delta_voltage_axis_limits(summary)
        if delta_voltage_limits is None
        else delta_voltage_limits
    )
    pd.DataFrame(
        {
            "plot_order": np.arange(1, len(selected) + 1),
            "cell_id": [cell.cell_id for cell in selected],
            "source_file": [cell.source_file for cell in selected],
        }
    ).to_csv(destination / "selected_cells.csv", index=False)
    audit.to_csv(destination / "data_audit.csv", index=False)
    save_config(config, destination / "resolved_config.yaml")
    write_json(
        destination / "plot_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "data_root": str(data_root),
            "git_commit": git_commit(),
            "plot": str(plot_path),
            "reference_cycle": args.reference_cycle,
            "selected_cells": [cell.cell_id for cell in selected],
            "seed": args.seed,
            "q_min": q_limits[0],
            "q_max": q_limits[1],
            "q_points": args.q_points,
            "axis_reference": args.axis_reference,
            "axis_limits": {
                "delta_v_min": used_delta_limits[0],
                "delta_v_max": used_delta_limits[1],
            },
        },
    )
    print(f"Selected cells: {', '.join(cell.cell_id for cell in selected)}")
    print(f"Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()

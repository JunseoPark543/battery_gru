"""Plot selected MATR discharge V-Q curves beside the full SOH trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, resolve_data_root
from .data import CellData, CycleData, load_dataset


DEFAULT_CELL_ID = "MATR_b1c26"
DEFAULT_CYCLES = (50, 130, 200, 500, 600)


def select_cell(cells: Sequence[CellData], cell_id: str) -> CellData:
    """Return one explicitly requested cell."""
    for cell in cells:
        if cell.cell_id == cell_id:
            return cell
    available = ", ".join(cell.cell_id for cell in cells[:10])
    raise ValueError(
        f"requested cell is unavailable: {cell_id}; first available cells: {available}"
    )


def select_cycles(cell: CellData, cycle_numbers: Sequence[int]) -> list[CycleData]:
    """Return unique requested cycles with valid discharge V-Q curves."""
    requested = list(dict.fromkeys(int(value) for value in cycle_numbers))
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("cycle numbers must be positive")
    by_number = {cycle.cycle_number: cycle for cycle in cell.cycles}
    missing = [value for value in requested if value not in by_number]
    if missing:
        raise ValueError(f"{cell.cell_id}: requested cycles are unavailable: {missing}")
    without_curve = [
        value for value in requested if by_number[value].discharge is None
    ]
    if without_curve:
        raise ValueError(
            f"{cell.cell_id}: requested cycles have no valid discharge V-Q curve: "
            f"{without_curve}"
        )
    return [by_number[value] for value in requested]


def selected_vq_frames(
    cell: CellData,
    selected: Sequence[CycleData],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build point-level V-Q data and one-row-per-cycle SOH summary."""
    point_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for cycle in selected:
        curve = cycle.discharge
        if curve is None:  # Protected by select_cycles; retained for direct callers.
            raise ValueError(f"{cell.cell_id}: cycle {cycle.cycle_number} has no V-Q curve")
        point_frames.append(
            pd.DataFrame(
                {
                    "cell_id": cell.cell_id,
                    "cycle": cycle.cycle_number,
                    "q_normalized": curve.q,
                    "discharge_capacity_ah": curve.q * cell.nominal_capacity_ah,
                    "voltage_v": curve.voltage_v,
                    "soh": cycle.soh,
                }
            )
        )
        summary_rows.append(
            {
                "cell_id": cell.cell_id,
                "cycle": cycle.cycle_number,
                "nominal_capacity_ah": cell.nominal_capacity_ah,
                "discharge_capacity_ah": cycle.discharge_capacity_ah,
                "soh": cycle.soh,
                "vq_q_end": float(curve.q[-1]),
                "vq_voltage_end_v": float(curve.voltage_v[-1]),
                "vq_q_end_minus_soh": float(curve.q[-1] - cycle.soh),
                "vq_points": len(curve.q),
            }
        )
    return pd.concat(point_frames, ignore_index=True), pd.DataFrame(summary_rows)


def plot_selected_vq_soh(
    cell: CellData,
    cycle_numbers: Sequence[int],
    destination: str | Path,
    *,
    q_limits: tuple[float, float] | None = None,
    voltage_limits: tuple[float, float] | None = None,
    eol_threshold: float = 0.8,
    dpi: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw selected V-Q curves and their positions on the observed SOH curve."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not 0.0 < eol_threshold <= 1.5:
        raise ValueError("eol_threshold must lie in (0, 1.5]")
    for name, limits in (("q", q_limits), ("voltage", voltage_limits)):
        if limits is not None and limits[0] >= limits[1]:
            raise ValueError(f"{name} minimum must be smaller than maximum")

    selected = select_cycles(cell, cycle_numbers)
    points, summary = selected_vq_frames(cell, selected)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.92, len(selected))
    )
    figure, (vq_axis, soh_axis) = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.3),
        layout="constrained",
    )

    for color, cycle in zip(colors, selected):
        curve = cycle.discharge
        assert curve is not None
        label = f"Cycle {cycle.cycle_number}  (SOH={cycle.soh:.4f})"
        vq_axis.plot(curve.q, curve.voltage_v, color=color, lw=2.0, label=label)
        vq_axis.scatter(
            curve.q[-1],
            curve.voltage_v[-1],
            color=color,
            s=42,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    vq_axis.set(
        xlabel="Normalized discharged capacity  q = Qd / Qnom",
        ylabel="Discharge voltage (V)",
        title="Selected discharge V-Q curves\n(dot = observed curve endpoint)",
    )
    if q_limits is not None:
        vq_axis.set_xlim(*q_limits)
    if voltage_limits is not None:
        vq_axis.set_ylim(*voltage_limits)
    vq_axis.grid(alpha=0.24)
    vq_axis.legend(fontsize=9, loc="best")

    all_cycles = cell.cycle_numbers
    all_soh = cell.soh
    soh_axis.plot(
        all_cycles,
        all_soh,
        color="0.32",
        lw=1.35,
        alpha=0.8,
        label="Observed SOH trajectory",
    )
    soh_axis.axhline(
        eol_threshold,
        color="black",
        ls="--",
        lw=1.0,
        alpha=0.75,
        label=f"EOL threshold = {eol_threshold:g}",
    )
    for color, cycle in zip(colors, selected):
        soh_axis.scatter(
            cycle.cycle_number,
            cycle.soh,
            color=color,
            s=58,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        soh_axis.annotate(
            f"C{cycle.cycle_number}\n{cycle.soh:.4f}",
            (cycle.cycle_number, cycle.soh),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )
    soh_axis.set(
        xlabel="Cycle number",
        ylabel="SOH = discharge capacity / nominal capacity",
        title="Full observed SOH trajectory\n(selected cycles are highlighted)",
    )
    soh_axis.grid(alpha=0.24)
    soh_axis.legend(fontsize=9, loc="best")
    figure.suptitle(
        f"{cell.cell_id}: V-Q curves and SOH | Qnom={cell.nominal_capacity_ah:.4g} Ah",
        fontsize=15,
    )

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return points, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected MATR discharge V-Q curves and their SOH values"
    )
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cell-id", default=DEFAULT_CELL_ID)
    parser.add_argument("--cycles", type=int, nargs="+", default=list(DEFAULT_CYCLES))
    parser.add_argument("--q-min", type=float)
    parser.add_argument("--q-max", type=float)
    parser.add_argument("--voltage-min", type=float)
    parser.add_argument("--voltage-max", type=float)
    parser.add_argument("--eol-threshold", type=float, default=0.8)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _optional_limits(
    minimum: float | None,
    maximum: float | None,
    name: str,
) -> tuple[float, float] | None:
    if minimum is None and maximum is None:
        return None
    if minimum is None or maximum is None:
        raise ValueError(f"both --{name}-min and --{name}-max are required")
    return minimum, maximum


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.data.dataset.upper() != "MATR":
        raise ValueError("this plot requires a MATR configuration")
    data_root = resolve_data_root(config, args.data_root)
    cells, _ = load_dataset(data_root, config.data, tolerate_invalid_cells=True)
    cell = select_cell(cells, args.cell_id)
    cycle_label = "-".join(str(value) for value in args.cycles)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config.paths.output_root).resolve()
        / "data_selected_vq_soh"
        / f"{cell.cell_id}_cycles{cycle_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cell.cell_id}_cycles{cycle_label}_vq_soh"
    plot_path = output_dir / f"{stem}.png"
    points, summary = plot_selected_vq_soh(
        cell,
        args.cycles,
        plot_path,
        q_limits=_optional_limits(args.q_min, args.q_max, "q"),
        voltage_limits=_optional_limits(
            args.voltage_min, args.voltage_max, "voltage"
        ),
        eol_threshold=args.eol_threshold,
        dpi=args.dpi,
    )
    points.to_csv(output_dir / f"{stem}_points.csv", index=False)
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    print(f"Cell: {cell.cell_id}")
    print(f"Cycles: {args.cycles}")
    print(f"Plot: {plot_path.resolve()}")
    print(f"SOH summary: {(output_dir / f'{stem}_summary.csv').resolve()}")
    print(f"V-Q points: {(output_dir / f'{stem}_points.csv').resolve()}")


if __name__ == "__main__":
    main()

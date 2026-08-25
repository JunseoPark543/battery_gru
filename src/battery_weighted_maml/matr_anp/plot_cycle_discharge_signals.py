"""Plot voltage, current, and discharged capacity for one MATR discharge cycle."""

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
from .data import CellData, load_dataset
from .plot_cycle_time_fraction_voltage import TimedDischargeCurve, load_timed_discharge


def select_cell(
    cells: Sequence[CellData],
    cycle_number: int,
    cell_id: str | None,
    seed: int,
) -> CellData:
    """Select an explicit or deterministic random cell containing the cycle."""
    eligible = [
        cell
        for cell in cells
        if any(
            cycle.cycle_number == cycle_number and cycle.discharge is not None
            for cycle in cell.cycles
        )
    ]
    if cell_id is not None:
        matches = [cell for cell in eligible if cell.cell_id == cell_id]
        if not matches:
            raise ValueError(f"{cell_id}: valid cycle {cycle_number} is unavailable")
        return matches[0]
    if not eligible:
        raise ValueError(f"no valid MATR cell contains cycle {cycle_number}")
    index = int(np.random.default_rng(seed).integers(0, len(eligible)))
    return eligible[index]


def discharge_signal_frame(curve: TimedDischargeCurve) -> pd.DataFrame:
    """Return the exact acquisition-order samples used by the plot."""
    if curve.current_a is None or curve.discharge_capacity_ah is None:
        raise ValueError("timed discharge curve does not contain current/capacity signals")
    size = len(curve.elapsed_time_s)
    if not all(
        len(values) == size
        for values in (
            curve.q,
            curve.voltage_v,
            curve.current_a,
            curve.discharge_capacity_ah,
        )
    ):
        raise ValueError("discharge signal lengths are inconsistent")
    return pd.DataFrame(
        {
            "elapsed_time_s": curve.elapsed_time_s,
            "elapsed_time_min": curve.elapsed_time_s / 60.0,
            "voltage_in_V": curve.voltage_v,
            "current_in_A": curve.current_a,
            "discharge_capacity_in_Ah": curve.discharge_capacity_ah,
            "q_normalized": curve.q,
        }
    )


def plot_discharge_signals(
    curve: TimedDischargeCurve,
    destination: str | Path,
    *,
    cell_id: str,
    cycle_number: int,
    dpi: int = 180,
) -> pd.DataFrame:
    """Draw three time-aligned discharge signals in one figure."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    frame = discharge_signal_frame(curve)
    time_min = frame["elapsed_time_min"]
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 9.0),
        sharex=True,
        layout="constrained",
    )
    specifications = (
        ("voltage_in_V", "Voltage (V)", "#1f77b4"),
        ("current_in_A", "Current (A)", "#d62728"),
        ("discharge_capacity_in_Ah", "Discharge capacity (Ah)", "#2ca02c"),
    )
    for axis, (column, label, color) in zip(axes, specifications):
        axis.plot(time_min, frame[column], color=color, linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.margins(x=0.0)
    axes[-1].set_xlabel("Elapsed discharge time (min)")
    duration_min = float(time_min.iloc[-1])
    figure.suptitle(
        f"{cell_id} - cycle {cycle_number} discharge signals\n"
        f"duration={duration_min:.2f} min, samples={len(frame):,}, "
        f"final Qd={frame['discharge_capacity_in_Ah'].iloc[-1]:.4f} Ah",
        fontsize=14,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot voltage, current, and discharge capacity versus time for one "
            "MATR discharge cycle"
        )
    )
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cell-id", help="omit for deterministic random selection")
    parser.add_argument("--cycle", type=int, default=130)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.data.dataset.upper() != "MATR":
        raise ValueError("this plot requires a MATR configuration")
    data_root = resolve_data_root(config, args.data_root)
    cells, _ = load_dataset(data_root, config.data, tolerate_invalid_cells=True)
    cell = select_cell(cells, args.cycle, args.cell_id, args.seed)
    curve = load_timed_discharge(cell, args.cycle)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config.paths.output_root).resolve() / "data_cycle_discharge_signals"
    )
    stem = f"{cell.cell_id}_cycle{args.cycle}_discharge_voltage_current_capacity"
    frame = plot_discharge_signals(
        curve,
        output_dir / f"{stem}.png",
        cell_id=cell.cell_id,
        cycle_number=args.cycle,
        dpi=args.dpi,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    print(f"Selected cell: {cell.cell_id}")
    print(f"Samples: {len(frame)}")
    print(f"Plot: {(output_dir / f'{stem}.png').resolve()}")
    print(f"CSV: {(output_dir / f'{stem}.csv').resolve()}")


if __name__ == "__main__":
    main()

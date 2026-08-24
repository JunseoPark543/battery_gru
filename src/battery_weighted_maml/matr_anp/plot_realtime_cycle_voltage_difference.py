"""Replay the online voltage-Q difference of one current cycle from a past cycle."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, resolve_data_root, save_config
from .data import CellData, load_dataset
from .plot_cycle_time_fraction_voltage import (
    TimedDischargeCurve,
    evenly_spaced_time_fractions,
    load_timed_discharge,
    time_fraction_prefix,
)
from .runtime import git_commit, write_json


def select_realtime_difference_cell(
    cells: Sequence[CellData],
    *,
    reference_cycle: int,
    current_cycle: int,
    seed: int,
    cell_id: str | None = None,
) -> CellData:
    """Choose a cell with valid reference/current discharge curves."""
    if reference_cycle >= current_cycle:
        raise ValueError("reference_cycle must precede current_cycle")
    candidates: list[CellData] = []
    for cell in cells:
        available = {
            cycle.cycle_number
            for cycle in cell.cycles
            if cycle.discharge is not None
        }
        if {reference_cycle, current_cycle}.issubset(available):
            candidates.append(cell)
    if cell_id is not None:
        matches = [cell for cell in candidates if cell.cell_id == cell_id]
        if not matches:
            raise ValueError(
                f"{cell_id}: cycles {reference_cycle} and {current_cycle} are unavailable"
            )
        return matches[0]
    if not candidates:
        raise ValueError(
            f"no cell has valid cycles {reference_cycle} and {current_cycle}"
        )
    return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]


def select_realtime_difference_cell_from_end(
    cells: Sequence[CellData],
    *,
    reference_cycle: int,
    rank_from_end: int,
    seed: int,
    cell_id: str | None = None,
) -> tuple[CellData, int]:
    """Choose a cell and its Nth valid discharge cycle counted from the end."""
    if rank_from_end <= 0:
        raise ValueError("rank_from_end must be positive")
    candidates: list[tuple[CellData, int]] = []
    for cell in cells:
        valid_cycles = sorted(
            cycle.cycle_number
            for cycle in cell.cycles
            if cycle.discharge is not None
        )
        if len(valid_cycles) < rank_from_end:
            continue
        current_cycle = valid_cycles[-rank_from_end]
        if current_cycle <= reference_cycle or reference_cycle not in valid_cycles:
            continue
        candidates.append((cell, current_cycle))
    if cell_id is not None:
        matches = [item for item in candidates if item[0].cell_id == cell_id]
        if not matches:
            raise ValueError(
                f"{cell_id}: cannot select the {rank_from_end}th valid cycle "
                f"from the end after reference cycle {reference_cycle}"
            )
        return matches[0]
    if not candidates:
        raise ValueError(
            f"no cell has a valid {rank_from_end}th discharge cycle from the end "
            f"after reference cycle {reference_cycle}"
        )
    return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]


def build_timed_voltage_difference(
    cell: CellData,
    *,
    reference_cycle: int = 10,
    current_cycle: int = 130,
) -> TimedDischargeCurve:
    """Return time-ordered V_current(q)-V_reference(q) without extrapolation."""
    reference_record = cell.cycle_by_number(reference_cycle)
    if reference_record.discharge is None:
        raise ValueError(f"{cell.cell_id}: cycle {reference_cycle} has no discharge curve")
    current = load_timed_discharge(cell, current_cycle)
    reference = reference_record.discharge
    valid = (
        np.isfinite(current.q)
        & np.isfinite(current.voltage_v)
        & (current.q >= reference.q[0])
        & (current.q <= reference.q[-1])
    )
    delta = np.full_like(current.voltage_v, np.nan)
    delta[valid] = current.voltage_v[valid] - np.interp(
        current.q[valid], reference.q, reference.voltage_v
    )
    if np.count_nonzero(np.isfinite(delta)) < 2:
        raise ValueError(
            f"{cell.cell_id}: cycles {reference_cycle}/{current_cycle} have no q overlap"
        )
    return TimedDischargeCurve(current.elapsed_time_s, current.q, delta)


def plot_realtime_voltage_difference(
    curve: TimedDischargeCurve,
    destination: str | Path,
    *,
    cell_id: str,
    reference_cycle: int,
    current_cycle: int,
    fractions: Sequence[float],
    q_limits: tuple[float, float] = (0.0, 1.0),
    delta_voltage_limits: tuple[float, float] = (-0.5, 0.1),
    dpi: int = 180,
) -> pd.DataFrame:
    """Draw fixed-axis snapshots containing only values received by each time."""
    values = sorted(dict.fromkeys(float(value) for value in fractions))
    if not values:
        raise ValueError("at least one time fraction is required")
    if q_limits[0] >= q_limits[1]:
        raise ValueError("q_limits must be increasing")
    if delta_voltage_limits[0] >= delta_voltage_limits[1]:
        raise ValueError("delta_voltage_limits must be increasing")
    prefixes = [time_fraction_prefix(curve, value) for value in values]
    columns = min(3, len(prefixes))
    rows_count = math.ceil(len(prefixes) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(5.0 * columns, 4.2 * rows_count),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    flat_axes = axes.ravel()
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(prefixes)))
    summary_rows: list[dict[str, float | int | str]] = []
    for axis, fraction, prefix, color in zip(flat_axes, values, prefixes, colors):
        valid = (
            np.isfinite(prefix.q)
            & np.isfinite(prefix.voltage_v)
            & (prefix.q >= q_limits[0])
            & (prefix.q <= q_limits[1])
        )
        if np.count_nonzero(valid) < 2:
            raise ValueError(f"t/T={fraction:.3f}: too few comparable q observations")
        q = prefix.q[valid]
        delta = prefix.voltage_v[valid]
        axis.plot(q, delta, color=color, lw=2.2, label="received ΔV")
        axis.scatter(q[-1], delta[-1], color=color, s=34, zorder=3)
        axis.axhline(0.0, color="black", lw=0.9, alpha=0.65)
        axis.axvline(q[-1], color=color, ls="--", lw=0.9, alpha=0.6)
        axis.set_xlim(*q_limits)
        axis.set_ylim(*delta_voltage_limits)
        axis.set_title(
            f"t/T = {fraction:.2f}  ({prefix.elapsed_time_s[-1] / 60.0:.1f} min)\n"
            f"q received = {prefix.q[-1]:.3f}",
            fontsize=11,
        )
        axis.set_xlabel("Normalized discharged capacity q = Qd / Qnom")
        axis.set_ylabel(
            f"ΔV = V(cycle {current_cycle}) − V(cycle {reference_cycle})  [V]"
        )
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=8)
        summary_rows.append(
            {
                "cell_id": cell_id,
                "reference_cycle": reference_cycle,
                "current_cycle": current_cycle,
                "time_fraction": fraction,
                "cutoff_time_s": float(prefix.elapsed_time_s[-1]),
                "total_discharge_time_s": float(curve.elapsed_time_s[-1]),
                "q_received": float(prefix.q[-1]),
                "received_points": len(prefix.q),
                "comparable_points": int(np.count_nonzero(valid)),
                "delta_v_last": float(delta[-1]),
                "delta_v_mean": float(np.mean(delta)),
                "delta_v_min": float(np.min(delta)),
                "delta_v_max": float(np.max(delta)),
            }
        )
    for axis in flat_axes[len(prefixes) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"{cell_id}: online replay of cycle {current_cycle} voltage–Q change "
        f"from cycle {reference_cycle}",
        fontsize=14,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay cycle 130 voltage-Q difference from MATR cycle 10"
    )
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cell-id")
    parser.add_argument("--reference-cycle", type=int, default=10)
    current = parser.add_mutually_exclusive_group()
    current.add_argument("--current-cycle", type=int)
    current.add_argument(
        "--current-cycle-from-end",
        type=int,
        help="1 selects the last valid discharge cycle; 100 selects the 100th from end",
    )
    parser.add_argument("--num-snapshots", type=int, default=5)
    parser.add_argument("--time-fractions", type=float, nargs="+")
    parser.add_argument("--q-min", type=float, default=0.0)
    parser.add_argument("--q-max", type=float, default=1.0)
    parser.add_argument("--delta-v-min", type=float, default=-0.5)
    parser.add_argument("--delta-v-max", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.data.dataset.upper() != "MATR":
        raise ValueError("this online replay requires a MATR configuration")
    data_root = resolve_data_root(config, args.data_root)
    cells, audit = load_dataset(data_root, config.data, tolerate_invalid_cells=True)
    if args.current_cycle_from_end is not None:
        cell, current_cycle = select_realtime_difference_cell_from_end(
            cells,
            reference_cycle=args.reference_cycle,
            rank_from_end=args.current_cycle_from_end,
            seed=args.seed,
            cell_id=args.cell_id,
        )
    else:
        current_cycle = 130 if args.current_cycle is None else args.current_cycle
        cell = select_realtime_difference_cell(
            cells,
            reference_cycle=args.reference_cycle,
            current_cycle=current_cycle,
            seed=args.seed,
            cell_id=args.cell_id,
        )
    curve = build_timed_voltage_difference(
        cell,
        reference_cycle=args.reference_cycle,
        current_cycle=current_cycle,
    )
    fractions = (
        args.time_fractions
        if args.time_fractions is not None
        else evenly_spaced_time_fractions(args.num_snapshots)
    )
    destination = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config.paths.output_root).resolve()
        / "data_realtime_voltage_difference"
        / f"cycle{current_cycle}_from_cycle{args.reference_cycle}_seed{args.seed}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{cell.cell_id}_cycle{current_cycle}_minus_cycle"
        f"{args.reference_cycle}_{len(fractions)}snapshots"
    )
    plot_path = destination / f"{stem}.png"
    summary = plot_realtime_voltage_difference(
        curve,
        plot_path,
        cell_id=cell.cell_id,
        reference_cycle=args.reference_cycle,
        current_cycle=current_cycle,
        fractions=fractions,
        q_limits=(args.q_min, args.q_max),
        delta_voltage_limits=(args.delta_v_min, args.delta_v_max),
        dpi=args.dpi,
    )
    summary.to_csv(destination / f"{stem}.csv", index=False)
    audit.to_csv(destination / "data_audit.csv", index=False)
    save_config(config, destination / "resolved_config.yaml")
    write_json(
        destination / "plot_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "MATR",
            "data_root": str(data_root),
            "git_commit": git_commit(),
            "cell_id": cell.cell_id,
            "reference_cycle": args.reference_cycle,
            "current_cycle": current_cycle,
            "current_cycle_from_end": args.current_cycle_from_end,
            "time_fractions": fractions,
            "q_limits": [args.q_min, args.q_max],
            "delta_voltage_limits": [args.delta_v_min, args.delta_v_max],
            "future_curve_displayed": False,
            "snapshot_policy": "offline equal-duration replay",
            "plot": str(plot_path),
        },
    )
    print(f"Selected cell: {cell.cell_id}")
    print(f"Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()

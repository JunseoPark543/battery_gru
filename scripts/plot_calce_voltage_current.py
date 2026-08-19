"""Plot cycle-level voltage and current statistics for one CALCE cell."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from battery_weighted_maml.data.calce_loader import load_calce_pickle


def _finite_values(record: object, key: str) -> np.ndarray:
    """Convert one signal to a flat array and discard NaN/inf samples."""
    values = np.asarray(record[key], dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def cycle_statistics(cell_path: Path) -> tuple[np.ndarray, ...]:
    """Return cycle number plus mean/min/max voltage and current arrays."""
    cell = load_calce_pickle(cell_path)
    rows: list[tuple[float, ...]] = []
    for index, record in enumerate(cell.cycle_records):
        for key in ("voltage_in_V", "current_in_A"):
            if key not in record:
                raise ValueError(
                    f"{cell.file_name}: cycle_data[{index}] missing required key '{key}'"
                )
        voltage = _finite_values(record, "voltage_in_V")
        current = _finite_values(record, "current_in_A")
        if voltage.size == 0 or current.size == 0:
            continue
        rows.append(
            (
                float(record["cycle_number"]),
                float(voltage.mean()),
                float(voltage.min()),
                float(voltage.max()),
                float(current.mean()),
                float(current.min()),
                float(current.max()),
            )
        )
    if not rows:
        raise ValueError(f"{cell.file_name}: no cycle has finite voltage and current samples")
    return tuple(np.asarray(column) for column in zip(*rows))


def plot_cell(
    cell_path: Path,
    output_path: Path,
    start_cycle: int | None = None,
    end_cycle: int | None = None,
) -> int:
    cycle, v_mean, v_min, v_max, i_mean, i_min, i_max = cycle_statistics(cell_path)
    mask = np.ones(cycle.shape, dtype=bool)
    if start_cycle is not None:
        mask &= cycle >= start_cycle
    if end_cycle is not None:
        mask &= cycle <= end_cycle
    if not np.any(mask):
        raise ValueError("the selected cycle range contains no data")

    cycle = cycle[mask]
    v_mean, v_min, v_max = v_mean[mask], v_min[mask], v_max[mask]
    i_mean, i_min, i_max = i_mean[mask], i_min[mask], i_max[mask]

    fig, voltage_axis = plt.subplots(figsize=(13, 6.5))
    current_axis = voltage_axis.twinx()
    voltage_color, current_color = "tab:blue", "tab:red"

    voltage_axis.fill_between(
        cycle, v_min, v_max, color=voltage_color, alpha=0.10, label="Voltage min-max"
    )
    voltage_axis.plot(cycle, v_mean, color=voltage_color, lw=1.2, label="Mean voltage")
    current_axis.fill_between(
        cycle, i_min, i_max, color=current_color, alpha=0.08, label="Current min-max"
    )
    current_axis.plot(cycle, i_mean, color=current_color, lw=1.0, label="Mean current")
    current_axis.axhline(0.0, color="0.4", lw=0.7, alpha=0.5)

    voltage_axis.set_xlabel("Cycle number")
    voltage_axis.set_ylabel("Voltage (V)", color=voltage_color)
    current_axis.set_ylabel("Current (A)", color=current_color)
    voltage_axis.tick_params(axis="y", colors=voltage_color)
    current_axis.tick_params(axis="y", colors=current_color)
    voltage_axis.grid(True, alpha=0.25)
    voltage_axis.set_title(f"{cell_path.stem}: cycle-level voltage and current")

    handles1, labels1 = voltage_axis.get_legend_handles_labels()
    handles2, labels2 = current_axis.get_legend_handles_labels()
    voltage_axis.legend(handles1 + handles2, labels1 + labels2, loc="best", ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return int(mask.sum())


def plot_all_cycle_profiles(
    cell_path: Path,
    output_dir: Path,
    start_cycle: int | None = None,
    end_cycle: int | None = None,
) -> tuple[Path, Path, int]:
    """Save separate voltage and current plots containing every cycle profile."""
    cell = load_calce_pickle(cell_path)
    profiles: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, record in enumerate(cell.cycle_records):
        cycle = float(record["cycle_number"])
        if start_cycle is not None and cycle < start_cycle:
            continue
        if end_cycle is not None and cycle > end_cycle:
            continue
        missing = {"time_in_s", "voltage_in_V", "current_in_A"} - set(record)
        if missing:
            raise ValueError(
                f"{cell.file_name}: cycle_data[{index}] missing: {', '.join(sorted(missing))}"
            )
        time = np.asarray(record["time_in_s"], dtype=float).reshape(-1)
        voltage = np.asarray(record["voltage_in_V"], dtype=float).reshape(-1)
        current = np.asarray(record["current_in_A"], dtype=float).reshape(-1)
        size = min(time.size, voltage.size, current.size)
        if size == 0:
            continue
        time, voltage, current = time[:size], voltage[:size], current[:size]
        finite = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
        if not np.any(finite):
            continue
        time, voltage, current = time[finite], voltage[finite], current[finite]
        elapsed_hours = (time - time[0]) / 3600.0
        profiles.append((cycle, elapsed_hours, voltage, current))

    if not profiles:
        raise ValueError("the selected cycle range contains no finite profile data")

    output_dir.mkdir(parents=True, exist_ok=True)
    voltage_path = output_dir / f"{cell_path.stem}_all_cycles_voltage.png"
    current_path = output_dir / f"{cell_path.stem}_all_cycles_current.png"
    cycle_values = np.asarray([profile[0] for profile in profiles])
    norm = Normalize(vmin=float(cycle_values.min()), vmax=float(cycle_values.max()))
    # Use one hue with changing intensity so cycle progression does not imply
    # unrelated categorical colors.
    cmap = plt.get_cmap("Blues")

    for signal_index, ylabel, title_signal, output_path in (
        (2, "Voltage (V)", "voltage", voltage_path),
        (3, "Current (A)", "current", current_path),
    ):
        fig, axis = plt.subplots(figsize=(11, 7))
        segments = [
            np.column_stack((profile[1], profile[signal_index])) for profile in profiles
        ]
        collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=0.45,
            alpha=0.35,
            rasterized=True,
        )
        collection.set_array(cycle_values)
        axis.add_collection(collection)
        axis.autoscale()
        axis.margins(x=0.01, y=0.03)
        axis.set_xlabel("Elapsed time within cycle (h)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{cell_path.stem}: {title_signal} profiles for all cycles")
        axis.grid(True, alpha=0.2)
        colorbar = fig.colorbar(collection, ax=axis, pad=0.02)
        colorbar.set_label("Cycle number")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    return voltage_path, current_path, len(profiles)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot voltage and current by cycle for one CALCE pickle"
    )
    parser.add_argument(
        "--cell",
        default="data/CALCE/CALCE_CX2_37.pkl",
        help="path to a CALCE pickle (default: CALCE_CX2_37.pkl)",
    )
    parser.add_argument("--output", help="output PNG path")
    parser.add_argument("--start-cycle", type=int)
    parser.add_argument("--end-cycle", type=int)
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="save two plots containing all raw cycle profiles",
    )
    args = parser.parse_args()

    cell_path = Path(args.cell).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else Path("outputs/data_inspection") / f"{cell_path.stem}_voltage_current.png"
    )
    if args.profiles:
        output_dir = Path(args.output).resolve() if args.output else Path("outputs/data_inspection")
        voltage_path, current_path, count = plot_all_cycle_profiles(
            cell_path, output_dir, args.start_cycle, args.end_cycle
        )
        print(f"Plotted {count} cycles")
        print(f"Voltage: {voltage_path.resolve()}")
        print(f"Current: {current_path.resolve()}")
    else:
        count = plot_cell(cell_path, output_path, args.start_cycle, args.end_cycle)
        print(f"Plotted {count} cycles: {output_path.resolve()}")


if __name__ == "__main__":
    main()

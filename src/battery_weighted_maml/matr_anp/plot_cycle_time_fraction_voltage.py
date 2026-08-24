"""Plot voltage-Q prefixes observed at fractions of one discharge cycle's time."""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, resolve_data_root
from .data import CellData, load_dataset


@dataclass(frozen=True)
class TimedDischargeCurve:
    """Raw discharge samples in acquisition order."""

    elapsed_time_s: np.ndarray
    q: np.ndarray
    voltage_v: np.ndarray


def _numeric(record: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    for key in keys:
        if key in record:
            try:
                return np.asarray(record[key], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                break
    return np.empty(0, dtype=np.float64)


def _records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("cycle_data")
    if isinstance(raw, pd.DataFrame):
        items: Sequence[Any] = raw.to_dict(orient="records")
    elif isinstance(raw, np.ndarray):
        items = raw.tolist()
    elif isinstance(raw, (list, tuple, pd.Series)):
        items = list(raw)
    else:
        raise ValueError("pickle has no supported cycle_data collection")
    records: list[Mapping[str, Any]] = []
    for item in items:
        if isinstance(item, pd.Series):
            item = item.to_dict()
        if isinstance(item, Mapping):
            records.append(item)
    return records


def _cycle_number(record: Mapping[str, Any]) -> int | None:
    for key in ("cycle_number", "cycle_index", "cycle"):
        if key in record:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                return None
    return None


def load_timed_discharge(cell: CellData, cycle_number: int) -> TimedDischargeCurve:
    """Load one cycle and retain only its capacity-increasing discharge segment."""
    source = Path(cell.source_file)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source.name}: pickle root must be a mapping")
    matches = [record for record in _records(payload) if _cycle_number(record) == cycle_number]
    if not matches:
        raise ValueError(f"{cell.cell_id}: cycle {cycle_number} is unavailable")
    record = matches[0]

    time_s = _numeric(record, ("time_in_s", "test_time_in_s", "time_s", "time"))
    voltage = _numeric(record, ("voltage_in_V", "voltage_v", "voltage"))
    current = _numeric(record, ("current_in_A", "current_a", "current"))
    capacity = _numeric(
        record,
        ("discharge_capacity_in_Ah", "discharge_capacity_ah", "discharge_capacity", "QD"),
    )
    size = min(time_s.size, voltage.size, current.size, capacity.size)
    if size < 2:
        raise ValueError(f"{cell.cell_id} cycle {cycle_number}: aligned time/I/V/Q data unavailable")
    time_s, voltage, current, capacity = (
        values[:size] for values in (time_s, voltage, current, capacity)
    )
    finite = (
        np.isfinite(time_s)
        & np.isfinite(voltage)
        & np.isfinite(current)
        & np.isfinite(capacity)
    )
    tolerance = max(1.0e-10, 1.0e-7 * cell.nominal_capacity_ah)
    growth = finite & (np.diff(capacity, prepend=capacity[0]) > tolerance)
    growth_indices = np.flatnonzero(growth)
    if growth_indices.size < 2:
        raise ValueError(f"{cell.cell_id} cycle {cycle_number}: discharge capacity does not increase")
    discharge = np.zeros(size, dtype=bool)
    discharge[max(0, growth_indices[0] - 1) : growth_indices[-1] + 1] = True

    for key in ("step_type", "stage", "state", "operation"):
        if key not in record:
            continue
        labels = np.asarray(record[key]).reshape(-1)
        if labels.size == size:
            stage = np.char.find(np.char.lower(labels.astype(str)), "discharg") >= 0
            if np.count_nonzero(discharge & stage & finite) >= 2:
                discharge &= stage
        break

    signed = current[growth & (np.abs(current) > 1.0e-8)]
    sign = int(np.sign(np.median(signed))) if signed.size else 0
    if sign:
        same_sign = np.sign(current) == sign
        if np.count_nonzero(discharge & same_sign & finite) >= 2:
            discharge &= same_sign

    selected = discharge & finite
    selected_time = time_s[selected]
    selected_q = capacity[selected] / cell.nominal_capacity_ah
    selected_voltage = voltage[selected]
    order = np.argsort(selected_time, kind="stable")
    selected_time = selected_time[order]
    selected_q = selected_q[order]
    selected_voltage = selected_voltage[order]

    unique_time, first_indices = np.unique(selected_time, return_index=True)
    selected_q = selected_q[first_indices]
    selected_voltage = selected_voltage[first_indices]
    if unique_time.size < 2 or unique_time[-1] <= unique_time[0]:
        raise ValueError(f"{cell.cell_id} cycle {cycle_number}: valid elapsed-time range unavailable")
    return TimedDischargeCurve(
        elapsed_time_s=unique_time - unique_time[0],
        q=selected_q,
        voltage_v=selected_voltage,
    )


def time_fraction_prefix(
    curve: TimedDischargeCurve, fraction: float
) -> TimedDischargeCurve:
    """Return a prefix ending exactly at a fraction of total discharge time."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("time fractions must lie in (0,1]")
    cutoff = float(fraction) * float(curve.elapsed_time_s[-1])
    right = int(np.searchsorted(curve.elapsed_time_s, cutoff, side="right"))
    time_s = curve.elapsed_time_s[:right].copy()
    q = curve.q[:right].copy()
    voltage = curve.voltage_v[:right].copy()
    if not np.isclose(time_s[-1], cutoff):
        upper = min(right, len(curve.elapsed_time_s) - 1)
        lower = max(0, upper - 1)
        span = curve.elapsed_time_s[upper] - curve.elapsed_time_s[lower]
        weight = 0.0 if span <= 0 else (cutoff - curve.elapsed_time_s[lower]) / span
        time_s = np.append(time_s, cutoff)
        q = np.append(q, curve.q[lower] + weight * (curve.q[upper] - curve.q[lower]))
        voltage = np.append(
            voltage,
            curve.voltage_v[lower]
            + weight * (curve.voltage_v[upper] - curve.voltage_v[lower]),
        )
    return TimedDischargeCurve(time_s, q, voltage)


def evenly_spaced_time_fractions(count: int) -> list[float]:
    """Return end points of equal, non-overlapping time sections."""
    if count <= 0:
        raise ValueError("snapshot count must be positive")
    return [float(value) for value in np.linspace(1.0 / count, 1.0, count)]


def plot_time_fraction_voltage_q(
    curve: TimedDischargeCurve,
    destination: str | Path,
    *,
    cell_id: str,
    cycle_number: int,
    fractions: Sequence[float],
    q_limits: tuple[float, float] | None = None,
    voltage_limits: tuple[float, float] | None = None,
    show_full_curve: bool = False,
    dpi: int = 180,
) -> pd.DataFrame:
    """Plot successive online prefixes without displaying future observations."""
    values = sorted(dict.fromkeys(float(value) for value in fractions))
    if not values:
        raise ValueError("at least one time fraction is required")
    prefixes = [time_fraction_prefix(curve, value) for value in values]
    columns = min(3, len(values))
    rows_count = math.ceil(len(values) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(5.0 * columns, 4.2 * rows_count),
        sharex=False,
        sharey=False,
        squeeze=False,
        layout="constrained",
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.85, len(values)))
    rows: list[dict[str, float | int | str]] = []
    flat_axes = axes.ravel()
    for axis, fraction, prefix, color in zip(flat_axes, values, prefixes, colors):
        if show_full_curve:
            axis.plot(
                curve.q,
                curve.voltage_v,
                color="0.78",
                lw=1.2,
                label="future/full (reference)",
            )
        axis.plot(prefix.q, prefix.voltage_v, color=color, lw=2.3, label="received so far")
        axis.scatter(prefix.q[-1], prefix.voltage_v[-1], color=color, s=35, zorder=3)
        axis.axvline(prefix.q[-1], color=color, ls="--", lw=0.9, alpha=0.65)
        axis.set_title(
            f"t/T = {fraction:.2f}  ({prefix.elapsed_time_s[-1] / 60.0:.1f} min)\n"
            f"q reached = {prefix.q[-1]:.3f}",
            fontsize=11,
        )
        axis.set_xlabel("Normalized discharged capacity q = Qd / Qnom")
        axis.set_ylabel("Voltage (V)")
        if q_limits is not None:
            if q_limits[0] >= q_limits[1]:
                raise ValueError("q_limits must be increasing")
            axis.set_xlim(*q_limits)
        else:
            axis.set_xlim(left=min(0.0, float(np.min(prefix.q))))
        if voltage_limits is not None:
            if voltage_limits[0] >= voltage_limits[1]:
                raise ValueError("voltage_limits must be increasing")
            axis.set_ylim(*voltage_limits)
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=8)
        rows.append(
            {
                "cell_id": cell_id,
                "cycle": cycle_number,
                "time_fraction": fraction,
                "cutoff_time_s": float(prefix.elapsed_time_s[-1]),
                "total_discharge_time_s": float(curve.elapsed_time_s[-1]),
                "cutoff_q": float(prefix.q[-1]),
                "cutoff_voltage_v": float(prefix.voltage_v[-1]),
                "observed_points": len(prefix.q),
            }
        )
    for axis in flat_axes[len(values) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"{cell_id} · cycle {cycle_number}: online voltage–Q replay",
        fontsize=14,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return pd.DataFrame(rows)


def _select_cell(
    cells: Sequence[CellData], cycle_number: int, cell_id: str | None, seed: int
) -> CellData:
    candidates = [
        cell
        for cell in cells
        if any(
            cycle.cycle_number == cycle_number and cycle.discharge is not None
            for cycle in cell.cycles
        )
    ]
    if cell_id is not None:
        candidates = [cell for cell in candidates if cell.cell_id == cell_id]
        if not candidates:
            raise ValueError(f"{cell_id}: valid cycle {cycle_number} is unavailable")
        return candidates[0]
    if not candidates:
        raise ValueError(f"no valid MATR cell contains cycle {cycle_number}")
    return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot voltage-Q observed at selected fractions of one cycle's time"
    )
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cell-id", help="omit to select one eligible cell reproducibly")
    parser.add_argument("--cycle", type=int, default=130)
    parser.add_argument(
        "--time-fractions",
        type=float,
        nargs="+",
        help="explicit t/T snapshots; omit to divide the duration equally",
    )
    parser.add_argument("--num-snapshots", type=int, default=5)
    parser.add_argument(
        "--q-min",
        type=float,
        default=0.0,
        help="display-axis minimum in nominal-capacity coordinates",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=1.0,
        help="display-axis maximum; independent of the model q-grid",
    )
    parser.add_argument("--voltage-min", type=float, default=2.0)
    parser.add_argument("--voltage-max", type=float, default=3.7)
    parser.add_argument(
        "--show-full-curve",
        action="store_true",
        help="show future/full data in gray for offline comparison",
    )
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
    cell = _select_cell(cells, args.cycle, args.cell_id, args.seed)
    curve = load_timed_discharge(cell, args.cycle)
    fractions = (
        args.time_fractions
        if args.time_fractions is not None
        else evenly_spaced_time_fractions(args.num_snapshots)
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config.paths.output_root).resolve() / "data_cycle_time_fraction"
    )
    stem = f"{cell.cell_id}_cycle{args.cycle}_voltage_q_{len(fractions)}snapshots"
    summary = plot_time_fraction_voltage_q(
        curve,
        output_dir / f"{stem}.png",
        cell_id=cell.cell_id,
        cycle_number=args.cycle,
        fractions=fractions,
        q_limits=(args.q_min, args.q_max),
        voltage_limits=(args.voltage_min, args.voltage_max),
        show_full_curve=args.show_full_curve,
        dpi=args.dpi,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{stem}.csv", index=False)
    print(f"Selected cell: {cell.cell_id}")
    print(f"Plot: {(output_dir / f'{stem}.png').resolve()}")


if __name__ == "__main__":
    main()

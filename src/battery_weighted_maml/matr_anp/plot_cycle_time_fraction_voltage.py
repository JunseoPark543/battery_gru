"""Plot voltage-Q prefixes observed at fractions of one discharge cycle's time."""

from __future__ import annotations

import argparse
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


def plot_time_fraction_voltage_q(
    curve: TimedDischargeCurve,
    destination: str | Path,
    *,
    cell_id: str,
    cycle_number: int,
    fractions: Sequence[float],
    dpi: int = 180,
) -> pd.DataFrame:
    """Plot each observed time prefix in a separate, common-scale panel."""
    values = sorted(dict.fromkeys(float(value) for value in fractions))
    if not values:
        raise ValueError("at least one time fraction is required")
    prefixes = [time_fraction_prefix(curve, value) for value in values]
    figure, axes = plt.subplots(
        1,
        len(values),
        figsize=(5.0 * len(values), 4.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.85, len(values)))
    rows: list[dict[str, float | int | str]] = []
    for axis, fraction, prefix, color in zip(axes[0], values, prefixes, colors):
        axis.plot(curve.q, curve.voltage_v, color="0.78", lw=1.2, label="unobserved/full")
        axis.plot(prefix.q, prefix.voltage_v, color=color, lw=2.3, label="observed")
        axis.scatter(prefix.q[-1], prefix.voltage_v[-1], color=color, s=35, zorder=3)
        axis.axvline(prefix.q[-1], color=color, ls="--", lw=0.9, alpha=0.65)
        axis.set_title(
            f"t/T = {fraction:.2f}\nq reached = {prefix.q[-1]:.3f}", fontsize=11
        )
        axis.set_xlabel("Normalized discharged capacity q = Qd / Qnom")
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
    axes[0, 0].set_ylabel("Voltage (V)")
    figure.suptitle(
        f"{cell_id} · cycle {cycle_number}: voltage–Q at partial discharge times",
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
    parser.add_argument("--time-fractions", type=float, nargs="+", default=[0.3, 0.5, 0.7])
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
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config.paths.output_root).resolve() / "data_cycle_time_fraction"
    )
    stem = f"{cell.cell_id}_cycle{args.cycle}_voltage_q_time_fraction"
    summary = plot_time_fraction_voltage_q(
        curve,
        output_dir / f"{stem}.png",
        cell_id=cell.cell_id,
        cycle_number=args.cycle,
        fractions=args.time_fractions,
        dpi=args.dpi,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{stem}.csv", index=False)
    print(f"Selected cell: {cell.cell_id}")
    print(f"Plot: {(output_dir / f'{stem}.png').resolve()}")


if __name__ == "__main__":
    main()

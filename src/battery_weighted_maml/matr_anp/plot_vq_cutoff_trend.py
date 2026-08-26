"""Analyse how discharge capacity at a fixed voltage cutoff changes by cycle."""

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
from .data import CellData, DischargeCurve, load_dataset
from .plot_selected_vq_soh import DEFAULT_CELL_ID, DEFAULT_CYCLES, select_cell


def q_at_voltage_cutoff(
    curve: DischargeCurve,
    cutoff_voltage: float,
) -> float:
    """Interpolate the first descending crossing of one fixed voltage cutoff."""
    q = np.asarray(curve.q, dtype=np.float64)
    voltage = np.asarray(curve.voltage_v, dtype=np.float64)
    finite = np.isfinite(q) & np.isfinite(voltage)
    q, voltage = q[finite], voltage[finite]
    if len(q) < 2:
        return float("nan")
    crossing = np.flatnonzero(
        (voltage[:-1] >= cutoff_voltage) & (voltage[1:] <= cutoff_voltage)
    )
    if crossing.size == 0:
        return float("nan")
    index = int(crossing[0])
    v0, v1 = voltage[index], voltage[index + 1]
    q0, q1 = q[index], q[index + 1]
    if np.isclose(v0, v1):
        return float(q1)
    fraction = (cutoff_voltage - v0) / (v1 - v0)
    return float(q0 + fraction * (q1 - q0))


def cutoff_trend_frame(
    cell: CellData,
    cutoff_voltage: float,
    *,
    rolling_window: int = 21,
) -> pd.DataFrame:
    """Return all valid cycle endpoints and fixed-cutoff crossing capacities."""
    if not np.isfinite(cutoff_voltage):
        raise ValueError("cutoff voltage must be finite")
    if rolling_window <= 0:
        raise ValueError("rolling window must be positive")
    rows: list[dict[str, float | int | str | bool]] = []
    for cycle in cell.cycles:
        curve = cycle.discharge
        if curve is None:
            continue
        q_cutoff = q_at_voltage_cutoff(curve, cutoff_voltage)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "cycle": cycle.cycle_number,
                "cutoff_voltage_v": cutoff_voltage,
                "q_at_cutoff": q_cutoff,
                "reached_cutoff": np.isfinite(q_cutoff),
                "vq_q_endpoint": float(curve.q[-1]),
                "vq_endpoint_voltage_v": float(curve.voltage_v[-1]),
                "capacity_soh": cycle.soh,
                "q_endpoint_minus_soh": float(curve.q[-1] - cycle.soh),
            }
        )
    frame = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{cell.cell_id}: no valid V-Q curves")
    frame["q_at_cutoff_rolling_median"] = frame["q_at_cutoff"].rolling(
        window=rolling_window,
        center=True,
        min_periods=1,
    ).median()
    return frame


def trend_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize the all-cycle cutoff trend without assuming strict monotonicity."""
    valid = frame.dropna(subset=["q_at_cutoff"])
    if len(valid) < 2:
        return {
            "valid_cutoff_cycles": len(valid),
            "missing_cutoff_cycles": len(frame) - len(valid),
            "linear_slope_q_per_cycle": float("nan"),
            "spearman_cycle_vs_q": float("nan"),
        }
    slope = float(np.polyfit(valid["cycle"], valid["q_at_cutoff"], 1)[0])
    spearman = float(
        valid["cycle"].rank(method="average").corr(
            valid["q_at_cutoff"].rank(method="average")
        )
    )
    return {
        "valid_cutoff_cycles": len(valid),
        "missing_cutoff_cycles": len(frame) - len(valid),
        "linear_slope_q_per_cycle": slope,
        "spearman_cycle_vs_q": spearman,
    }


def plot_vq_cutoff_trend(
    cell: CellData,
    destination: str | Path,
    *,
    cutoff_voltage: float,
    selected_cycles: Sequence[int] = DEFAULT_CYCLES,
    rolling_window: int = 21,
    dpi: int = 200,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Plot selected crossings, all-cycle q trend, and endpoint voltage audit."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    frame = cutoff_trend_frame(
        cell,
        cutoff_voltage,
        rolling_window=rolling_window,
    )
    statistics = trend_statistics(frame)
    by_cycle = {cycle.cycle_number: cycle for cycle in cell.cycles}
    requested = list(dict.fromkeys(int(value) for value in selected_cycles))
    missing = [
        value
        for value in requested
        if value not in by_cycle or by_cycle[value].discharge is None
    ]
    if missing:
        raise ValueError(f"{cell.cell_id}: selected V-Q cycles unavailable: {missing}")

    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(requested)))
    figure, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), layout="constrained")
    vq_axis, trend_axis, voltage_axis = axes

    for color, cycle_number in zip(colors, requested):
        cycle = by_cycle[cycle_number]
        curve = cycle.discharge
        assert curve is not None
        q_cutoff = q_at_voltage_cutoff(curve, cutoff_voltage)
        vq_axis.plot(
            curve.q,
            curve.voltage_v,
            color=color,
            lw=1.8,
            label=f"C{cycle_number}: qcut={q_cutoff:.4f}",
        )
        if np.isfinite(q_cutoff):
            vq_axis.scatter(q_cutoff, cutoff_voltage, color=color, s=45, zorder=3)
    vq_axis.axhline(
        cutoff_voltage,
        color="black",
        ls="--",
        lw=1.0,
        label=f"fixed cutoff = {cutoff_voltage:g} V",
    )
    vq_axis.set(
        xlabel="Normalized discharged capacity q = Qd / Qnom",
        ylabel="Discharge voltage (V)",
        title="Selected V-Q cutoff crossings",
    )
    vq_axis.grid(alpha=0.23)
    vq_axis.legend(fontsize=8)

    valid = frame[frame["reached_cutoff"]]
    trend_axis.scatter(
        valid["cycle"],
        valid["q_at_cutoff"],
        color="#1f77b4",
        s=8,
        alpha=0.38,
        label="q at fixed voltage cutoff",
    )
    trend_axis.plot(
        frame["cycle"],
        frame["q_at_cutoff_rolling_median"],
        color="#d62728",
        lw=2.0,
        label=f"rolling median ({rolling_window} cycles)",
    )
    trend_axis.plot(
        frame["cycle"],
        frame["capacity_soh"],
        color="0.25",
        lw=1.0,
        alpha=0.7,
        label="capacity-based SOH",
    )
    trend_axis.set(
        xlabel="Cycle number",
        ylabel="Normalized capacity q",
        title=(
            "Cutoff-capacity trend\n"
            f"slope={statistics['linear_slope_q_per_cycle']:.3g}/cycle, "
            f"Spearman={statistics['spearman_cycle_vs_q']:.3f}"
        ),
    )
    trend_axis.grid(alpha=0.23)
    trend_axis.legend(fontsize=8)

    voltage_axis.scatter(
        frame["cycle"],
        frame["vq_endpoint_voltage_v"],
        color="#9467bd",
        s=8,
        alpha=0.45,
    )
    voltage_axis.axhline(cutoff_voltage, color="black", ls="--", lw=1.0)
    voltage_axis.set(
        xlabel="Cycle number",
        ylabel="Observed endpoint voltage (V)",
        title=(
            "Endpoint-voltage audit\n"
            f"missing fixed-cutoff crossings={statistics['missing_cutoff_cycles']}"
        ),
    )
    voltage_axis.grid(alpha=0.23)

    figure.suptitle(
        f"{cell.cell_id}: capacity reached at a common discharge-voltage cutoff",
        fontsize=15,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return frame, statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all-cycle q trend at one fixed MATR discharge cutoff"
    )
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cell-id", default=DEFAULT_CELL_ID)
    parser.add_argument("--cutoff-voltage", type=float, default=2.0)
    parser.add_argument("--selected-cycles", type=int, nargs="+", default=list(DEFAULT_CYCLES))
    parser.add_argument("--rolling-window", type=int, default=21)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.data.dataset.upper() != "MATR":
        raise ValueError("this analysis requires a MATR configuration")
    data_root = resolve_data_root(config, args.data_root)
    cells, _ = load_dataset(data_root, config.data, tolerate_invalid_cells=True)
    cell = select_cell(cells, args.cell_id)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config.paths.output_root).resolve()
        / "data_vq_cutoff_trend"
        / cell.cell_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff_label = str(args.cutoff_voltage).replace(".", "p")
    stem = f"{cell.cell_id}_vq_cutoff_{cutoff_label}V_trend"
    frame, statistics = plot_vq_cutoff_trend(
        cell,
        output_dir / f"{stem}.png",
        cutoff_voltage=args.cutoff_voltage,
        selected_cycles=args.selected_cycles,
        rolling_window=args.rolling_window,
        dpi=args.dpi,
    )
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    pd.DataFrame([statistics]).to_csv(
        output_dir / f"{stem}_statistics.csv", index=False
    )
    print(f"Cell: {cell.cell_id}")
    print(f"Fixed cutoff voltage: {args.cutoff_voltage:g} V")
    print(f"Valid cutoff cycles: {statistics['valid_cutoff_cycles']}")
    print(f"Missing cutoff cycles: {statistics['missing_cutoff_cycles']}")
    print(f"Linear slope: {statistics['linear_slope_q_per_cycle']:.8g} q/cycle")
    print(f"Spearman correlation: {statistics['spearman_cycle_vs_q']:.6g}")
    print(f"Plot: {(output_dir / f'{stem}.png').resolve()}")
    print(f"CSV: {(output_dir / f'{stem}.csv').resolve()}")


if __name__ == "__main__":
    main()

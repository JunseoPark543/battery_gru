"""SOH extraction, cycle interpolation, CSV export, and diagnostic plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calce_loader import RawCell, load_calce_pickle, load_eol_labels
from .task_views import FullCellTrajectory, infer_family


def _last_finite(value: Any) -> float:
    if isinstance(value, pd.DataFrame):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, (pd.Series, pd.Index)):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, np.ndarray):
        raw = value.reshape(-1)
    elif isinstance(value, (list, tuple)):
        raw = np.asarray(value, dtype=object).reshape(-1)
    else:
        raw = np.asarray([value], dtype=object)
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return float(finite[-1]) if finite.size else float("nan")


def _finite_mean(value: Any) -> float:
    if isinstance(value, pd.DataFrame):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, (pd.Series, pd.Index)):
        raw = value.to_numpy().reshape(-1)
    elif isinstance(value, np.ndarray):
        raw = value.reshape(-1)
    elif isinstance(value, (list, tuple)):
        raw = np.asarray(value, dtype=object).reshape(-1)
    else:
        raw = np.asarray([value], dtype=object)
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return float(finite.mean()) if finite.size else float("nan")


def preprocess_cell(
    raw: RawCell,
    true_eol_cycle: int | None = None,
    logger: logging.Logger | None = None,
) -> FullCellTrajectory:
    """Convert one raw cell to a sorted, gap-filled SOH trajectory."""
    log = logger or logging.getLogger("battery_weighted_maml")
    by_cycle: dict[int, float] = {}
    voltage_by_cycle: dict[int, float] = {}
    duplicates: set[int] = set()
    for index, record in enumerate(raw.cycle_records):
        try:
            cycle_float = float(record["cycle_number"])
            cycle = int(cycle_float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{raw.file_name}: invalid key 'cycle_number' in cycle_data[{index}]"
            ) from exc
        if not np.isfinite(cycle_float) or cycle_float != cycle or cycle <= 0:
            raise ValueError(
                f"{raw.file_name}: cycle_data[{index}] cycle_number must be a positive integer"
            )
        capacity = _last_finite(record["discharge_capacity_in_Ah"])
        voltage = _finite_mean(record.get("voltage_in_V", []))
        if cycle in by_cycle:
            duplicates.add(cycle)
        # The last valid duplicate wins; an invalid duplicate does not erase a valid value.
        if np.isfinite(capacity) or cycle not in by_cycle:
            by_cycle[cycle] = capacity
        if np.isfinite(voltage) or cycle not in voltage_by_cycle:
            voltage_by_cycle[cycle] = voltage
    if duplicates:
        log.warning(
            "%s: duplicate cycle_number values %s; retained each last valid record",
            raw.file_name,
            sorted(duplicates),
        )
    if not by_cycle:
        raise ValueError(f"{raw.file_name}: no cycle records could be parsed")
    first_cycle, last_cycle = min(by_cycle), max(by_cycle)
    grid = np.arange(first_cycle, last_cycle + 1, dtype=np.int64)
    original = pd.Series(by_cycle, dtype=float).reindex(grid)
    missing_before = int(original.isna().sum())
    interpolated = original.interpolate(method="linear", limit_direction="both")
    missing_after = int(interpolated.isna().sum())
    if missing_after:
        raise ValueError(
            f"{raw.file_name}: {missing_after} capacity values remain missing after interpolation"
        )
    is_interpolated = original.isna().to_numpy(dtype=bool)
    capacities = interpolated.to_numpy(dtype=np.float64)
    raw_voltage = pd.Series(voltage_by_cycle, dtype=float).reindex(grid)
    if raw_voltage.notna().any():
        mean_voltage_v = raw_voltage.interpolate(
            method="linear", limit_direction="both"
        ).to_numpy(dtype=np.float64)
    else:
        mean_voltage_v = np.full(len(grid), np.nan, dtype=np.float64)
    soh = capacities / raw.nominal_capacity_ah
    if not np.all(np.isfinite(soh)):
        raise ValueError(f"{raw.file_name}: SOH contains non-finite values after preprocessing")
    log.info(
        "%s: raw=%d processed=%d missing before/after=%d/%d",
        raw.file_name, len(raw.cycle_records), len(grid), missing_before, missing_after,
    )
    return FullCellTrajectory(
        file_name=raw.file_name,
        cell_id=raw.cell_id,
        family=infer_family(raw.file_name),
        nominal_capacity_ah=raw.nominal_capacity_ah,
        cycles=grid,
        capacities_ah=capacities,
        soh=soh,
        is_interpolated=is_interpolated,
        true_eol_cycle=true_eol_cycle,
        raw_cycle_count=len(raw.cycle_records),
        missing_count_before=missing_before,
        missing_count_after=missing_after,
        mean_voltage_v=mean_voltage_v,
    )


def trajectory_frame(cell: FullCellTrajectory) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_name": cell.file_name,
            "cell_id": cell.cell_id,
            "family": cell.family,
            "cycle": cell.cycles,
            "discharge_capacity_Ah": cell.capacities_ah,
            "soh": cell.soh,
            "mean_voltage_V": cell.mean_voltage_v,
            "is_interpolated": cell.is_interpolated,
            "true_eol_cycle": cell.true_eol_cycle,
        }
    )


def summary_record(cell: FullCellTrajectory) -> dict[str, Any]:
    at_eol = float("nan")
    if cell.true_eol_cycle is not None:
        match = np.flatnonzero(cell.cycles == cell.true_eol_cycle)
        if match.size:
            at_eol = float(cell.soh[match[0]])
    return {
        "file_name": cell.file_name,
        "cell_id": cell.cell_id,
        "family": cell.family,
        "nominal_capacity_Ah": cell.nominal_capacity_ah,
        "raw_cycle_count": cell.raw_cycle_count,
        "processed_cycle_count": len(cell.cycles),
        "missing_count_before": cell.missing_count_before,
        "missing_count_after": cell.missing_count_after,
        "first_cycle": int(cell.cycles[0]),
        "last_cycle": int(cell.cycles[-1]),
        "true_eol_cycle": cell.true_eol_cycle,
        "soh_at_true_eol": at_eol,
    }


def _plot_cell(cell: FullCellTrajectory, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(cell.cycles, cell.soh, linewidth=1.4, label="SOH")
    if np.any(cell.is_interpolated):
        axis.scatter(
            cell.cycles[cell.is_interpolated], cell.soh[cell.is_interpolated],
            s=12, color="tab:orange", label="interpolated",
        )
    axis.axhline(0.8, color="tab:red", linestyle="--", linewidth=1, label="EOL threshold")
    axis.set(xlabel="Cycle", ylabel="SOH", title=cell.file_name)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def preprocess_dataset(
    calce_dir: str | Path,
    label_path: str | Path,
    output_dir: str | Path,
    logger: logging.Logger | None = None,
) -> dict[str, FullCellTrajectory]:
    """Preprocess every PKL, write aggregate CSVs/figures, and return trajectories."""
    log = logger or logging.getLogger("battery_weighted_maml")
    data_dir = Path(calce_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"CALCE data directory not found: {data_dir}")
    labels = load_eol_labels(label_path)
    files = sorted(data_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no .pkl files found in CALCE data directory: {data_dir}")
    missing_labels = [path.name for path in files if path.name not in labels]
    if missing_labels:
        raise ValueError(f"missing EOL label(s) for: {', '.join(missing_labels)}")
    trajectories: dict[str, FullCellTrajectory] = {}
    for path in files:
        trajectories[path.name] = preprocess_cell(
            load_calce_pickle(path), labels[path.name], logger=log
        )
    destination = Path(output_dir)
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pd.concat([trajectory_frame(cell) for cell in trajectories.values()], ignore_index=True).to_csv(
        destination / "all_cells_soh.csv", index=False
    )
    pd.DataFrame([summary_record(cell) for cell in trajectories.values()]).to_csv(
        destination / "cell_summary.csv", index=False
    )
    for cell in trajectories.values():
        _plot_cell(cell, figures / f"{Path(cell.file_name).stem}_soh.png")
    return trajectories

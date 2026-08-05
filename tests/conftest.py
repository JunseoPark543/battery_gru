from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from battery_weighted_maml.data.calce_loader import load_calce_pickle
from battery_weighted_maml.data.preprocess import preprocess_cell
from battery_weighted_maml.data.task_views import FullCellTrajectory, infer_family


def write_pickle(
    path: Path,
    capacities: list[tuple[int, object]],
    nominal: float = 2.0,
    cell_id: str | None = None,
    voltages: list[object] | None = None,
    currents: list[object] | None = None,
) -> Path:
    if voltages is not None and len(voltages) != len(capacities):
        raise ValueError("voltages and capacities must have equal lengths")
    if currents is not None and len(currents) != len(capacities):
        raise ValueError("currents and capacities must have equal lengths")
    payload = {
        "cell_id": cell_id or path.stem,
        "nominal_capacity_in_Ah": nominal,
        "cycle_data": [
            {
                "cycle_number": cycle,
                "discharge_capacity_in_Ah": value,
                "current_in_A": [] if currents is None else currents[index],
                "voltage_in_V": [] if voltages is None else voltages[index],
                "charge_capacity_in_Ah": [],
                "time_in_s": [],
            }
            for index, (cycle, value) in enumerate(capacities)
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def make_trajectory(
    name: str,
    soh: list[float],
    true_eol: int = 8,
    mean_voltage_v: list[float] | None = None,
    mean_current_a: list[float] | None = None,
) -> FullCellTrajectory:
    values = np.asarray(soh, dtype=float)
    cycles = np.arange(1, len(values) + 1)
    return FullCellTrajectory(
        file_name=name,
        cell_id=Path(name).stem,
        family=infer_family(name),
        nominal_capacity_ah=2.0,
        cycles=cycles,
        capacities_ah=values * 2.0,
        soh=values,
        is_interpolated=np.zeros(len(values), dtype=bool),
        true_eol_cycle=true_eol,
        raw_cycle_count=len(values),
        missing_count_before=0,
        missing_count_after=0,
        mean_voltage_v=(
            None if mean_voltage_v is None else np.asarray(mean_voltage_v, dtype=float)
        ),
        mean_current_a=(
            None if mean_current_a is None else np.asarray(mean_current_a, dtype=float)
        ),
    )


@pytest.fixture
def parsed_cell(tmp_path: Path):
    path = write_pickle(
        tmp_path / "CALCE_CX2_33.pkl",
        [(1, [0.1, 2.1]), (2, np.array([2.0])), (3, [1.9])],
    )
    return preprocess_cell(load_calce_pickle(path), true_eol_cycle=3)

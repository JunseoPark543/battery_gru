from __future__ import annotations

import numpy as np
import pytest

from battery_weighted_maml.matr_anp.data import CellData, CycleData
from battery_weighted_maml.matr_anp.plot_matr_batch_soh_trajectories import (
    batch_from_cell_id,
    batch_summary,
    plot_batch_trajectories,
    trajectory_frame,
)


def _cell(cell_id: str, length: int) -> CellData:
    cycles = tuple(
        CycleData(
            cycle_number=index,
            discharge_capacity_ah=float(1.1 - index / (length * 10)),
            soh=float(1.0 - index / (length * 10)),
            discharge=None,
            raw_signal_length=0,
        )
        for index in range(1, length + 1)
    )
    return CellData(cell_id, f"{cell_id}.pkl", 1.1, cycles)


def test_matr_file_batch_trajectory_plot_has_five_panels(tmp_path) -> None:
    cells = [
        _cell("MATR_b1c0", 30),
        _cell("MATR_b1c1", 34),
        _cell("MATR_b2c0", 36),
        _cell("MATR_b3c0", 40),
        _cell("MATR_b4c0", 44),
    ]
    trajectories = trajectory_frame(cells)
    summary = batch_summary(trajectories)
    output = plot_batch_trajectories(
        trajectories,
        summary,
        tmp_path / "five_panels.png",
        eol_threshold=0.8,
        y_min=0.7,
        y_max=1.05,
        dpi=72,
    )

    assert set(trajectories["batch"]) == {"b1", "b2", "b3", "b4"}
    assert summary.set_index("batch").loc["b1", "num_cells"] == 2
    assert np.isclose(summary.set_index("batch").loc["b2", "median_last_cycle"], 36)
    assert output.is_file()


def test_matr_batch_parser_rejects_ambiguous_ids() -> None:
    assert batch_from_cell_id("MATR_b4c24") == "b4"
    with pytest.raises(ValueError, match="expected MATR"):
        batch_from_cell_id("HUST_4-2")

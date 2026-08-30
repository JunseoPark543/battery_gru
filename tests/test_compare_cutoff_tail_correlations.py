from __future__ import annotations

import numpy as np
import pandas as pd

from battery_weighted_maml.matr_anp.compare_cutoff_tail_correlations import (
    cutoff_tail_cell_summary,
    cutoff_tail_cycle_frame,
    cutoff_tail_dataset_summary,
    normalized_tail_quantiles,
    plot_cutoff_tail_comparison,
)
from battery_weighted_maml.matr_anp.data import CellData, CycleData, DischargeCurve


def _curve(tail_length: float) -> DischargeCurve:
    endpoint = 1.0
    tail_start = endpoint - tail_length
    q_before = np.linspace(0.0, tail_start, 40)
    voltage_before = np.linspace(3.5, 2.01, 40)
    q_tail = np.linspace(tail_start + 1.0e-4, endpoint, 12)
    voltage_tail = np.full_like(q_tail, 2.0)
    q = np.concatenate([q_before, q_tail])
    voltage = np.concatenate([voltage_before, voltage_tail])
    return DischargeCurve(q, voltage, np.ones_like(q), -1, True, 0)


def _cell(cell_id: str, increasing: bool = True) -> CellData:
    values = np.linspace(0.01, 0.08, 30)
    if not increasing:
        values = values[::-1]
    cycles = tuple(
        CycleData(index + 1, 1.0, 1.0, _curve(float(tail)), 52)
        for index, tail in enumerate(values)
    )
    return CellData(cell_id, f"{cell_id}.pkl", 1.0, cycles)


def test_per_cell_cutoff_tail_correlation_and_plot(tmp_path) -> None:
    per_cycle = pd.concat(
        [
            cutoff_tail_cycle_frame(
                [_cell("MATR_A")], "MATR",
                cutoff_voltage=2.0, endpoint_tolerance_v=0.01,
            ),
            cutoff_tail_cycle_frame(
                [_cell("HUST_A", increasing=False)], "HUST",
                cutoff_voltage=2.0, endpoint_tolerance_v=0.01,
            ),
        ],
        ignore_index=True,
    )
    assert per_cycle["valid_cutoff_tail"].all()
    per_cell = cutoff_tail_cell_summary(
        per_cycle,
        minimum_valid_cycles=20,
        edge_fraction=0.1,
        minimum_edge_cycles=3,
    )
    matr = per_cell[per_cell["dataset"] == "MATR"].iloc[0]
    hust = per_cell[per_cell["dataset"] == "HUST"].iloc[0]
    assert matr["spearman_cycle_vs_tail"] > 0.99
    assert bool(matr["strong_positive_trend"])
    assert hust["spearman_cycle_vs_tail"] < -0.99
    assert not bool(hust["positive_trend"])

    summary = cutoff_tail_dataset_summary(per_cell)
    assert summary.loc[summary["dataset"] == "MATR", "positive_trend_cells"].item() == 1
    assert summary.loc[summary["dataset"] == "HUST", "positive_trend_cells"].item() == 0
    normalized = normalized_tail_quantiles(per_cycle, per_cell, grid_points=21)
    destination = tmp_path / "comparison.png"
    assert plot_cutoff_tail_comparison(
        per_cell, summary, normalized, destination, dpi=72
    ).is_file()


def test_endpoint_outside_cutoff_band_is_excluded() -> None:
    curve = _curve(0.05)
    shifted = DischargeCurve(
        curve.q,
        curve.voltage_v - np.linspace(0.0, 0.2, len(curve.q)),
        curve.current_a_magnitude,
        curve.original_current_sign,
        curve.monotonic_before_cleanup,
        curve.duplicate_q_count,
    )
    cycle = CycleData(1, 1.0, 1.0, shifted, len(shifted.q))
    cell = CellData("MATR_BAD", "bad.pkl", 1.0, (cycle,))
    frame = cutoff_tail_cycle_frame(
        [cell], "MATR", cutoff_voltage=2.0, endpoint_tolerance_v=0.01
    )
    assert not bool(frame.loc[0, "valid_cutoff_tail"])
    assert np.isnan(frame.loc[0, "cutoff_tail_q_length"])

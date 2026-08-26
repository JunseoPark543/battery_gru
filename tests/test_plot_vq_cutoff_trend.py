from __future__ import annotations

import numpy as np

from battery_weighted_maml.matr_anp.data import CellData, CycleData, DischargeCurve
from battery_weighted_maml.matr_anp.plot_vq_cutoff_trend import (
    cutoff_trend_frame,
    plot_vq_cutoff_trend,
    q_at_voltage_cutoff,
    trend_statistics,
)


def _curve(q_end: float) -> DischargeCurve:
    q = np.linspace(0.0, q_end, 50)
    voltage = 3.5 - 1.5 * q / q_end
    return DischargeCurve(q, voltage, np.ones_like(q), -1, True, 0)


def _cell() -> CellData:
    cycles = tuple(
        CycleData(number, q_end, q_end, _curve(q_end), 50)
        for number, q_end in ((50, 1.0), (130, 0.95), (200, 0.9))
    )
    return CellData("MATR_TEST", "synthetic.pkl", 1.0, cycles)


def test_cutoff_interpolation_and_decreasing_trend(tmp_path) -> None:
    assert np.isclose(q_at_voltage_cutoff(_curve(1.0), 2.0), 1.0)
    frame = cutoff_trend_frame(_cell(), 2.0, rolling_window=3)
    assert np.allclose(frame["q_at_cutoff"], [1.0, 0.95, 0.9])
    statistics = trend_statistics(frame)
    assert statistics["linear_slope_q_per_cycle"] < 0
    assert np.isclose(statistics["spearman_cycle_vs_q"], -1.0)

    destination = tmp_path / "cutoff_trend.png"
    plotted, _ = plot_vq_cutoff_trend(
        _cell(),
        destination,
        cutoff_voltage=2.0,
        selected_cycles=[50, 130, 200],
        rolling_window=3,
        dpi=72,
    )
    assert destination.is_file()
    assert len(plotted) == 3

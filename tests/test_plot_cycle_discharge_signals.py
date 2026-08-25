from __future__ import annotations

import numpy as np

from battery_weighted_maml.matr_anp.plot_cycle_discharge_signals import (
    discharge_signal_frame,
    plot_discharge_signals,
    plot_multiple_cycle_discharge_signals,
)
from battery_weighted_maml.matr_anp.plot_cycle_time_fraction_voltage import (
    TimedDischargeCurve,
    time_fraction_prefix,
)


def _curve() -> TimedDischargeCurve:
    time_s = np.linspace(0.0, 1200.0, 50)
    capacity = np.linspace(0.0, 1.0, 50)
    return TimedDischargeCurve(
        elapsed_time_s=time_s,
        q=capacity / 1.1,
        voltage_v=3.6 - 0.7 * capacity,
        current_a=-np.ones(50),
        discharge_capacity_ah=capacity,
    )


def test_signal_frame_and_plot(tmp_path) -> None:
    curve = _curve()
    frame = plot_discharge_signals(
        curve,
        tmp_path / "signals.png",
        cell_id="MATR_TEST",
        cycle_number=130,
        dpi=72,
    )
    assert (tmp_path / "signals.png").is_file()
    assert list(frame.columns) == [
        "elapsed_time_s",
        "elapsed_time_min",
        "voltage_in_V",
        "current_in_A",
        "discharge_capacity_in_Ah",
        "q_normalized",
    ]
    assert frame["elapsed_time_min"].iloc[-1] == 20.0


def test_time_prefix_preserves_current_and_capacity() -> None:
    prefix = time_fraction_prefix(_curve(), 0.5)
    frame = discharge_signal_frame(prefix)
    assert prefix.current_a is not None
    assert prefix.discharge_capacity_ah is not None
    assert len(frame) == len(prefix.elapsed_time_s)
    assert np.isclose(frame["elapsed_time_s"].iloc[-1], 600.0)


def test_multiple_cycles_are_combined_by_row(tmp_path) -> None:
    destination = tmp_path / "multiple.png"
    frame = plot_multiple_cycle_discharge_signals(
        [(50, _curve()), (130, _curve()), (470, _curve())],
        destination,
        cell_id="MATR_TEST",
        dpi=72,
    )
    assert destination.is_file()
    assert frame["cycle_number"].drop_duplicates().tolist() == [50, 130, 470]
    assert frame.groupby("cycle_number").size().to_dict() == {50: 50, 130: 50, 470: 50}

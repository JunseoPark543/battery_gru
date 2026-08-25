from __future__ import annotations

import numpy as np

from battery_weighted_maml.matr_anp.data import CellData, CycleData, DischargeCurve
from battery_weighted_maml.matr_anp.plot_selected_vq_soh import (
    plot_selected_vq_soh,
    select_cycles,
)


def _cell() -> CellData:
    nominal = 1.1
    cycles = []
    for cycle_number, soh in ((50, 0.98), (130, 0.94), (200, 0.91)):
        q = np.linspace(0.0, soh, 30)
        curve = DischargeCurve(
            q=q,
            voltage_v=3.6 - 0.8 * q,
            current_a_magnitude=np.ones_like(q),
            original_current_sign=-1,
            monotonic_before_cleanup=True,
            duplicate_q_count=0,
        )
        cycles.append(
            CycleData(
                cycle_number=cycle_number,
                discharge_capacity_ah=soh * nominal,
                soh=soh,
                discharge=curve,
                raw_signal_length=len(q),
            )
        )
    return CellData("MATR_TEST", "synthetic.pkl", nominal, tuple(cycles))


def test_selected_vq_soh_plot_and_tables(tmp_path) -> None:
    destination = tmp_path / "vq_soh.png"
    points, summary = plot_selected_vq_soh(
        _cell(),
        [50, 130, 200],
        destination,
        dpi=72,
    )
    assert destination.is_file()
    assert summary["cycle"].tolist() == [50, 130, 200]
    assert np.allclose(summary["vq_q_end"], summary["soh"])
    assert points.groupby("cycle").size().to_dict() == {50: 30, 130: 30, 200: 30}


def test_missing_cycle_is_reported() -> None:
    try:
        select_cycles(_cell(), [50, 500])
    except ValueError as exc:
        assert "500" in str(exc)
    else:
        raise AssertionError("missing cycle should fail")

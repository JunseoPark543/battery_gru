from __future__ import annotations

import numpy as np

from battery_weighted_maml.data.calce_loader import load_calce_pickle
from battery_weighted_maml.data.preprocess import preprocess_cell
from conftest import write_pickle


def test_soh_uses_last_finite_capacity_without_clipping(tmp_path):
    path = write_pickle(
        tmp_path / "CALCE_CX2_33.pkl",
        [(1, [np.nan, 2.2]), (2, [2.0])], nominal=2.0,
    )
    cell = preprocess_cell(load_calce_pickle(path), true_eol_cycle=2)
    np.testing.assert_allclose(cell.capacities_ah, [2.2, 2.0])
    np.testing.assert_allclose(cell.soh, [1.1, 1.0])


def test_missing_cycles_are_linearly_interpolated(tmp_path):
    path = write_pickle(
        tmp_path / "CALCE_CS2_33.pkl", [(1, [2.0]), (3, [1.6])], nominal=2.0
    )
    cell = preprocess_cell(load_calce_pickle(path), true_eol_cycle=3)
    np.testing.assert_array_equal(cell.cycles, [1, 2, 3])
    np.testing.assert_allclose(cell.soh, [1.0, 0.9, 0.8])
    np.testing.assert_array_equal(cell.is_interpolated, [False, True, False])
    assert cell.missing_count_before == 1
    assert cell.missing_count_after == 0


def test_last_valid_duplicate_record_wins(tmp_path):
    path = write_pickle(
        tmp_path / "CALCE_CS2_33.pkl",
        [(1, [2.0]), (1, [1.8]), (1, [np.nan]), (2, [1.7])],
    )
    cell = preprocess_cell(load_calce_pickle(path), true_eol_cycle=2)
    assert cell.soh[0] == 0.9


from __future__ import annotations

import numpy as np
import pytest

from conftest import make_trajectory


def test_source_support_query_split(parsed_cell):
    task = parsed_cell.source_task(2)
    np.testing.assert_allclose(task.support_soh, [1.05, 1.0])
    np.testing.assert_allclose(task.query_soh, [0.95])


def test_cell_requires_l_plus_one_cycles(parsed_cell):
    with pytest.raises(ValueError, match=r"L\+1"):
        parsed_cell.source_task(3)


def test_voltage_is_normalized_from_support_prefix_only():
    common = [3.5, 3.7, 3.6, 3.4]
    first = make_trajectory(
        "CALCE_CX2_33.pkl", [1.0, 0.98, 0.96, 0.94], mean_voltage_v=common
    )
    second = make_trajectory(
        "CALCE_CX2_33.pkl", [1.0, 0.98, 0.96, 0.94],
        mean_voltage_v=[3.5, 3.7, 1000.0, -1000.0],
    )
    first_view = first.target_support(2, ["soh", "voltage_mean"])
    second_view = second.target_support(2, ["soh", "voltage_mean"])
    np.testing.assert_allclose(first_view.features, second_view.features)
    np.testing.assert_allclose(first_view.features[:, 1], [-1.0, 1.0])
    assert not first_view.features.flags.writeable


def test_current_is_normalized_from_support_prefix_only():
    first = make_trajectory(
        "CALCE_CX2_33.pkl",
        [1.0, 0.98, 0.96, 0.94],
        mean_voltage_v=[3.5, 3.7, 3.6, 3.4],
        mean_current_a=[1.5, 1.7, 1.6, 1.4],
    )
    second = make_trajectory(
        "CALCE_CX2_33.pkl",
        [1.0, 0.98, 0.96, 0.94],
        mean_voltage_v=[3.5, 3.7, 1000.0, -1000.0],
        mean_current_a=[1.5, 1.7, 1000.0, -1000.0],
    )
    features = ["soh", "voltage_mean", "current_mean"]
    first_view = first.target_support(2, features)
    second_view = second.target_support(2, features)
    np.testing.assert_allclose(first_view.features, second_view.features)
    np.testing.assert_allclose(first_view.features[:, 1], [-1.0, 1.0])
    np.testing.assert_allclose(first_view.features[:, 2], [-1.0, 1.0])
    assert first_view.features.shape == (2, 3)

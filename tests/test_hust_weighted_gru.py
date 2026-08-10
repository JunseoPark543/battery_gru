from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from hust_weighted_gru.config import load_config
from hust_weighted_gru.data import (
    load_hust_pickle,
    parse_protocol,
    preprocess_cell,
    select_source_names,
)


def _write_cell(path: Path, final_cycle: int = 103, first_cycle: int = 1) -> None:
    records = []
    for cycle in range(first_cycle, final_cycle + 1):
        records.append(
            {
                "cycle_number": cycle,
                "discharge_capacity_in_Ah": [1.1 - 0.001 * cycle],
                "voltage_in_V": [3.0 + 0.001 * cycle, 3.2 + 0.001 * cycle],
                "current_in_A": [-1.0, 0.5],
            }
        )
    with path.open("wb") as handle:
        pickle.dump(
            {
                "cell_id": path.stem,
                "nominal_capacity_in_Ah": 1.1,
                "cycle_data": records,
            },
            handle,
        )


def test_default_config_is_exact_l100_multivariate_weighted_maml() -> None:
    config = load_config("hust_weighted_gru/config.yaml")
    assert config.data.history_lengths == [100]
    assert config.data.max_forecast_cycle is None
    assert config.model.features == ["soh", "voltage_mean", "current_mean"]
    assert config.model.input_size == 3
    assert config.maml.algorithm == "maml"
    assert config.maml.full_maml is True
    assert config.weights.method == "mmd_qp"
    assert config.source_mode == "same_protocol"


def test_preprocess_builds_cycle_level_features(tmp_path: Path) -> None:
    source = tmp_path / "HUST_2-3.pkl"
    _write_cell(source)
    raw = load_hust_pickle(source)
    cell = preprocess_cell(raw, true_eol_cycle=103)
    assert parse_protocol(source.name) == ("protocol_2", 3)
    assert cell.family == "protocol_2"
    assert np.array_equal(cell.cycles, np.arange(1, 104))
    assert cell.soh.shape == (103,)
    assert cell.mean_voltage_v[0] == pytest.approx((3.001 + 3.201) / 2)
    assert cell.mean_current_a[0] == pytest.approx(-0.25)
    support = cell.target_support(100, ["soh", "voltage_mean", "current_mean"])
    assert support.features.shape == (100, 3)
    assert support.features[:, 1].mean() == pytest.approx(0.0, abs=1e-12)


def test_same_protocol_excludes_target_and_other_protocol() -> None:
    class Cell:
        def __init__(self, family: str) -> None:
            self.family = family

    trajectories = {
        "HUST_1-1.pkl": Cell("protocol_1"),
        "HUST_1-2.pkl": Cell("protocol_1"),
        "HUST_2-1.pkl": Cell("protocol_2"),
    }
    assert select_source_names(
        trajectories, "HUST_1-1.pkl", "same_protocol"
    ) == ["HUST_1-2.pkl"]
    assert select_source_names(
        trajectories, "HUST_1-1.pkl", "leave_protocol_out"
    ) == ["HUST_2-1.pkl"]


def test_preprocess_fills_missing_leading_cycle_without_shifting_axis(
    tmp_path: Path,
) -> None:
    source = tmp_path / "HUST_7-5.pkl"
    _write_cell(source, first_cycle=2)
    cell = preprocess_cell(load_hust_pickle(source), true_eol_cycle=103)
    assert np.array_equal(cell.cycles, np.arange(1, 104))
    assert bool(cell.is_interpolated[0]) is True
    assert cell.capacities_ah[0] == pytest.approx(cell.capacities_ah[1])
    assert cell.mean_voltage_v[0] == pytest.approx(cell.mean_voltage_v[1])
    assert cell.mean_current_a[0] == pytest.approx(cell.mean_current_a[1])


def test_labels_example_uses_filename_mapping(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"HUST_1-1": 1200}), encoding="utf-8")
    # This assertion documents the accepted server label convention.
    from hust_weighted_gru.data import load_labels

    assert load_labels(labels) == {"HUST_1-1.pkl": 1200}

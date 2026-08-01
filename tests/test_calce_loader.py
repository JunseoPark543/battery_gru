from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from battery_weighted_maml.data.calce_loader import load_calce_pickle
from conftest import write_pickle


def test_pkl_schema_parsing_accepts_numpy_and_pandas(tmp_path):
    path = tmp_path / "CALCE_CX2_33.pkl"
    payload = {
        "cell_id": "CX2_33",
        "nominal_capacity_in_Ah": np.float64(2.0),
        "cycle_data": pd.DataFrame(
            {
                "cycle_number": [1, 2],
                "discharge_capacity_in_Ah": [pd.Series([1.9]), np.array([1.8])],
            }
        ),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    result = load_calce_pickle(path)
    assert result.cell_id == "CX2_33"
    assert result.nominal_capacity_ah == 2.0
    assert len(result.cycle_records) == 2


def test_missing_schema_key_names_file_and_key(tmp_path):
    path = tmp_path / "broken.pkl"
    with path.open("wb") as handle:
        pickle.dump({"cell_id": "x", "cycle_data": []}, handle)
    with pytest.raises(ValueError, match=r"broken.pkl.*nominal_capacity_in_Ah"):
        load_calce_pickle(path)


def test_nonpositive_nominal_capacity_is_rejected(tmp_path):
    path = write_pickle(tmp_path / "CALCE_CX2_33.pkl", [(1, [1.0])], nominal=0)
    with pytest.raises(ValueError, match="must be finite and > 0"):
        load_calce_pickle(path)


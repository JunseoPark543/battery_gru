from __future__ import annotations

from battery_weighted_maml.config import load_config
from battery_weighted_maml.gru_baseline import load_gru_baseline_config


def test_non_meta_pair_differs_only_in_history_length() -> None:
    l100 = load_gru_baseline_config(
        "configs/calce_gru_baselines/nometa_soh_l100.yaml"
    )
    l500 = load_gru_baseline_config(
        "configs/calce_gru_baselines/nometa_soh_l500.yaml"
    )
    assert l100.data.history_length == 100
    assert l500.data.history_length == 500
    l100_dict = l100.to_dict()
    l500_dict = l500.to_dict()
    l100_dict["data"]["history_length"] = 500
    assert l100_dict == l500_dict


def test_weighted_meta_pair_differs_only_in_history_length() -> None:
    l100 = load_config(
        "configs/calce_gru_baselines/weighted_meta_soh_l100.yaml"
    )
    l500 = load_config(
        "configs/calce_gru_baselines/weighted_meta_soh_l500.yaml"
    )
    assert l100.data.history_lengths == [100]
    assert l500.data.history_lengths == [500]
    assert l100.model.features == ["soh"]
    assert l100.maml.full_maml is True
    assert l100.weights.method == "mmd_qp"
    l100_dict = l100.to_dict()
    l500_dict = l500.to_dict()
    l100_dict["data"]["history_lengths"] = [500]
    assert l100_dict == l500_dict


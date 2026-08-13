from __future__ import annotations

from battery_weighted_maml.config import load_config
from battery_weighted_maml.recursive_gru_baseline import load_recursive_gru_config


def test_non_meta_pair_differs_only_in_history_length() -> None:
    l100 = load_recursive_gru_config(
        "configs/calce_gru_baselines/nometa_soh_l100.yaml"
    )
    l500 = load_recursive_gru_config(
        "configs/calce_gru_baselines/nometa_soh_l500.yaml"
    )
    assert l100.data.history_length == 100
    assert l500.data.history_length == 500
    l100_dict = l100.to_dict()
    l500_dict = l500.to_dict()
    l100_dict["data"]["history_length"] = 500
    assert l100_dict == l500_dict
    assert l100.training.teacher_forcing_ratio == 0.0


def test_l100_no_early_stopping_config_has_exact_step_budget() -> None:
    config = load_recursive_gru_config(
        "configs/calce_gru_baselines/"
        "nometa_soh_l100_10000steps_no_early_stopping.yaml"
    )
    assert config.data.history_length == 100
    assert config.training.max_steps == 10_000
    assert config.training.early_stopping is False
    assert config.training.teacher_forcing_ratio == 0.0


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
    assert l100.model.teacher_forcing_ratio == 0.0
    l100_dict = l100.to_dict()
    l500_dict = l500.to_dict()
    l100_dict["data"]["history_lengths"] = [500]
    assert l100_dict == l500_dict

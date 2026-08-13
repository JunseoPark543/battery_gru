from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from paper_reproduction.config import load_config
from paper_reproduction.run_paper_recursive_suite import (
    _combine_results,
    _main_args,
)
from paper_reproduction.main import _new_run_dir


ROOT = Path(__file__).resolve().parents[1]


def test_paper_recursive_configs_are_isolated_and_aligned():
    for history_length in (500, 100):
        config = load_config(
            ROOT / f"paper_reproduction/configs/paper_recursive_l{history_length}.yaml"
        )
        assert config.data.history_length == history_length
        assert config.data.train_cells == [
            "CALCE_CX2_16.pkl",
            "CALCE_CX2_33.pkl",
            "CALCE_CX2_34.pkl",
            "CALCE_CX2_35.pkl",
            "CALCE_CX2_36.pkl",
        ]
        assert config.data.test_cells == ["CALCE_CX2_37.pkl", "CALCE_CX2_38.pkl"]
        assert config.model.input_size == 1
        assert config.model.hidden_size == 64
        assert config.model.num_layers == 1
        assert config.model.predicted_input_probability == 0.5
        assert config.maml.meta_batch_size == 5
        assert config.maml.inner_steps == 1
        assert config.maml.inner_batch_size == 64
        assert config.maml.inner_learning_rate == 0.05
        assert config.maml.max_epochs == 500
        assert config.maml.gradient_clip_norm is None
        assert config.adaptation.fast_steps == [0, 1, 3, 5]
        assert config.adaptation.complete_learning_rate == 0.05
        assert config.adaptation.sampling_mode == "random"
        assert config.adaptation.checkpoint_selection == "paper_query_early_stopping"
        assert config.adaptation.gradient_clip_norm is None
        assert config.evaluation.forecast_mode == "paper"
        assert config.evaluation.max_prediction_length is None
        assert config.evaluation.eol_threshold == 0.70


def test_suite_namespace_does_not_override_disabled_gradient_clipping(tmp_path):
    args = _main_args(tmp_path / "config.yaml", "all", "cuda")
    assert isinstance(args, Namespace)
    assert not hasattr(args, "gradient_clip_norm")
    assert args.mode == "all"
    assert args.history_length is None


def test_paper_recursive_run_name_stays_short(tmp_path):
    config = load_config(
        ROOT / "paper_reproduction/configs/paper_recursive_l500.yaml"
    )
    config.maml.experiment_label = "paper-rec"
    path = _new_run_dir(config, tmp_path, "all")
    assert "_all_L500_paper-rec_s42_c" in path.name
    assert len(path.name) < 90


def test_suite_combines_results_and_compares_reference(tmp_path):
    runs: dict[int, Path] = {}
    for history_length in (500, 100):
        run_dir = tmp_path / f"run_l{history_length}"
        output = run_dir / "meta_test"
        output.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "cell": cell,
                    "mode": "complete_paper_query_selected",
                    "mae_percent": float(history_length) / 1000,
                    "rmse_percent": 1.0,
                    "r2": 0.9,
                    "rul_error_actual_minus_predicted": 1,
                }
                for cell in ("CALCE_CX2_37.pkl", "CALCE_CX2_38.pkl")
            ]
        ).to_csv(output / "meta_test_summary.csv", index=False)
        runs[history_length] = run_dir
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        [
            {
                "history_length": history_length,
                "cell": cell,
                "mode": "complete_paper_query_selected",
                "mae_percent": 0.6,
                "rmse_percent": 1.1,
                "r2": 0.8,
                "rul_error_actual_minus_predicted": 2,
                "source": "test",
            }
            for history_length in (500, 100)
            for cell in ("CALCE_CX2_37.pkl", "CALCE_CX2_38.pkl")
        ]
    ).to_csv(reference, index=False)

    comparison_path = _combine_results(runs, tmp_path, reference)

    comparison = pd.read_csv(comparison_path)
    assert len(comparison) == 4
    assert "mae_percent_difference" in comparison
    assert (tmp_path / "paper_recursive_results.csv").is_file()
    assert (tmp_path / "paper_recursive_comparison.png").is_file()

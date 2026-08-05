from __future__ import annotations

import json

from battery_weighted_maml.cli import run_experiment
from battery_weighted_maml.config import ExperimentConfig
from battery_weighted_maml.evaluation.plots import performance_title
from conftest import make_trajectory


def test_prediction_figure_title_contains_scaled_performance_metrics():
    title = performance_title(
        "CX2_37 full adaptation",
        {
            "mae": 0.0226056591,
            "rmse": 0.0300993291,
            "r2": 0.85777355,
            "absolute_rul_error": 38,
        },
    )
    assert "MAE=2.261%" in title
    assert "RMSE=3.010%" in title
    assert "R²=0.858" in title
    assert "absolute RUL error=38 cycles" in title


def test_end_to_end_smoke_pipeline(tmp_path):
    trajectories = {
        "CALCE_CX2_37.pkl": make_trajectory(
            "CALCE_CX2_37.pkl", [1.02, 0.99, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76],
            mean_voltage_v=[3.80, 3.78, 3.76, 3.74, 3.72, 3.70, 3.68, 3.66],
        ),
        "CALCE_CX2_33.pkl": make_trajectory(
            "CALCE_CX2_33.pkl", [1.01, 0.98, 0.95, 0.91, 0.87, 0.83, 0.79, 0.75],
            mean_voltage_v=[3.79, 3.77, 3.75, 3.73, 3.71, 3.69, 3.67, 3.65],
        ),
        "CALCE_CX2_34.pkl": make_trajectory(
            "CALCE_CX2_34.pkl", [1.03, 1.00, 0.97, 0.94, 0.90, 0.86, 0.82, 0.78],
            mean_voltage_v=[3.81, 3.79, 3.77, 3.75, 3.73, 3.71, 3.69, 3.67],
        ),
    }
    config = ExperimentConfig()
    config.device = "cpu"
    config.model.hidden_size = 4
    config.model.input_size = 2
    config.model.features = ["soh", "voltage_mean"]
    config.data.history_lengths = [4]
    config.data.max_forecast_cycle = 14
    config.maml.inner_batch_size = 4
    run_dir = run_experiment(
        config,
        "CALCE_CX2_37.pkl",
        4,
        "same_family",
        project_root=tmp_path,
        smoke_test=True,
        trajectories=trajectories,
    )
    required = [
        "checkpoints/best_source_meta_loss.pt",
        "checkpoints/last.pt",
        "weights/final_alpha.csv",
        "predictions/target_fast_prediction.csv",
        "predictions/target_full_prediction.csv",
        "metrics/fast_metrics.json",
        "metrics/fast_1_metrics.json",
        "metrics/fast_2_metrics.json",
        "metrics/fast_metrics_by_step.csv",
        "metrics/full_metrics.json",
        "figures/target_soh_fast_1.png",
        "figures/target_soh_fast_2.png",
        "figures/target_soh_full.png",
    ]
    for relative in required:
        assert (run_dir / relative).is_file(), relative
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["history_length"] == 4
    assert manifest["resolved_config"]["model"]["features"] == ["soh", "voltage_mean"]
    assert manifest["fast_steps"] == [1, 2]
    assert set(manifest["fast_metrics_by_step"]) == {"1", "2"}

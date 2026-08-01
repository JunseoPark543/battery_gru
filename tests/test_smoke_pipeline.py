from __future__ import annotations

import json

from battery_weighted_maml.cli import run_experiment
from battery_weighted_maml.config import ExperimentConfig
from conftest import make_trajectory


def test_end_to_end_smoke_pipeline(tmp_path):
    trajectories = {
        "CALCE_CX2_37.pkl": make_trajectory(
            "CALCE_CX2_37.pkl", [1.02, 0.99, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76]
        ),
        "CALCE_CX2_33.pkl": make_trajectory(
            "CALCE_CX2_33.pkl", [1.01, 0.98, 0.95, 0.91, 0.87, 0.83, 0.79, 0.75]
        ),
        "CALCE_CX2_34.pkl": make_trajectory(
            "CALCE_CX2_34.pkl", [1.03, 1.00, 0.97, 0.94, 0.90, 0.86, 0.82, 0.78]
        ),
    }
    config = ExperimentConfig()
    config.device = "cpu"
    config.model.hidden_size = 4
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
        "metrics/full_metrics.json",
        "figures/target_soh_full.png",
    ]
    for relative in required:
        assert (run_dir / relative).is_file(), relative
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["history_length"] == 4


from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from battery_weighted_maml.source_pretrained_gru_baseline import (
    SourceTrajectoryTask,
    SourcePretrainedGRUConfig,
    load_source_pretrained_gru_config,
    run_source_pretrained_gru_baseline,
    sample_source_cut_points,
)
from conftest import make_trajectory


def test_source_pretrained_default_config_matches_requested_protocol() -> None:
    config = load_source_pretrained_gru_config(
        "configs/baseline/source_pretrained_gru_l100_soh.yaml"
    )
    assert config.data.history_length == 100
    assert config.data.source_mode == "same_family"
    assert config.model.teacher_forcing_ratio == 0.5
    assert config.fine_tuning.learning_rate == 0.05
    assert config.fine_tuning.fast_steps == [1, 3, 5, 10, 15, 20]


def test_variable_cutpoint_config_and_sampling_cover_lifetime_and_l100() -> None:
    config = load_source_pretrained_gru_config(
        "configs/baseline/source_varcut_gru_l100_soh.yaml"
    )
    task = SourceTrajectoryTask(
        "CALCE_CX2_16.pkl",
        torch.linspace(1.0, 0.7, 1000),
        torch.linspace(1.0, 0.7, 1000).unsqueeze(-1),
    )
    points = sample_source_cut_points(
        task, config, torch.Generator(device="cpu").manual_seed(42)
    )
    assert config.pretraining.prefix_mode == "variable_cutpoint"
    assert len(points) == 16
    assert points == sorted(set(points))
    assert 100 in points
    assert min(points) >= 1
    assert max(points) <= 999
    assert any(point < 100 for point in points)
    assert any(point > 500 for point in points)


def test_source_pretrained_smoke_pipeline_uses_sources_then_target_support(
    tmp_path,
) -> None:
    def curve(offset: float) -> list[float]:
        return (1.0 + offset - 0.012 * np.arange(14)).tolist()

    trajectories = {
        "CALCE_CX2_16.pkl": make_trajectory(
            "CALCE_CX2_16.pkl", curve(0.01), true_eol=12
        ),
        "CALCE_CX2_33.pkl": make_trajectory(
            "CALCE_CX2_33.pkl", curve(-0.01), true_eol=11
        ),
        "CALCE_CX2_37.pkl": make_trajectory(
            "CALCE_CX2_37.pkl", curve(0.0), true_eol=12
        ),
    }
    config = SourcePretrainedGRUConfig()
    config.device = "cpu"
    config.data.history_length = 6
    config.model.hidden_size = 4
    config.pretraining.max_epochs = 2
    config.pretraining.early_stopping = False
    config.fine_tuning.batch_size = 4
    config.fine_tuning.fast_steps = [1, 2]
    config.fine_tuning.full_max_steps = 2
    config.fine_tuning.full_patience = 2
    run_dir = run_source_pretrained_gru_baseline(
        config,
        target_name="CALCE_CX2_37.pkl",
        project_root=tmp_path,
        trajectories=trajectories,
        smoke_test=True,
    )

    assert run_dir.parent == tmp_path / "outputs" / "baseline"
    assert run_dir.name == "cx2_37_gru_l6_soh_transfer_samefam_s42"
    required = [
        "checkpoints/pretrain_best.pt",
        "checkpoints/pretrain_last.pt",
        "checkpoints/target_fast_1.pt",
        "checkpoints/target_full_best.pt",
        "pretraining/epoch_history.csv",
        "pretraining/source_loss_history.csv",
        "adaptation/fast_history.csv",
        "metrics/transfer_0_metrics.json",
        "metrics/transfer_fast_1_metrics.json",
        "metrics/transfer_full_metrics.json",
        "metrics/transfer_metrics_summary.csv",
        "figures/pretraining_loss.png",
        "figures/target_soh_transfer_full.png",
        "figures/target_soh_transfer_0_vs_full.png",
    ]
    for relative in required:
        assert (run_dir / relative).is_file(), relative
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["meta_learning"] is False
    assert manifest["weighted_meta_learning"] is False
    assert manifest["source_weighting"] == "uniform_task_balanced"
    assert manifest["target_future_used_for_training_or_selection"] is False
    assert manifest["sources"] == ["CALCE_CX2_16.pkl", "CALCE_CX2_33.pkl"]


def test_variable_cutpoint_smoke_pipeline_reports_zero_to_full(tmp_path) -> None:
    def curve(offset: float) -> list[float]:
        return (1.0 + offset - 0.012 * np.arange(14)).tolist()

    trajectories = {
        "CALCE_CX2_16.pkl": make_trajectory(
            "CALCE_CX2_16.pkl", curve(0.01), true_eol=12
        ),
        "CALCE_CX2_33.pkl": make_trajectory(
            "CALCE_CX2_33.pkl", curve(-0.01), true_eol=11
        ),
        "CALCE_CX2_37.pkl": make_trajectory(
            "CALCE_CX2_37.pkl", curve(0.0), true_eol=12
        ),
    }
    config = SourcePretrainedGRUConfig()
    config.device = "cpu"
    config.data.history_length = 6
    config.model.hidden_size = 4
    config.pretraining.prefix_mode = "variable_cutpoint"
    config.pretraining.cut_points_per_source_per_epoch = 4
    config.pretraining.cut_point_batch_size = 2
    config.pretraining.selection_cut_points = [2, 4, 6]
    config.pretraining.max_epochs = 2
    config.pretraining.early_stopping = False
    config.fine_tuning.batch_size = 4
    config.fine_tuning.fast_steps = [1, 2]
    config.fine_tuning.full_max_steps = 2
    config.fine_tuning.full_patience = 2

    run_dir = run_source_pretrained_gru_baseline(
        config,
        target_name="CALCE_CX2_37.pkl",
        project_root=tmp_path,
        trajectories=trajectories,
        smoke_test=True,
    )

    assert run_dir.name == "cx2_37_gru_l6_soh_varcut_samefam_s42"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_prefix_mode"] == "variable_cutpoint"
    assert manifest["source_cut_point_generalization"] is True
    assert set(manifest["metrics_by_mode"]) == {
        "transfer_0",
        "transfer_fast_1",
        "transfer_fast_2",
        "transfer_full",
    }
    source_history = pd.read_csv(run_dir / "pretraining/source_loss_history.csv")
    assert (source_history["cut_point_count"] == 4).all()

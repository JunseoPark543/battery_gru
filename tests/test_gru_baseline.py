from __future__ import annotations

import json

import numpy as np
import torch

from battery_weighted_maml.data.soh_window_dataset import SOHWindowDataset
from battery_weighted_maml.gru_baseline import GRUBaselineConfig, run_gru_baseline
from battery_weighted_maml.models.gru_encoder_decoder_baseline import SOHGRUEncoderDecoder
from conftest import make_trajectory


def test_plain_gru_encoder_decoder_shapes():
    model = SOHGRUEncoderDecoder(hidden_size=8)
    history = torch.tensor(
        [[[1.0], [0.98], [0.96]], [[1.01], [0.99], [0.97]]]
    )
    targets = torch.zeros(2, 4, 1)
    output = model(
        history,
        future_targets=targets,
        teacher_forcing_ratio=0.5,
        generator=torch.Generator().manual_seed(42),
    )
    forecast = model.recursive_forecast([1.0, 0.98, 0.96], 5)
    assert output.shape == (2, 4, 1)
    assert forecast.shape == (1, 5, 1)
    assert torch.isfinite(output).all()


def test_soh_window_dataset_uses_only_scalar_soh():
    values = np.linspace(1.0, 0.8, 10)
    dataset = SOHWindowDataset(values, encoder_window=4, forecast_horizon=2)
    history, future = dataset[0]
    assert len(dataset) == 5
    assert history.shape == (4, 1)
    assert future.shape == (2, 1)
    np.testing.assert_allclose(history[:, 0], values[:4], rtol=1e-6)
    np.testing.assert_allclose(future[:, 0], values[4:6], rtol=1e-6)


def test_plain_gru_baseline_smoke_pipeline(tmp_path):
    soh = np.linspace(1.02, 0.75, 25).tolist()
    trajectory = make_trajectory("CALCE_CX2_37.pkl", soh, true_eol=21)
    config = GRUBaselineConfig()
    config.device = "cpu"
    config.data.history_length = 20
    config.data.max_forecast_cycle = 30
    config.model.hidden_size = 4
    config.training.max_epochs = 2
    config.training.batch_size = 4
    config.training.encoder_window = 5
    config.training.forecast_horizon = 3
    config.training.validation_cycles = 4
    config.training.early_stopping_patience = 2
    run_dir = run_gru_baseline(
        config,
        target_name="CALCE_CX2_37.pkl",
        project_root=tmp_path,
        smoke_test=True,
        target_trajectory=trajectory,
    )
    required = [
        "checkpoints/best_validation.pt",
        "checkpoints/last.pt",
        "training/epoch_history.csv",
        "predictions/target_gru_baseline_prediction.csv",
        "metrics/gru_baseline_metrics.json",
        "figures/training_loss.png",
        "figures/target_soh_gru_baseline.png",
    ]
    for relative in required:
        assert (run_dir / relative).is_file(), relative
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["weighted_meta_learning"] is False
    assert manifest["input_features"] == ["soh"]
    assert manifest["history_length"] == 20

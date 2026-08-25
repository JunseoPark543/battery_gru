from __future__ import annotations

import torch

from battery_weighted_maml.matr_anp.config import DataConfig, QGridConfig
from battery_weighted_maml.matr_anp.data import load_dataset
from battery_weighted_maml.matr_anp.synthetic import write_synthetic_matr_dataset
from battery_weighted_maml.streaming_soh.config import EpisodeConfig, ModelConfig
from battery_weighted_maml.streaming_soh.episodes import EpisodeSampler
from battery_weighted_maml.streaming_soh.features import CycleGridProcessor, fit_signal_scaler
from battery_weighted_maml.streaming_soh.model import build_model
from battery_weighted_maml.streaming_soh.rolling_context_demo import (
    _plot_overlay,
    _plot_panels,
    _predict_rolling_context,
    _select_cell,
    _validate_cycles,
)


def test_rolling_context_compares_common_future_horizon(tmp_path) -> None:
    root = write_synthetic_matr_dataset(
        tmp_path, num_cells=4, num_cycles=40, signal_points=72
    )
    cells, _ = load_dataset(
        root,
        DataConfig(
            dataset="MATR",
            file_globs=["**/*.pkl"],
            minimum_valid_cycles=20,
            minimum_discharge_points=8,
        ),
    )
    scaler = fit_signal_scaler(cells[:3])
    processor = CycleGridProcessor(
        QGridConfig(minimum=0.0, maximum=1.2, num_points=64), 4, 4
    )
    episode_config = EpisodeConfig(
        minimum_current_cycle=10,
        minimum_history_cycles=5,
        maximum_history_cycles=24,
        minimum_future_cycles=5,
        maximum_training_future_points=20,
        training_cycle_alpha_range=[0.3, 0.7],
        training_beta_range=[0.2, 0.8],
        evaluation_cycle_alphas=[0.5],
        evaluation_betas=[0.5],
        minimum_observed_q_points=4,
        minimum_future_q_points=4,
        cycle_scale=100.0,
    )
    sampler = EpisodeSampler(episode_config, processor, scaler)
    cycles = _validate_cycles([25, 15, 20, 20])
    assert cycles == [15, 20, 25]
    cell = _select_cell([cells[3]], sampler, cycles, cells[3].cell_id)
    episodes = [
        sampler.evaluation(cell, 0.5, 0.5, current_cycle=cycle) for cycle in cycles
    ]
    model = build_model(
        ModelConfig(
            convolution_channels=[8, 16],
            kernel_size=5,
            curve_embedding_dim=16,
            cycle_feature_dim=20,
            gru_hidden_dim=24,
            gru_layers=2,
            decoder_hidden_dim=32,
            dropout=0.0,
            minimum_soh_std=0.003,
        )
    )
    metrics, points = _predict_rolling_context(
        model,
        episodes,
        torch.device("cpu"),
        cycle_scale=100.0,
        interval_level=0.95,
    )
    assert metrics["current_cycle"].tolist() == cycles
    assert metrics["completed_through_cycle"].tolist() == [14, 19, 24]
    assert set(metrics["common_horizon_start"]) == {25}
    assert metrics["common_horizon_soh_rmse"].notna().all()
    assert points.groupby("current_cycle").size().loc[15] > points.groupby(
        "current_cycle"
    ).size().loc[25]
    _plot_panels(cell, metrics, points, tmp_path / "panels.png", 72)
    _plot_overlay(cell, metrics, points, tmp_path / "overlay.png", 72)
    assert (tmp_path / "panels.png").is_file()
    assert (tmp_path / "overlay.png").is_file()

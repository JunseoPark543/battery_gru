from __future__ import annotations

import numpy as np
import torch

from battery_weighted_maml.matr_anp.config import DataConfig, QGridConfig
from battery_weighted_maml.matr_anp.data import load_dataset
from battery_weighted_maml.matr_anp.synthetic import write_synthetic_matr_dataset
from battery_weighted_maml.streaming_soh.config import EpisodeConfig
from battery_weighted_maml.streaming_soh.episodes import EpisodeSampler, collate_episodes
from battery_weighted_maml.streaming_soh.features import CycleGridProcessor, fit_signal_scaler
from battery_weighted_maml.streaming_soh_anp.config import ModelConfig
from battery_weighted_maml.streaming_soh_anp.evaluate import (
    _plot_calibration,
    _plot_examples,
    _plot_heatmaps,
    _predict,
)
from battery_weighted_maml.streaming_soh_anp.losses import latent_anp_loss
from battery_weighted_maml.streaming_soh_anp.model import build_model
from battery_weighted_maml.streaming_soh_anp.online import OnlineLatentANPSession
from battery_weighted_maml.streaming_soh_anp.train import model_forward


def _setup(tmp_path):
    root = write_synthetic_matr_dataset(
        tmp_path, num_cells=4, num_cycles=36, signal_points=72
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
        maximum_history_cycles=16,
        minimum_future_cycles=5,
        maximum_training_future_points=20,
        training_cycle_alpha_range=[0.3, 0.7],
        training_beta_range=[0.2, 0.8],
        evaluation_cycle_alphas=[0.5],
        evaluation_betas=[0.3, 0.7],
        minimum_observed_q_points=4,
        minimum_future_q_points=4,
        cycle_scale=100.0,
    )
    sampler = EpisodeSampler(episode_config, processor, scaler)
    first = sampler.evaluation(cells[3], 0.5, 0.3)
    second = sampler.evaluation(cells[3], 0.5, 0.7, current_cycle=first.current_cycle)
    model = build_model(
        ModelConfig(
            convolution_channels=[8, 16],
            kernel_size=5,
            curve_embedding_dim=16,
            cycle_feature_dim=20,
            gru_hidden_dim=24,
            gru_layers=2,
            attention_heads=4,
            latent_dim=8,
            latent_hidden_dim=24,
            decoder_hidden_dim=32,
            dropout=0.0,
            minimum_latent_std=0.02,
            minimum_observation_std=0.003,
        )
    )
    return cells, scaler, processor, episode_config, first, second, model


def test_anp_training_has_posterior_kl_and_finite_gradients(tmp_path) -> None:
    _, _, _, _, first, second, model = _setup(tmp_path)
    batch = collate_episodes([first, second])
    model.train()
    output = model_forward(model, batch, use_posterior=True, num_latent_samples=1)
    assert output["soh_mean"].shape == batch.target_soh.shape
    assert output["prior_mean"].shape == (2, 8)
    assert output["posterior_mean"].shape == (2, 8)
    assert torch.all(output["prior_std"] > 0)
    assert torch.all(output["posterior_std"] > 0)
    loss, parts = latent_anp_loss(
        output,
        batch,
        soh_huber_delta=0.02,
        soh_huber_weight=0.2,
        voltage_huber_delta=0.5,
        endpoint_huber_delta=0.05,
        kl_coefficient=0.01,
        kl_free_bits=0.02,
        voltage_completion_weight=0.2,
        endpoint_weight=0.1,
        monotonic_weight=0.01,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "soh_nll", "soh_huber", "kl", "future_voltage", "endpoint", "monotonic"
    }
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_inference_prior_ignores_future_targets_and_decomposes_uncertainty(tmp_path) -> None:
    _, _, _, _, first, second, model = _setup(tmp_path)
    batch = collate_episodes([first, second])
    model.eval()
    torch.manual_seed(123)
    original = model_forward(model, batch, use_posterior=False, num_latent_samples=20)
    batch.target_soh.add_(0.4)
    torch.manual_seed(123)
    changed = model_forward(model, batch, use_posterior=False, num_latent_samples=20)
    torch.testing.assert_close(original["soh_mean"], changed["soh_mean"])
    torch.testing.assert_close(original["prior_mean"], changed["prior_mean"])
    assert "posterior_mean" not in original
    assert torch.any(original["soh_epistemic_std"] > 0)
    reconstructed = torch.sqrt(
        original["soh_epistemic_std"].square()
        + original["soh_aleatoric_std"].square()
    )
    torch.testing.assert_close(original["soh_std"], reconstructed)
    torch.testing.assert_close(
        original["completed_state"][0], original["completed_state"][1]
    )
    metrics, points = _predict(
        model,
        [(0.5, 0.3, first), (0.5, 0.7, second)],
        torch.device("cpu"),
        cycle_scale=100.0,
        latent_samples=5,
        interval_level=0.95,
    )
    assert {"soh_rmse", "soh_nll", "soh_crps", "coverage_95"}.issubset(metrics)
    assert {"epistemic_std", "aleatoric_std", "predictive_std"}.issubset(points)
    _plot_heatmaps(metrics, tmp_path / "heatmap.png", 72)
    _plot_calibration(metrics, tmp_path / "calibration.png", 72)
    _plot_examples(metrics, points, tmp_path / "examples.png", 1, 72)
    assert (tmp_path / "heatmap.png").is_file()
    assert (tmp_path / "calibration.png").is_file()
    assert (tmp_path / "examples.png").is_file()


def test_online_prefix_updates_context_prior_without_weight_update(tmp_path) -> None:
    cells, scaler, processor, config, first, _, model = _setup(tmp_path)
    model.eval()
    cell = cells[3]
    current = cell.cycle_by_number(first.current_cycle)
    assert current.discharge is not None
    session = OnlineLatentANPSession(
        model,
        processor,
        scaler,
        cell,
        current_cycle=first.current_cycle,
        forecast_cycles=list(range(first.current_cycle, first.current_cycle + 5)),
        maximum_history_cycles=config.maximum_history_cycles,
        cycle_scale=config.cycle_scale,
        latent_samples=20,
        device=torch.device("cpu"),
    )
    early_cut = len(current.discharge.q) // 3
    late_cut = 2 * len(current.discharge.q) // 3
    early = session.observe(
        current.discharge.q[:early_cut],
        current.discharge.voltage_v[:early_cut],
        current.discharge.current_a_magnitude[:early_cut],
    )
    late = session.observe(
        current.discharge.q[:late_cut],
        current.discharge.voltage_v[:late_cut],
        current.discharge.current_a_magnitude[:late_cut],
    )
    assert early.soh_mean.shape == (5,)
    assert np.all(np.isfinite(early.predictive_std))
    assert not np.allclose(early.prior_mean, late.prior_mean)

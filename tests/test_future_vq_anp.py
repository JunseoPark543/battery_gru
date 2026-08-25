from __future__ import annotations

import numpy as np
import torch

from battery_weighted_maml.future_vq_anp.config import EpisodeConfig, ModelConfig
from battery_weighted_maml.future_vq_anp.episodes import EpisodeSampler, collate_episodes
from battery_weighted_maml.future_vq_anp.features import CurveGridProcessor, fit_voltage_scaler
from battery_weighted_maml.future_vq_anp.losses import future_vq_loss
from battery_weighted_maml.future_vq_anp.model import build_model
from battery_weighted_maml.future_vq_anp.train import model_forward
from battery_weighted_maml.matr_anp.config import DataConfig, QGridConfig
from battery_weighted_maml.matr_anp.data import load_dataset
from battery_weighted_maml.matr_anp.synthetic import write_synthetic_matr_dataset


def _setup(tmp_path):
    root = write_synthetic_matr_dataset(
        tmp_path, num_cells=4, num_cycles=40, signal_points=72
    )
    cells, _ = load_dataset(
        root,
        DataConfig(
            dataset="MATR",
            file_globs=["**/*.pkl"],
            minimum_valid_cycles=30,
            minimum_discharge_points=8,
        ),
    )
    scaler = fit_voltage_scaler(cells[:3])
    processor = CurveGridProcessor(
        QGridConfig(minimum=0.0, maximum=1.2, num_points=64), minimum_q_points=4
    )
    episode_config = EpisodeConfig(
        history_cycles=10,
        minimum_future_cycles=5,
        maximum_training_future_cycles=12,
        training_cut_alpha_range=[0.2, 0.8],
        evaluation_cut_cycles=[15, 20],
        minimum_q_points=4,
        cycle_scale=100.0,
    )
    sampler = EpisodeSampler(episode_config, processor, scaler)
    first = sampler.evaluation(cells[3], 15)
    second = sampler.evaluation(cells[3], 20)
    model = build_model(
        ModelConfig(
            convolution_channels=[8, 16],
            kernel_size=5,
            curve_embedding_dim=16,
            cycle_feature_dim=20,
            gru_hidden_dim=24,
            gru_layers=2,
            attention_heads=4,
            q_embedding_dim=8,
            latent_dim=8,
            latent_hidden_dim=24,
            decoder_hidden_dim=32,
            dropout=0.0,
        )
    )
    return cells, scaler, sampler, first, second, model


def test_episode_uses_recent_completed_curves_and_all_later_targets(tmp_path) -> None:
    _, scaler, _, first, _, _ = _setup(tmp_path)
    assert first.history_curve.shape == (10, 64, 2)
    assert first.cut_cycle == 15
    np.testing.assert_array_equal(first.query_cycle_numbers, np.arange(16, 41))
    assert first.target_voltage.shape == (25, 64)
    assert first.target_q_mask.shape == (25, 64)
    assert set(scaler.fit_cell_ids) == {
        "MATR_SYNTH_00", "MATR_SYNTH_01", "MATR_SYNTH_02"
    }


def test_training_posterior_loss_and_gradients_are_finite(tmp_path) -> None:
    _, _, _, first, second, model = _setup(tmp_path)
    batch = collate_episodes([first, second])
    output = model_forward(model, batch, use_posterior=True, num_latent_samples=1)
    assert output["sample_voltage_mean"].shape == (1, 2, 25, 64)
    assert output["sample_endpoint_mean"].shape == (1, 2, 25)
    assert output["prior_mean"].shape == (2, 8)
    assert output["posterior_mean"].shape == (2, 8)
    loss, parts = future_vq_loss(
        output,
        batch,
        voltage_huber_delta=0.5,
        voltage_huber_weight=0.2,
        endpoint_huber_delta=0.03,
        endpoint_weight=0.5,
        kl_coefficient=0.01,
        kl_free_bits=0.02,
        q_monotonic_weight=0.01,
        endpoint_monotonic_weight=0.01,
        temporal_smoothness_weight=0.001,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "voltage_nll", "voltage_huber", "endpoint_nll", "endpoint_huber", "kl",
        "q_monotonic", "endpoint_monotonic", "temporal_smoothness",
    }
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_prior_inference_ignores_future_targets_and_decomposes_uncertainty(tmp_path) -> None:
    _, _, _, first, second, model = _setup(tmp_path)
    batch = collate_episodes([first, second])
    model.eval()
    torch.manual_seed(7)
    original = model_forward(model, batch, use_posterior=False, num_latent_samples=12)
    batch.target_voltage.add_(10.0)
    batch.target_endpoint_fraction.zero_()
    torch.manual_seed(7)
    changed = model_forward(model, batch, use_posterior=False, num_latent_samples=12)
    torch.testing.assert_close(original["voltage_mean"], changed["voltage_mean"])
    torch.testing.assert_close(original["endpoint_mean"], changed["endpoint_mean"])
    torch.testing.assert_close(original["prior_mean"], changed["prior_mean"])
    assert "posterior_mean" not in original
    reconstructed = torch.sqrt(
        original["voltage_epistemic_std"].square()
        + original["voltage_aleatoric_std"].square()
    )
    torch.testing.assert_close(original["voltage_std"], reconstructed)
    assert torch.any(original["voltage_epistemic_std"] > 0)


def test_query_chunking_reuses_one_latent_surface(tmp_path) -> None:
    _, _, _, first, _, model = _setup(tmp_path)
    batch = collate_episodes([first])
    model.eval()
    with torch.no_grad():
        encoded = model.encode_history(
            batch.history_curve,
            batch.history_endpoint_fraction,
            batch.history_cycle_scaled,
            batch.history_gap_scaled,
            batch.history_mask,
            batch.q_coordinate,
        )
        latent = model.sample_latent(encoded["prior_mean"], encoded["prior_std"], 5)
        whole = model.decode_queries(
            encoded, batch.query_cycle_scaled, batch.q_coordinate, latent
        )["sample_voltage_mean"]
        chunks = []
        for start in range(0, whole.shape[2], 7):
            chunks.append(
                model.decode_queries(
                    encoded,
                    batch.query_cycle_scaled[:, start : start + 7],
                    batch.q_coordinate,
                    latent,
                )["sample_voltage_mean"]
            )
        torch.testing.assert_close(whole, torch.cat(chunks, dim=2))

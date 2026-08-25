from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from battery_weighted_maml.matr_anp.config import DataConfig, QGridConfig
from battery_weighted_maml.matr_anp.data import DischargeCurve, load_dataset
from battery_weighted_maml.matr_anp.synthetic import write_synthetic_matr_dataset
from battery_weighted_maml.streaming_soh.config import EpisodeConfig, ModelConfig
from battery_weighted_maml.streaming_soh.episodes import EpisodeSampler, collate_episodes
from battery_weighted_maml.streaming_soh.features import (
    CycleGridProcessor,
    SignalScaler,
    fit_signal_scaler,
    integrate_discharge_q,
)
from battery_weighted_maml.streaming_soh.losses import streaming_soh_loss
from battery_weighted_maml.streaming_soh.model import build_model
from battery_weighted_maml.streaming_soh.online import OnlineSOHSession
from battery_weighted_maml.streaming_soh.train import model_forward


def _curve() -> DischargeCurve:
    q = np.linspace(0.0, 1.0, 101)
    return DischargeCurve(
        q=q,
        voltage_v=3.7 - 0.65 * q - 0.04 * q**2,
        current_a_magnitude=1.0 + 0.1 * q,
        original_current_sign=-1,
        monotonic_before_cleanup=True,
        duplicate_q_count=0,
    )


def _scaler() -> SignalScaler:
    return SignalScaler(
        voltage_mean=3.3,
        voltage_std=0.2,
        current_mean=1.0,
        current_std=0.1,
        fit_cell_ids=("train",),
    )


def test_streaming_prefix_does_not_use_future_voltage_or_current() -> None:
    curve = _curve()
    changed_voltage = curve.voltage_v.copy()
    changed_current = curve.current_a_magnitude.copy()
    changed_voltage[51:] += 0.4
    changed_current[51:] += 0.8
    changed = replace(
        curve,
        voltage_v=changed_voltage,
        current_a_magnitude=changed_current,
    )
    processor = CycleGridProcessor(
        QGridConfig(minimum=0.0, maximum=1.2, num_points=64), 4, 4
    )
    first = processor.build_prefix(curve, 0.5, _scaler())
    second = processor.build_prefix(changed, 0.5, _scaler())
    np.testing.assert_allclose(first.feature, second.feature)
    live_feature, live_fraction = processor.observed_samples(
        curve.q[:51], curve.voltage_v[:51], curve.current_a_magnitude[:51], _scaler()
    )
    np.testing.assert_allclose(first.feature, live_feature)
    assert np.isclose(live_fraction, first.q_cut / processor.q_max)
    q_integrated = integrate_discharge_q(
        np.asarray([0.0, 1800.0, 3600.0]), np.asarray([-1.0, -1.0, -1.0]), 1.0
    )
    np.testing.assert_allclose(q_integrated, [0.0, 0.5, 1.0])
    assert not np.allclose(
        first.target_voltage[first.future_mask], second.target_voltage[second.future_mask]
    )


def test_model_reuses_immutable_completed_state_and_has_finite_gradients(tmp_path) -> None:
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
    assert set(scaler.fit_cell_ids) == {cell.cell_id for cell in cells[:3]}
    processor = CycleGridProcessor(
        QGridConfig(minimum=0.0, maximum=1.2, num_points=64), 4, 4
    )
    sampler = EpisodeSampler(
        EpisodeConfig(
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
        ),
        processor,
        scaler,
    )
    first = sampler.evaluation(cells[3], 0.5, 0.3)
    second = sampler.evaluation(
        cells[3], 0.5, 0.7, current_cycle=first.current_cycle
    )
    batch = collate_episodes([first, second])
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
    model.eval()
    output = model_forward(model, batch)
    assert output["soh_mean"].shape == batch.target_soh.shape
    assert output["soh_std"].shape == batch.target_soh.shape
    assert output["voltage"].shape == batch.target_voltage.shape
    assert output["endpoint_fraction"].shape == (2,)
    torch.testing.assert_close(output["completed_state"][0], output["completed_state"][1])
    assert not torch.allclose(output["candidate_state"][0], output["candidate_state"][1])
    assert torch.all(output["endpoint_fraction"] >= batch.prefix_fraction)
    loss, parts = streaming_soh_loss(
        output,
        batch,
        soh_huber_delta=0.02,
        voltage_huber_delta=0.5,
        endpoint_huber_delta=0.05,
        uncertainty_weight=0.05,
        voltage_completion_weight=0.2,
        endpoint_weight=0.1,
        monotonic_weight=0.01,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "soh_huber", "uncertainty_nll", "future_voltage", "endpoint", "monotonic"
    }
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    current_cycle = first.current_cycle
    raw_cycle = cells[3].cycle_by_number(current_cycle)
    assert raw_cycle.discharge is not None
    session = OnlineSOHSession(
        model,
        processor,
        scaler,
        cells[3],
        current_cycle=current_cycle,
        forecast_cycles=list(range(current_cycle, current_cycle + 5)),
        maximum_history_cycles=16,
        cycle_scale=100.0,
        device=torch.device("cpu"),
    )
    cut = len(raw_cycle.discharge.q) // 2
    online = session.observe(
        raw_cycle.discharge.q[:cut],
        raw_cycle.discharge.voltage_v[:cut],
        raw_cycle.discharge.current_a_magnitude[:cut],
    )
    assert online.soh_mean.shape == (5,)
    assert online.soh_std.shape == (5,)
    assert np.all(np.isfinite(online.soh_mean))

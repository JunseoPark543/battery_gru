from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from battery_weighted_maml.matr_anp.config import DataConfig, QGridConfig
from battery_weighted_maml.matr_anp.data import DischargeCurve, load_dataset
from battery_weighted_maml.matr_anp.synthetic import write_synthetic_matr_dataset
from battery_weighted_maml.partial_vq_forecasting.config import (
    EpisodeConfig,
    ModelConfig,
)
from battery_weighted_maml.partial_vq_forecasting.episodes import (
    EpisodeSampler,
    collate_episodes,
)
from battery_weighted_maml.partial_vq_forecasting.features import (
    PartialVQProcessor,
    VoltageScaler,
    fit_voltage_scaler,
)
from battery_weighted_maml.partial_vq_forecasting.losses import forecasting_loss
from battery_weighted_maml.partial_vq_forecasting.model import build_model


def _curve() -> DischargeCurve:
    q = np.linspace(0.0, 1.0, 101)
    voltage = 3.7 - 0.65 * q - 0.04 * q**2
    return DischargeCurve(
        q=q,
        voltage_v=voltage,
        current_a_magnitude=np.ones_like(q),
        original_current_sign=-1,
        monotonic_before_cleanup=True,
        duplicate_q_count=0,
    )


def test_prefix_input_never_uses_future_voltage() -> None:
    curve = _curve()
    changed = curve.voltage_v.copy()
    changed[51:] += 0.4
    future_changed = replace(curve, voltage_v=changed)
    scaler = VoltageScaler(mean=3.3, std=0.2, fit_cell_ids=("train",))
    processor = PartialVQProcessor(
        QGridConfig(minimum=0.0, maximum=1.2, num_points=121), 4, 4
    )
    first = processor.build(curve, 0.5, scaler)
    second = processor.build(future_changed, 0.5, scaler)
    np.testing.assert_allclose(first.input_feature, second.input_feature)
    assert not np.allclose(
        first.target_voltage[first.future_mask], second.target_voltage[second.future_mask]
    )


def test_model_predicts_curve_and_endpoint_after_observed_cut() -> None:
    scaler = VoltageScaler(mean=3.3, std=0.2, fit_cell_ids=("train",))
    processor = PartialVQProcessor(
        QGridConfig(minimum=0.0, maximum=1.2, num_points=64), 4, 4
    )
    partial = processor.build(_curve(), 0.4, scaler)
    from battery_weighted_maml.partial_vq_forecasting.episodes import VQEpisode

    episode = VQEpisode(
        cell_id="cell",
        cycle_number=10,
        beta=0.4,
        input_feature=partial.input_feature,
        q_coordinate=partial.q_coordinate,
        target_voltage=partial.target_voltage,
        observed_mask=partial.observed_mask,
        future_mask=partial.future_mask,
        valid_mask=partial.valid_mask,
        endpoint_fraction=partial.endpoint_fraction,
        q_cut=partial.q_cut,
        q_end=partial.q_end,
        observed_points=partial.observed_points,
        future_points=partial.future_points,
    )
    batch = collate_episodes([episode])
    model = build_model(
        ModelConfig(
            convolution_channels=[8, 16],
            hidden_dim=16,
            attention_layers=1,
            attention_heads=4,
            feedforward_dim=32,
            decoder_hidden_dim=24,
            dropout=0.0,
        )
    )
    output = model(batch.input_feature, batch.q_coordinate)
    assert output["voltage"].shape == (1, 64)
    assert output["endpoint_fraction"].shape == (1,)
    q_cut_fraction = batch.q_coordinate.masked_fill(~batch.observed_mask, 0.0).amax()
    assert output["endpoint_fraction"].item() >= q_cut_fraction.item()
    loss, parts = forecasting_loss(
        output,
        batch.target_voltage,
        batch.observed_mask,
        batch.future_mask,
        batch.valid_mask,
        batch.endpoint_fraction,
        voltage_huber_delta=0.5,
        endpoint_huber_delta=0.05,
        endpoint_weight=1.0,
        observed_reconstruction_weight=0.1,
        monotonic_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "future_voltage", "observed_reconstruction", "endpoint", "monotonic"
    }


def test_synthetic_cell_episode_and_train_only_scaler(tmp_path) -> None:
    root = write_synthetic_matr_dataset(
        tmp_path, num_cells=4, num_cycles=30, signal_points=80
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
    train_cells = cells[:3]
    scaler = fit_voltage_scaler(train_cells, minimum_position=10)
    assert set(scaler.fit_cell_ids) == {cell.cell_id for cell in train_cells}
    processor = PartialVQProcessor(
        QGridConfig(minimum=0.0, maximum=1.2, num_points=64), 4, 4
    )
    sampler = EpisodeSampler(
        EpisodeConfig(
            minimum_cycle_position=10,
            training_beta_range=[0.2, 0.7],
            evaluation_betas=[0.4],
            evaluation_cycle_alphas=[0.5],
            minimum_observed_points=4,
            minimum_future_points=4,
        ),
        processor,
        scaler,
    )
    episode = sampler.evaluation(cells[3], 0.5, 0.4)
    assert episode.observed_mask.any()
    assert episode.future_mask.any()
    assert not np.any(episode.observed_mask & episode.future_mask)
    assert episode.q_cut < episode.q_end

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from battery_weighted_maml.matr_anp.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PathsConfig,
    QGridConfig,
    SplitConfig,
    resolve_data_root,
)
from battery_weighted_maml.matr_anp.data import (
    CellData,
    DischargeCurve,
    extract_discharge_curve,
    load_matr_dataset,
)
from battery_weighted_maml.matr_anp.episodes import EpisodeSampler, collate_episodes
from battery_weighted_maml.matr_anp.features import FoldScalers, PartialIVProcessor
from battery_weighted_maml.matr_anp.inference import predict_episode
from battery_weighted_maml.matr_anp.losses import anp_elbo_loss
from battery_weighted_maml.matr_anp.model import build_model
from battery_weighted_maml.matr_anp.runtime import parameter_checksum
from battery_weighted_maml.matr_anp.smoke_test import run_smoke
from battery_weighted_maml.matr_anp.splits import make_splits
from battery_weighted_maml.matr_anp.synthetic import write_synthetic_matr_dataset


@pytest.fixture()
def synthetic(tmp_path: Path):
    root = write_synthetic_matr_dataset(
        tmp_path, num_cells=6, num_cycles=28, signal_points=32
    )
    data_config = DataConfig(
        minimum_valid_cycles=24,
        minimum_discharge_points=8,
        short_signal_threshold=12,
    )
    cells, audit = load_matr_dataset(root, data_config)
    processor = PartialIVProcessor(QGridConfig(num_points=32), data_config)
    scalers = FoldScalers.fit(cells[:4], processor, minimum_current_position=9)
    config = ExperimentConfig(data=data_config)
    config.q_grid.num_points = 32
    config.episode.minimum_current_cycle_position = 10
    config.episode.training_alpha_range = [0.3, 0.7]
    config.episode.min_context_points = 4
    config.episode.max_context_points = 16
    config.episode.max_target_points = 16
    return root, cells, audit, processor, scalers, config


def test_data_root_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli, environment, configured = (tmp_path / name for name in ("cli", "env", "config"))
    for path in (cli, environment, configured):
        path.mkdir()
    config = ExperimentConfig(paths=PathsConfig(data_root=str(configured)))
    monkeypatch.setenv("BATTERYLIFE_DATA_ROOT", str(environment))
    assert resolve_data_root(config, str(cli)) == cli.resolve()
    assert resolve_data_root(config, None) == environment.resolve()
    monkeypatch.delenv("BATTERYLIFE_DATA_ROOT")
    assert resolve_data_root(config, None) == configured.resolve()


def test_fixed_grid_reference_beta_and_train_only_scaler(synthetic) -> None:
    _, cells, audit, processor, scalers, _ = synthetic
    assert (audit["status"] == "valid").all()
    cell = cells[0]
    assert np.isclose(cell.cycles[-1].soh, cell.cycles[-1].discharge_capacity_ah / cell.nominal_capacity_ah)
    assert cell.cycles[-1].discharge is not None
    # q is nominal-capacity based: an aged cycle does not end at one by construction.
    assert cell.cycles[-1].discharge.q[-1] < 1.0
    delta, current, valid, references = processor.raw_feature(cell, 20)
    assert references and max(references) < 20
    assert np.median(delta[valid]) < 0.0  # current voltage minus an earlier/higher reference
    assert np.all(current[valid] > 0.0)  # current polarity is converted to magnitude
    beta0 = processor.build(cell, 20, 0.0, scalers)
    beta50 = processor.build(cell, 20, 0.5, scalers)
    assert not beta0.mask.any()
    expected = valid & (processor.grid <= 0.4 + 1.0e-12)
    assert np.array_equal(beta50.mask, expected)
    assert np.all(beta50.values[:2, ~expected] == 0.0)
    assert scalers.fit_cell_ids == sorted(cell.cell_id for cell in cells[:4])
    assert not set(scalers.fit_cell_ids) & {cells[4].cell_id, cells[5].cell_id}
    assert scalers.transform_cycles(np.asarray([scalers.max_cycle_train + 10]))[0] > 1.0


def test_interpolation_never_extrapolates_and_missing_reference_falls_back(synthetic) -> None:
    _, cells, _, _, scalers, _ = synthetic
    processor = PartialIVProcessor(
        QGridConfig(minimum=0.0, maximum=1.1, num_points=23), DataConfig(
            minimum_valid_cycles=24,
            minimum_discharge_points=8,
            reference_cycles=[5, 6, 7, 8, 9, 10],
            minimum_reference_cycles=3,
        )
    )
    curve = DischargeCurve(
        q=np.asarray([0.1, 0.3, 0.5]),
        voltage_v=np.asarray([3.7, 3.5, 3.3]),
        current_a_magnitude=np.asarray([1.0, 1.0, 1.0]),
        original_current_sign=-1,
        monotonic_before_cleanup=True,
        duplicate_q_count=0,
    )
    grid = processor.interpolate(curve)
    assert not grid.mask[0] and not grid.mask[-1]
    assert grid.voltage_v[0] == 0.0 and grid.voltage_v[-1] == 0.0

    cell = cells[0]
    # Remove preferred 5..10 cycles; fallback must use only earlier available cycles.
    fallback = CellData(
        cell.cell_id,
        cell.source_file,
        cell.nominal_capacity_ah,
        tuple(cycle for cycle in cell.cycles if cycle.cycle_number not in range(5, 11)),
    )
    _, _, _, references = processor.raw_feature(fallback, 20)
    assert len(references) >= 3 and max(references) < 20
    assert not set(references) & set(range(5, 11))
    with pytest.raises(ValueError, match="only"):
        extract_discharge_curve(
            {"voltage": [3.5], "current": [-1.0], "discharge_capacity": [0.1]},
            1.1,
            8,
        )


def test_cell_splits_and_episode_boundaries(synthetic) -> None:
    _, cells, _, processor, scalers, config = synthetic
    split_config = SplitConfig(num_folds=3, validation_fraction=0.25, seed=17)
    first = make_splits([cell.cell_id for cell in cells], split_config)
    second = make_splits([cell.cell_id for cell in cells], split_config)
    assert first == second
    for split in first:
        split.validate([cell.cell_id for cell in cells])
        assert not set(split.train_cells) & set(split.test_cells)

    sampler = EpisodeSampler(config.episode, processor, scalers)
    episode = sampler.evaluation(cells[0], 0.5, 0.5)
    context_cycles = np.rint(episode.context_x[:, 0] * scalers.max_cycle_train).astype(int)
    assert np.all(context_cycles < episode.current_cycle)
    assert episode.target_cycles[0] == episode.current_cycle
    assert episode.current_cycle not in context_cycles
    other = sampler.evaluation(cells[1], 0.7, 1.0)
    batch = collate_episodes([episode, other])
    assert batch.context_mask.sum(1).tolist() == [len(episode.context_x), len(other.context_x)]
    assert batch.target_mask.sum(1).tolist() == [len(episode.target_x), len(other.target_x)]


def _small_model_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=16,
        wide_hidden_min=16,
        wide_hidden_max=64,
        latent_dim=8,
        attention_heads=4,
        mlp_layers=2,
        iv_channels=[4, 8],
        iv_embedding_dim=8,
    )


def test_models_loss_padding_and_masked_iv(synthetic) -> None:
    _, cells, _, processor, scalers, config = synthetic
    sampler = EpisodeSampler(config.episode, processor, scalers)
    episodes = [sampler.evaluation(cells[0], 0.3, 0.5), sampler.evaluation(cells[1], 0.7, 0.5)]
    batch = collate_episodes(episodes)
    model_config = _small_model_config()
    for model_name in ("soh_only_anp", "partial_iv_anp"):
        model, _ = build_model(model_name, model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        output = model(
            batch.context_x,
            batch.context_y,
            batch.context_mask,
            batch.target_x,
            target_y=batch.target_y,
            target_mask=batch.target_mask,
            iv_feature=batch.iv_feature,
        )
        assert output["mean"].shape == batch.target_y.shape
        assert output["std"].shape == batch.target_y.shape
        assert torch.all(output["std"] > 0)
        loss = anp_elbo_loss(output, batch.target_y, batch.target_mask, 0.5)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()

    model, _ = build_model("partial_iv_anp", model_config)
    model.eval()
    with torch.no_grad():
        base = model(
            batch.context_x,
            batch.context_y,
            batch.context_mask,
            batch.target_x,
            iv_feature=batch.iv_feature,
            sample_latent=False,
        )["mean"]
        changed_iv = batch.iv_feature.clone()
        unobserved = changed_iv[:, 2:3, :] == 0
        changed_iv[:, :2, :] = torch.where(unobserved, torch.full_like(changed_iv[:, :2, :], 999.0), changed_iv[:, :2, :])
        iv_result = model(
            batch.context_x,
            batch.context_y,
            batch.context_mask,
            batch.target_x,
            iv_feature=changed_iv,
            sample_latent=False,
        )["mean"]
        assert torch.allclose(base, iv_result, atol=1.0e-6)

        padded_x, padded_y = batch.context_x.clone(), batch.context_y.clone()
        padding = ~batch.context_mask
        padded_x[padding] = 999.0
        padded_y[padding] = -999.0
        padded_result = model(
            padded_x,
            padded_y,
            batch.context_mask,
            batch.target_x,
            iv_feature=batch.iv_feature,
            sample_latent=False,
        )["mean"]
        assert torch.allclose(base, padded_result, atol=1.0e-6)


def test_parameter_match_and_inference_does_not_update(synthetic) -> None:
    _, cells, _, processor, scalers, config = synthetic
    model_config = _small_model_config()
    # The production dimensions admit an attention-compatible width within ±5%.
    _, wide = build_model("soh_only_anp_wide", ModelConfig())
    assert wide.parameter_match_relative_error is not None
    assert wide.parameter_match_relative_error <= 0.05
    model, _ = build_model("partial_iv_anp", model_config)
    sampler = EpisodeSampler(config.episode, processor, scalers)
    episode = sampler.evaluation(cells[0], 0.5, 0.0)
    checksum = parameter_checksum(model)
    result = predict_episode(
        model,
        episode,
        scalers,
        torch.device("cpu"),
        mc_samples=3,
        interval_level=0.95,
        seed=42,
    )
    assert result.mean.shape == episode.target_soh_raw.shape
    assert result.standard_deviation.shape == episode.target_soh_raw.shape
    assert np.isfinite(result.mean).all()
    assert checksum == parameter_checksum(model)


def test_end_to_end_smoke(tmp_path: Path) -> None:
    destination = run_smoke(tmp_path / "smoke", "cpu")
    assert (destination / "smoke_manifest.json").is_file()
    assert (destination / "evaluation/aggregate_metrics.csv").is_file()
    assert (destination / "streaming/streaming_manifest.json").is_file()

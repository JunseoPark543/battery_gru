from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
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
from battery_weighted_maml.matr_anp.context_streaming import (
    _aggregate,
    _unique_betas,
    cycle_schedule,
)
from battery_weighted_maml.matr_anp.data import (
    CellData,
    DischargeCurve,
    extract_discharge_curve,
    load_dataset,
    load_matr_dataset,
)
from battery_weighted_maml.matr_anp.episodes import EpisodeSampler, collate_episodes
from battery_weighted_maml.matr_anp.features import FoldScalers, PartialIVProcessor
from battery_weighted_maml.matr_anp.inference import predict_episode
from battery_weighted_maml.matr_anp.losses import anp_elbo_loss
from battery_weighted_maml.matr_anp.model import build_model
from battery_weighted_maml.matr_anp.plotting import plot_context_streaming_summary
from battery_weighted_maml.matr_anp.plot_data_trajectories import (
    matr_batch,
    plot_trajectories,
    trajectory_frame,
    trajectory_summary,
)
from battery_weighted_maml.matr_anp.plot_voltage_cycles import (
    plot_voltage_grid,
    select_cells,
)
from battery_weighted_maml.matr_anp.plot_cycle_time_fraction_voltage import (
    evenly_spaced_time_fractions,
    load_timed_discharge,
    plot_time_fraction_voltage_q,
    time_fraction_prefix,
)
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
    _, cells, audit, processor, scalers, config = synthetic
    assert (audit["status"] == "valid").all()
    cell = cells[0]
    assert np.isclose(cell.cycles[-1].soh, cell.cycles[-1].discharge_capacity_ah / cell.nominal_capacity_ah)
    assert cell.cycles[-1].discharge is not None
    # q is nominal-capacity based: an aged cycle does not end at one by construction.
    assert cell.cycles[-1].discharge.q[-1] < 1.0
    delta, current, valid, references = processor.raw_feature(cell, 20)
    cache_after_first = processor.cache_info()
    cached_delta, cached_current, cached_valid, cached_references = (
        processor.raw_feature(cell, 20)
    )
    assert cached_delta is delta
    assert cached_current is current
    assert cached_valid is valid
    assert cached_references == references
    assert processor.cache_info() == cache_after_first
    assert not delta.flags.writeable and not current.flags.writeable
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

    sampler = EpisodeSampler(config=config.episode, processor=processor, scalers=scalers)
    first_positions = sampler._eligible_positions(cell)
    second_positions = sampler._eligible_positions(cell)
    assert first_positions is second_positions
    assert len(sampler._eligible_positions_cache) == 1


def test_calce_uses_the_same_cell_and_partial_iv_pipeline(tmp_path: Path) -> None:
    matr_root = write_synthetic_matr_dataset(
        tmp_path, num_cells=4, num_cycles=28, signal_points=32
    )
    calce_root = tmp_path / "CALCE"
    calce_root.mkdir()
    for index, source in enumerate(sorted(matr_root.glob("*.pkl"))):
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        payload["dataset"] = "CALCE"
        payload["cell_id"] = f"CALCE_SYNTH_{index:02d}"
        destination = calce_root / f"CALCE_SYNTH_{index:02d}.pkl"
        with destination.open("wb") as handle:
            pickle.dump(payload, handle)

    data_config = DataConfig(
        dataset="CALCE",
        minimum_valid_cycles=24,
        minimum_discharge_points=8,
        short_signal_threshold=12,
    )
    config = ExperimentConfig(data=data_config)
    config.validate()
    cells, audit = load_dataset(calce_root, data_config)
    assert len(cells) == 4
    assert (audit["status"] == "valid").all()
    assert all(cell.cell_id.startswith("CALCE_") for cell in cells)

    processor = PartialIVProcessor(QGridConfig(num_points=32), data_config)
    scalers = FoldScalers.fit(cells[:3], processor, minimum_current_position=9)
    config.episode.minimum_current_cycle_position = 10
    config.episode.training_alpha_range = [0.3, 0.7]
    config.episode.min_context_points = 4
    config.episode.max_context_points = 16
    config.episode.max_target_points = 16
    episode = EpisodeSampler(config.episode, processor, scalers).evaluation(
        cells[-1], 0.5, 0.5
    )
    assert episode.iv_feature.shape == (3, 32)
    assert episode.iv_feature[2].sum() > 0


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


def test_absolute_cycle_streaming_context_and_schedules(synthetic) -> None:
    _, cells, _, processor, scalers, config = synthetic
    sampler = EpisodeSampler(config.episode, processor, scalers)
    episode = sampler.evaluation_after_cycle(cells[0], 10, 0.0)
    context_cycles = np.rint(
        episode.context_x[:, 0] * scalers.max_cycle_train
    ).astype(int)
    assert context_cycles.tolist() == list(range(1, 11))
    assert episode.current_cycle == 11
    assert episode.target_cycles[0] == 11
    capped = sampler.evaluation_after_cycle(cells[0], 20, 0.0)
    assert len(capped.context_x) == config.episode.max_context_points
    assert capped.current_cycle == 21
    assert cycle_schedule(100, 125, 10) == [100, 110, 120]
    assert cycle_schedule(100, 103, 1) == [100, 101, 102, 103]
    assert _unique_betas([0.0, 0.25, 0.25, 1.0]) == [0.0, 0.25, 1.0]
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _unique_betas([1.01])


def test_multibeta_streaming_aggregation_and_plot(tmp_path: Path) -> None:
    rows = []
    for beta, rmse in ((0.0, 0.05), (0.25, 0.04)):
        for cutoff in (100, 101):
            rows.append(
                {
                    "status": "ok",
                    "schedule": "step1",
                    "cycle_step": 1,
                    "beta": beta,
                    "requested_observed_cycle": cutoff,
                    "cell_id": "MATR_test",
                    "future_rmse": rmse,
                    "current_soh_abs_error": 0.001,
                    "nll": -2.0,
                    "coverage_95": 0.9,
                    "interval_width_95": 0.1,
                    "num_context_points": cutoff,
                    "num_available_context_points": cutoff,
                    "num_target_points": 200 - cutoff,
                }
            )
    aggregate = _aggregate(pd.DataFrame(rows))
    assert len(aggregate) == 4
    assert sorted(aggregate["beta"].unique()) == [0.0, 0.25]
    destination = tmp_path / "multibeta_summary.png"
    plot_context_streaming_summary(aggregate, destination)
    assert destination.is_file()


def test_matr_data_trajectory_plot(synthetic, tmp_path: Path) -> None:
    _, cells, _, _, _, _ = synthetic
    trajectories = trajectory_frame(cells)
    assert trajectories["cell_id"].nunique() == len(cells)
    assert matr_batch("MATR_b3c17") == "b3"
    assert matr_batch(cells[0].cell_id) == "other"
    summary = trajectory_summary(trajectories, eol_threshold=0.9)
    assert len(summary) == len(cells)
    assert summary["last_cycle"].eq(28).all()
    destination = plot_trajectories(
        trajectories,
        tmp_path / "matr_trajectories.png",
        x_axis="cycle",
        eol_threshold=0.9,
        highlight_cells=[cells[0].cell_id],
        dpi=80,
    )
    assert destination.is_file()


def test_random_cell_all_cycle_voltage_grid(synthetic, tmp_path: Path) -> None:
    _, cells, _, _, _, _ = synthetic
    first = select_cells(cells, count=4, seed=42)
    second = select_cells(cells, count=4, seed=42)
    assert [cell.cell_id for cell in first] == [cell.cell_id for cell in second]
    assert len({cell.cell_id for cell in first}) == 4

    destination = tmp_path / "matr_voltage_cycles.png"
    summary = plot_voltage_grid(
        first,
        destination,
        columns=2,
        q_min=0.0,
        q_max=0.8,
        dpi=80,
    )
    assert destination.is_file()
    assert len(summary) == 4
    assert summary["plotted_cycles"].gt(0).all()


def test_cycle_time_fraction_voltage_q_plot(synthetic, tmp_path: Path) -> None:
    _, cells, _, _, _, _ = synthetic
    curve = load_timed_discharge(cells[0], 20)
    prefix = time_fraction_prefix(curve, 0.3)
    assert np.isclose(prefix.elapsed_time_s[-1], 0.3 * curve.elapsed_time_s[-1])
    assert prefix.q[-1] < curve.q[-1]
    fractions = evenly_spaced_time_fractions(5)
    assert np.allclose(fractions, [0.2, 0.4, 0.6, 0.8, 1.0])

    destination = tmp_path / "cycle20_voltage_q_time_fraction.png"
    summary = plot_time_fraction_voltage_q(
        curve,
        destination,
        cell_id=cells[0].cell_id,
        cycle_number=20,
        fractions=fractions,
        q_limits=(0.0, 0.8),
        voltage_limits=(2.0, 3.8),
        dpi=80,
    )
    assert destination.is_file()
    assert np.allclose(summary["time_fraction"], fractions)
    assert summary["cutoff_q"].is_monotonic_increasing


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

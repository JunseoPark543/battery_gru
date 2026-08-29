from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from battery_weighted_maml.horizon_lifetime_iv_anp.config import (
    DataConfig,
    EvaluationConfig,
    LifetimeIVConfig,
    ModelConfig,
    TaskConfig,
    TrainingConfig,
)
from battery_weighted_maml.horizon_lifetime_iv_anp.data import (
    LifetimeIVPrefixStore,
    LifetimeIVScalers,
    load_labeled_cells,
)
from battery_weighted_maml.horizon_lifetime_iv_anp.evaluate import evaluate_checkpoint
from battery_weighted_maml.horizon_lifetime_iv_anp.inference import predict_batch
from battery_weighted_maml.horizon_lifetime_iv_anp.losses import lifetime_elbo
from battery_weighted_maml.horizon_lifetime_iv_anp.model import build_model
from battery_weighted_maml.horizon_lifetime_iv_anp.tasks import (
    LifetimeTaskSampler,
    collate_tasks,
)
from battery_weighted_maml.horizon_lifetime_iv_anp.train import train_run
from battery_weighted_maml.matr_anp.config import QGridConfig
from battery_weighted_maml.matr_anp.splits import make_splits
from battery_weighted_maml.matr_anp.synthetic import write_synthetic_matr_dataset


def _fixture(tmp_path: Path):
    root = write_synthetic_matr_dataset(
        tmp_path, num_cells=12, num_cycles=36, signal_points=24
    )
    label_path = tmp_path / "MATR_labels.json"
    label_path.write_text(json.dumps({
        path.name: 28 + index % 9
        for index, path in enumerate(sorted(root.glob("*.pkl")))
    }), encoding="utf-8")
    config = LifetimeIVConfig(
        data=DataConfig(
            minimum_valid_cycles=24,
            minimum_discharge_points=8,
            short_signal_threshold=12,
            lifetime_source="label_file",
            label_path=str(label_path),
        ),
        q_grid=QGridConfig(minimum=0.0, maximum=1.2, num_points=256),
        task=TaskConfig(
            horizons=[5, 10], context_size_min=2, context_size_max=3,
            query_size=2, min_cells_per_task=4, max_resample_attempts=30,
        ),
        model=ModelConfig(
            curve_d_model=16, curve_attention_heads=4, curve_layers=1,
            curve_patch_size=32, temporal_d_model=32,
            temporal_attention_heads=4, temporal_layers=1, dropout=0.0,
            latent_dim=8, anp_mlp_layers=2,
            gradient_checkpoint_curves=False,
        ),
        training=TrainingConfig(
            max_steps=1, task_batch_size=1, kl_warmup_steps=1,
            validation_interval=1, early_stopping_patience=2,
            checkpoint_interval=1, log_interval=1, use_amp=False,
        ),
        evaluation=EvaluationConfig(
            horizons=[5, 10], context_size=3, mc_samples=2,
        ),
    )
    config.split.num_folds = 3
    config.split.validation_fraction = 0.25
    config.validate()
    cells, audit = load_labeled_cells(root, config.data)  # type: ignore[arg-type]
    split = make_splits([item.cell_id for item in cells], config.split)[0]
    by_id = {item.cell_id: item for item in cells}
    train = [by_id[value] for value in split.train_cells]
    validation = [by_id[value] for value in split.validation_cells]
    test = [by_id[value] for value in split.test_cells]
    scalers = LifetimeIVScalers.fit(train, max(config.task.horizons))
    store = LifetimeIVPrefixStore(scalers, config.q_grid, max(config.task.horizons))
    sampler = LifetimeTaskSampler(config.task, scalers, store)
    return root, config, audit, split, train, validation, test, scalers, sampler


def _arguments(batch):
    return (
        batch.context_cycles, batch.context_cycle_mask,
        batch.context_curves, batch.context_curve_mask,
        batch.context_point_mask, batch.context_y,
        batch.query_cycles, batch.query_cycle_mask,
        batch.query_curves, batch.query_curve_mask,
        batch.query_point_mask,
    )


def test_lifetime_context_and_256_point_iv_prefix(tmp_path: Path) -> None:
    _, config, audit, split, train, _, _, scalers, sampler = _fixture(tmp_path)
    assert (audit["label_status"] == "valid").all()
    assert set(scalers.fit_cell_ids) == set(split.train_cells)
    task = sampler.sample_training(train, np.random.default_rng(42))
    task.validate()
    assert {point.cell_id for point in task.context}.isdisjoint(
        point.cell_id for point in task.query
    )
    for point in (*task.context, *task.query):
        assert point.curves.shape[1:] == (256, 3)
        assert point.curve_masks.shape == point.curves.shape[:2]
        assert point.cycle_features.shape[1] == 2
        assert np.isclose(
            point.lifetime_normalized,
            scalers.transform_lifetime(point.lifetime_cycles),
        )
    assert config.model.curve_input_dim == 3

    early, late = sampler.sample_training_pair(
        train, np.random.default_rng(43), horizon_gap=5
    )
    assert (early.horizon, late.horizon) == (5, 10)
    assert [point.cell_id for point in early.context] == [
        point.cell_id for point in late.context
    ]
    assert [point.cell_id for point in early.query] == [
        point.cell_id for point in late.query
    ]
    assert np.allclose(
        [point.lifetime_cycles for point in early.query],
        [point.lifetime_cycles for point in late.query],
    )


def test_hierarchical_attention_lifetime_output_and_no_query_label_leakage(
    tmp_path: Path,
) -> None:
    _, config, _, _, train, validation, _, scalers, sampler = _fixture(tmp_path)
    task = sampler.sample_training(train, np.random.default_rng(7))
    batch = collate_tasks([task])
    model, spec = build_model(config.model, 256)
    output = model(*_arguments(batch), query_y=batch.query_y, return_representations=True)
    assert output["mean"].shape == batch.query_y.shape
    assert output["context_h"].shape[-1] == config.model.temporal_d_model
    assert output["query_h"].shape[-1] == config.model.temporal_d_model
    assert spec.q_points == 256 and spec.algorithm.endswith("lifetime_iv_anp")
    assert (output["std"] > 0).all()
    loss = lifetime_elbo(output, batch.query_y, batch.query_point_mask, 0.5)
    assert all(torch.isfinite(value) for value in loss.values())
    loss["loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    evaluation_task = sampler.evaluation(
        10, train, validation, context_size=3, seed=42
    )
    evaluation_batch = collate_tasks([evaluation_task])
    model.eval()
    with torch.no_grad():
        first = model(*_arguments(evaluation_batch), sample_latent=False)["mean"]
        evaluation_batch.query_y.add_(999.0)
        second = model(*_arguments(evaluation_batch), sample_latent=False)["mean"]
    assert torch.equal(first, second)
    prediction = predict_batch(
        model, evaluation_batch, scalers,
        mc_samples=2, interval_level=0.95, seed=42,
    )
    assert np.isfinite(prediction.lifetime_mean).all()
    assert np.allclose(
        prediction.rul_mean,
        prediction.lifetime_mean - evaluation_batch.horizons.numpy()[:, None],
    )


def test_one_step_train_and_held_out_evaluation(tmp_path: Path) -> None:
    root, config, _, _, _, _, _, _, _ = _fixture(tmp_path)
    config.training.paired_horizon_training = True
    config.training.consistency_weight = 0.1
    config.training.consistency_horizon_gap = 5
    config.training.consistency_warmup_steps = 1
    config.validate()
    run_dir = train_run(
        config, 0, root, max_steps=1, output_root=tmp_path / "runs"
    )
    assert (run_dir / "checkpoints/best.pt").is_file()
    assert (run_dir / "checkpoints/last.pt").is_file()
    history = pd.read_csv(run_dir / "training/history.csv")
    assert len(history) == 1
    assert np.isfinite(history.loc[0, "consistency_loss"])
    assert np.isclose(history.loc[0, "consistency_weight"], 0.1)
    assert "->" in history.loc[0, "horizons"]
    destination = evaluate_checkpoint(
        config, run_dir / "checkpoints/best.pt", root,
        horizons=[5], mc_samples=2,
    )
    aggregate = pd.read_csv(destination / "aggregate_metrics.csv")
    predictions = pd.read_csv(destination / "per_cell_predictions.csv")
    assert (aggregate["status"] == "ok").all()
    assert not predictions["query_label_used_as_input"].any()
    assert np.allclose(
        predictions["predicted_rul_mean_cycles"],
        predictions["predicted_lifetime_mean_cycles"] - predictions["horizon"],
    )
    assert (destination / "plots/metrics_by_horizon.png").is_file()

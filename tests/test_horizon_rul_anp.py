from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from battery_weighted_maml.horizon_rul_anp.config import (
    DataConfig,
    EvaluationConfig,
    HorizonRULConfig,
    ModelConfig,
    TaskConfig,
    TrainingConfig,
)
from battery_weighted_maml.horizon_rul_anp.data import (
    RULScalers,
    load_labeled_cells,
)
from battery_weighted_maml.horizon_rul_anp.evaluate import (
    LoadedExperiment,
    evaluate_checkpoint,
)
from battery_weighted_maml.horizon_rul_anp.inference import predict_batch
from battery_weighted_maml.horizon_rul_anp.losses import horizon_rul_elbo
from battery_weighted_maml.horizon_rul_anp.model import build_model
from battery_weighted_maml.horizon_rul_anp.streaming import predict_streaming
from battery_weighted_maml.horizon_rul_anp.tasks import (
    HorizonTaskSampler,
    collate_tasks,
)
from battery_weighted_maml.horizon_rul_anp.train import train_run
from battery_weighted_maml.matr_anp.runtime import parameter_checksum
from battery_weighted_maml.matr_anp.splits import make_splits
from battery_weighted_maml.matr_anp.synthetic import write_synthetic_matr_dataset


def _fixture(tmp_path: Path):
    root = write_synthetic_matr_dataset(
        tmp_path,
        num_cells=12,
        num_cycles=36,
        signal_points=24,
    )
    label_path = tmp_path / "MATR_labels.json"
    labels = {
        path.name: 28 + index % 9
        for index, path in enumerate(sorted(root.glob("*.pkl")))
    }
    label_path.write_text(json.dumps(labels), encoding="utf-8")
    data = DataConfig(
        minimum_valid_cycles=24,
        minimum_discharge_points=8,
        short_signal_threshold=12,
        lifetime_source="label_file",
        label_path=str(label_path),
    )
    task = TaskConfig(
        min_horizon=5,
        max_horizon=15,
        context_size_min=2,
        context_size_max=3,
        query_size=2,
        min_cells_per_task=4,
        max_resample_attempts=30,
    )
    model = ModelConfig(
        prefix_feature_dim=3,
        d_model=32,
        attention_heads=4,
        prefix_layers=1,
        dropout=0.0,
        latent_dim=8,
        mlp_layers=2,
    )
    training = TrainingConfig(
        max_steps=2,
        task_batch_size=1,
        kl_warmup_steps=1,
        validation_interval=1,
        validation_horizons=[5, 10, 15],
        early_stopping_patience=2,
        checkpoint_interval=1,
        log_interval=1,
        use_amp=False,
    )
    evaluation = EvaluationConfig(
        horizons=[5, 10, 15],
        context_size=3,
        mc_samples=2,
    )
    config = HorizonRULConfig(
        data=data,
        task=task,
        model=model,
        training=training,
        evaluation=evaluation,
    )
    config.split.num_folds = 3
    config.split.validation_fraction = 0.25
    config.validate()
    cells, audit = load_labeled_cells(root, config.data)
    splits = make_splits([item.cell_id for item in cells], config.split)
    split = splits[0]
    by_id = {item.cell_id: item for item in cells}
    train = [by_id[cell_id] for cell_id in split.train_cells]
    validation = [by_id[cell_id] for cell_id in split.validation_cells]
    test = [by_id[cell_id] for cell_id in split.test_cells]
    scalers = RULScalers.fit(train, config.task)
    sampler = HorizonTaskSampler(config.task, scalers)
    return root, config, cells, audit, split, train, validation, test, scalers, sampler


def test_label_mapping_cell_split_and_horizon_task(tmp_path: Path) -> None:
    _, config, cells, audit, split, train, validation, test, scalers, sampler = (
        _fixture(tmp_path)
    )
    assert len(cells) == 12
    assert (audit["label_status"] == "valid").all()
    assert set(split.train_cells).isdisjoint(split.validation_cells)
    assert set(split.train_cells).isdisjoint(split.test_cells)
    assert set(scalers.fit_cell_ids) == set(split.train_cells)

    task = sampler.sample_training(train, np.random.default_rng(42))
    task.validate()
    assert {point.horizon for point in (*task.context, *task.query)} == {
        task.horizon
    }
    assert {point.cell_id for point in task.context}.isdisjoint(
        point.cell_id for point in task.query
    )
    for point in (*task.context, *task.query):
        assert point.rul_cycles == point.lifetime - task.horizon
        assert point.prefix.shape[1] == config.model.prefix_feature_dim
        assert point.prefix.shape[0] == task.horizon
    short_lived = min(cells, key=lambda item: item.lifetime)
    assert short_lived not in sampler.eligible(cells, short_lived.lifetime)

    evaluation_task = sampler.evaluation(
        10,
        train,
        validation,
        context_size=3,
        seed=42,
    )
    assert {point.cell_id for point in evaluation_task.context} <= set(
        split.train_cells
    )
    assert {point.cell_id for point in evaluation_task.query} <= set(
        split.validation_cells
    )


def test_model_shapes_query_only_elbo_and_no_query_label_inference(tmp_path: Path) -> None:
    _, config, _, _, _, train, validation, _, scalers, sampler = _fixture(tmp_path)
    training_task = sampler.sample_training(train, np.random.default_rng(7))
    batch = collate_tasks([training_task])
    model, spec = build_model(config.model)
    output = model(
        batch.context_prefix,
        batch.context_prefix_mask,
        batch.context_mask,
        batch.context_y,
        batch.query_prefix,
        batch.query_prefix_mask,
        batch.query_mask,
        query_y=batch.query_y,
        return_representations=True,
    )
    assert output["mean"].shape == batch.query_y.shape
    assert output["std"].shape == batch.query_y.shape
    assert (output["std"] > 0).all()
    assert output["context_x"].shape[-1] == config.model.d_model
    assert output["query_x"].shape[-1] == config.model.d_model
    assert spec.parameter_count == sum(p.numel() for p in model.parameters())
    attention_sum = output["inter_cell_attention"].sum(dim=-1)
    assert torch.allclose(attention_sum[batch.query_mask], torch.ones_like(attention_sum[batch.query_mask]), atol=1e-5)
    losses = horizon_rul_elbo(output, batch.query_y, batch.query_mask, 0.5)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
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
        first = model(
            evaluation_batch.context_prefix,
            evaluation_batch.context_prefix_mask,
            evaluation_batch.context_mask,
            evaluation_batch.context_y,
            evaluation_batch.query_prefix,
            evaluation_batch.query_prefix_mask,
            evaluation_batch.query_mask,
            sample_latent=False,
        )["mean"]
        evaluation_batch.query_y.add_(999.0)
        second = model(
            evaluation_batch.context_prefix,
            evaluation_batch.context_prefix_mask,
            evaluation_batch.context_mask,
            evaluation_batch.context_y,
            evaluation_batch.query_prefix,
            evaluation_batch.query_prefix_mask,
            evaluation_batch.query_mask,
            sample_latent=False,
        )["mean"]
    assert torch.equal(first, second)
    prediction = predict_batch(
        model,
        evaluation_batch,
        scalers,
        mc_samples=2,
        interval_level=0.95,
        seed=42,
    )
    assert np.isfinite(prediction.mean_cycles).all()
    assert (prediction.std_cycles > 0).all()


def test_checkpoint_roundtrip_and_streaming_without_parameter_updates(
    tmp_path: Path,
) -> None:
    _, config, _, audit, split, train, validation, test, scalers, sampler = (
        _fixture(tmp_path)
    )
    model, spec = build_model(config.model)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded, reloaded_spec = build_model(config.model)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert spec == reloaded_spec
    assert parameter_checksum(model) == parameter_checksum(reloaded)

    experiment = LoadedExperiment(
        config=config,
        model=reloaded,
        device=torch.device("cpu"),
        scalers=scalers,
        sampler=sampler,
        train_cells=train,
        validation_cells=validation,
        test_cells=test,
        audit=audit,
        payload={"fold_split": {"fold": split.fold}},
    )
    before = parameter_checksum(reloaded)
    frame = predict_streaming(
        experiment,
        test[0],
        [5, 6, 7],
        mc_samples=2,
    )
    assert (frame["status"] == "ok").all()
    assert list(frame["horizon"]) == [5, 6, 7]
    assert not frame["query_label_used_as_input"].any()
    assert not frame["model_parameter_update"].any()
    assert parameter_checksum(reloaded) == before


def test_two_step_train_and_held_out_evaluation_smoke(tmp_path: Path) -> None:
    root, config, _, _, _, _, _, _, _, _ = _fixture(tmp_path)
    run_dir = train_run(
        config,
        0,
        root,
        max_steps=2,
        output_root=tmp_path / "runs",
    )
    assert (run_dir / "checkpoints/best.pt").is_file()
    assert (run_dir / "checkpoints/last.pt").is_file()
    history = pd.read_csv(run_dir / "training/history.csv")
    assert len(history) == 2
    destination = evaluate_checkpoint(
        config,
        run_dir / "checkpoints/best.pt",
        root,
        horizons=[5, 10],
        mc_samples=2,
    )
    assert (destination / "aggregate_metrics.csv").is_file()
    assert (destination / "per_cell_predictions.csv").is_file()
    assert (destination / "plots/metrics_by_horizon.png").is_file()

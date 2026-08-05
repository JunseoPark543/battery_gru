from __future__ import annotations

import copy
import logging

import higher
import numpy as np
import pytest
import torch

from paper_reproduction.adapt_and_test import _evaluate_one, evaluate_test_tasks
from paper_reproduction.config import ExperimentConfig
from paper_reproduction.data import (
    CellTask,
    RecursivePairDataset,
    build_recursive_pairs,
    sample_support_batch,
    variable_length_collate,
)
from paper_reproduction.losses import masked_mae, masked_mse
from paper_reproduction.maml_train import (
    _model_loss,
    _task_post_adaptation_loss,
    evaluate_meta_objective_deterministic,
    load_meta_checkpoint,
    train_meta_model,
)
from paper_reproduction.metrics import evaluate_prediction, last_hitting_eol, soh_metrics
from paper_reproduction.model import GRUEncoderDecoder


def test_paper_model_parameter_count_and_batch_shapes():
    model = GRUEncoderDecoder(hidden_size=64, num_layers=1)
    assert sum(parameter.numel() for parameter in model.parameters()) == 25_793
    history = torch.tensor(
        [[[1.0], [0.98], [0.96]], [[1.01], [0.99], [0.0]]]
    )
    lengths = torch.tensor([3, 2])
    target = torch.zeros(2, 4, 1)
    output = model(
        history,
        input_lengths=lengths,
        target=target,
        predicted_input_probability=0.5,
        generator=torch.Generator().manual_seed(42),
    )
    assert output.shape == (2, 4, 1)
    assert torch.isfinite(output).all()
    single = model.recursive_forecast([1.0, 0.98], 3)
    assert single.shape == (1, 3, 1)


def test_predicted_input_probability_has_paper_semantics():
    model = GRUEncoderDecoder(hidden_size=4)
    model.train()
    history = torch.tensor([[[1.0], [0.9]]])
    target = torch.full((1, 4, 1), 0.75)
    ground_truth_driven = model(
        history, target=target, predicted_input_probability=0.0
    )
    prediction_driven = model(
        history, target=target, predicted_input_probability=1.0
    )
    torch.testing.assert_close(ground_truth_driven[:, :1], prediction_driven[:, :1])
    assert not torch.allclose(ground_truth_driven[:, 1:], prediction_driven[:, 1:])
    model.eval()
    eval_with_target = model(
        history, target=target, predicted_input_probability=0.0
    )
    torch.testing.assert_close(eval_with_target, prediction_driven)


def test_recursive_pairs_padding_masks_and_losses():
    pairs = build_recursive_pairs([1.0, 0.9, 0.8, 0.7])
    assert len(pairs) == 3
    batch = variable_length_collate([pairs[0], pairs[2]])
    assert batch["history"].shape == (2, 3, 1)
    assert batch["target"].shape == (2, 3, 1)
    assert batch["input_lengths"].tolist() == [1, 3]
    assert batch["target_mask"].tolist() == [
        [True, True, True], [True, False, False]
    ]
    prediction = torch.tensor([[[2.0], [100.0]], [[4.0], [100.0]]])
    target = torch.tensor([[[1.0], [0.0]], [[2.0], [0.0]]])
    mask = torch.tensor([[True, False], [True, False]])
    assert torch.isclose(masked_mse(prediction, target, mask), torch.tensor(2.5))
    assert torch.isclose(masked_mae(prediction, target, mask), torch.tensor(1.5))


def test_masked_loss_padding_shape_and_nan_guards():
    prediction = torch.tensor([[[2.0], [50.0]], [[4.0], [60.0]]])
    target = torch.tensor([[[1.0], [-10.0]], [[2.0], [-20.0]]])
    mask_2d = torch.tensor([[True, False], [True, False]])
    expected = masked_mse(prediction, target, mask_2d)
    changed_padding_prediction = prediction.clone()
    changed_padding_target = target.clone()
    changed_padding_prediction[:, 1] = -9999.0
    changed_padding_target[:, 1] = 9999.0
    assert torch.equal(
        masked_mse(changed_padding_prediction, changed_padding_target, mask_2d), expected
    )
    mask_3d = mask_2d.unsqueeze(-1)
    assert torch.equal(masked_mse(prediction, target, mask_3d), expected)
    changed_valid = target.clone()
    changed_valid[0, 0, 0] = 10.0
    assert not torch.equal(masked_mse(prediction, changed_valid, mask_2d), expected)
    with pytest.raises(ValueError, match="no valid"):
        masked_mse(prediction, target, torch.zeros_like(mask_2d))
    with pytest.raises(ValueError, match="share shape"):
        masked_mse(prediction[:, :1], target, mask_2d)
    nan_prediction = prediction.clone()
    nan_prediction[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN"):
        masked_mse(nan_prediction, target, mask_2d)


def test_last_hitting_ignores_early_oscillating_crossing():
    cycles = np.arange(500, 505)
    soh = np.array([0.75, 0.69, 0.72, 0.68, 0.67])
    assert last_hitting_eol(cycles, soh, 0.70) == 502


def test_last_hitting_eol_edge_cases():
    assert last_hitting_eol([1, 2, 3, 4], [0.80, 0.72, 0.69, 0.68], 0.70) == 2
    assert last_hitting_eol([1, 2, 3], [0.80, 0.75, 0.71], 0.70) == 3
    assert last_hitting_eol([1, 2, 3], [0.70, 0.69, 0.60], 0.70) is None
    assert last_hitting_eol([1, 2, 3, 4], [0.80, np.nan, 0.72, 0.69], 0.70) == 3
    with np.testing.assert_raises(ValueError):
        last_hitting_eol([1, 2], [0.8], 0.70)
    with np.testing.assert_raises(ValueError):
        last_hitting_eol([], [], 0.70)


def test_soh_metrics_return_fraction_and_percent_units():
    result = soh_metrics(np.array([1.0, 0.9]), np.array([0.99, 0.88]))
    assert result["mae_percent"] == pytest.approx(100.0 * result["mae"])
    assert result["rmse_percent"] == pytest.approx(100.0 * result["rmse"])


def test_last_hitting_rul_error_is_actual_minus_predicted():
    result = evaluate_prediction(
        actual_cycles=np.array([1, 2, 3, 4, 5]),
        actual_soh=np.array([0.80, 0.72, 0.69, 0.68, 0.67]),
        forecast_cycles=np.array([2, 3, 4, 5]),
        forecast_soh=np.array([0.76, 0.71, 0.69, 0.68]),
        current_cycle=1,
        current_soh=0.80,
        threshold=0.70,
    )
    assert result["actual_eol_cycle_last_hitting"] == 2
    assert result["predicted_eol_cycle_last_hitting"] == 3
    assert result["actual_rul"] == 1
    assert result["predicted_rul"] == 2
    assert result["rul_error_actual_minus_predicted"] == -1


class _RecordingForecastModel:
    def __init__(self) -> None:
        self.horizons: list[int] = []

    def recursive_forecast(self, history, prediction_length):
        self.horizons.append(prediction_length)
        return torch.full((1, prediction_length, 1), 0.65)


def test_paper_horizon_uses_query_length_with_irregular_cycles(tmp_path):
    task = CellTask(
        "irregular.pkl",
        np.array([1, 2, 4, 7, 8, 12]),
        np.array([1.0, 0.95, 0.90, 0.82, 0.74, 0.68]),
    )
    config = _small_config()
    config.evaluation.forecast_mode = "paper"
    model = _RecordingForecastModel()
    _evaluate_one(model, task, 2, "paper_test", config, tmp_path)
    assert model.horizons == [4]
    frame = np.genfromtxt(
        tmp_path / "irregular/predictions/paper_test.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    np.testing.assert_array_equal(frame["cycle"][-4:], [4, 7, 8, 12])


def test_deployment_horizon_remains_available(tmp_path):
    task = _task("deployment.pkl")
    config = _small_config()
    config.evaluation.forecast_mode = "deployment"
    config.evaluation.max_prediction_length = 3
    model = _RecordingForecastModel()
    _evaluate_one(model, task, 4, "deployment_test", config, tmp_path)
    assert model.horizons == [3]


def _small_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.device = "cpu"
    config.data.history_length = 4
    config.data.train_cells = [f"train_{index}.pkl" for index in range(5)]
    config.data.test_cells = ["test_0.pkl", "test_1.pkl"]
    config.model.hidden_size = 4
    config.maml.max_epochs = 1
    config.maml.inner_batch_size = 2
    config.maml.early_stopping = False
    config.adaptation.batch_size = 2
    config.adaptation.fast_steps = [0, 1]
    config.adaptation.complete_max_steps = 2
    config.adaptation.complete_patience = 2
    config.evaluation.max_forecast_cycle = 12
    config.validate()
    return config


def _task(name: str, offset: float = 0.0) -> CellTask:
    soh = np.linspace(1.0 + offset, 0.65 + offset, 10)
    return CellTask(name, np.arange(1, 11), soh)


def test_inner_update_keeps_outer_second_order_gradient():
    config = _small_config()
    model = GRUEncoderDecoder(hidden_size=4)
    task = _task(config.data.train_cells[0])
    support = task.soh[:config.data.history_length]
    dataset = RecursivePairDataset(support)
    batch = sample_support_batch(
        dataset,
        config.maml.inner_batch_size,
        torch.Generator().manual_seed(42),
        torch.device("cpu"),
    )
    inner_optimizer = torch.optim.SGD(
        model.parameters(), lr=config.maml.inner_learning_rate
    )
    with higher.innerloop_ctx(
        model,
        inner_optimizer,
        copy_initial_weights=False,
        track_higher_grads=True,
    ) as (task_model, differentiable_optimizer):
        before_inner = [parameter.detach().clone() for parameter in task_model.parameters()]
        support_loss = _model_loss(
            task_model, batch, masked_mse, 0.5, torch.Generator().manual_seed(7)
        )
        differentiable_optimizer.step(support_loss)
        after_inner = [parameter.detach().clone() for parameter in task_model.parameters()]
        assert any(
            not torch.equal(before, after)
            for before, after in zip(before_inner, after_inner)
        )

    outer_optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    before_outer = [parameter.detach().clone() for parameter in model.parameters()]
    task_query_losses = []
    for task_index, train_name in enumerate(config.data.train_cells):
        _, query_loss = _task_post_adaptation_loss(
            model,
            _task(train_name, task_index * 0.001),
            config,
            epoch=1,
            task_index=task_index,
            device=torch.device("cpu"),
            loss_function=masked_mse,
        )
        task_query_losses.append(query_loss)
    meta_loss = torch.stack(task_query_losses).mean()
    outer_optimizer.zero_grad(set_to_none=True)
    meta_loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    outer_optimizer.step()
    after_outer = [parameter.detach().clone() for parameter in model.parameters()]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_outer, after_outer)
    )


def test_single_task_second_order_gradient_reaches_encoder():
    config = _small_config()
    model = GRUEncoderDecoder(hidden_size=4)
    task = _task(config.data.train_cells[0])
    _, query_loss = _task_post_adaptation_loss(
        model,
        task,
        config,
        epoch=1,
        task_index=0,
        device=torch.device("cpu"),
        loss_function=masked_mse,
    )
    query_loss.backward()
    gradient = model.encoder.gru.weight_ih_l0.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_synthetic_full_maml_and_meta_test_outputs(tmp_path):
    config = _small_config()
    model = GRUEncoderDecoder(hidden_size=4)
    train_tasks = [_task(name, index * 0.002) for index, name in enumerate(config.data.train_cells)]
    logger = logging.getLogger("paper_reproduction_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    result = train_meta_model(
        model,
        train_tasks,
        config,
        torch.device("cpu"),
        tmp_path,
        logger,
    )
    assert (tmp_path / "checkpoints/best_meta_model.pt").is_file()
    assert result.history.shape[0] == 1
    checkpoint = torch.load(
        tmp_path / "checkpoints/best_meta_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["random_seed"] == config.seed
    assert checkpoint["checkpoint_selection_metric"] == (
        "deterministic_post_update_meta_query_loss"
    )
    assert checkpoint["best_meta_loss"] == pytest.approx(
        result.history.loc[0, "checkpoint_selection_meta_loss"]
    )
    loaded = GRUEncoderDecoder(hidden_size=4)
    load_meta_checkpoint(
        tmp_path / "checkpoints/best_meta_model.pt", loaded, torch.device("cpu")
    )
    for expected, actual in zip(result.model.parameters(), loaded.parameters()):
        torch.testing.assert_close(expected, actual)
    recomputed_selection_loss = evaluate_meta_objective_deterministic(
        loaded,
        train_tasks,
        config,
        torch.device("cpu"),
        masked_mse,
    )
    assert recomputed_selection_loss == pytest.approx(
        checkpoint["best_meta_loss"], rel=0, abs=1.0e-12
    )
    test_tasks = [_task(name, index * 0.001) for index, name in enumerate(config.data.test_cells)]
    summary = evaluate_test_tasks(
        result.model,
        test_tasks,
        config,
        torch.device("cpu"),
        tmp_path / "meta_test",
        logger,
    )
    assert len(summary) == 6
    assert set(summary["mode"]) == {"fast_0_steps", "fast_1_steps", "complete"}
    assert set(summary["point_count"]) == {6}
    np.testing.assert_allclose(summary["mae_percent"], 100.0 * summary["mae"])
    assert (tmp_path / "meta_test/meta_test_summary.csv").is_file()


def test_checkpoint_resume_matches_continuous_same_seed(tmp_path):
    config = _small_config()
    config.maml.max_epochs = 2
    train_tasks = [_task(name, index * 0.002) for index, name in enumerate(config.data.train_cells)]
    logger = logging.getLogger("paper_reproduction_resume_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    torch.manual_seed(config.seed)
    continuous_model = GRUEncoderDecoder(hidden_size=4)
    continuous = train_meta_model(
        continuous_model,
        train_tasks,
        config,
        torch.device("cpu"),
        tmp_path / "continuous",
        logger,
    )

    first_epoch_config = copy.deepcopy(config)
    first_epoch_config.maml.max_epochs = 1
    torch.manual_seed(config.seed)
    split_model = GRUEncoderDecoder(hidden_size=4)
    train_meta_model(
        split_model,
        train_tasks,
        first_epoch_config,
        torch.device("cpu"),
        tmp_path / "resumed",
        logger,
    )
    resumed_model = GRUEncoderDecoder(hidden_size=4)
    resumed = train_meta_model(
        resumed_model,
        train_tasks,
        config,
        torch.device("cpu"),
        tmp_path / "resumed",
        logger,
        resume=tmp_path / "resumed/checkpoints/last.pt",
    )
    assert resumed.history.shape[0] == 2
    np.testing.assert_allclose(
        continuous.history["meta_loss"], resumed.history["meta_loss"],
        rtol=0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        continuous.history["checkpoint_selection_meta_loss"],
        resumed.history["checkpoint_selection_meta_loss"],
        rtol=0,
        atol=1.0e-15,
    )
    for expected, actual in zip(continuous.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

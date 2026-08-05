from __future__ import annotations

import logging

import numpy as np
import torch

from paper_reproduction.adapt_and_test import evaluate_test_tasks
from paper_reproduction.config import ExperimentConfig
from paper_reproduction.data import CellTask, build_recursive_pairs, variable_length_collate
from paper_reproduction.losses import masked_mae, masked_mse
from paper_reproduction.maml_train import _task_post_adaptation_loss, train_meta_model
from paper_reproduction.metrics import last_hitting_eol
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


def test_last_hitting_ignores_early_oscillating_crossing():
    cycles = np.arange(500, 505)
    soh = np.array([0.75, 0.69, 0.72, 0.68, 0.67])
    assert last_hitting_eol(cycles, soh, 0.70) == 503
    assert last_hitting_eol(cycles, np.full(5, 0.8), 0.70) is None


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
    assert (tmp_path / "meta_test/meta_test_summary.csv").is_file()

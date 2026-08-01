from __future__ import annotations

import numpy as np
import torch

from battery_weighted_maml.data.task_views import SourceTaskView
from battery_weighted_maml.meta.maml import adapt_source_task, weighted_meta_loss
from battery_weighted_maml.models.gru_seq2seq import GRUSeq2Seq


def _task(name: str, offset: float = 0.0) -> SourceTaskView:
    support = np.array([1.0, 0.98, 0.95, 0.92]) + offset
    query = np.array([0.89, 0.86]) + offset
    return SourceTaskView(name, np.arange(1, 5), support, np.arange(5, 7), query)


def test_weighted_meta_loss_propagates_to_model_parameters():
    model = GRUSeq2Seq(hidden_size=4)
    losses = [
        adapt_source_task(
            model, _task("a"), 1, 0.05, 8, 0.5, torch.device("cpu"),
            torch.Generator().manual_seed(4), full_maml=True,
        ),
        adapt_source_task(
            model, _task("b", 0.02), 1, 0.05, 8, 0.5, torch.device("cpu"),
            torch.Generator().manual_seed(5), full_maml=True,
        ),
    ]
    loss = weighted_meta_loss(losses, torch.tensor([0.7, 0.3]))
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_full_maml_retains_second_order_graph():
    model = GRUSeq2Seq(hidden_size=3)
    result = adapt_source_task(
        model, _task("second-order"), 1, 0.05, 8, 1.0, torch.device("cpu"),
        torch.Generator().manual_seed(7), full_maml=True,
    )
    parameter = next(model.parameters())
    meta_gradient = torch.autograd.grad(result.query_loss, parameter, create_graph=True)[0]
    curvature_probe = torch.autograd.grad(meta_gradient.square().sum(), parameter)[0]
    assert curvature_probe is not None
    assert torch.isfinite(curvature_probe).all()


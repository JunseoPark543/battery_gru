from __future__ import annotations

import torch

from battery_weighted_maml.models.gru_seq2seq import GRUSeq2Seq


def test_gru_forward_shape_with_padded_history():
    model = GRUSeq2Seq(hidden_size=8)
    history = torch.tensor(
        [[[1.0], [0.9], [0.8]], [[1.0], [0.9], [0.0]]], dtype=torch.float32
    )
    lengths = torch.tensor([3, 2])
    target = torch.zeros(2, 4, 1)
    output = model(history, lengths, future_targets=target, teacher_forcing_ratio=0.5)
    assert output.shape == (2, 4, 1)
    assert torch.isfinite(output).all()


def test_recursive_forecast_shape():
    model = GRUSeq2Seq(hidden_size=8)
    output = model.recursive_forecast([1.0, 0.98, 0.96], horizon=7)
    assert output.shape == (1, 7, 1)
    assert not output.requires_grad


from __future__ import annotations

import torch

from battery_weighted_maml.data.collate import collate_support_pairs
from battery_weighted_maml.data.support_dataset import PrefixFutureDataset
from battery_weighted_maml.models.gru_seq2seq import masked_mse


def test_prefix_future_pairs_and_padded_shapes():
    dataset = PrefixFutureDataset([1.0, 0.9, 0.8, 0.7])
    assert len(dataset) == 3
    assert dataset[0].history.shape == (1, 1)
    assert dataset[0].future.shape == (3, 1)
    assert dataset[2].history.shape == (3, 1)
    assert dataset[2].future.shape == (1, 1)
    batch = collate_support_pairs([dataset[0], dataset[2]])
    assert batch["history"].shape == (2, 3, 1)
    assert batch["future"].shape == (2, 3, 1)
    assert batch["future_mask"].tolist() == [[True, True, True], [True, False, False]]


def test_masked_mse_excludes_padding():
    prediction = torch.tensor([[[2.0], [100.0]], [[4.0], [100.0]]])
    target = torch.tensor([[[1.0], [0.0]], [[2.0], [0.0]]])
    mask = torch.tensor([[True, False], [True, False]])
    assert torch.isclose(masked_mse(prediction, target, mask), torch.tensor(2.5))


def test_random_support_sampling_uses_generator_device():
    dataset = PrefixFutureDataset(torch.linspace(1.0, 0.8, 100))
    generator = torch.Generator(device="cpu").manual_seed(42)
    indices = dataset.sample_indices(64, generator)
    assert indices.device.type == "cpu"
    assert indices.shape == (64,)
    assert len(torch.unique(indices)) == 64

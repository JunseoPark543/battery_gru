"""Padding and masking for variable-length recursive pairs."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from .support_dataset import SupportPair


def collate_support_pairs(pairs: Sequence[SupportPair]) -> dict[str, torch.Tensor]:
    if not pairs:
        raise ValueError("cannot collate an empty collection")
    histories = [pair.history for pair in pairs]
    futures = [pair.future for pair in pairs]
    history_lengths = torch.tensor([len(item) for item in histories], dtype=torch.long)
    future_lengths = torch.tensor([len(item) for item in futures], dtype=torch.long)
    history = pad_sequence(histories, batch_first=True, padding_value=0.0)
    future = pad_sequence(futures, batch_first=True, padding_value=0.0)
    positions = torch.arange(future.shape[1]).unsqueeze(0)
    future_mask = positions < future_lengths.unsqueeze(1)
    return {
        "history": history,
        "history_lengths": history_lengths,
        "future": future,
        "future_lengths": future_lengths,
        "future_mask": future_mask,
    }


def sample_support_batch(
    dataset: "PrefixFutureDataset",
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    from .support_dataset import PrefixFutureDataset

    if not isinstance(dataset, PrefixFutureDataset):
        raise TypeError("dataset must be PrefixFutureDataset")
    indices = dataset.sample_indices(batch_size, generator)
    batch = collate_support_pairs([dataset[int(index)] for index in indices])
    return {key: value.to(device) for key, value in batch.items()}


"""Leakage-safe fast and full target support adaptation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
import torch

from ..data.collate import sample_support_batch
from ..data.support_dataset import PrefixFutureDataset
from ..data.task_views import TargetSupportView
from ..models.gru_seq2seq import GRUSeq2Seq, masked_mse


@dataclass
class AdaptationResult:
    model: GRUSeq2Seq
    history: pd.DataFrame
    best_loss: float
    best_step: int
    snapshots: dict[int, GRUSeq2Seq] = field(default_factory=dict)


def adapt_target(
    model: GRUSeq2Seq,
    target: TargetSupportView,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    teacher_forcing_ratio: float,
    device: torch.device,
    generator: torch.Generator,
    patience: int | None = None,
    capture_steps: Sequence[int] | None = None,
) -> AdaptationResult:
    """Fine-tune every model parameter using only ``TargetSupportView`` pairs."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    requested_steps = set(int(step) for step in (capture_steps or ()))
    if any(step <= 0 or step > max_steps for step in requested_steps):
        raise ValueError("capture_steps must lie between 1 and max_steps")
    adapted = copy.deepcopy(model).to(device)
    adapted.train()
    optimizer = torch.optim.SGD(adapted.parameters(), lr=learning_rate)
    dataset = PrefixFutureDataset(target.soh, target.features)
    best_state = copy.deepcopy(adapted.state_dict())
    best_loss = float("inf")
    best_step = 0
    stale = 0
    records: list[dict[str, float | int]] = []
    snapshots: dict[int, GRUSeq2Seq] = {}
    for step in range(1, max_steps + 1):
        batch = sample_support_batch(dataset, batch_size, generator, device)
        predictions = adapted(
            batch["history"],
            batch["history_lengths"],
            future_targets=batch["future"],
            teacher_forcing_ratio=teacher_forcing_ratio,
            generator=generator,
        )
        loss = masked_mse(predictions, batch["future"], batch["future_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite target support loss at adaptation step {step}: {loss}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in requested_steps:
            snapshot = copy.deepcopy(adapted)
            snapshot.eval()
            snapshots[step] = snapshot
        value = float(loss.detach().cpu())
        records.append({"step": step, "support_loss": value})
        if value < best_loss - 1e-12:
            best_loss = value
            best_step = step
            best_state = copy.deepcopy(adapted.state_dict())
            stale = 0
        else:
            stale += 1
        if patience is not None and stale >= patience:
            break
    adapted.load_state_dict(best_state)
    adapted.eval()
    return AdaptationResult(
        model=adapted,
        history=pd.DataFrame(records, columns=["step", "support_loss"]),
        best_loss=best_loss,
        best_step=best_step,
        snapshots=snapshots,
    )

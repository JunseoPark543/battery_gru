"""Differentiable full-MAML source adaptation and weighted query objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import higher
import torch

from ..data.collate import sample_support_batch
from ..data.support_dataset import PrefixFutureDataset
from ..data.task_views import SourceTaskView
from ..models.gru_seq2seq import GRUSeq2Seq, masked_mse


@dataclass
class TaskMetaLoss:
    task_name: str
    support_loss: torch.Tensor
    query_loss: torch.Tensor


def _loss_on_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    teacher_forcing_ratio: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    predictions = model(
        batch["history"],
        batch["history_lengths"],
        future_targets=batch["future"],
        teacher_forcing_ratio=teacher_forcing_ratio,
        generator=generator,
    )
    return masked_mse(predictions, batch["future"], batch["future_mask"])


def adapt_source_task(
    model: GRUSeq2Seq,
    task: SourceTaskView,
    inner_steps: int,
    inner_lr: float,
    inner_batch_size: int,
    teacher_forcing_ratio: float,
    device: torch.device,
    generator: torch.Generator,
    full_maml: bool = True,
) -> TaskMetaLoss:
    """Adapt on one source's support and evaluate its mean query loss."""
    dataset = PrefixFutureDataset(task.support_soh)
    inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
    support_losses: list[torch.Tensor] = []
    with higher.innerloop_ctx(
        model,
        inner_optimizer,
        copy_initial_weights=False,
        track_higher_grads=full_maml,
    ) as (functional_model, differentiable_optimizer):
        for _ in range(inner_steps):
            batch = sample_support_batch(dataset, inner_batch_size, generator, device)
            support_loss = _loss_on_batch(
                functional_model, batch, teacher_forcing_ratio, generator
            )
            if not torch.isfinite(support_loss):
                raise FloatingPointError(
                    f"non-finite source support loss for task {task.file_name}: {support_loss}"
                )
            differentiable_optimizer.step(support_loss)
            support_losses.append(support_loss)
        history = torch.tensor(task.support_soh, dtype=torch.float32, device=device).view(1, -1, 1)
        query = torch.tensor(task.query_soh, dtype=torch.float32, device=device).view(1, -1, 1)
        lengths = torch.tensor([history.shape[1]], dtype=torch.long, device=device)
        predictions = functional_model(
            history,
            lengths,
            future_targets=query,
            teacher_forcing_ratio=teacher_forcing_ratio,
            generator=generator,
        )
        query_loss = (predictions - query).square().mean()
        if not torch.isfinite(query_loss):
            raise FloatingPointError(
                f"non-finite source query loss for task {task.file_name}: {query_loss}"
            )
        mean_support = torch.stack(support_losses).mean()
    return TaskMetaLoss(task.file_name, mean_support, query_loss)


def weighted_meta_loss(losses: Sequence[TaskMetaLoss], alpha: torch.Tensor) -> torch.Tensor:
    """Weight only adapted task query losses; each task loss is already a mean."""
    if len(losses) != len(alpha) or len(losses) == 0:
        raise ValueError("loss and alpha counts must match and be nonempty")
    if alpha.requires_grad:
        raise ValueError("alpha must be detached from the encoder/QP computation")
    query_losses = torch.stack([item.query_loss for item in losses])
    result = torch.sum(alpha.to(query_losses.device) * query_losses)
    if not torch.isfinite(result):
        raise FloatingPointError(f"weighted meta loss is non-finite: {result}")
    return result

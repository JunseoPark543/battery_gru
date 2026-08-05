"""Differentiable full-MAML source adaptation and weighted query objective."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
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
    # This is the task contribution used by the weighted outer objective. It is
    # the ordinary final-step query loss in legacy mode and the robust path loss
    # when query_losses_by_step contains multiple checkpoints.
    query_loss: torch.Tensor
    query_losses_by_step: dict[int, torch.Tensor] = field(default_factory=dict)
    path_mean_loss: torch.Tensor | None = None
    path_dispersion: torch.Tensor | None = None
    path_worst_loss: torch.Tensor | None = None


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


def robust_adaptation_path_loss(
    query_losses_by_step: dict[int, torch.Tensor],
    worst_weight: float,
    dispersion_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine losses along one adaptation path without changing model scale.

    The mean/worst terms form a convex combination, so equal step losses reduce
    exactly to the legacy query loss. Population standard deviation adds a
    same-unit penalty only when performance varies across adaptation depths.
    """
    if not query_losses_by_step:
        raise ValueError("query_losses_by_step cannot be empty")
    if not 0.0 <= worst_weight <= 1.0:
        raise ValueError("worst_weight must be in [0, 1]")
    if dispersion_weight < 0.0:
        raise ValueError("dispersion_weight cannot be negative")
    ordered_steps = sorted(query_losses_by_step)
    losses = torch.stack([query_losses_by_step[step] for step in ordered_steps])
    if not torch.isfinite(losses).all():
        raise FloatingPointError(f"non-finite adaptation path losses: {losses}")
    mean_loss = losses.mean()
    worst_loss = losses.max()
    if len(losses) == 1:
        dispersion = losses.new_zeros(())
    else:
        variance = (losses - mean_loss).square().mean()
        # Use a scale-relative numerical floor. A raw float32 epsilon is larger
        # than converged MSE values in this project and would suppress the very
        # path differences that this term is intended to measure.
        floor = mean_loss.detach().abs().clamp_min(torch.finfo(losses.dtype).tiny) * 1.0e-6
        dispersion = torch.sqrt(variance + floor.square()) - floor
    result = (
        (1.0 - worst_weight) * mean_loss
        + worst_weight * worst_loss
        + dispersion_weight * dispersion
    )
    if not torch.isfinite(result):
        raise FloatingPointError(f"non-finite robust adaptation path loss: {result}")
    return result, mean_loss, dispersion, worst_loss


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
    robust_path_steps: Sequence[int] | None = None,
    robust_path_worst_weight: float = 0.0,
    robust_path_dispersion_weight: float = 0.0,
) -> TaskMetaLoss:
    """Adapt on source support and evaluate one or more path checkpoints."""
    query_steps = (
        tuple(robust_path_steps) if robust_path_steps is not None else (inner_steps,)
    )
    if (
        not query_steps
        or any(step <= 0 for step in query_steps)
        or tuple(sorted(set(query_steps))) != query_steps
    ):
        raise ValueError("robust_path_steps must be unique, increasing, and positive")
    if robust_path_steps is not None and inner_steps not in query_steps:
        raise ValueError("inner_steps must be included in robust_path_steps")
    total_inner_steps = max(query_steps)
    dataset = PrefixFutureDataset(task.support_soh, task.support_features)
    inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
    support_losses: list[torch.Tensor] = []
    # CuDNN RNN kernels do not implement the double backward required by full
    # MAML. Disable CuDNN only for the differentiable source adaptation/query
    # forwards; CUDA tensors and the rest of the experiment still use the GPU.
    rnn_backend = (
        torch.backends.cudnn.flags(enabled=False)
        if device.type == "cuda" and full_maml
        else nullcontext()
    )
    with rnn_backend:
        with higher.innerloop_ctx(
            model,
            inner_optimizer,
            copy_initial_weights=False,
            track_higher_grads=full_maml,
        ) as (functional_model, differentiable_optimizer):
            history = torch.tensor(
                task.support_features, dtype=torch.float32, device=device
            ).unsqueeze(0)
            query = torch.tensor(
                task.query_soh, dtype=torch.float32, device=device
            ).view(1, -1, 1)
            lengths = torch.tensor([history.shape[1]], dtype=torch.long, device=device)
            query_generator_state = generator.get_state().clone()
            query_losses_by_step: dict[int, torch.Tensor] = {}
            query_step_set = set(query_steps)
            for step in range(1, total_inner_steps + 1):
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
                if step in query_step_set:
                    # Every checkpoint uses the same teacher-forcing random mask.
                    # Restore the support RNG afterward so query evaluation never
                    # changes subsequent support batches on the shared path.
                    support_generator_state = generator.get_state().clone()
                    generator.set_state(query_generator_state)
                    predictions = functional_model(
                        history,
                        lengths,
                        future_targets=query,
                        teacher_forcing_ratio=teacher_forcing_ratio,
                        generator=generator,
                    )
                    generator.set_state(support_generator_state)
                    step_query_loss = (predictions - query).square().mean()
                    if not torch.isfinite(step_query_loss):
                        raise FloatingPointError(
                            "non-finite source query loss for task "
                            f"{task.file_name} at inner step {step}: {step_query_loss}"
                        )
                    query_losses_by_step[step] = step_query_loss
            query_loss, path_mean, path_dispersion, path_worst = (
                robust_adaptation_path_loss(
                    query_losses_by_step,
                    worst_weight=robust_path_worst_weight,
                    dispersion_weight=robust_path_dispersion_weight,
                )
            )
            mean_support = torch.stack(support_losses).mean()
    return TaskMetaLoss(
        task_name=task.file_name,
        support_loss=mean_support,
        query_loss=query_loss,
        query_losses_by_step=query_losses_by_step,
        path_mean_loss=path_mean,
        path_dispersion=path_dispersion,
        path_worst_loss=path_worst,
    )


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

"""Target-aware weighted full-MAML training loop."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..config import ExperimentConfig
from ..data.task_views import SourceTaskView, TargetSupportView
from ..meta.kernel_weights import KernelWeightResult, compute_target_aware_weights
from ..meta.maml import TaskMetaLoss, adapt_source_task, weighted_meta_loss
from ..models.gru_seq2seq import GRUSeq2Seq
from ..seed import make_generator
from .checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint
from .history import TrainingHistory


@dataclass
class MetaTrainingResult:
    model: GRUSeq2Seq
    history: TrainingHistory
    final_weights: KernelWeightResult
    best_metric: float
    last_iteration: int


class WeightedMAMLTrainer:
    """Trainer whose target argument is statically restricted to support-only data."""

    def __init__(
        self,
        model: GRUSeq2Seq,
        source_tasks: Sequence[SourceTaskView],
        target_support: TargetSupportView,
        config: ExperimentConfig,
        device: torch.device,
        run_dir: str | Path,
        source_mode: str,
        logger: logging.Logger,
    ) -> None:
        if not source_tasks:
            raise ValueError("at least one source task is required")
        if not isinstance(target_support, TargetSupportView):
            raise TypeError("trainer accepts only TargetSupportView, never target evaluation data")
        support_length = target_support.history_length
        if any(len(task.support_soh) != support_length for task in source_tasks):
            raise ValueError("all source/target support sequences must have the same length L")
        self.model = model.to(device)
        self.source_tasks = list(source_tasks)
        self.target_support = target_support
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.source_mode = source_mode
        self.logger = logger
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.maml.outer_lr)
        self.history = TrainingHistory()
        self.generator = make_generator(config.seed + 101, device)

    def compute_weights(self) -> KernelWeightResult:
        """Compute current target-aware weights from source/target support only."""
        self.model.eval()
        with torch.no_grad():
            source_points = [
                self.model.empirical_points(
                    torch.tensor(
                        task.support_features, dtype=torch.float32, device=self.device
                    )
                )
                for task in self.source_tasks
            ]
            target_points = self.model.empirical_points(
                torch.tensor(
                    self.target_support.features, dtype=torch.float32, device=self.device
                )
            )
        weight_config = self.config.weights
        return compute_target_aware_weights(
            source_points,
            target_points,
            sigma=weight_config.sigma,
            diagonal_jitter=weight_config.diagonal_jitter,
            primary_solver=weight_config.qp_solver_primary,
            fallback_solver=weight_config.qp_solver_fallback,
            device=self.device,
            logger=self.logger,
        )

    def _validate_resume_objective(self, payload: dict[str, object]) -> None:
        """Prevent silently changing the meta objective inside one run."""
        saved_config = payload.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("resume checkpoint has no valid resolved config")
        saved_maml = saved_config.get("maml")
        if not isinstance(saved_maml, dict):
            raise ValueError("resume checkpoint has no valid MAML config")
        current = self.config.maml
        expected = {
            "inner_steps": current.inner_steps,
            "robust_path_steps": current.robust_path_steps,
            "robust_path_worst_weight": current.robust_path_worst_weight,
            "robust_path_dispersion_weight": current.robust_path_dispersion_weight,
        }
        actual = {
            "inner_steps": saved_maml.get("inner_steps"),
            "robust_path_steps": saved_maml.get("robust_path_steps"),
            "robust_path_worst_weight": saved_maml.get(
                "robust_path_worst_weight", 0.0
            ),
            "robust_path_dispersion_weight": saved_maml.get(
                "robust_path_dispersion_weight", 0.0
            ),
        }
        if actual != expected:
            raise ValueError(
                "resume checkpoint robust adaptation objective differs from config: "
                f"checkpoint={actual}, requested={expected}"
            )

    def _payload(
        self, iteration: int, best_metric: float, ema_metric: float, alpha: torch.Tensor
    ) -> dict[str, object]:
        return checkpoint_payload(
            self.model,
            self.optimizer,
            iteration,
            best_metric,
            ema_metric,
            self.config.to_dict(),
            self.target_support.file_name,
            [task.file_name for task in self.source_tasks],
            self.target_support.history_length,
            self.source_mode,
            self.config.seed,
            alpha,
        )

    def train(self, resume: str | Path | None = None) -> MetaTrainingResult:
        checkpoints = self.run_dir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        start_iteration = 1
        best_metric = float("inf")
        ema: float | None = None
        latest: KernelWeightResult | None = None
        if resume is not None:
            payload = load_checkpoint(
                resume,
                self.model,
                self.optimizer,
                restore_rng=True,
                map_location=self.device,
            )
            if payload["target_file_name"] != self.target_support.file_name:
                raise ValueError("resume checkpoint target does not match requested target")
            if int(payload["L"]) != self.target_support.history_length:
                raise ValueError("resume checkpoint L does not match requested history length")
            if payload["source_mode"] != self.source_mode:
                raise ValueError("resume checkpoint source mode does not match")
            if list(payload["source_file_names"]) != [task.file_name for task in self.source_tasks]:
                raise ValueError("resume checkpoint source list does not match")
            self._validate_resume_objective(payload)
            start_iteration = int(payload["meta_iteration"]) + 1
            best_metric = float(payload["best_metric"])
            ema = float(payload.get("ema_metric", best_metric))
            self.history = TrainingHistory.load(self.run_dir)
            self.logger.info("Resuming meta-training at iteration %d", start_iteration)
        total = self.config.maml.meta_iterations
        if start_iteration > total:
            self.logger.info("Checkpoint already reached configured meta_iterations=%d", total)
            latest = self.compute_weights()
            return MetaTrainingResult(
                self.model, self.history, latest, best_metric, start_iteration - 1
            )
        began = time.perf_counter()
        progress = tqdm(range(start_iteration, total + 1), desc="meta-training", unit="iter")
        for iteration in progress:
            iteration_start = time.perf_counter()
            if latest is None or self.config.weights.recompute_every_iteration:
                latest = self.compute_weights()
            self.model.train()
            task_losses: list[TaskMetaLoss] = []
            for task_index, task in enumerate(self.source_tasks):
                task_generator = make_generator(
                    self.config.seed + iteration * 1009 + task_index, self.device
                )
                task_losses.append(
                    adapt_source_task(
                        self.model,
                        task,
                        inner_steps=self.config.maml.inner_steps,
                        inner_lr=self.config.maml.inner_lr,
                        inner_batch_size=self.config.maml.inner_batch_size,
                        teacher_forcing_ratio=self.config.model.teacher_forcing_ratio,
                        device=self.device,
                        generator=task_generator,
                        full_maml=self.config.maml.full_maml,
                        robust_path_steps=self.config.maml.robust_path_steps,
                        robust_path_worst_weight=(
                            self.config.maml.robust_path_worst_weight
                        ),
                        robust_path_dispersion_weight=(
                            self.config.maml.robust_path_dispersion_weight
                        ),
                    )
                )
            meta_loss = weighted_meta_loss(task_losses, latest.alpha)
            alpha_on_device = latest.alpha.to(self.device)
            weighted_path_mean = torch.sum(
                alpha_on_device
                * torch.stack([item.path_mean_loss for item in task_losses])
            )
            weighted_path_dispersion = torch.sum(
                alpha_on_device
                * torch.stack([item.path_dispersion for item in task_losses])
            )
            weighted_path_worst = torch.sum(
                alpha_on_device
                * torch.stack([item.path_worst_loss for item in task_losses])
            )
            self.optimizer.zero_grad(set_to_none=True)
            meta_loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.maml.gradient_clip_norm
            )
            grad_norm = float(grad_norm_tensor.detach().cpu())
            if not math.isfinite(grad_norm):
                summaries = {
                    name: {
                        "min": float(parameter.detach().min().cpu()),
                        "max": float(parameter.detach().max().cpu()),
                        "mean": float(parameter.detach().mean().cpu()),
                    }
                    for name, parameter in self.model.named_parameters()
                }
                self.logger.error(
                    "Non-finite gradient at iteration %d; parameter summaries=%s",
                    iteration, summaries,
                )
                raise FloatingPointError(f"non-finite gradient norm at iteration {iteration}")
            self.optimizer.step()
            value = float(meta_loss.detach().cpu())
            ema = value if ema is None else 0.98 * ema + 0.02 * value
            elapsed = time.perf_counter() - began
            done = iteration - start_iteration + 1
            eta = elapsed / done * (total - iteration)
            alpha_cpu = latest.alpha.detach().cpu().numpy()
            entropy = float(-np.sum(alpha_cpu * np.log(alpha_cpu + 1e-12)))
            effective = float(1.0 / np.sum(alpha_cpu ** 2))
            record = {
                "iteration": iteration,
                "weighted_meta_loss": value,
                "ema_source_meta_loss": ema,
                "weighted_path_mean_query_loss": float(
                    weighted_path_mean.detach().cpu()
                ),
                "weighted_path_dispersion": float(
                    weighted_path_dispersion.detach().cpu()
                ),
                "weighted_path_worst_query_loss": float(
                    weighted_path_worst.detach().cpu()
                ),
                "alpha_entropy": entropy,
                "effective_sources": effective,
                "mmd_objective": latest.objective,
                "rbf_sigma": latest.sigma,
                "qp_solver_status": latest.solver_status,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "iteration_seconds": time.perf_counter() - iteration_start,
            }
            if self.device.type == "cuda":
                record["cuda_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
                record["cuda_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
            self.history.iterations.append(record)
            self.history.gradients.append({"iteration": iteration, "gradient_norm": grad_norm})
            for task_loss, alpha in zip(task_losses, alpha_cpu):
                source_record = {
                    "iteration": iteration,
                    "source": task_loss.task_name,
                    "support_loss": float(task_loss.support_loss.detach().cpu()),
                    "query_loss": float(task_loss.query_loss.detach().cpu()),
                    "path_mean_query_loss": float(
                        task_loss.path_mean_loss.detach().cpu()
                    ),
                    "path_dispersion": float(
                        task_loss.path_dispersion.detach().cpu()
                    ),
                    "path_worst_query_loss": float(
                        task_loss.path_worst_loss.detach().cpu()
                    ),
                    "alpha": float(alpha),
                }
                for step, step_loss in task_loss.query_losses_by_step.items():
                    source_record[f"query_loss_step_{step}"] = float(
                        step_loss.detach().cpu()
                    )
                self.history.source_losses.append(source_record)
            if iteration % self.config.logging.save_alpha_interval == 0 or iteration == total:
                for task, alpha in zip(self.source_tasks, alpha_cpu):
                    self.history.alphas.append(
                        {"iteration": iteration, "source": task.file_name, "alpha": float(alpha)}
                    )
            if ema < best_metric:
                best_metric = ema
                save_checkpoint(
                    self._payload(iteration, best_metric, ema, latest.alpha),
                    checkpoints / "best_source_meta_loss.pt",
                )
            if iteration % self.config.logging.checkpoint_interval == 0 or iteration == total:
                save_checkpoint(
                    self._payload(iteration, best_metric, ema, latest.alpha), checkpoints / "last.pt"
                )
                self.history.save(self.run_dir)
            progress.set_postfix(loss=f"{value:.5g}", sigma=f"{latest.sigma:.3g}")
            if iteration % self.config.logging.log_interval == 0 or iteration in {start_iteration, total}:
                support_log = {
                    item.task_name: float(item.support_loss.detach().cpu()) for item in task_losses
                }
                query_log = {
                    item.task_name: float(item.query_loss.detach().cpu()) for item in task_losses
                }
                path_log = {
                    item.task_name: {
                        step: float(step_loss.detach().cpu())
                        for step, step_loss in item.query_losses_by_step.items()
                    }
                    for item in task_losses
                }
                top = int(np.argmax(alpha_cpu))
                self.logger.info(
                    "iter=%d/%d loss=%.7g ema=%.7g support=%s query=%s path=%s alpha=%s "
                    "top=%s:%.5f entropy=%.5f effective=%.3f mmd=%.7g sigma=%.7g "
                    "solver=%s grad=%.5g lr=%.5g elapsed=%.1fs eta=%.1fs",
                    iteration, total, value, ema, support_log, query_log, path_log,
                    np.array2string(alpha_cpu, precision=5),
                    self.source_tasks[top].file_name, alpha_cpu[top], entropy, effective,
                    latest.objective, latest.sigma, latest.solver_status, grad_norm,
                    self.optimizer.param_groups[0]["lr"], elapsed, eta,
                )
        self.history.save(self.run_dir)
        assert latest is not None
        return MetaTrainingResult(self.model, self.history, latest, best_metric, total)

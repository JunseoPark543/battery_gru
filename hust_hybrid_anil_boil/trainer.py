"""Leakage-auditable training loop for all five architecture-matched methods."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from hust_direct_rul_boil.data import CellSample, InputNormalizer, protocol_sort_key
from hust_direct_rul_boil.metrics import save_json
from hust_direct_rul_boil.trainer import resolve_device, set_global_seed

from .analysis import save_training_diagnostics
from .config import ExperimentConfig
from .losses import LossBreakdown, outer_objective
from .meta import (
    adapt_task,
    concatenate_outputs,
    forward_with_parameters,
    parameter_policy,
)
from .model import GeneralSpecificRULModel


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: GeneralSpecificRULModel
    normalizer: InputNormalizer
    best_iteration: int
    best_validation_mae_cycles: float
    stopped_iteration: int
    checkpoint_path: Path
    history: list[dict[str, Any]]
    source_domain_metrics: dict[str, float]


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in state["cuda"]]
        )


class HybridTrainer:
    def __init__(
        self,
        source_samples: Sequence[CellSample],
        held_out_protocol: str,
        config: ExperimentConfig,
        device: torch.device,
        run_dir: str | Path,
        seed: int,
    ) -> None:
        self.all_source_samples = list(source_samples)
        self.held_out_protocol = held_out_protocol
        self.config = config
        self.method = config.method
        self.device = device
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        if any(sample.protocol == held_out_protocol for sample in self.all_source_samples):
            raise RuntimeError("held-out target protocol leaked into the source trainer")
        self.source_protocols = sorted(
            {sample.protocol for sample in self.all_source_samples},
            key=protocol_sort_key,
        )
        if len(self.source_protocols) < 2:
            raise ValueError("hybrid meta-learning requires at least two source protocols")
        self.protocol_to_index = {
            protocol: index for index, protocol in enumerate(self.source_protocols)
        }
        self.train_samples, self.validation_samples = self._split_source_validation()
        self.normalizer = InputNormalizer.fit(
            self.train_samples, config.data.normalization_epsilon
        )
        self.model = GeneralSpecificRULModel(
            waveform_channels=len(config.data.profile_channels),
            scalar_features=len(config.data.scalar_features),
            history_length=config.data.history_length,
            source_domains=len(self.source_protocols),
            model_config=config.model,
            ablation=config.ablation,
        ).to(device)
        source_rul_median = float(np.median([sample.rul_cycles for sample in self.train_samples]))
        self.model.initialize_prediction_bias(source_rul_median)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.outer_lr,
            weight_decay=config.train.weight_decay,
        )
        self.waveforms: dict[str, Tensor] = {}
        self.scalars: dict[str, Tensor] = {}
        self.targets: dict[str, Tensor] = {}
        for sample in self.all_source_samples:
            self.waveforms[sample.file_name] = torch.as_tensor(
                self.normalizer.transform_waveforms(sample.waveforms),
                dtype=torch.float32,
                device=device,
            )
            self.scalars[sample.file_name] = torch.as_tensor(
                self.normalizer.transform_scalars(sample.scalars),
                dtype=torch.float32,
                device=device,
            )
            self.targets[sample.file_name] = torch.as_tensor(
                sample.rul_cycles, dtype=torch.float32, device=device
            )
        self.history: list[dict[str, Any]] = []
        self.best_score = float("inf")
        self.best_iteration = 0
        self.stale_evaluations = 0
        self.latest_validation_metrics: dict[str, float] = {}
        save_json(
            {
                "held_out_target_protocol": held_out_protocol,
                "meta_task_definition": "one charging protocol",
                "support_query_definition": (
                    "disjoint battery cells from the same source protocol"
                ),
                "why_not_one_cell_per_task": (
                    "each direct-RUL cell provides only one (first-100-cycles, RUL) label; "
                    "splitting it would duplicate the same target"
                ),
                "training_files": [sample.file_name for sample in self.train_samples],
                "source_validation_files": [
                    sample.file_name for sample in self.validation_samples
                ],
                "normalization_fit_files": [
                    sample.file_name for sample in self.train_samples
                ],
                "target_protocol_used_for_training_or_selection": False,
            },
            self.run_dir / "source_split.json",
        )
        save_json(parameter_policy(self.model, self.method), self.run_dir / "inner_policy.json")

    def _split_source_validation(self) -> tuple[list[CellSample], list[CellSample]]:
        training: list[CellSample] = []
        validation: list[CellSample] = []
        count = self.config.train.validation_cells_per_protocol
        required_training = (
            self.config.train.support_cells_per_task
            + self.config.train.query_cells_per_task
        )
        for protocol in self.source_protocols:
            group = sorted(
                (sample for sample in self.all_source_samples if sample.protocol == protocol),
                key=lambda sample: (sample.replicate, sample.file_name),
            )
            if len(group) - count < required_training:
                raise ValueError(
                    f"{protocol}: {len(group)} cells cannot provide {count} validation "
                    f"and {required_training} disjoint train support/query cells"
                )
            offset = (self.seed + protocol_sort_key(protocol)) % len(group)
            validation_indices = {(offset + index) % len(group) for index in range(count)}
            validation.extend(
                sample for index, sample in enumerate(group) if index in validation_indices
            )
            training.extend(
                sample for index, sample in enumerate(group) if index not in validation_indices
            )
        return training, validation

    def batch(self, samples: Sequence[CellSample]) -> tuple[Tensor, Tensor, Tensor]:
        if not samples:
            raise ValueError("cannot create an empty cell batch")
        return (
            torch.stack([self.waveforms[sample.file_name] for sample in samples]),
            torch.stack([self.scalars[sample.file_name] for sample in samples]),
            torch.stack([self.targets[sample.file_name] for sample in samples]),
        )

    def _augment(self, waveforms: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        config = self.config.augmentation
        if not config.enabled:
            return waveforms, scalars
        output_waveforms = waveforms
        output_scalars = scalars
        if config.waveform_gaussian_std:
            output_waveforms = output_waveforms + torch.randn_like(output_waveforms) * config.waveform_gaussian_std
        if config.scalar_gaussian_std:
            output_scalars = output_scalars + torch.randn_like(output_scalars) * config.scalar_gaussian_std
        if config.profile_channel_mask_probability:
            keep = torch.rand(
                output_waveforms.shape[0], 1, 1, output_waveforms.shape[-1],
                device=self.device,
            ) >= config.profile_channel_mask_probability
            output_waveforms = output_waveforms * keep.to(output_waveforms.dtype)
        if config.cycle_mask_probability:
            keep = torch.rand(
                output_waveforms.shape[0], output_waveforms.shape[1], 1, 1,
                device=self.device,
            ) >= config.cycle_mask_probability
            output_waveforms = output_waveforms * keep.to(output_waveforms.dtype)
            output_scalars = output_scalars * keep.squeeze(-1).to(output_scalars.dtype)
        return output_waveforms, output_scalars

    def _grl_strength(self, iteration: int) -> float:
        progress = iteration / max(1, self.config.train.iterations)
        return self.config.loss.grl_max_strength * (
            2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        )

    def _sample_episode(self, protocol: str) -> tuple[list[CellSample], list[CellSample]]:
        group = [sample for sample in self.train_samples if sample.protocol == protocol]
        support_count = self.config.train.support_cells_per_task
        query_count = self.config.train.query_cells_per_task
        chosen = random.sample(group, support_count + query_count)
        support = chosen[:support_count]
        query = chosen[support_count:]
        if {sample.file_name for sample in support} & {sample.file_name for sample in query}:
            raise RuntimeError("support/query cell leakage detected")
        return support, query

    def _supervised_batch(self) -> tuple[list[CellSample], Tensor]:
        samples: list[CellSample] = []
        domains: list[int] = []
        count = self.config.train.supervised_cells_per_domain
        for protocol in self.source_protocols:
            group = [sample for sample in self.train_samples if sample.protocol == protocol]
            selected = random.sample(group, min(count, len(group)))
            samples.extend(selected)
            domains.extend([self.protocol_to_index[protocol]] * len(selected))
        return samples, torch.as_tensor(domains, dtype=torch.long, device=self.device)

    def _training_objective(self, iteration: int) -> tuple[LossBreakdown, dict[str, float]]:
        if self.method == "supervised":
            samples, domains = self._supervised_batch()
            waveforms, scalars, targets = self.batch(samples)
            waveforms, scalars = self._augment(waveforms, scalars)
            output = self.model(waveforms, scalars, grl_strength=self._grl_strength(iteration))
            loss = outer_objective(output, targets, domains, self.config.loss, self.config.ablation)
            return loss, {"support_loss": 0.0}

        task_count = min(self.config.train.tasks_per_iteration, len(self.source_protocols))
        protocols = random.sample(self.source_protocols, task_count)
        outputs = []
        target_batches = []
        domain_batches = []
        support_losses: list[Tensor] = []
        update_norms: dict[str, list[Tensor]] = {}
        for protocol in protocols:
            support_samples, query_samples = self._sample_episode(protocol)
            support_waveforms, support_scalars, support_targets = self.batch(support_samples)
            query_waveforms, query_scalars, query_targets = self.batch(query_samples)
            support_waveforms, support_scalars = self._augment(
                support_waveforms, support_scalars
            )
            adapted = adapt_task(
                self.model,
                support_waveforms,
                support_scalars,
                support_targets,
                method=self.method,
                steps=self.config.train.inner_steps,
                config=self.config,
            )
            outputs.append(
                forward_with_parameters(
                    self.model,
                    adapted.parameters,
                    query_waveforms,
                    query_scalars,
                    grl_strength=self._grl_strength(iteration),
                )
            )
            target_batches.append(query_targets)
            domain_batches.append(
                torch.full(
                    (len(query_samples),),
                    self.protocol_to_index[protocol],
                    dtype=torch.long,
                    device=self.device,
                )
            )
            if adapted.support_losses:
                support_losses.append(adapted.support_losses[-1])
            for module, value in adapted.update_norms.items():
                update_norms.setdefault(module, []).append(value)
        combined = concatenate_outputs(outputs)
        targets = torch.cat(target_batches)
        domains = torch.cat(domain_batches)
        loss = outer_objective(
            combined, targets, domains, self.config.loss, self.config.ablation
        )
        diagnostics = {
            "support_loss": float(
                torch.stack(support_losses).mean().detach().cpu()
            ) if support_losses else 0.0
        }
        for module, values in update_norms.items():
            diagnostics[f"inner_update_{module}"] = float(
                torch.stack(values).mean().detach().cpu()
            )
        return loss, diagnostics

    def _source_validation(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        general_correct = 0
        specific_correct = 0
        count = 0
        with torch.enable_grad():
            for protocol in self.source_protocols:
                group = sorted(
                    (sample for sample in self.validation_samples if sample.protocol == protocol),
                    key=lambda sample: (sample.replicate, sample.file_name),
                )
                support = group[:1]
                query = group[1:]
                if not query:
                    raise ValueError(f"{protocol}: source validation needs at least two cells")
                support_waveforms, support_scalars, support_targets = self.batch(support)
                query_waveforms, query_scalars, query_targets = self.batch(query)
                adapted = adapt_task(
                    self.model,
                    support_waveforms,
                    support_scalars,
                    support_targets,
                    method=self.method,
                    steps=(
                        0
                        if self.method == "supervised"
                        else self.config.train.validation_adaptation_steps
                    ),
                    config=self.config,
                    second_order=False,
                )
                output = forward_with_parameters(
                    self.model,
                    adapted.parameters,
                    query_waveforms,
                    query_scalars,
                    grl_strength=0.0,
                )
                predictions.append(output.prediction.detach())
                targets.append(query_targets.detach())
                domain_index = self.protocol_to_index[protocol]
                general_correct += int(
                    output.general_domain_logits.argmax(dim=-1).eq(domain_index).sum().detach().cpu()
                )
                if output.specific_domain_logits is not None:
                    specific_correct += int(
                        output.specific_domain_logits.argmax(dim=-1).eq(domain_index).sum().detach().cpu()
                    )
                count += len(query)
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        score = float(torch.mean(torch.abs(prediction - target)).cpu())
        metrics = {
            "general_domain_accuracy": general_correct / max(1, count),
            "specific_domain_accuracy": (
                specific_correct / max(1, count)
                if self.config.ablation.use_specific_domain_classifier
                else 0.0
            ),
        }
        self.model.train()
        return score, metrics

    def _checkpoint_payload(self, iteration: int) -> dict[str, Any]:
        return {
            "format_version": 1,
            "experiment": "hust_hybrid_anil_boil",
            "iteration": iteration,
            "method": self.method,
            "held_out_protocol": self.held_out_protocol,
            "source_protocols": self.source_protocols,
            "training_files": [sample.file_name for sample in self.train_samples],
            "validation_files": [sample.file_name for sample in self.validation_samples],
            "seed": self.seed,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "stale_evaluations": self.stale_evaluations,
            "latest_validation_metrics": self.latest_validation_metrics,
            "history": self.history,
            "config": self.config.to_dict(),
            "parameter_policy": parameter_policy(self.model, self.method),
            "rng_state": _capture_rng_state(),
        }

    def _save_checkpoint(self, name: str, iteration: int) -> Path:
        path = self.checkpoint_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(iteration), temporary)
        temporary.replace(path)
        return path

    def _load_checkpoint(self, path: str | Path) -> int:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint not found: {source}")
        payload = torch.load(source, map_location=self.device, weights_only=False)
        if payload.get("experiment") != "hust_hybrid_anil_boil":
            raise ValueError("checkpoint belongs to a different experiment")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported hybrid checkpoint format")
        if payload["method"] != self.method:
            raise ValueError("checkpoint method does not match")
        if payload["held_out_protocol"] != self.held_out_protocol:
            raise ValueError("checkpoint held-out protocol does not match")
        if int(payload["seed"]) != self.seed:
            raise ValueError("checkpoint seed does not match")
        if payload["training_files"] != [sample.file_name for sample in self.train_samples]:
            raise ValueError("checkpoint training split does not match")
        if payload["validation_files"] != [sample.file_name for sample in self.validation_samples]:
            raise ValueError("checkpoint validation split does not match")
        if payload["normalizer"] != self.normalizer.state_dict():
            raise ValueError("checkpoint input normalization does not match")
        if payload["parameter_policy"] != parameter_policy(self.model, self.method):
            raise ValueError("checkpoint inner-loop policy does not match")
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.best_score = float(payload["best_score"])
        self.best_iteration = int(payload["best_iteration"])
        self.stale_evaluations = int(payload["stale_evaluations"])
        self.latest_validation_metrics = dict(payload.get("latest_validation_metrics", {}))
        self.history = list(payload["history"])
        _restore_rng_state(payload["rng_state"])
        return int(payload["iteration"]) + 1

    def train(self, resume: str | Path | None = None) -> TrainingResult:
        start_iteration = 1 if resume is None else self._load_checkpoint(resume)
        if resume is not None:
            LOGGER.info("Resumed %s at iteration %d", resume, start_iteration)
        stopped_iteration = start_iteration - 1
        already_stopped = self.stale_evaluations >= self.config.train.early_stopping_patience_evaluations
        iteration_range = (
            range(start_iteration, self.config.train.iterations + 1)
            if not already_stopped
            else ()
        )
        self.model.train()
        for iteration in iteration_range:
            self.optimizer.zero_grad(set_to_none=True)
            losses, diagnostics = self._training_objective(iteration)
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"non-finite outer loss at iteration {iteration}")
            losses.total.backward()
            gradient_norm = float(
                clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_norm)
            )
            self.optimizer.step()
            should_evaluate = (
                iteration % self.config.train.evaluation_interval == 0
                or iteration == self.config.train.iterations
            )
            validation_score: float | None = None
            domain_metrics: dict[str, float] = {}
            if should_evaluate:
                validation_score, domain_metrics = self._source_validation()
                self.latest_validation_metrics = {
                    "mae_cycles": validation_score,
                    **domain_metrics,
                }
                if validation_score < self.best_score:
                    self.best_score = validation_score
                    self.best_iteration = iteration
                    self.stale_evaluations = 0
                else:
                    self.stale_evaluations += 1
            row = {
                "iteration": iteration,
                "method": self.method,
                "total_loss": float(losses.total.detach().cpu()),
                "query_loss": float(losses.query_loss.detach().cpu()),
                "L_T": float(losses.total_prediction.detach().cpu()),
                "L_GY": float(losses.general_prediction.detach().cpu()),
                "L_G": float(losses.general_domain.detach().cpu()),
                "L_S": float(losses.specific_domain.detach().cpu()),
                "L_R": float(losses.reconstruction.detach().cpu()),
                "L_C": float(losses.consistency.detach().cpu()),
                "L_O": float(losses.orthogonal.detach().cpu()),
                "L_delta": float(losses.residual.detach().cpu()),
                "general_domain_accuracy_batch": float(losses.general_domain_accuracy.detach().cpu()),
                "specific_domain_accuracy_batch": float(losses.specific_domain_accuracy.detach().cpu()),
                "mean_absolute_specific_residual": float(losses.mean_absolute_residual.detach().cpu()),
                "outer_gradient_norm": gradient_norm,
                "grl_strength": self._grl_strength(iteration),
                "source_validation_mae_cycles": validation_score,
                "source_validation_general_domain_accuracy": domain_metrics.get("general_domain_accuracy"),
                "source_validation_specific_domain_accuracy": domain_metrics.get("specific_domain_accuracy"),
                **diagnostics,
            }
            self.history.append(row)
            if should_evaluate and self.best_iteration == iteration:
                self._save_checkpoint("best.pt", iteration)
            if iteration % self.config.train.checkpoint_interval == 0 or iteration == self.config.train.iterations:
                self._save_checkpoint("last.pt", iteration)
                pd.DataFrame(self.history).to_csv(self.run_dir / "training_history.csv", index=False)
            if iteration % self.config.train.log_interval == 0 or should_evaluate:
                LOGGER.info(
                    "method=%s fold=%s iter=%d/%d total=%.6g query=%.6g "
                    "L_GY=%.6g L_G=%.6g L_S=%.6g L_R=%.6g L_C=%.6g "
                    "L_O=%.6g L_delta=%.6g val_mae=%s best=%.3f@%d stale=%d",
                    self.method,
                    self.held_out_protocol,
                    iteration,
                    self.config.train.iterations,
                    row["total_loss"],
                    row["query_loss"],
                    row["L_GY"],
                    row["L_G"],
                    row["L_S"],
                    row["L_R"],
                    row["L_C"],
                    row["L_O"],
                    row["L_delta"],
                    "-" if validation_score is None else f"{validation_score:.3f}",
                    self.best_score,
                    self.best_iteration,
                    self.stale_evaluations,
                )
            stopped_iteration = iteration
            if should_evaluate and self.stale_evaluations >= self.config.train.early_stopping_patience_evaluations:
                LOGGER.info(
                    "Early stopping method=%s fold=%s at %d; best %.3f cycles at %d",
                    self.method,
                    self.held_out_protocol,
                    iteration,
                    self.best_score,
                    self.best_iteration,
                )
                self._save_checkpoint("last.pt", iteration)
                break
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("training finished without a best checkpoint")
        best_payload = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(best_payload["model_state"], strict=True)
        _, source_domain_metrics = self._source_validation()
        pd.DataFrame(self.history).to_csv(self.run_dir / "training_history.csv", index=False)
        save_training_diagnostics(self.history, self.run_dir / "training_diagnostics.png")
        save_json(
            {
                "method": self.method,
                "held_out_protocol": self.held_out_protocol,
                "target_protocol_used_for_training_or_checkpoint_selection": False,
                "auxiliary_loss_data": "post-adaptation query cells from source protocols only",
                "consistency_pairs": "different source protocols with similar RUL/500 state",
                "label_normalization": "none; raw RUL cycles",
                "best_iteration": self.best_iteration,
                "best_source_validation_mae_cycles": self.best_score,
                "stopped_iteration": stopped_iteration,
                **source_domain_metrics,
            },
            self.run_dir / "training_summary.json",
        )
        return TrainingResult(
            model=self.model,
            normalizer=self.normalizer,
            best_iteration=self.best_iteration,
            best_validation_mae_cycles=self.best_score,
            stopped_iteration=stopped_iteration,
            checkpoint_path=best_path,
            history=list(self.history),
            source_domain_metrics=source_domain_metrics,
        )


__all__ = ["HybridTrainer", "TrainingResult", "resolve_device", "set_global_seed"]

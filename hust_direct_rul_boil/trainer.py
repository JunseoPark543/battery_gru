"""Leakage-safe HUST protocol-DG trainer with raw-cycle BOIL regression."""

from __future__ import annotations

import copy
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

from .boil import boil_episode
from .config import ExperimentConfig, PROFILE_CHANNELS, SCALAR_FEATURES
from .data import CellSample, InputNormalizer, protocol_sort_key
from .losses import joint_loss
from .metrics import save_json, save_training_figure
from .model import HUSTDirectRULModel


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: HUSTDirectRULModel
    normalizer: InputNormalizer
    best_iteration: int
    best_validation_mae_cycles: float
    stopped_iteration: int
    checkpoint_path: Path
    history: list[dict[str, Any]]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


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
    torch.set_rng_state(
        torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu")
    )
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [
                torch.as_tensor(item, dtype=torch.uint8, device="cpu")
                for item in state["cuda"]
            ]
        )


class SourceOnlyTrainer:
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
        self.device = device
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        if any(sample.protocol == held_out_protocol for sample in self.all_source_samples):
            raise RuntimeError("held-out target protocol leaked into trainer")
        self.source_protocols = sorted(
            {sample.protocol for sample in self.all_source_samples},
            key=protocol_sort_key,
        )
        if len(self.source_protocols) < 2:
            raise ValueError("BOIL requires at least two source protocols")
        self.train_samples, self.validation_samples = self._split_source_validation()
        self.normalizer = InputNormalizer.fit(
            self.train_samples, config.data.normalization_epsilon
        )
        self.protocol_to_index = {
            protocol: index for index, protocol in enumerate(self.source_protocols)
        }
        self.model = HUSTDirectRULModel(
            waveform_channels=len(PROFILE_CHANNELS),
            scalar_features=len(SCALAR_FEATURES),
            history_length=config.data.history_length,
            source_domains=len(self.source_protocols),
            config=config.model,
        ).to(device)
        source_rul_median = float(
            np.median([sample.rul_cycles for sample in self.train_samples])
        )
        self.model.predictor.initialize_bias(source_rul_median)
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
        save_json(
            {
                "held_out_target_protocol": held_out_protocol,
                "target_protocol_cells_in_trainer": False,
                "training_files": [sample.file_name for sample in self.train_samples],
                "source_validation_files": [
                    sample.file_name for sample in self.validation_samples
                ],
                "normalization_fit_files": [
                    sample.file_name for sample in self.train_samples
                ],
            },
            self.run_dir / "source_split.json",
        )

    def _split_source_validation(self) -> tuple[list[CellSample], list[CellSample]]:
        validation: list[CellSample] = []
        training: list[CellSample] = []
        count = self.config.train.validation_cells_per_protocol
        for protocol in self.source_protocols:
            group = sorted(
                (sample for sample in self.all_source_samples if sample.protocol == protocol),
                key=lambda sample: (sample.replicate, sample.file_name),
            )
            if len(group) <= count:
                raise ValueError(
                    f"{protocol} has {len(group)} cells but validation requires {count}"
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

    def _batch(
        self, samples: Sequence[CellSample]
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not samples:
            raise ValueError("cannot create an empty cell batch")
        return (
            torch.stack([self.waveforms[sample.file_name] for sample in samples]),
            torch.stack([self.scalars[sample.file_name] for sample in samples]),
            torch.stack([self.targets[sample.file_name] for sample in samples]),
        )

    def _sample_protocols(
        self, protocols: Sequence[str], cells_per_protocol: int
    ) -> list[CellSample]:
        selected: list[CellSample] = []
        for protocol in protocols:
            group = [
                sample for sample in self.train_samples if sample.protocol == protocol
            ]
            selected.extend(random.sample(group, min(cells_per_protocol, len(group))))
        return selected

    def _augment(self, waveforms: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        config = self.config.augmentation
        if not config.enabled:
            return waveforms, scalars
        augmented_waveforms = waveforms
        augmented_scalars = scalars
        if config.waveform_gaussian_std:
            augmented_waveforms = augmented_waveforms + (
                torch.randn_like(augmented_waveforms) * config.waveform_gaussian_std
            )
        if config.scalar_gaussian_std:
            augmented_scalars = augmented_scalars + (
                torch.randn_like(augmented_scalars) * config.scalar_gaussian_std
            )
        if config.profile_channel_mask_probability:
            keep_channel = (
                torch.rand(
                    augmented_waveforms.shape[0],
                    1,
                    1,
                    augmented_waveforms.shape[3],
                    device=self.device,
                )
                >= config.profile_channel_mask_probability
            )
            augmented_waveforms = augmented_waveforms * keep_channel.to(
                augmented_waveforms.dtype
            )
        if config.cycle_mask_probability:
            keep_cycle = (
                torch.rand(
                    augmented_waveforms.shape[0],
                    augmented_waveforms.shape[1],
                    1,
                    1,
                    device=self.device,
                )
                >= config.cycle_mask_probability
            )
            augmented_waveforms = augmented_waveforms * keep_cycle.to(
                augmented_waveforms.dtype
            )
            augmented_scalars = augmented_scalars * keep_cycle.squeeze(-1).to(
                augmented_scalars.dtype
            )
        return augmented_waveforms, augmented_scalars

    def _grl_strength(self, iteration: int) -> float:
        progress = iteration / max(1, self.config.train.iterations)
        return self.config.loss.grl_max_strength * (
            2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        )

    def _validation_score(self) -> float:
        self.model.eval()
        waveforms, scalars, targets = self._batch(self.validation_samples)
        with torch.no_grad():
            prediction = self.model(waveforms, scalars, grl_strength=0.0).prediction
        score = float(torch.mean(torch.abs(prediction - targets)).cpu())
        self.model.train()
        return score

    def _checkpoint_payload(self, iteration: int) -> dict[str, Any]:
        return {
            "format_version": 1,
            "iteration": iteration,
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
            "history": self.history,
            "config": self.config.to_dict(),
            "rng_state": _capture_rng_state(),
        }

    def _save_checkpoint(self, name: str, iteration: int) -> Path:
        path = self.checkpoint_dir / name
        torch.save(self._checkpoint_payload(iteration), path)
        return path

    def _load_checkpoint(self, path: str | Path) -> int:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint not found: {source}")
        payload = torch.load(source, map_location=self.device, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported HUST checkpoint format")
        if payload["held_out_protocol"] != self.held_out_protocol:
            raise ValueError("checkpoint held-out protocol does not match")
        if int(payload["seed"]) != self.seed:
            raise ValueError("checkpoint seed does not match")
        if payload["training_files"] != [
            sample.file_name for sample in self.train_samples
        ]:
            raise ValueError("checkpoint training cell split does not match")
        if payload["validation_files"] != [
            sample.file_name for sample in self.validation_samples
        ]:
            raise ValueError("checkpoint validation cell split does not match")
        if payload["normalizer"] != self.normalizer.state_dict():
            raise ValueError("checkpoint input normalization does not match")
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.best_score = float(payload["best_score"])
        self.best_iteration = int(payload["best_iteration"])
        self.stale_evaluations = int(payload["stale_evaluations"])
        self.history = list(payload["history"])
        _restore_rng_state(payload["rng_state"])
        return int(payload["iteration"]) + 1

    def train(self, resume: str | Path | None = None) -> TrainingResult:
        start_iteration = 1
        if resume is not None:
            start_iteration = self._load_checkpoint(resume)
            LOGGER.info("Resumed %s at iteration %d", resume, start_iteration)
        stopped_iteration = start_iteration - 1
        already_stopped = (
            self.stale_evaluations
            >= self.config.train.early_stopping_patience_evaluations
        )
        iteration_range = (
            range(start_iteration, self.config.train.iterations + 1)
            if not already_stopped
            else ()
        )
        if already_stopped:
            LOGGER.info("Checkpoint already early-stopped; evaluating best.pt directly")
        self.model.train()
        for iteration in iteration_range:
            query_protocol = self.source_protocols[
                (iteration - 1) % len(self.source_protocols)
            ]
            meta_train_protocols = [
                protocol
                for protocol in self.source_protocols
                if protocol != query_protocol
            ]

            joint_samples = self._sample_protocols(
                meta_train_protocols, self.config.train.joint_cells_per_domain
            )
            joint_waveforms, joint_scalars, joint_targets = self._batch(joint_samples)
            joint_domain = torch.as_tensor(
                [self.protocol_to_index[sample.protocol] for sample in joint_samples],
                dtype=torch.long,
                device=self.device,
            )
            waveform_views: list[Tensor] = []
            scalar_views: list[Tensor] = []
            target_views: list[Tensor] = []
            domain_views: list[Tensor] = []
            for _ in range(self.config.augmentation.joint_views_per_cell):
                augmented_waveforms, augmented_scalars = self._augment(
                    joint_waveforms, joint_scalars
                )
                waveform_views.append(augmented_waveforms)
                scalar_views.append(augmented_scalars)
                target_views.append(joint_targets)
                domain_views.append(joint_domain)
            joint_waveforms = torch.cat(waveform_views, dim=0)
            joint_scalars = torch.cat(scalar_views, dim=0)
            joint_targets = torch.cat(target_views, dim=0)
            joint_domain = torch.cat(domain_views, dim=0)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(
                joint_waveforms, joint_scalars, self._grl_strength(iteration)
            )
            joint = joint_loss(output, joint_targets, joint_domain, self.config.loss)
            joint.total.backward()
            joint_grad = float(
                clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_norm)
            )
            self.optimizer.step()

            support_samples = self._sample_protocols(
                meta_train_protocols,
                self.config.train.meta_support_cells_per_domain,
            )
            query_samples = self._sample_protocols(
                [query_protocol], self.config.train.joint_cells_per_domain
            )
            support_waveforms, support_scalars, support_targets = self._batch(
                support_samples
            )
            query_waveforms, query_scalars, query_targets = self._batch(query_samples)
            support_waveforms, support_scalars = self._augment(
                support_waveforms, support_scalars
            )
            self.optimizer.zero_grad(set_to_none=True)
            meta = boil_episode(
                self.model,
                support_waveforms,
                support_scalars,
                support_targets,
                query_waveforms,
                query_scalars,
                query_targets,
                inner_steps=self.config.train.inner_steps,
                inner_lr=self.config.train.inner_lr,
                second_order=self.config.train.second_order,
                loss_config=self.config.loss,
            )
            meta.total.backward()
            meta_grad = float(
                clip_grad_norm_(
                    self.model.meta_parameters(), self.config.train.gradient_clip_norm
                )
            )
            self.optimizer.step()

            should_evaluate = (
                iteration % self.config.train.evaluation_interval == 0
                or iteration == self.config.train.iterations
            )
            validation_score: float | None = None
            if should_evaluate:
                validation_score = self._validation_score()
                if validation_score < self.best_score:
                    self.best_score = validation_score
                    self.best_iteration = iteration
                    self.stale_evaluations = 0
                else:
                    self.stale_evaluations += 1
            row = {
                "iteration": iteration,
                "joint_total": float(joint.total.detach()),
                "joint_task": float(joint.task.detach()),
                "joint_domain": float(joint.domain.detach()),
                "joint_fuzzy": float(joint.fuzzy.detach()),
                "joint_orthogonality": float(joint.orthogonality.detach()),
                "meta_total": float(meta.total.detach()),
                "meta_support": float(meta.support.detach()),
                "meta_query": float(meta.query.detach()),
                "meta_query_protocol": query_protocol,
                "body_update_norm": float(meta.update_norm.detach()),
                "joint_gradient_norm": joint_grad,
                "meta_gradient_norm": meta_grad,
                "grl_strength": self._grl_strength(iteration),
                "source_validation_mae_cycles": validation_score,
            }
            self.history.append(row)
            if should_evaluate and self.best_iteration == iteration:
                self._save_checkpoint("best.pt", iteration)
            if (
                iteration % self.config.train.checkpoint_interval == 0
                or iteration == self.config.train.iterations
            ):
                self._save_checkpoint("last.pt", iteration)
                pd.DataFrame(self.history).to_csv(
                    self.run_dir / "training_history.csv", index=False
                )
            if iteration % self.config.train.log_interval == 0 or should_evaluate:
                LOGGER.info(
                    "fold=%s iter=%d/%d joint=%.6g raw_task=%.6g "
                    "meta_query=%.6g meta_test=%s val_mae=%s best=%.3f@%d stale=%d",
                    self.held_out_protocol,
                    iteration,
                    self.config.train.iterations,
                    row["joint_total"],
                    row["joint_task"],
                    row["meta_query"],
                    query_protocol,
                    "-" if validation_score is None else f"{validation_score:.3f}",
                    self.best_score,
                    self.best_iteration,
                    self.stale_evaluations,
                )
            stopped_iteration = iteration
            if (
                should_evaluate
                and self.stale_evaluations
                >= self.config.train.early_stopping_patience_evaluations
            ):
                LOGGER.info(
                    "Early stopping at %d; best source-cell validation MAE "
                    "%.3f cycles at %d",
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
        pd.DataFrame(self.history).to_csv(self.run_dir / "training_history.csv", index=False)
        save_json(
            {
                "held_out_protocol": self.held_out_protocol,
                "target_labels_used_for_training_or_selection": False,
                "label_normalization": "none; prediction is raw RUL cycles",
                "best_iteration": self.best_iteration,
                "best_source_validation_mae_cycles": self.best_score,
                "stopped_iteration": stopped_iteration,
            },
            self.run_dir / "training_summary.json",
        )
        save_training_figure(self.history, self.run_dir / "training_diagnostics.png")
        return TrainingResult(
            model=self.model,
            normalizer=self.normalizer,
            best_iteration=self.best_iteration,
            best_validation_mae_cycles=self.best_score,
            stopped_iteration=stopped_iteration,
            checkpoint_path=best_path,
            history=copy.deepcopy(self.history),
        )

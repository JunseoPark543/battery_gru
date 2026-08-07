"""Source-only joint DG and BOIL trainer.

The trainer receives source cells only. Held-out target cells and their life
labels remain outside this object until final evaluation.
"""

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

from .boil import adapted_prediction, boil_episode
from .config import ExperimentConfig
from .data import CellSample, FeatureNormalizer, RULNormalizer
from .losses import joint_loss
from .metrics import save_json, save_training_figure
from .model import DirectRULBOILModel


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: DirectRULBOILModel
    feature_normalizer: FeatureNormalizer
    rul_normalizer: RULNormalizer
    best_iteration: int
    best_source_cv_mae_cycles: float
    stopped_iteration: int
    history: list[dict[str, Any]]
    checkpoint_path: Path


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
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
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
    cpu_state = torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu")
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = [
            torch.as_tensor(item, dtype=torch.uint8, device="cpu")
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


class SourceOnlyTrainer:
    def __init__(
        self,
        source_samples: Sequence[CellSample],
        held_out_domain: str,
        config: ExperimentConfig,
        device: torch.device,
        run_dir: str | Path,
        seed: int,
    ) -> None:
        self.samples = list(source_samples)
        self.held_out_domain = held_out_domain
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        if not self.samples:
            raise ValueError("source-only trainer requires at least one sample")
        if any(sample.domain == held_out_domain for sample in self.samples):
            raise RuntimeError("target-domain leakage: trainer received a held-out cell")
        self.source_domains = sorted({sample.domain for sample in self.samples})
        if len(self.source_domains) < 2:
            raise ValueError("BOIL domain episodes require at least two source domains")
        self.domain_to_index = {
            domain: index for index, domain in enumerate(self.source_domains)
        }
        self.feature_normalizer = FeatureNormalizer.fit(
            self.samples, config.data.normalization_epsilon
        )
        self.rul_normalizer = RULNormalizer.fit(
            self.samples, config.data.normalization_epsilon
        )
        self.model = DirectRULBOILModel(
            input_size=len(config.data.features),
            history_length=config.data.history_length,
            num_source_domains=len(self.source_domains),
            config=config.model,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.outer_lr,
            weight_decay=config.train.weight_decay,
        )
        self.features = {
            sample.file_name: torch.as_tensor(
                self.feature_normalizer.transform(sample.features),
                dtype=torch.float32,
                device=device,
            )
            for sample in self.samples
        }
        self.targets = {
            sample.file_name: torch.as_tensor(
                float(self.rul_normalizer.transform(sample.rul_cycles)),
                dtype=torch.float32,
                device=device,
            )
            for sample in self.samples
        }
        self.history: list[dict[str, Any]] = []
        self.best_score = float("inf")
        self.best_iteration = 0
        self.stale_evaluations = 0

    def _batch(self, samples: Sequence[CellSample]) -> tuple[Tensor, Tensor]:
        features = torch.stack([self.features[sample.file_name] for sample in samples])
        targets = torch.stack([self.targets[sample.file_name] for sample in samples])
        return features, targets

    def _augment(self, features: Tensor) -> Tensor:
        config = self.config.augmentation
        if not config.enabled:
            return features
        augmented = features
        if config.gaussian_std:
            augmented = augmented + torch.randn_like(augmented) * config.gaussian_std
        if config.feature_mask_probability:
            keep_feature = (
                torch.rand(
                    augmented.shape[0], 1, augmented.shape[2], device=augmented.device
                )
                >= config.feature_mask_probability
            )
            augmented = augmented * keep_feature.to(augmented.dtype)
        if config.cycle_mask_probability:
            keep_cycle = (
                torch.rand(
                    augmented.shape[0], augmented.shape[1], 1, device=augmented.device
                )
                >= config.cycle_mask_probability
            )
            augmented = augmented * keep_cycle.to(augmented.dtype)
        return augmented

    def _joint_batch(self) -> tuple[Tensor, Tensor, Tensor]:
        features, targets = self._batch(self.samples)
        feature_views = []
        target_views = []
        domain_views = []
        domain_targets = torch.as_tensor(
            [self.domain_to_index[sample.domain] for sample in self.samples],
            dtype=torch.long,
            device=self.device,
        )
        for _ in range(self.config.augmentation.joint_views_per_cell):
            feature_views.append(self._augment(features))
            target_views.append(targets)
            domain_views.append(domain_targets)
        return (
            torch.cat(feature_views, dim=0),
            torch.cat(target_views, dim=0),
            torch.cat(domain_views, dim=0),
        )

    def _meta_episode(self, iteration: int) -> tuple[list[CellSample], list[CellSample], str]:
        query_domain = self.source_domains[(iteration - 1) % len(self.source_domains)]
        support = [sample for sample in self.samples if sample.domain != query_domain]
        query = [sample for sample in self.samples if sample.domain == query_domain]
        if not support or not query:
            raise RuntimeError("invalid source-domain BOIL episode")
        return support, query, query_domain

    def _grl_strength(self, iteration: int) -> float:
        progress = iteration / max(1, self.config.train.iterations)
        scheduled = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        return self.config.loss.grl_max_strength * scheduled

    def _source_cv_score(self) -> float:
        """Source-domain meta-CV used only for checkpoint selection/early stopping."""
        self.model.eval()
        absolute_errors: list[float] = []
        for query_domain in self.source_domains:
            support = [sample for sample in self.samples if sample.domain != query_domain]
            query = [sample for sample in self.samples if sample.domain == query_domain]
            support_x, support_y = self._batch(support)
            query_x, _ = self._batch(query)
            prediction = adapted_prediction(
                self.model,
                support_x,
                support_y,
                query_x,
                inner_steps=self.config.train.inner_steps,
                inner_lr=self.config.train.inner_lr,
                huber_delta=self.config.loss.huber_delta,
            )
            predicted_cycles = self.rul_normalizer.inverse(
                prediction.detach().cpu().numpy()
            )
            actual_cycles = np.asarray([sample.rul_cycles for sample in query])
            absolute_errors.extend(np.abs(predicted_cycles - actual_cycles).tolist())
        self.model.train()
        return float(np.mean(absolute_errors))

    def _checkpoint_payload(self, iteration: int) -> dict[str, Any]:
        return {
            "format_version": 1,
            "iteration": iteration,
            "held_out_domain": self.held_out_domain,
            "source_domains": self.source_domains,
            "source_files": [sample.file_name for sample in self.samples],
            "seed": self.seed,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "feature_normalizer": self.feature_normalizer.state_dict(),
            "rul_normalizer": self.rul_normalizer.state_dict(),
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
            raise ValueError("unsupported checkpoint format")
        if payload["held_out_domain"] != self.held_out_domain:
            raise ValueError("resume checkpoint held-out domain does not match")
        if int(payload["seed"]) != self.seed:
            raise ValueError("resume checkpoint seed does not match")
        expected_files = [sample.file_name for sample in self.samples]
        if payload["source_files"] != expected_files:
            raise ValueError("resume checkpoint source-cell split does not match")
        if payload["feature_normalizer"] != self.feature_normalizer.state_dict():
            raise ValueError("resume feature-normalization statistics do not match")
        if payload["rul_normalizer"] != self.rul_normalizer.state_dict():
            raise ValueError("resume RUL-normalization statistics do not match")
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
            LOGGER.info("Resumed %s at outer iteration %d", resume, start_iteration)
        stopped_iteration = start_iteration - 1
        self.model.train()
        already_early_stopped = (
            self.stale_evaluations
            >= self.config.train.early_stopping_patience_evaluations
        )
        if already_early_stopped:
            LOGGER.info(
                "Checkpoint had already early-stopped at %d; loading best.pt "
                "without additional training",
                stopped_iteration,
            )
        iteration_range = (
            range(start_iteration, self.config.train.iterations + 1)
            if not already_early_stopped
            else ()
        )
        for iteration in iteration_range:
            joint_x, joint_y, joint_domain = self._joint_batch()
            self.optimizer.zero_grad(set_to_none=True)
            joint_output = self.model(joint_x, self._grl_strength(iteration))
            joint = joint_loss(joint_output, joint_y, joint_domain, self.config.loss)
            joint.total.backward()
            joint_grad = float(
                clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_norm)
            )
            self.optimizer.step()

            support_samples, query_samples, query_domain = self._meta_episode(iteration)
            support_x, support_y = self._batch(support_samples)
            query_x, query_y = self._batch(query_samples)
            support_x = self._augment(support_x)
            self.optimizer.zero_grad(set_to_none=True)
            meta = boil_episode(
                self.model,
                support_x,
                support_y,
                query_x,
                query_y,
                inner_steps=self.config.train.inner_steps,
                inner_lr=self.config.train.inner_lr,
                huber_delta=self.config.loss.huber_delta,
                second_order=self.config.train.second_order,
                support_weight=self.config.loss.meta_support_weight,
                query_weight=self.config.loss.meta_query_weight,
            )
            meta.total_loss.backward()
            meta_grad = float(
                clip_grad_norm_(self.model.meta_parameters(), self.config.train.gradient_clip_norm)
            )
            self.optimizer.step()

            should_evaluate = (
                iteration % self.config.train.evaluation_interval == 0
                or iteration == self.config.train.iterations
            )
            source_score: float | None = None
            if should_evaluate:
                source_score = self._source_cv_score()
                if source_score < self.best_score:
                    self.best_score = source_score
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
                "meta_total": float(meta.total_loss.detach()),
                "meta_support": float(meta.support_loss.detach()),
                "meta_query": float(meta.query_loss.detach()),
                "meta_query_domain": query_domain,
                "body_update_norm": float(meta.body_update_norm.detach()),
                "joint_gradient_norm": joint_grad,
                "meta_gradient_norm": meta_grad,
                "grl_strength": self._grl_strength(iteration),
                "source_cv_meta_mae_cycles": source_score,
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
                    "fold=%s iter=%d/%d joint=%.6g task=%.6g meta_query=%.6g "
                    "meta_test=%s source_cv_mae=%s best=%.4f@%d stale=%d",
                    self.held_out_domain,
                    iteration,
                    self.config.train.iterations,
                    row["joint_total"],
                    row["joint_task"],
                    row["meta_query"],
                    query_domain,
                    "-" if source_score is None else f"{source_score:.4f}",
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
                    "Source-only early stopping at %d; best %.4f cycles at %d",
                    iteration,
                    self.best_score,
                    self.best_iteration,
                )
                self._save_checkpoint("last.pt", iteration)
                break

        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("training finished without a source-selected best checkpoint")
        best_payload = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(best_payload["model_state"], strict=True)
        pd.DataFrame(self.history).to_csv(self.run_dir / "training_history.csv", index=False)
        save_json(
            {
                "held_out_domain": self.held_out_domain,
                "target_labels_used_for_selection": False,
                "best_iteration": self.best_iteration,
                "best_source_cv_meta_mae_cycles": self.best_score,
                "stopped_iteration": stopped_iteration,
            },
            self.run_dir / "training_summary.json",
        )
        save_training_figure(self.history, self.run_dir / "training_diagnostics.png")
        return TrainingResult(
            model=self.model,
            feature_normalizer=self.feature_normalizer,
            rul_normalizer=self.rul_normalizer,
            best_iteration=self.best_iteration,
            best_source_cv_mae_cycles=self.best_score,
            stopped_iteration=stopped_iteration,
            history=copy.deepcopy(self.history),
            checkpoint_path=best_path,
        )

"""Functional selective inner loops for MAML, ANIL, BOIL, and the hybrid."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.func import functional_call

from .config import ExperimentConfig, METHODS
from .losses import inner_objective
from .model import GeneralSpecificRULModel, HybridOutput


@dataclass
class AdaptationResult:
    parameters: OrderedDict[str, Tensor]
    support_losses: list[Tensor]
    update_norms: dict[str, Tensor]


def inner_module_names(method: str, prediction_mode: str) -> tuple[str, ...]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    heads = (
        ("general_head", "specific_head")
        if prediction_mode == "residual"
        else ("concat_head",)
    )
    if method == "supervised":
        return ()
    if method == "maml":
        return ("general_encoder", "specific_encoder", *heads)
    if method == "anil":
        return heads
    if method == "boil":
        return ("general_encoder", "specific_encoder")
    # Proposed: the prediction mapping of the general path is ANIL-adapted,
    # while the specific feature extractor is BOIL-adapted.
    general_mapping = "general_head" if prediction_mode == "residual" else "concat_head"
    return (general_mapping, "specific_encoder")


def inner_parameter_names(
    model: GeneralSpecificRULModel,
    method: str,
) -> tuple[str, ...]:
    prefixes = tuple(
        f"{name}." for name in inner_module_names(method, model.ablation.prediction_mode)
    )
    return tuple(
        name for name, _ in model.named_parameters() if name.startswith(prefixes)
    )


def parameter_policy(model: GeneralSpecificRULModel, method: str) -> dict[str, object]:
    adapted_modules = list(inner_module_names(method, model.ablation.prediction_mode))
    adapted_names = list(inner_parameter_names(model, method))
    named_parameters = dict(model.named_parameters())
    all_modules = [
        "general_encoder",
        "general_head",
        "specific_encoder",
        "specific_head",
        "concat_head",
        "general_domain_classifier",
        "specific_domain_classifier",
        "reconstruction_decoder",
    ]
    return {
        "method": method,
        "prediction_mode": model.ablation.prediction_mode,
        "inner_updated_modules": adapted_modules,
        "inner_frozen_modules": [name for name in all_modules if name not in adapted_modules],
        "inner_updated_parameter_names": adapted_names,
        "inner_updated_parameter_count": sum(
            named_parameters[name].numel() for name in adapted_names
        ),
        "outer_updated_parameter_names": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "domain_and_reconstruction_losses_in_inner_loop": False,
    }


def _learning_rate(name: str, config: ExperimentConfig) -> float:
    if name.startswith("general_head.") or name.startswith("concat_head."):
        return config.train.inner_lr_general_head
    if name.startswith("specific_encoder."):
        return config.train.inner_lr_specific_encoder
    return config.train.inner_lr_other


def forward_with_parameters(
    model: GeneralSpecificRULModel,
    parameters: Mapping[str, Tensor],
    waveforms: Tensor,
    scalars: Tensor,
    *,
    grl_strength: float = 0.0,
) -> HybridOutput:
    if not parameters:
        return model(waveforms, scalars, grl_strength=grl_strength)
    return functional_call(
        model,
        parameters,
        (waveforms, scalars),
        {"grl_strength": grl_strength},
        strict=False,
    )


def adapt_task(
    model: GeneralSpecificRULModel,
    support_waveforms: Tensor,
    support_scalars: Tensor,
    support_targets: Tensor,
    *,
    method: str,
    steps: int,
    config: ExperimentConfig,
    second_order: bool | None = None,
) -> AdaptationResult:
    if steps < 0:
        raise ValueError("adaptation steps cannot be negative")
    selected = inner_parameter_names(model, method)
    fast = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name in selected
    )
    if steps > 0 and not fast:
        if method == "supervised":
            return AdaptationResult(fast, [], {})
        raise RuntimeError(f"method={method} selected no inner-loop parameters")
    create_graph = (not config.train.first_order) if second_order is None else second_order
    support_losses: list[Tensor] = []
    norm_squares: dict[str, list[Tensor]] = {}
    for _ in range(steps):
        output = forward_with_parameters(
            model,
            fast,
            support_waveforms,
            support_scalars,
            grl_strength=0.0,
        )
        loss = inner_objective(output, support_targets, config.loss)
        gradients = torch.autograd.grad(
            loss,
            tuple(fast.values()),
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        updated: OrderedDict[str, Tensor] = OrderedDict()
        for (name, parameter), gradient in zip(fast.items(), gradients):
            learning_rate = _learning_rate(name, config)
            update = learning_rate * gradient
            updated[name] = parameter - update
            module = name.split(".", 1)[0]
            norm_squares.setdefault(module, []).append(update.square().sum())
        fast = updated
        support_losses.append(loss)
    update_norms = {
        module: torch.sqrt(torch.stack(values).sum().clamp_min(0.0))
        for module, values in norm_squares.items()
    }
    return AdaptationResult(fast, support_losses, update_norms)


def concatenate_outputs(outputs: Sequence[HybridOutput]) -> HybridOutput:
    if not outputs:
        raise ValueError("cannot concatenate an empty output sequence")

    def cat(name: str) -> Tensor:
        return torch.cat([getattr(output, name) for output in outputs], dim=0)

    specific_logits = [output.specific_domain_logits for output in outputs]
    reconstructions = [output.reconstruction for output in outputs]
    return HybridOutput(
        prediction=cat("prediction"),
        general_prediction=cat("general_prediction"),
        specific_residual=cat("specific_residual"),
        general_embedding=cat("general_embedding"),
        specific_embedding=cat("specific_embedding"),
        general_domain_logits=cat("general_domain_logits"),
        specific_domain_logits=(
            None
            if any(value is None for value in specific_logits)
            else torch.cat([value for value in specific_logits if value is not None], dim=0)
        ),
        reconstruction=(
            None
            if any(value is None for value in reconstructions)
            else torch.cat([value for value in reconstructions if value is not None], dim=0)
        ),
        reconstruction_target=cat("reconstruction_target"),
    )


def representation_change(before: Tensor, after: Tensor) -> dict[str, float]:
    if before.shape != after.shape:
        raise ValueError("representation tensors must have equal shapes")
    cosine = 1.0 - torch.nn.functional.cosine_similarity(before, after, dim=-1)
    l2 = torch.linalg.vector_norm(after - before, dim=-1)
    return {
        "cosine_distance": float(cosine.mean().detach().cpu()),
        "l2_distance": float(l2.mean().detach().cpu()),
    }

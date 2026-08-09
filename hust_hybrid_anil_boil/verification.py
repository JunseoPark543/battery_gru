"""Synthetic checks for the user to run before starting long experiments.

This module intentionally uses no HUST files and performs no optimization loop.
It checks one forward/backward and one selective inner update.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from hust_direct_rul_boil.model import gradient_reverse

from .config import ExperimentConfig
from .losses import outer_objective
from .meta import adapt_task, forward_with_parameters, parameter_policy
from .model import GeneralSpecificRULModel


def _tiny_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.waveform_hidden = 8
    config.model.waveform_embedding = 8
    config.model.scalar_embedding = 8
    config.model.d_model = 16
    config.model.nhead = 4
    config.model.attention_stages = 1
    config.model.dim_feedforward = 32
    config.model.embedding_dim = 8
    config.model.predictor_hidden = 8
    config.model.domain_hidden = 8
    config.model.reconstruction_hidden = 16
    config.model.dropout = 0.0
    config.train.first_order = True
    config.train.inner_steps = 1
    config.method = "hybrid"
    config.validate()
    return config


def run_checks() -> dict[str, object]:
    torch.manual_seed(7)
    config = _tiny_config()
    model = GeneralSpecificRULModel(8, 14, 100, 3, config.model, config.ablation)
    model.eval()
    waveforms = torch.randn(4, 100, 16, 8)
    scalars = torch.randn(4, 100, 14)
    targets = torch.tensor([300.0, 450.0, 600.0, 750.0])
    domains = torch.tensor([0, 0, 1, 2])
    output = model(waveforms, scalars, grl_strength=1.0)
    assert output.prediction.shape == (4,)
    assert output.general_prediction.shape == (4,)
    assert output.specific_residual.shape == (4,)
    assert output.reconstruction is not None
    assert output.reconstruction.shape == output.reconstruction_target.shape
    assert output.reconstruction.shape == (4, 100, 30)
    assert all(torch.isfinite(value).all() for value in (
        output.prediction,
        output.general_embedding,
        output.specific_embedding,
        output.reconstruction,
    ))
    loss = outer_objective(output, targets, domains, config.loss, config.ablation)
    loss.total.backward()
    assert all(
        any(parameter.grad is not None for parameter in module.parameters())
        for module in (
            model.general_encoder,
            model.specific_encoder,
            model.general_head,
            model.specific_head,
        )
    )
    model.zero_grad(set_to_none=True)

    # GRL must negate, and only negate, the encoder-side gradient.
    direct_x = torch.tensor([1.0, -2.0], requires_grad=True)
    direct_x.sum().backward()
    direct_gradient = direct_x.grad.detach().clone()
    reversed_x = torch.tensor([1.0, -2.0], requires_grad=True)
    gradient_reverse(reversed_x, 1.0).sum().backward()
    assert torch.allclose(reversed_x.grad, -direct_gradient)

    adapted = adapt_task(
        model,
        waveforms[:2],
        scalars[:2],
        targets[:2],
        method="hybrid",
        steps=1,
        config=config,
    )
    policy = parameter_policy(model, "hybrid")
    selected = tuple(adapted.parameters)
    assert selected
    assert all(name.startswith(("general_head.", "specific_encoder.")) for name in selected)
    assert not any(name.startswith("general_encoder.") for name in selected)
    assert not any(name.startswith("specific_head.") for name in selected)
    assert adapted.update_norms["general_head"] > 0
    assert adapted.update_norms["specific_encoder"] > 0
    assert set(adapted.parameter_trajectory) == {0, 1}
    query_output = forward_with_parameters(
        model, adapted.parameters, waveforms[2:], scalars[2:]
    )
    query_loss = outer_objective(
        query_output, targets[2:], domains[2:], config.loss, config.ablation
    ).total
    query_loss.backward()
    outer_modules = (
        model.general_encoder,
        model.general_head,
        model.specific_encoder,
        model.specific_head,
    )
    assert all(any(parameter.grad is not None for parameter in module.parameters()) for module in outer_modules)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "roundtrip.pt"
        payload = {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "parameter_policy": policy,
        }
        torch.save(payload, checkpoint)
        restored = GeneralSpecificRULModel(8, 14, 100, 3, config.model, config.ablation)
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"],
            strict=True,
        )
    return {
        "import": "ok",
        "forward_shape": list(output.prediction.shape),
        "reconstruction_shape": list(output.reconstruction.shape),
        "backward": "ok",
        "grl_direction": "ok",
        "inner_updated_modules": policy["inner_updated_modules"],
        "inner_frozen_general_encoder": True,
        "inner_frozen_specific_head": True,
        "inner_trajectory_steps": sorted(adapted.parameter_trajectory),
        "outer_gradient_all_prediction_modules": True,
        "finite_outputs": True,
        "checkpoint_roundtrip": "ok",
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))

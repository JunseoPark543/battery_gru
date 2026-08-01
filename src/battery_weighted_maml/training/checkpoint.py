"""Complete, resumable training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..seed import capture_rng_state, restore_rng_state


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    meta_iteration: int,
    best_metric: float,
    ema_metric: float,
    config: dict[str, Any],
    target_file_name: str,
    source_file_names: list[str],
    history_length: int,
    source_mode: str,
    seed: int,
    latest_alpha: torch.Tensor,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "outer_optimizer_state_dict": optimizer.state_dict(),
        "meta_iteration": meta_iteration,
        "best_metric": best_metric,
        "ema_metric": ema_metric,
        "config": config,
        "target_file_name": target_file_name,
        "source_file_names": source_file_names,
        "L": history_length,
        "source_mode": source_mode,
        "seed": seed,
        "latest_alpha": latest_alpha.detach().cpu(),
        "rng_states": capture_rng_state(),
    }


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    required = {
        "model_state_dict", "meta_iteration", "best_metric", "config",
        "target_file_name", "source_file_names", "L", "source_mode", "seed",
        "latest_alpha", "rng_states",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint {source} missing keys: {sorted(missing)}")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        if "outer_optimizer_state_dict" not in payload:
            raise ValueError(f"checkpoint {source} has no optimizer state")
        optimizer.load_state_dict(payload["outer_optimizer_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_states"])
    return payload

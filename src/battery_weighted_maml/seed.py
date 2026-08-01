"""Reproducible random-state helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(..., map_location='cuda')`` also moves serialized RNG
    # ByteTensors to CUDA, while the RNG restoration APIs require CPU state
    # tensors. Normalize them explicitly so CUDA resume remains portable.
    torch_state = state["torch"].detach().to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            item.detach().to(device="cpu", dtype=torch.uint8)
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    generator = torch.Generator(device=torch.device(device).type)
    generator.manual_seed(seed)
    return generator

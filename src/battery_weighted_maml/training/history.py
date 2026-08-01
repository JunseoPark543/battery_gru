"""Structured CSV histories collected during meta-training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TrainingHistory:
    iterations: list[dict[str, Any]] = field(default_factory=list)
    source_losses: list[dict[str, Any]] = field(default_factory=list)
    gradients: list[dict[str, Any]] = field(default_factory=list)
    alphas: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, run_dir: str | Path) -> "TrainingHistory":
        """Load existing CSV histories so resumed training appends instead of overwriting."""
        root = Path(run_dir)

        def records(path: Path) -> list[dict[str, Any]]:
            if not path.is_file() or path.stat().st_size == 0:
                return []
            return pd.read_csv(path).to_dict(orient="records")

        return cls(
            iterations=records(root / "training/iteration_history.csv"),
            source_losses=records(root / "training/source_loss_history.csv"),
            gradients=records(root / "training/gradient_history.csv"),
            alphas=records(root / "weights/alpha_history.csv"),
        )

    def save(self, run_dir: str | Path) -> None:
        root = Path(run_dir)
        (root / "training").mkdir(parents=True, exist_ok=True)
        (root / "weights").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.iterations).to_csv(root / "training/iteration_history.csv", index=False)
        pd.DataFrame(self.source_losses).to_csv(
            root / "training/source_loss_history.csv", index=False
        )
        pd.DataFrame(self.gradients).to_csv(
            root / "training/gradient_history.csv", index=False
        )
        pd.DataFrame(self.alphas).to_csv(root / "weights/alpha_history.csv", index=False)

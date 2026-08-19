"""Leakage-safe deterministic cell-level cross-validation splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import SplitConfig


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_cells: list[str]
    validation_cells: list[str]
    test_cells: list[str]

    def validate(self, all_cells: Sequence[str]) -> None:
        train, validation, test = map(
            set, (self.train_cells, self.validation_cells, self.test_cells)
        )
        if train & validation or train & test or validation & test:
            raise ValueError(f"fold {self.fold}: cell leakage across splits")
        if train | validation | test != set(all_cells):
            raise ValueError(f"fold {self.fold}: split does not cover every cell exactly once")
        if not train or not validation or not test:
            raise ValueError(f"fold {self.fold}: train/validation/test must all be non-empty")


def make_splits(cell_ids: Sequence[str], config: SplitConfig) -> list[FoldSplit]:
    identifiers = sorted(set(map(str, cell_ids)))
    if len(identifiers) != len(cell_ids):
        raise ValueError("cell IDs must be unique")
    number_of_folds = len(identifiers) if config.strategy == "loocv" else config.num_folds
    if len(identifiers) < number_of_folds or len(identifiers) < 3:
        raise ValueError(
            f"{len(identifiers)} cells cannot form {number_of_folds} folds with validation"
        )
    rng = np.random.default_rng(config.seed)
    shuffled = np.asarray(identifiers, dtype=object)
    rng.shuffle(shuffled)
    test_chunks = np.array_split(shuffled, number_of_folds)
    splits: list[FoldSplit] = []
    for fold, test_chunk in enumerate(test_chunks):
        test = sorted(map(str, test_chunk.tolist()))
        remaining = [cell for cell in shuffled.tolist() if cell not in set(test)]
        fold_rng = np.random.default_rng(config.seed + 10_007 * (fold + 1))
        remaining_array = np.asarray(remaining, dtype=object)
        fold_rng.shuffle(remaining_array)
        validation_count = max(1, int(round(len(remaining) * config.validation_fraction)))
        validation_count = min(validation_count, len(remaining) - 1)
        validation = sorted(map(str, remaining_array[:validation_count].tolist()))
        train = sorted(map(str, remaining_array[validation_count:].tolist()))
        split = FoldSplit(fold, train, validation, test)
        split.validate(identifiers)
        splits.append(split)
    return splits


def save_splits(splits: Sequence[FoldSplit], path: str | Path, config: SplitConfig) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": config.strategy,
        "seed": config.seed,
        "num_folds": len(splits),
        "folds": [asdict(split) for split in splits],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> list[FoldSplit]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FoldSplit(**record) for record in payload["folds"]]

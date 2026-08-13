"""Output-directory naming for non-meta-learning baselines."""

from __future__ import annotations

import re
from pathlib import Path


_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _target_token(target_name: str) -> str:
    stem = Path(target_name).stem
    if stem.lower().startswith("calce_"):
        stem = stem[len("calce_") :]
    token = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    if not token:
        raise ValueError(f"cannot build a run name from target: {target_name!r}")
    return token


def _compact_count(value: int) -> str:
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def recursive_gru_run_name(
    target_name: str,
    history_length: int,
    max_steps: int | None,
    max_epochs: int,
    early_stopping: bool,
    seed: int,
) -> str:
    """Return a short, condition-based name for a recursive GRU baseline."""
    if max_steps is not None:
        budget = f"{_compact_count(max_steps)}step"
    else:
        budget = f"{_compact_count(max_epochs)}ep"
    stopping = "es" if early_stopping else "noes"
    return (
        f"{_target_token(target_name)}_gru_l{history_length}_soh_rec_"
        f"{budget}_{stopping}_s{seed}"
    )


def window_gru_run_name(
    target_name: str,
    history_length: int,
    max_epochs: int,
    seed: int,
) -> str:
    """Return a short, condition-based name for the legacy window GRU baseline."""
    return (
        f"{_target_token(target_name)}_gru_l{history_length}_soh_window_"
        f"{_compact_count(max_epochs)}ep_es_s{seed}"
    )


def transfer_gru_run_name(
    target_name: str,
    history_length: int,
    source_mode: str,
    seed: int,
    prefix_mode: str = "fixed_history",
) -> str:
    """Return a short name for source pretraining plus target fine-tuning."""
    source_tag = {
        "same_family": "samefam",
        "all_calce": "allcalce",
    }.get(source_mode)
    if source_tag is None:
        raise ValueError("source_mode must be 'same_family' or 'all_calce'")
    prefix_tag = {
        "fixed_history": "transfer",
        "variable_cutpoint": "varcut",
    }.get(prefix_mode)
    if prefix_tag is None:
        raise ValueError(
            "prefix_mode must be 'fixed_history' or 'variable_cutpoint'"
        )
    return (
        f"{_target_token(target_name)}_gru_l{history_length}_soh_{prefix_tag}_"
        f"{source_tag}_s{seed}"
    )


def new_baseline_run_dir(
    project_root: str | Path,
    automatic_name: str,
    requested_name: str | None = None,
) -> Path:
    """Resolve a new run below ``outputs/baseline`` without overwriting data."""
    name = requested_name or automatic_name
    if not _SAFE_RUN_NAME.fullmatch(name):
        raise ValueError(
            "run name may contain only letters, numbers, '.', '_' and '-' "
            "and must start with a letter or number"
        )
    run_dir = Path(project_root).resolve() / "outputs" / "baseline" / name
    if run_dir.exists():
        raise FileExistsError(
            f"baseline run already exists: {run_dir}. "
            "Use --resume RUN/checkpoints/last.pt to continue it, or choose a "
            "distinct --run-name for a new condition."
        )
    return run_dir

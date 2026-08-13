from __future__ import annotations

import pytest

from battery_weighted_maml.baseline_paths import (
    new_baseline_run_dir,
    recursive_gru_run_name,
    window_gru_run_name,
)


def test_recursive_name_uses_conditions_without_timestamp() -> None:
    name = recursive_gru_run_name(
        "CALCE_CX2_37.pkl",
        history_length=100,
        max_steps=10_000,
        max_epochs=10_000,
        early_stopping=False,
        seed=42,
    )
    assert name == "cx2_37_gru_l100_soh_rec_10kstep_noes_s42"
    assert "2026" not in name


def test_window_name_uses_conditions_without_timestamp() -> None:
    assert window_gru_run_name("CALCE_CX2_37.pkl", 500, 300, 42) == (
        "cx2_37_gru_l500_soh_window_300ep_es_s42"
    )


def test_new_baseline_directory_rejects_overwrite(tmp_path) -> None:
    first = new_baseline_run_dir(tmp_path, "automatic_name", "short_run")
    first.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="--resume"):
        new_baseline_run_dir(tmp_path, "automatic_name", "short_run")


def test_custom_name_cannot_escape_baseline_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="run name"):
        new_baseline_run_dir(tmp_path, "automatic_name", "../outside")

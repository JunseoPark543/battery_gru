from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_reproduction.config import load_config
from paper_reproduction.run_l100_hparam_suite import (
    CANDIDATES,
    EIGHT_HOUR_CANDIDATES,
    EIGHT_HOUR_INNER_LEARNING_RATES,
    EIGHT_HOUR_PREDICTED_INPUT_PROBABILITIES,
    combine_results,
    configure_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_l100_hparam_candidates_are_isolated_and_valid(tmp_path):
    base = load_config(ROOT / "paper_reproduction/configs/paper_recursive_l100.yaml")
    original = base.to_dict()

    for name, candidate in CANDIDATES.items():
        config = configure_candidate(base, name, tmp_path / "runs", max_epochs=12)
        assert config.data.history_length == 100
        assert config.maml.meta_batch_size == 5
        assert config.maml.max_epochs == 12
        assert config.maml.outer_learning_rate == 0.001
        assert config.maml.inner_learning_rate == candidate.inner_learning_rate
        assert config.maml.inner_steps == candidate.inner_steps
        assert config.maml.multi_step_query_weights == {candidate.inner_steps: 1.0}
        assert config.model.predicted_input_probability == candidate.predicted_input_probability
        assert config.adaptation.fast_learning_rate == candidate.inner_learning_rate
        assert config.adaptation.complete_learning_rate == candidate.inner_learning_rate

    assert base.to_dict() == original


def test_eight_hour_preset_is_a_sixteen_run_one_step_grid(tmp_path):
    assert len(EIGHT_HOUR_CANDIDATES) == 16
    combinations = {
        (
            CANDIDATES[name].predicted_input_probability,
            CANDIDATES[name].inner_learning_rate,
            CANDIDATES[name].inner_steps,
        )
        for name in EIGHT_HOUR_CANDIDATES
    }
    expected = {
        (probability, learning_rate, 1)
        for probability in EIGHT_HOUR_PREDICTED_INPUT_PROBABILITIES
        for learning_rate in EIGHT_HOUR_INNER_LEARNING_RATES
    }
    assert combinations == expected

    base = load_config(ROOT / "paper_reproduction/configs/paper_recursive_l100.yaml")
    for name in EIGHT_HOUR_CANDIDATES:
        config = configure_candidate(
            base,
            name,
            tmp_path / "runs",
            max_epochs=500,
            early_stopping=False,
        )
        assert config.maml.early_stopping is False
        assert config.maml.max_epochs == 500


def test_l100_hparam_combines_source_ranking_and_target_diagnostics(tmp_path):
    records = []
    for index, name in enumerate(("recursive", "gentle"), start=1):
        run_dir = tmp_path / name
        output = run_dir / "meta_test"
        output.mkdir(parents=True)
        rows = []
        for cell in ("CALCE_CX2_37.pkl", "CALCE_CX2_38.pkl"):
            for step in (0, 1, 3, 5):
                rows.append(
                    {
                        "cell": cell,
                        "mode": f"fast_{step}_steps",
                        "mae_percent": float(index + step),
                    }
                )
        pd.DataFrame(rows).to_csv(output / "meta_test_summary.csv", index=False)
        records.append(
            {
                "candidate": name,
                "best_source_meta_loss": float(3 - index),
                "best_epoch": index,
                "predicted_input_probability": CANDIDATES[name].predicted_input_probability,
                "inner_learning_rate": CANDIDATES[name].inner_learning_rate,
                "inner_steps": CANDIDATES[name].inner_steps,
                "elapsed_minutes": 1.0,
                "run_dir": str(run_dir),
            }
        )

    combine_results(records, tmp_path)

    ranking = pd.read_csv(tmp_path / "source_selection_ranking.csv")
    assert ranking["candidate"].tolist() == ["gentle", "recursive"]
    diagnostics = pd.read_csv(tmp_path / "target_diagnostics.csv")
    assert len(diagnostics) == 16
    assert (tmp_path / "source_selection_ranking.png").is_file()
    assert (tmp_path / "target_mae_comparison.png").is_file()

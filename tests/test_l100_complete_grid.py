from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_reproduction.plot_l100_complete_grid import (
    find_latest_complete_suite,
    plot_complete_grid,
)


def _make_suite(root: Path) -> Path:
    suite = root / "20260814_000000_000000"
    records = []
    for probability in (0.5, 0.7, 0.85, 1.0):
        for learning_rate in (0.005, 0.01, 0.025, 0.05):
            name = f"p{probability}_lr{learning_rate}"
            run = suite / "runs" / name
            prediction_dir = run / "meta_test/CALCE_CX2_37/predictions"
            prediction_dir.mkdir(parents=True)
            cycles = np.arange(1, 121)
            observed = 1.0 - 0.002 * cycles
            predicted = observed.copy()
            pd.DataFrame(
                {
                    "cycle": cycles,
                    "observed_soh": observed,
                    "predicted_soh": predicted,
                    "split": ["support"] * 100 + ["future"] * 20,
                    "eol_threshold": 0.7,
                    "mode": "complete_paper_query_selected",
                }
            ).to_csv(
                prediction_dir / "complete_paper_query_selected.csv", index=False
            )
            summary_dir = run / "meta_test"
            pd.DataFrame(
                [
                    {
                        "cell": "CALCE_CX2_37.pkl",
                        "mode": "complete_paper_query_selected",
                        "current_cycle": 100,
                        "adaptation_best_step": 1,
                        "mae_percent": 1.0,
                        "rmse_percent": 1.2,
                        "r2": 0.9,
                    }
                ]
            ).to_csv(summary_dir / "meta_test_summary.csv", index=False)
            records.append(
                {
                    "candidate": name,
                    "predicted_input_probability": probability,
                    "inner_learning_rate": learning_rate,
                    "inner_steps": 1,
                    # Deliberately foreign path: portable resolution must use basename.
                    "run_dir": f"/home/server/project/runs/{name}",
                }
            )
    manifest = {
        "status": "completed",
        "fixed_parameters": {"outer_learning_rate": 0.001},
        "runs": records,
    }
    (suite / "suite_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return suite


def test_complete_grid_supports_portable_run_paths(tmp_path):
    suite = _make_suite(tmp_path)

    output = plot_complete_grid(suite)

    assert output == suite / "CALCE_CX2_37_complete_16_trajectory_grid.png"
    assert output.is_file()
    assert output.stat().st_size > 10_000


def test_find_latest_completed_sixteen_run_suite(tmp_path):
    older = _make_suite(tmp_path)
    newer = tmp_path / "20260815_000000_000000"
    newer.mkdir()
    manifest = json.loads((older / "suite_manifest.json").read_text(encoding="utf-8"))
    (newer / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert find_latest_complete_suite(tmp_path) == newer

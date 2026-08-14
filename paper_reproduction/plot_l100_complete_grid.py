"""Combine the 16 L=100 hyperparameter complete forecasts into one figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPLETE_MODE = "complete_paper_query_selected"
DEFAULT_SUITES_ROOT = Path(
    "outputs/paper_recursive_reproduction/l100_hparam_suites"
)


def _resolve_run_dir(suite_dir: Path, record: dict[str, Any]) -> Path:
    """Resolve a run after results have been copied between Linux and Windows."""
    recorded = Path(str(record["run_dir"]))
    if recorded.is_dir():
        return recorded
    portable = suite_dir / "runs" / recorded.name
    if portable.is_dir():
        return portable
    raise FileNotFoundError(
        f"run directory not found at recorded or portable path: {recorded}, {portable}"
    )


def find_latest_complete_suite(root: Path) -> Path:
    """Return the newest completed suite containing exactly 16 runs."""
    if not root.is_dir():
        raise FileNotFoundError(f"hyperparameter suite root not found: {root}")
    matches: list[Path] = []
    for candidate in root.iterdir():
        manifest_path = candidate / "suite_manifest.json"
        if not candidate.is_dir() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed" and len(manifest.get("runs", [])) == 16:
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(f"no completed 16-run suite found under: {root}")
    return max(matches, key=lambda path: path.name)


def plot_complete_grid(
    suite_dir: str | Path,
    target_cell: str = "CALCE_CX2_37.pkl",
    destination: str | Path | None = None,
) -> Path:
    """Create a 4x4 complete-only forecast grid for one target cell."""
    suite = Path(suite_dir).resolve()
    manifest_path = suite / "suite_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"suite manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("runs", [])
    if len(records) != 16:
        raise ValueError(f"the combined figure requires exactly 16 runs, found {len(records)}")

    metric_cell = target_cell if target_cell.endswith(".pkl") else f"{target_cell}.pkl"
    cell_dir = Path(metric_cell).stem
    panels: list[dict[str, Any]] = []
    for record in records:
        run_dir = _resolve_run_dir(suite, record)
        prediction_path = (
            run_dir
            / "meta_test"
            / cell_dir
            / "predictions"
            / f"{COMPLETE_MODE}.csv"
        )
        summary_path = run_dir / "meta_test/meta_test_summary.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"complete prediction not found: {prediction_path}")
        if not summary_path.is_file():
            raise FileNotFoundError(f"meta-test summary not found: {summary_path}")
        prediction = pd.read_csv(prediction_path)
        summary = pd.read_csv(summary_path)
        selected = summary[
            (summary["cell"] == metric_cell) & (summary["mode"] == COMPLETE_MODE)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one {metric_cell}/{COMPLETE_MODE} row in {summary_path}, "
                f"found {len(selected)}"
            )
        panels.append(
            {
                "candidate": record["candidate"],
                "predicted_input_probability": float(
                    record["predicted_input_probability"]
                ),
                "inner_learning_rate": float(record["inner_learning_rate"]),
                "inner_steps": int(record["inner_steps"]),
                "prediction": prediction,
                "metrics": selected.iloc[0],
            }
        )

    probabilities = sorted({panel["predicted_input_probability"] for panel in panels})
    learning_rates = sorted({panel["inner_learning_rate"] for panel in panels})
    if len(probabilities) != 4 or len(learning_rates) != 4:
        raise ValueError(
            "expected a 4x4 predicted-input-probability/inner-learning-rate grid; "
            f"found {len(probabilities)}x{len(learning_rates)}"
        )
    lookup = {
        (panel["predicted_input_probability"], panel["inner_learning_rate"]): panel
        for panel in panels
    }
    if len(lookup) != 16:
        raise ValueError("the 16 runs do not define 16 unique hyperparameter combinations")

    all_observed = np.concatenate(
        [panel["prediction"]["observed_soh"].dropna().to_numpy() for panel in panels]
    )
    all_predicted = np.concatenate(
        [
            panel["prediction"].loc[
                panel["prediction"]["split"] == "future", "predicted_soh"
            ].dropna().to_numpy()
            for panel in panels
        ]
    )
    y_min = float(min(all_observed.min(), all_predicted.min()))
    y_max = float(max(all_observed.max(), all_predicted.max()))
    margin = max(0.02, 0.04 * (y_max - y_min))

    figure, axes = plt.subplots(
        4,
        4,
        figsize=(24, 19),
        sharex=True,
        sharey=True,
    )
    legend_handles = None
    for row_index, probability in enumerate(probabilities):
        for column_index, learning_rate in enumerate(learning_rates):
            axis = axes[row_index, column_index]
            panel = lookup[(probability, learning_rate)]
            prediction = panel["prediction"]
            metrics = panel["metrics"]
            observed = prediction["observed_soh"].notna()
            support = prediction["split"] == "support"
            future = prediction["split"] == "future"
            threshold = float(prediction["eol_threshold"].dropna().iloc[0])
            current_cycle = int(metrics["current_cycle"])

            actual_line = axis.plot(
                prediction.loc[observed, "cycle"],
                prediction.loc[observed, "observed_soh"],
                color="#303030",
                linewidth=1.25,
                label="Actual SOH",
                zorder=2,
            )[0]
            support_line = axis.plot(
                prediction.loc[support, "cycle"],
                prediction.loc[support, "observed_soh"],
                color="#2ca02c",
                linewidth=2.2,
                label="Observed input (cycles 1-100)",
                zorder=3,
            )[0]
            forecast_line = axis.plot(
                prediction.loc[future, "cycle"],
                prediction.loc[future, "predicted_soh"],
                color="#d62728",
                linewidth=1.7,
                label="Complete prediction",
                zorder=4,
            )[0]
            threshold_line = axis.axhline(
                threshold,
                color="#777777",
                linestyle="--",
                linewidth=1.0,
                label=f"EOL threshold ({threshold:g})",
                zorder=1,
            )
            axis.axvline(current_cycle, color="#2ca02c", linestyle=":", linewidth=1.0)
            if legend_handles is None:
                legend_handles = [
                    actual_line,
                    support_line,
                    forecast_line,
                    threshold_line,
                ]
            axis.set_title(
                f"p(pred)={probability:g} | inner LR={learning_rate:g}\n"
                f"complete step={int(metrics['adaptation_best_step'])} | "
                f"MAE={float(metrics['mae_percent']):.2f}% | "
                f"RMSE={float(metrics['rmse_percent']):.2f}% | "
                f"R²={float(metrics['r2']):.3f}",
                fontsize=10,
            )
            axis.set_ylim(y_min - margin, y_max + margin)
            axis.grid(alpha=0.20)
            if row_index == 3:
                axis.set_xlabel("Cycle")
            if column_index == 0:
                axis.set_ylabel(f"SOH\np(pred)={probability:g}")

    fixed = manifest.get("fixed_parameters", {})
    outer_lr = fixed.get("outer_learning_rate", 0.001)
    figure.suptitle(
        f"{cell_dir}: complete adaptation trajectories for 16 L=100 MAML runs\n"
        f"Rows: predicted-input probability | Columns: inner learning rate | "
        f"outer LR={outer_lr:g}, meta inner steps=1",
        fontsize=17,
        y=0.992,
    )
    if legend_handles is not None:
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.946),
            frameon=False,
        )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.055,
        top=0.885,
        hspace=0.38,
        wspace=0.06,
    )

    output = (
        Path(destination)
        if destination is not None
        else suite / f"{cell_dir}_complete_16_trajectory_grid.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine 16 L100 complete forecasts for CX2_37 into a 4x4 plot"
    )
    parser.add_argument(
        "--suite-dir",
        help="suite directory; omitted means the latest completed 16-run suite",
    )
    parser.add_argument("--target-cell", default="CALCE_CX2_37.pkl")
    parser.add_argument("--output", help="optional destination PNG")
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    root = Path.cwd().resolve()
    suite = (
        Path(args.suite_dir).resolve()
        if args.suite_dir
        else find_latest_complete_suite(root / DEFAULT_SUITES_ROOT)
    )
    output = plot_complete_grid(suite, args.target_cell, args.output)
    print(f"Combined complete trajectory plot: {output}")
    return output


if __name__ == "__main__":
    main()

"""Run a focused, leakage-aware L=100 MAML hyperparameter ablation.

The candidates deliberately change one factor at a time before combining the
changes.  Model checkpoints are still selected only with the five source-cell
meta objective.  Held-out CX2_37/CX2_38 results are reported as diagnostics;
they are not used to select a checkpoint or rank the training configurations.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import ExperimentConfig, load_config, save_config
from .main import run as run_experiment
from .plot_l100_complete_grid import plot_complete_grid
from .run_paper_recursive_suite import _main_args


BASE_CONFIG = Path("paper_reproduction/configs/paper_recursive_l100.yaml")


@dataclass(frozen=True)
class Candidate:
    predicted_input_probability: float
    inner_learning_rate: float
    inner_steps: int
    description: str


CANDIDATES: dict[str, Candidate] = {
    "paper_ref": Candidate(
        predicted_input_probability=0.5,
        inner_learning_rate=0.05,
        inner_steps=1,
        description="Current paper-aligned reference",
    ),
    "recursive": Candidate(
        predicted_input_probability=1.0,
        inner_learning_rate=0.05,
        inner_steps=1,
        description="Fully recursive training; original inner update",
    ),
    "gentle": Candidate(
        predicted_input_probability=0.5,
        inner_learning_rate=0.01,
        inner_steps=1,
        description="Original teacher forcing; gentler inner update",
    ),
    "recursive_gentle": Candidate(
        predicted_input_probability=1.0,
        inner_learning_rate=0.01,
        inner_steps=1,
        description="Fully recursive training and gentler inner update",
    ),
    "recursive_3step": Candidate(
        predicted_input_probability=1.0,
        inner_learning_rate=0.01,
        inner_steps=3,
        description="Fully recursive training and three gentle inner updates",
    ),
}


def _number_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


# A 4 x 4 grid focused on the two hyperparameters most directly related to the
# observed L=100 failure: train/inference exposure bias and adaptation size.
# With the user's measured 30 minutes per complete L=100 run, 16 candidates
# consume approximately eight hours on the same server/GPU.
EIGHT_HOUR_PREDICTED_INPUT_PROBABILITIES = [0.5, 0.7, 0.85, 1.0]
EIGHT_HOUR_INNER_LEARNING_RATES = [0.005, 0.01, 0.025, 0.05]
EIGHT_HOUR_CANDIDATES: list[str] = []
for predicted_probability in EIGHT_HOUR_PREDICTED_INPUT_PROBABILITIES:
    for inner_learning_rate in EIGHT_HOUR_INNER_LEARNING_RATES:
        candidate_name = (
            f"p{_number_tag(predicted_probability)}_"
            f"ilr{_number_tag(inner_learning_rate)}"
        )
        CANDIDATES[candidate_name] = Candidate(
            predicted_input_probability=predicted_probability,
            inner_learning_rate=inner_learning_rate,
            inner_steps=1,
            description=(
                f"predicted-input probability {predicted_probability:g}; "
                f"inner learning rate {inner_learning_rate:g}"
            ),
        )
        EIGHT_HOUR_CANDIDATES.append(candidate_name)

DEFAULT_CANDIDATES = [
    "recursive",
    "gentle",
    "recursive_gentle",
    "recursive_3step",
]


def configure_candidate(
    base: ExperimentConfig,
    name: str,
    output_dir: Path,
    max_epochs: int,
    early_stopping: bool = True,
) -> ExperimentConfig:
    """Return an isolated config for one ablation candidate."""
    candidate = CANDIDATES[name]
    config = copy.deepcopy(base)
    config.paths.output_dir = str(output_dir)
    config.model.predicted_input_probability = candidate.predicted_input_probability
    config.maml.max_epochs = max_epochs
    config.maml.early_stopping = early_stopping
    config.maml.outer_learning_rate = 1.0e-3
    config.maml.inner_learning_rate = candidate.inner_learning_rate
    config.maml.inner_steps = candidate.inner_steps
    # Standard k-step MAML: only the query after the final inner update enters
    # the outer objective. This keeps the 3-step candidate interpretable and
    # avoids evaluating the very long query three times per source task.
    config.maml.multi_step_query_weights = {candidate.inner_steps: 1.0}
    config.maml.optuna_trials = 0
    config.maml.experiment_label = f"hp-{name.replace('_', '-')}"

    # At meta-test, use the same SGD step size that the initialization was
    # trained for. Comparing a 0.01 inner loop with 0.05 adaptation would no
    # longer be a valid MAML train/test comparison.
    config.adaptation.learning_rate = None
    config.adaptation.fast_learning_rate = candidate.inner_learning_rate
    config.adaptation.complete_learning_rate = candidate.inner_learning_rate
    config.validate()
    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def combine_results(records: list[dict[str, Any]], suite_dir: Path) -> None:
    """Write source-only ranking and held-out diagnostic tables/figures."""
    source_ranking = pd.DataFrame(
        [
            {
                "candidate": record["candidate"],
                "best_source_meta_loss": record["best_source_meta_loss"],
                "best_epoch": record["best_epoch"],
                "predicted_input_probability": record[
                    "predicted_input_probability"
                ],
                "inner_learning_rate": record["inner_learning_rate"],
                "inner_steps": record["inner_steps"],
                "elapsed_minutes": record.get("elapsed_minutes"),
                "run_dir": record["run_dir"],
            }
            for record in records
        ]
    ).sort_values("best_source_meta_loss", ignore_index=True)
    source_ranking.insert(0, "source_rank", range(1, len(source_ranking) + 1))
    source_ranking.to_csv(suite_dir / "source_selection_ranking.csv", index=False)

    target_frames: list[pd.DataFrame] = []
    for record in records:
        frame = pd.read_csv(Path(record["run_dir"]) / "meta_test/meta_test_summary.csv")
        frame.insert(0, "candidate", record["candidate"])
        frame.insert(1, "best_source_meta_loss", record["best_source_meta_loss"])
        target_frames.append(frame)
    target = pd.concat(target_frames, ignore_index=True)
    target.to_csv(suite_dir / "target_diagnostics.csv", index=False)
    _plot_source_ranking(source_ranking, suite_dir / "source_selection_ranking.png")
    _plot_target_diagnostics(target, suite_dir / "target_mae_comparison.png")
    if len(records) >= 4:
        _plot_grid_heatmaps(
            source_ranking,
            target,
            suite_dir / "hyperparameter_heatmaps.png",
        )
    if len(records) == 16:
        plot_complete_grid(suite_dir, target_cell="CALCE_CX2_37.pkl")


def _plot_source_ranking(frame: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(8.5, 0.62 * len(frame)), 5.2))
    bars = axis.bar(frame["candidate"], frame["best_source_meta_loss"])
    axis.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    axis.set_ylabel("Best deterministic source meta-query loss")
    axis.set_title("L=100 MAML hyperparameters: source-only model selection")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_target_diagnostics(frame: pd.DataFrame, destination: Path) -> None:
    modes = ["fast_0_steps", "fast_1_steps", "fast_3_steps", "fast_5_steps"]
    visible = frame[frame["mode"].isin(modes)].copy()
    cells = list(dict.fromkeys(visible["cell"]))
    figure, axes = plt.subplots(1, len(cells), figsize=(14, 5), sharey=True)
    if len(cells) == 1:
        axes = [axes]
    displayed_candidates = list(dict.fromkeys(visible["candidate"]))
    width = 0.8 / max(1, len(displayed_candidates))
    for axis, cell in zip(axes, cells):
        selected = visible[visible["cell"] == cell]
        candidates = list(dict.fromkeys(selected["candidate"]))
        for index, candidate in enumerate(candidates):
            rows = selected[selected["candidate"] == candidate].set_index("mode")
            values = [float(rows.loc[mode, "mae_percent"]) for mode in modes]
            offsets = [position - 0.4 + width / 2 + index * width for position in range(4)]
            axis.bar(offsets, values, width=width, label=candidate)
        axis.set_xticks(range(4), ["0", "1", "3", "5"])
        axis.set_xlabel("Target adaptation steps")
        axis.set_title(Path(cell).stem.replace("CALCE_", ""))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Future-trajectory MAE (%)")
    axes[-1].legend(fontsize=8)
    figure.suptitle(
        "Held-out target diagnostics (do not use these values for checkpoint selection)"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_grid_heatmaps(
    source: pd.DataFrame,
    target: pd.DataFrame,
    destination: Path,
) -> None:
    """Plot compact heatmaps when candidates form a probability/LR grid."""
    grid_source = source.dropna(
        subset=["predicted_input_probability", "inner_learning_rate"]
    )
    if grid_source.empty:
        return
    probabilities = sorted(grid_source["predicted_input_probability"].unique())
    learning_rates = sorted(grid_source["inner_learning_rate"].unique())
    if len(probabilities) * len(learning_rates) != len(grid_source):
        return

    diagnostics = target[target["mode"] == "fast_1_steps"]
    cells = list(dict.fromkeys(diagnostics["cell"]))
    figure, axes = plt.subplots(
        1,
        1 + len(cells),
        figsize=(5.2 * (1 + len(cells)), 4.8),
        constrained_layout=True,
    )
    if not isinstance(axes, (list, tuple)) and not hasattr(axes, "flat"):
        axes = [axes]
    else:
        axes = list(axes.flat)

    source_pivot = grid_source.pivot(
        index="predicted_input_probability",
        columns="inner_learning_rate",
        values="best_source_meta_loss",
    ).reindex(index=probabilities, columns=learning_rates)
    _draw_heatmap(
        axes[0],
        source_pivot,
        "Source selection loss\n(lower is better)",
        ".4f",
    )
    for axis, cell in zip(axes[1:], cells):
        rows = diagnostics[diagnostics["cell"] == cell].merge(
            grid_source[
                ["candidate", "predicted_input_probability", "inner_learning_rate"]
            ],
            on="candidate",
            how="left",
        )
        pivot = rows.pivot(
            index="predicted_input_probability",
            columns="inner_learning_rate",
            values="mae_percent",
        ).reindex(index=probabilities, columns=learning_rates)
        _draw_heatmap(
            axis,
            pivot,
            f"{Path(cell).stem.replace('CALCE_', '')} fast-1 MAE (%)\n(diagnostic only)",
            ".2f",
        )
    figure.suptitle("L=100 MAML 8-hour hyperparameter grid")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _draw_heatmap(
    axis: Any,
    pivot: pd.DataFrame,
    title: str,
    value_format: str,
) -> None:
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = float(pivot.iloc[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
    axis.set_xlabel("Inner learning rate")
    axis.set_ylabel("Predicted-input probability")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def run(args: argparse.Namespace) -> Path:
    root = Path.cwd().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    relative_suite = Path(
        f"outputs/paper_recursive_reproduction/l100_hparam_suites/{timestamp}"
    )
    suite_dir = root / relative_suite
    suite_dir.mkdir(parents=True, exist_ok=False)
    base = load_config(root / BASE_CONFIG)
    records: list[dict[str, Any]] = []
    candidate_names = (
        args.candidates
        if args.candidates is not None
        else (
            EIGHT_HOUR_CANDIDATES
            if args.preset == "8h"
            else DEFAULT_CANDIDATES
        )
    )
    disable_early_stopping = args.preset == "8h" and args.candidates is None
    expected_minutes = len(candidate_names) * args.estimated_minutes_per_run
    suite_started = time.perf_counter()
    print(
        f"Starting {len(candidate_names)} L100 candidates; estimated "
        f"duration={expected_minutes / 60:.2f}h "
        f"({args.estimated_minutes_per_run:g} min/run)."
    )

    for candidate_index, name in enumerate(candidate_names, start=1):
        config = configure_candidate(
            base,
            name,
            relative_suite / "runs",
            args.max_epochs,
            early_stopping=not disable_early_stopping,
        )
        config_path = suite_dir / "configs" / f"{name}.yaml"
        save_config(config, config_path)
        print(f"[{candidate_index}/{len(candidate_names)}] Starting {name}")
        candidate_started = time.perf_counter()
        run_dir = run_experiment(_main_args(config_path, "all", args.device))
        candidate_minutes = (time.perf_counter() - candidate_started) / 60.0
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        candidate = CANDIDATES[name]
        records.append(
            {
                "candidate": name,
                "description": candidate.description,
                "predicted_input_probability": candidate.predicted_input_probability,
                "inner_learning_rate": candidate.inner_learning_rate,
                "inner_steps": candidate.inner_steps,
                "outer_learning_rate": config.maml.outer_learning_rate,
                "best_source_meta_loss": float(manifest["best_meta_loss"]),
                "best_epoch": int(manifest["best_epoch"]),
                "last_epoch": int(manifest["last_epoch"]),
                "elapsed_minutes": candidate_minutes,
                "run_dir": str(run_dir),
                "config": str(config_path),
            }
        )
        elapsed_hours = (time.perf_counter() - suite_started) / 3600.0
        average_hours = elapsed_hours / candidate_index
        remaining_hours = average_hours * (len(candidate_names) - candidate_index)
        print(
            f"[{candidate_index}/{len(candidate_names)}] Completed {name} in "
            f"{candidate_minutes:.1f} min; measured ETA={remaining_hours:.2f}h"
        )

    combine_results(records, suite_dir)
    manifest = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_length": 100,
        "weighted_meta_learning": False,
        "algorithm": "full_second_order_maml",
        "selection_rule": (
            "Rank candidates only by deterministic post-adaptation meta-query "
            "loss on the five source cells. Target metrics are diagnostics."
        ),
        "preset": args.preset,
        "candidate_count": len(candidate_names),
        "estimated_minutes_per_run": args.estimated_minutes_per_run,
        "estimated_total_hours": expected_minutes / 60.0,
        "actual_total_hours": (time.perf_counter() - suite_started) / 3600.0,
        "fixed_parameters": {
            "outer_learning_rate": 1.0e-3,
            "hidden_size": 64,
            "inner_batch_size": 64,
            "maximum_epochs": args.max_epochs,
            "early_stopping": not disable_early_stopping,
        },
        "runs": records,
    }
    _write_json(suite_dir / "suite_manifest.json", manifest)
    print(f"Completed L100 hyperparameter suite: {suite_dir}")
    print(f"Source ranking: {suite_dir / 'source_selection_ranking.csv'}")
    print(f"Target diagnostics: {suite_dir / 'target_diagnostics.csv'}")
    return suite_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused L=100 second-order MAML hyperparameter ablation"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--preset",
        choices=["focused", "8h"],
        default="focused",
        help="8h runs a 4x4 recursive-probability/inner-LR grid (default: focused)",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=sorted(CANDIDATES),
        default=None,
        help="explicit candidates; overrides the selected preset",
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument(
        "--estimated-minutes-per-run",
        type=float,
        default=30.0,
        help="ETA calibration only; does not interrupt training (default: 30)",
    )
    args = parser.parse_args()
    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive")
    if args.estimated_minutes_per_run <= 0:
        parser.error("--estimated-minutes-per-run must be positive")
    if args.candidates is not None and len(set(args.candidates)) != len(args.candidates):
        parser.error("--candidates must not contain duplicates")
    return args


if __name__ == "__main__":
    run(parse_args())

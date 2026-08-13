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
) -> ExperimentConfig:
    """Return an isolated config for one ablation candidate."""
    candidate = CANDIDATES[name]
    config = copy.deepcopy(base)
    config.paths.output_dir = str(output_dir)
    config.model.predicted_input_probability = candidate.predicted_input_probability
    config.maml.max_epochs = max_epochs
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


def _plot_source_ranking(frame: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.bar(frame["candidate"], frame["best_source_meta_loss"])
    axis.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    axis.set_ylabel("Best deterministic source meta-query loss")
    axis.set_title("L=100 MAML hyperparameters: source-only model selection")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
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
    width = 0.8 / max(1, len(CANDIDATES))
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

    for name in args.candidates:
        config = configure_candidate(
            base,
            name,
            relative_suite / "runs",
            args.max_epochs,
        )
        config_path = suite_dir / "configs" / f"{name}.yaml"
        save_config(config, config_path)
        run_dir = run_experiment(_main_args(config_path, "all", args.device))
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
                "run_dir": str(run_dir),
                "config": str(config_path),
            }
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
        "fixed_parameters": {
            "outer_learning_rate": 1.0e-3,
            "hidden_size": 64,
            "inner_batch_size": 64,
            "maximum_epochs": args.max_epochs,
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
        "--candidates",
        nargs="+",
        choices=sorted(CANDIDATES),
        default=DEFAULT_CANDIDATES,
        help="default: recursive gentle recursive_gentle recursive_3step",
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    args = parser.parse_args()
    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive")
    if len(set(args.candidates)) != len(args.candidates):
        parser.error("--candidates must not contain duplicates")
    return args


if __name__ == "__main__":
    run(parse_args())

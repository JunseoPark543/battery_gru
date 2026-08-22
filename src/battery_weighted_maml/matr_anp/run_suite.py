"""Sequential portable runner for all MATR/CALCE ANP models and cell folds."""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .compare_results import compare_evaluations
from .config import load_config, resolve_data_root
from .evaluate import evaluate_run
from .model import MODEL_NAMES
from .runtime import write_json
from .train import train_run


def parse_args(default_config: str = "configs/matr_partial_iv_anp.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ANP model/fold suite")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main(default_config: str = "configs/matr_partial_iv_anp.yaml") -> None:
    args = parse_args(default_config)
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    data_root = resolve_data_root(config, args.data_root)
    fold_count = config.split.num_folds if config.split.strategy == "kfold" else None
    if args.folds is None:
        if fold_count is None:
            raise ValueError("LOOCV suite requires explicit --folds because cell count is data-dependent")
        folds = list(range(fold_count))
    else:
        folds = args.folds
    if len(set(folds)) != len(folds) or any(fold < 0 for fold in folds):
        raise ValueError("--folds must contain unique non-negative integers")
    if fold_count is not None and any(fold >= fold_count for fold in folds):
        raise ValueError(f"fold index exceeds configured fold count {fold_count}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suite_directory = Path(config.paths.output_root).resolve() / f"suite_{timestamp}"
    suite_directory.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.data.dataset.upper(),
        "models": args.models,
        "folds": folds,
        "max_steps_override": args.max_steps,
        "evaluate": args.evaluate,
        "comparison_directory": None,
        "runs": records,
    }
    write_json(suite_directory / "suite_manifest.json", manifest)
    for model_name in args.models:
        for fold in folds:
            record = {
                "model": model_name,
                "fold": fold,
                "status": "running",
                "run_directory": None,
                "best_checkpoint": None,
                "evaluation_directory": None,
            }
            records.append(record)
            write_json(suite_directory / "suite_manifest.json", manifest)
            try:
                run_directory = train_run(
                    config,
                    model_name,
                    fold,
                    data_root,
                    max_steps=args.max_steps,
                )
                record["run_directory"] = str(run_directory)
                record["best_checkpoint"] = str(
                    run_directory / "checkpoints/best.pt"
                )
                if args.evaluate:
                    record["evaluation_directory"] = str(
                        evaluate_run(
                            config,
                            run_directory / "checkpoints/best.pt",
                            data_root,
                        )
                    )
                record["status"] = "completed"
            except Exception as exc:
                record["status"] = "failed"
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)
                record["traceback"] = traceback.format_exc()
                manifest["status"] = "failed"
                manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(suite_directory / "suite_manifest.json", manifest)
                print(f"Suite failed; diagnostic manifest: {suite_directory / 'suite_manifest.json'}")
                raise
            write_json(suite_directory / "suite_manifest.json", manifest)
    evaluation_directories = [
        record["evaluation_directory"]
        for record in records
        if record.get("evaluation_directory")
    ]
    if len(evaluation_directories) >= 2:
        manifest["comparison_directory"] = str(
            compare_evaluations(
                evaluation_directories,
                suite_directory / "model_comparison",
            )
        )
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(suite_directory / "suite_manifest.json", manifest)
    print(f"Suite directory: {suite_directory}")


if __name__ == "__main__":
    main()

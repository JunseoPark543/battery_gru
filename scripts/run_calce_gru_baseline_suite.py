"""Run the four SOH-only CALCE GRU baselines in a reproducible order."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_weighted_maml.cli import run_experiment
from battery_weighted_maml.config import load_config
from battery_weighted_maml.gru_baseline import (
    load_gru_baseline_config,
    run_gru_baseline,
)


EXPERIMENTS = {
    "nometa_l500": (
        "nometa",
        "configs/calce_gru_baselines/nometa_soh_l500.yaml",
        500,
    ),
    "nometa_l100": (
        "nometa",
        "configs/calce_gru_baselines/nometa_soh_l100.yaml",
        100,
    ),
    "meta_l500": (
        "meta",
        "configs/calce_gru_baselines/weighted_meta_soh_l500.yaml",
        500,
    ),
    "meta_l100": (
        "meta",
        "configs/calce_gru_baselines/weighted_meta_soh_l100.yaml",
        100,
    ),
}


def _read_metrics(mode: str, run_dir: Path) -> dict[str, object]:
    file_name = "gru_baseline_metrics.json" if mode == "nometa" else "full_metrics.json"
    source = run_dir / "metrics" / file_name
    if not source.is_file():
        raise FileNotFoundError(f"completed run has no metrics file: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metrics file must contain an object: {source}")
    return payload


def _write_comparison(manifest: dict[str, object], suite_dir: Path) -> None:
    completed = manifest["completed_runs"]
    if not isinstance(completed, dict) or not completed:
        return
    rows: list[dict[str, object]] = []
    for name in manifest["selected_experiments"]:
        if name not in completed:
            continue
        mode, _, history_length = EXPERIMENTS[name]
        run_dir = Path(str(completed[name]))
        metrics = _read_metrics(mode, run_dir)
        rows.append(
            {
                "experiment": name,
                "meta_learning": mode == "meta",
                "history_length": history_length,
                "run_dir": str(run_dir),
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(suite_dir / "baseline_comparison.csv", index=False)
    (suite_dir / "baseline_comparison.json").write_text(
        frame.to_json(orient="records", indent=2), encoding="utf-8"
    )

    labels = frame["experiment"].str.replace("_", "\n", regex=False)
    x = np.arange(len(frame))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - width / 2, frame["mae"] * 100.0, width, label="MAE")
    axes[0].bar(x + width / 2, frame["rmse"] * 100.0, width, label="RMSE")
    axes[0].set_xticks(x, labels)
    axes[0].set(ylabel="SOH error (%)", title="Future trajectory error")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(x, frame["absolute_rul_error"], color="tab:purple")
    axes[1].set_xticks(x, labels)
    axes[1].set(ylabel="Absolute error (cycles)", title="RUL error")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(f"CALCE SOH-only GRU baselines: {manifest['target']}")
    figure.tight_layout()
    figure.savefig(suite_dir / "baseline_comparison.png", dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched CALCE SOH-only GRU baselines"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[*EXPERIMENTS, "all"],
    )
    parser.add_argument("--target", default="CALCE_CX2_37.pkl")
    parser.add_argument(
        "--source-mode",
        default="same_family",
        choices=["same_family", "all_calce"],
        help="used only by weighted-meta experiments",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    selected = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suite_dir = root / "outputs/calce_gru_baseline_suites" / timestamp
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "target": args.target,
        "source_mode": args.source_mode,
        "device": args.device,
        "selected_experiments": selected,
        "completed_runs": {},
    }
    manifest_path = suite_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name in selected:
        mode, config_path, history_length = EXPERIMENTS[name]
        if mode == "nometa":
            config = load_gru_baseline_config(root / config_path)
            config.device = args.device
            run_dir = run_gru_baseline(
                config,
                target_name=args.target,
                project_root=root,
                smoke_test=args.smoke_test,
            )
        else:
            config = load_config(root / config_path)
            config.device = args.device
            run_dir = run_experiment(
                config,
                target_name=args.target,
                history_length=history_length,
                source_mode=args.source_mode,
                project_root=root,
                smoke_test=args.smoke_test,
            )
        manifest["completed_runs"][name] = str(run_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _write_comparison(manifest, suite_dir)

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_comparison(manifest, suite_dir)
    print(f"Completed suite manifest: {manifest_path}")


if __name__ == "__main__":
    main()

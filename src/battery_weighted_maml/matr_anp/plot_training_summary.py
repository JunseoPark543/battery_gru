"""Create a compact training/validation dashboard from one MATR ANP run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .runtime import write_json


BATCH_COLORS = {
    "b1": "#4C78A8",
    "b2": "#59A14F",
    "b3": "#F28E2B",
    "b4": "#E15759",
    "unknown": "#9C9C9C",
}


def _batch_name(cell_id: str) -> str:
    match = re.match(r"MATR_(b\d+)c", cell_id)
    return match.group(1) if match else "unknown"


def _parse_cell_metrics(value: object, step: int) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value:
        return []
    rows = []
    for item in value.split("|"):
        if ":" not in item:
            continue
        cell_id, raw_metric = item.rsplit(":", 1)
        rows.append(
            {
                "step": step,
                "cell_id": cell_id,
                "batch": _batch_name(cell_id),
                "validation_rmse": float(raw_metric),
            }
        )
    return rows


def _load_run(run_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history_path = run_directory / "training/history.csv"
    manifest_path = run_directory / "run_manifest.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"training history not found: {history_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    history = pd.read_csv(history_path)
    required = {
        "step", "loss", "nll", "kl", "gradient_norm", "validation_rmse",
        "elapsed_seconds",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"training history is missing columns: {sorted(missing)}")
    validation_rows = history[history["validation_rmse"].notna()]
    cell_rows: list[dict[str, Any]] = []
    if "validation_cell_rmse" in validation_rows:
        for row in validation_rows.itertuples(index=False):
            cell_rows.extend(
                _parse_cell_metrics(row.validation_cell_rmse, int(row.step))
            )
    cells = pd.DataFrame(cell_rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return history, cells, manifest


def create_training_summary(
    run_dir: str | Path,
    destination: str | Path | None = None,
) -> Path:
    run_directory = Path(run_dir).resolve()
    history, cell_history, manifest = _load_run(run_directory)
    resolved_config_path = run_directory / "resolved_config.yaml"
    resolved_config = (
        load_config(resolved_config_path) if resolved_config_path.is_file() else None
    )
    output = (
        Path(destination).resolve()
        if destination
        else run_directory / "plots/training_summary.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    skipped = (
        history["optimizer_step_skipped"].fillna(False).astype(bool)
        if "optimizer_step_skipped" in history
        else pd.Series(False, index=history.index)
    )
    successful = history[~skipped].copy()
    validation = history[history["validation_rmse"].notna()].copy()
    if validation.empty:
        raise ValueError("training history contains no validation measurements")
    best_index = validation["validation_rmse"].idxmin()
    best = validation.loc[best_index]
    final = validation.iloc[-1]
    best_step = int(best["step"])
    best_rmse = float(best["validation_rmse"])
    final_rmse = float(final["validation_rmse"])
    initial_rmse = float(validation.iloc[0]["validation_rmse"])
    runtime_hours = float(history["elapsed_seconds"].dropna().iloc[-1]) / 3600.0
    rolling_window = min(200, max(20, len(successful) // 100))

    figure, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    loss_axis, validation_axis, cell_axis, stability_axis = axes.ravel()

    # Panel A: noisy episode ELBO and its trend.
    loss_axis.scatter(
        successful["step"], successful["loss"], s=2, alpha=0.06,
        color="#6B6B6B", rasterized=True, label="episode ELBO",
    )
    loss_roll = successful["loss"].rolling(rolling_window, min_periods=20).mean()
    nll_roll = successful["nll"].rolling(rolling_window, min_periods=20).mean()
    loss_axis.plot(successful["step"], loss_roll, color="#1F77B4", lw=2, label=f"ELBO mean ({rolling_window})")
    loss_axis.plot(successful["step"], nll_roll, color="#FF7F0E", lw=1.4, label=f"NLL mean ({rolling_window})")
    loss_axis.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    warmup_end = (
        resolved_config.training.kl_warmup_steps if resolved_config else 5000
    )
    loss_axis.axvline(warmup_end, color="#9467BD", ls="--", lw=1.2, label="KL warm-up end")
    limits = loss_roll.dropna().quantile([0.01, 0.99])
    if len(limits) == 2 and np.isfinite(limits).all() and limits.iloc[0] < limits.iloc[1]:
        margin = 0.15 * (limits.iloc[1] - limits.iloc[0])
        loss_axis.set_ylim(limits.iloc[0] - margin, limits.iloc[1] + margin)
    loss_axis.set(title="A. Training objective", xlabel="Training step", ylabel="Normalized ELBO / NLL")
    loss_axis.legend(fontsize=8)
    loss_axis.grid(alpha=0.2)

    # Panel B: cell-averaged validation and batch-specific domain behavior.
    validation_axis.plot(
        validation["step"], validation["validation_rmse"],
        color="black", marker="o", ms=3, lw=2.2, label="all validation cells",
    )
    if not cell_history.empty:
        batch_history = (
            cell_history.groupby(["step", "batch"], as_index=False)["validation_rmse"].mean()
        )
        for batch_name, rows in batch_history.groupby("batch"):
            validation_axis.plot(
                rows["step"], rows["validation_rmse"],
                color=BATCH_COLORS.get(batch_name, BATCH_COLORS["unknown"]),
                lw=1.5, alpha=0.9, label=f"{batch_name} mean",
            )
    validation_axis.scatter([best_step], [best_rmse], s=90, marker="*", color="#2CA02C", zorder=5, label="best checkpoint")
    validation_axis.scatter([int(final["step"])], [final_rmse], s=45, marker="X", color="#D62728", zorder=5, label="last checkpoint")
    validation_axis.annotate(
        f"best {best_rmse:.5f} @ {best_step}",
        (best_step, best_rmse), xytext=(8, -18), textcoords="offset points", fontsize=9,
    )
    validation_axis.set(title="B. Validation trajectory RMSE", xlabel="Training step", ylabel="Raw SOH RMSE")
    validation_axis.legend(fontsize=8, ncol=2)
    validation_axis.grid(alpha=0.2)

    # Panel C: which validation cells dominate the best checkpoint error.
    best_cells = cell_history[cell_history["step"] == best_step].copy()
    if not best_cells.empty:
        best_cells = best_cells.sort_values("validation_rmse", ascending=False)
        cell_axis.barh(
            best_cells["cell_id"],
            best_cells["validation_rmse"],
            color=[BATCH_COLORS.get(name, BATCH_COLORS["unknown"]) for name in best_cells["batch"]],
        )
        cell_axis.invert_yaxis()
        cell_axis.axvline(best_rmse, color="black", ls="--", lw=1, label=f"all-cell mean={best_rmse:.4f}")
        cell_axis.tick_params(axis="y", labelsize=7)
        cell_axis.legend(fontsize=8)
    cell_axis.set(title=f"C. Per-cell RMSE at best step {best_step}", xlabel="Raw SOH RMSE", ylabel="Validation cell")
    cell_axis.grid(axis="x", alpha=0.2)

    # Panel D: unscaled gradient norm, clipping pressure, AMP scale and overflow.
    gradient_median = successful["gradient_norm"].rolling(rolling_window, min_periods=20).median()
    gradient_p95 = successful["gradient_norm"].rolling(rolling_window, min_periods=20).quantile(0.95)
    stability_axis.plot(successful["step"], gradient_median, color="#4C78A8", lw=1.8, label=f"gradient median ({rolling_window})")
    stability_axis.plot(successful["step"], gradient_p95, color="#E15759", lw=1.2, label=f"gradient p95 ({rolling_window})")
    clip_norm = (
        resolved_config.training.gradient_clip_norm if resolved_config else 1.0
    )
    stability_axis.axhline(clip_norm, color="black", ls="--", lw=1, label="clip norm=1")
    stability_axis.set_yscale("log")
    stability_axis.set(title="D. Optimization stability", xlabel="Training step", ylabel="Gradient norm before clipping (log)")
    stability_axis.grid(alpha=0.2, which="both")
    scale_axis = stability_axis.twinx()
    if "amp_scale_after" in history:
        scale_axis.step(history["step"], history["amp_scale_after"], where="post", color="#59A14F", alpha=0.65, lw=1, label="AMP scale")
        if skipped.any():
            overflow = history[skipped]
            scale_axis.scatter(overflow["step"], overflow["amp_scale_after"], marker="x", s=35, color="#B00020", label="AMP overflow")
    scale_axis.set_ylabel("AMP scale")
    first_handles, first_labels = stability_axis.get_legend_handles_labels()
    second_handles, second_labels = scale_axis.get_legend_handles_labels()
    stability_axis.legend(first_handles + second_handles, first_labels + second_labels, fontsize=8, ncol=2)

    improvement = (initial_rmse - best_rmse) / initial_rmse * 100.0
    final_gap = (final_rmse - best_rmse) / best_rmse * 100.0
    clip_fraction = float((successful["gradient_norm"] > clip_norm).mean() * 100.0)
    late = successful[successful["step"] > successful["step"].max() - 500]
    late_loss = float(late["loss"].mean())
    late_kl = float(late["kl"].mean())
    model_name = str(manifest.get("model", "unknown"))
    fold = int(manifest.get("fold", -1))
    test_evaluated = any(
        (run_directory / "evaluation").rglob("aggregate_metrics.csv")
    )
    test_status = "TEST EVALUATED" if test_evaluated else "TEST NOT RUN"
    figure.suptitle(
        f"MATR {model_name} · fold {fold} · training/validation summary\n"
        f"best validation RMSE={best_rmse:.5f} @ step {best_step} · "
        f"last={final_rmse:.5f} · runtime={runtime_hours:.2f} h · "
        f"AMP skips={int(skipped.sum())} · {test_status}",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)

    best_batch_means = (
        best_cells.groupby("batch")["validation_rmse"].mean().to_dict()
        if not best_cells.empty
        else {}
    )
    summary = {
        "run_directory": str(run_directory),
        "model": model_name,
        "fold": fold,
        "status": manifest.get("status"),
        "last_step": int(history["step"].max()),
        "successful_optimizer_updates": int((~skipped).sum()),
        "amp_overflow_skips": int(skipped.sum()),
        "runtime_hours": runtime_hours,
        "initial_validation_rmse": initial_rmse,
        "best_validation_step": best_step,
        "best_validation_rmse": best_rmse,
        "last_validation_rmse": final_rmse,
        "validation_improvement_percent": improvement,
        "last_vs_best_degradation_percent": final_gap,
        "best_batch_mean_rmse": {key: float(value) for key, value in best_batch_means.items()},
        "gradient_clipped_fraction_percent": clip_fraction,
        "last_500_loss_mean": late_loss,
        "last_500_kl_mean": late_kl,
        "held_out_test_evaluated": test_evaluated,
        "plot": str(output),
    }
    write_json(output.parent / "training_summary.json", summary)
    if not cell_history.empty:
        cell_history.to_csv(output.parent / "validation_per_cell_history.csv", index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a MATR ANP training summary")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = create_training_summary(args.run_dir, args.output)
    print(f"Training summary plot: {output}")


if __name__ == "__main__":
    main()

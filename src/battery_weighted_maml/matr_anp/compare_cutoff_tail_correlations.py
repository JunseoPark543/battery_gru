"""Compare cycle-dependent fixed-voltage discharge tails in MATR and HUST."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import load_config, resolve_data_root
from .data import CellData, load_dataset
from .plot_vq_cutoff_trend import q_at_voltage_cutoff
from .runtime import git_commit, write_json


COLORS = {"MATR": "#4C78A8", "HUST": "#F28E2B"}


def cutoff_tail_cycle_frame(
    cells: list[CellData],
    dataset: str,
    *,
    cutoff_voltage: float,
    endpoint_tolerance_v: float,
) -> pd.DataFrame:
    """Measure capacity accumulated after the first descending cutoff crossing.

    A cycle is eligible for correlation analysis only when its final voltage is
    close to the requested cutoff. This avoids treating continued discharge
    below the cutoff as a constant-voltage tail.
    """
    if not np.isfinite(cutoff_voltage):
        raise ValueError("cutoff_voltage must be finite")
    if not np.isfinite(endpoint_tolerance_v) or endpoint_tolerance_v <= 0:
        raise ValueError("endpoint_tolerance_v must be positive and finite")

    rows: list[dict[str, object]] = []
    for cell in cells:
        for cycle in cell.cycles:
            curve = cycle.discharge
            if curve is None:
                continue
            q_cutoff = q_at_voltage_cutoff(curve, cutoff_voltage)
            q_endpoint = float(curve.q[-1])
            endpoint_voltage = float(curve.voltage_v[-1])
            reached_cutoff = bool(np.isfinite(q_cutoff))
            endpoint_near_cutoff = bool(
                np.isfinite(endpoint_voltage)
                and abs(endpoint_voltage - cutoff_voltage) <= endpoint_tolerance_v
            )
            tail = (
                max(0.0, q_endpoint - float(q_cutoff))
                if reached_cutoff and endpoint_near_cutoff
                else np.nan
            )
            rows.append(
                {
                    "dataset": dataset,
                    "cell_id": cell.cell_id,
                    "cycle": int(cycle.cycle_number),
                    "cutoff_voltage_v": float(cutoff_voltage),
                    "endpoint_tolerance_v": float(endpoint_tolerance_v),
                    "q_at_first_cutoff": q_cutoff,
                    "q_endpoint": q_endpoint,
                    "endpoint_voltage_v": endpoint_voltage,
                    "reached_cutoff": reached_cutoff,
                    "endpoint_near_cutoff": endpoint_near_cutoff,
                    "valid_cutoff_tail": reached_cutoff and endpoint_near_cutoff,
                    "cutoff_tail_q_length": tail,
                    "cutoff_tail_fraction_of_endpoint": (
                        tail / q_endpoint
                        if np.isfinite(tail) and q_endpoint > 0
                        else np.nan
                    ),
                    "soh": float(cycle.soh),
                }
            )
    if not rows:
        raise ValueError(f"{dataset} contains no discharge curves")
    return pd.DataFrame(rows).sort_values(
        ["dataset", "cell_id", "cycle"]
    ).reset_index(drop=True)


def _edge_count(size: int, edge_fraction: float, minimum_edge_cycles: int) -> int:
    requested = max(minimum_edge_cycles, int(np.ceil(size * edge_fraction)))
    return min(requested, max(1, size // 2))


def cutoff_tail_cell_summary(
    per_cycle: pd.DataFrame,
    *,
    minimum_valid_cycles: int,
    edge_fraction: float,
    minimum_edge_cycles: int,
) -> pd.DataFrame:
    """Calculate one cycle-vs-tail correlation and early/late change per cell."""
    if minimum_valid_cycles < 3:
        raise ValueError("minimum_valid_cycles must be at least three")
    if not 0.0 < edge_fraction <= 0.5:
        raise ValueError("edge_fraction must lie in (0, 0.5]")
    if minimum_edge_cycles <= 0:
        raise ValueError("minimum_edge_cycles must be positive")

    rows: list[dict[str, object]] = []
    for (dataset, cell_id), group in per_cycle.groupby(
        ["dataset", "cell_id"], sort=True
    ):
        ordered = group.sort_values("cycle")
        valid = ordered.dropna(subset=["cutoff_tail_q_length"])
        valid_count = int(len(valid))
        record: dict[str, object] = {
            "dataset": dataset,
            "cell_id": cell_id,
            "observed_curve_cycles": int(len(ordered)),
            "valid_cutoff_tail_cycles": valid_count,
            "valid_fraction": float(valid_count / len(ordered)),
            "first_valid_cycle": int(valid["cycle"].iloc[0]) if valid_count else np.nan,
            "last_valid_cycle": int(valid["cycle"].iloc[-1]) if valid_count else np.nan,
            "analyzed": valid_count >= minimum_valid_cycles,
        }
        if valid_count < minimum_valid_cycles:
            record.update(
                {
                    "pearson_cycle_vs_tail": np.nan,
                    "spearman_cycle_vs_tail": np.nan,
                    "spearman_pvalue": np.nan,
                    "linear_slope_q_per_cycle": np.nan,
                    "edge_cycles": 0,
                    "early_tail_mean": np.nan,
                    "late_tail_mean": np.nan,
                    "late_minus_early": np.nan,
                    "late_to_early_ratio": np.nan,
                    "positive_trend": False,
                    "strong_positive_trend": False,
                }
            )
            rows.append(record)
            continue

        cycles = valid["cycle"].to_numpy(dtype=np.float64)
        tails = valid["cutoff_tail_q_length"].to_numpy(dtype=np.float64)
        edge_count = _edge_count(valid_count, edge_fraction, minimum_edge_cycles)
        early_mean = float(np.mean(tails[:edge_count]))
        late_mean = float(np.mean(tails[-edge_count:]))
        pearson = float(np.corrcoef(cycles, tails)[0, 1])
        spearman_result = spearmanr(cycles, tails)
        spearman = float(spearman_result.statistic)
        pvalue = float(spearman_result.pvalue)
        slope = float(np.polyfit(cycles, tails, 1)[0])
        delta = late_mean - early_mean
        record.update(
            {
                "pearson_cycle_vs_tail": pearson,
                "spearman_cycle_vs_tail": spearman,
                "spearman_pvalue": pvalue,
                "linear_slope_q_per_cycle": slope,
                "edge_cycles": edge_count,
                "early_tail_mean": early_mean,
                "late_tail_mean": late_mean,
                "late_minus_early": delta,
                "late_to_early_ratio": (
                    late_mean / early_mean if early_mean > 0 else np.nan
                ),
                "positive_trend": bool(spearman > 0 and delta > 0),
                "strong_positive_trend": bool(spearman >= 0.5 and delta > 0),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["dataset", "cell_id"]).reset_index(drop=True)


def cutoff_tail_dataset_summary(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cell-level conclusions without pooling cycles across cells."""
    rows: list[dict[str, object]] = []
    for dataset, group in per_cell.groupby("dataset", sort=True):
        valid = group[group["analyzed"]].copy()
        analyzed = int(len(valid))
        rows.append(
            {
                "dataset": dataset,
                "total_cells": int(len(group)),
                "analyzed_cells": analyzed,
                "excluded_cells": int(len(group) - analyzed),
                "median_valid_fraction": float(group["valid_fraction"].median()),
                "median_spearman": (
                    float(valid["spearman_cycle_vs_tail"].median())
                    if analyzed
                    else np.nan
                ),
                "spearman_q25": (
                    float(valid["spearman_cycle_vs_tail"].quantile(0.25))
                    if analyzed
                    else np.nan
                ),
                "spearman_q75": (
                    float(valid["spearman_cycle_vs_tail"].quantile(0.75))
                    if analyzed
                    else np.nan
                ),
                "positive_trend_cells": int(valid["positive_trend"].sum()),
                "positive_trend_fraction": (
                    float(valid["positive_trend"].mean()) if analyzed else np.nan
                ),
                "strong_positive_trend_cells": int(
                    valid["strong_positive_trend"].sum()
                ),
                "strong_positive_trend_fraction": (
                    float(valid["strong_positive_trend"].mean())
                    if analyzed
                    else np.nan
                ),
                "median_early_tail": (
                    float(valid["early_tail_mean"].median()) if analyzed else np.nan
                ),
                "median_late_tail": (
                    float(valid["late_tail_mean"].median()) if analyzed else np.nan
                ),
                "median_late_minus_early": (
                    float(valid["late_minus_early"].median()) if analyzed else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def normalized_tail_quantiles(
    per_cycle: pd.DataFrame,
    per_cell: pd.DataFrame,
    *,
    grid_points: int = 101,
) -> pd.DataFrame:
    """Interpolate each eligible cell onto normalized cycle for visualization."""
    if grid_points < 10:
        raise ValueError("grid_points must be at least ten")
    eligible = set(
        zip(
            per_cell.loc[per_cell["analyzed"], "dataset"],
            per_cell.loc[per_cell["analyzed"], "cell_id"],
        )
    )
    grid = np.linspace(0.0, 1.0, grid_points)
    rows: list[dict[str, object]] = []
    for dataset, dataset_rows in per_cycle.groupby("dataset", sort=True):
        curves: list[np.ndarray] = []
        for cell_id, group in dataset_rows.groupby("cell_id", sort=True):
            if (dataset, cell_id) not in eligible:
                continue
            valid = group.dropna(subset=["cutoff_tail_q_length"]).sort_values("cycle")
            cycle = valid["cycle"].to_numpy(dtype=np.float64)
            normalized_cycle = (cycle - cycle[0]) / max(cycle[-1] - cycle[0], 1.0)
            curves.append(
                np.interp(
                    grid,
                    normalized_cycle,
                    valid["cutoff_tail_q_length"].to_numpy(dtype=np.float64),
                )
            )
        if not curves:
            continue
        matrix = np.stack(curves)
        quantiles = np.quantile(matrix, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
        for index, value in enumerate(grid):
            rows.append(
                {
                    "dataset": dataset,
                    "normalized_cycle": float(value),
                    "q10": float(quantiles[0, index]),
                    "q25": float(quantiles[1, index]),
                    "median": float(quantiles[2, index]),
                    "q75": float(quantiles[3, index]),
                    "q90": float(quantiles[4, index]),
                    "num_cells": int(matrix.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def plot_cutoff_tail_comparison(
    per_cell: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    normalized: pd.DataFrame,
    destination: str | Path,
    *,
    dpi: int,
) -> Path:
    """Create one figure containing the main cross-dataset conclusions."""
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    trend_axis, corr_axis, edge_axis, rate_axis = axes.ravel()

    for dataset, group in normalized.groupby("dataset", sort=True):
        color = COLORS.get(str(dataset), "0.4")
        trend_axis.fill_between(
            group["normalized_cycle"], group["q10"], group["q90"],
            color=color, alpha=0.12,
        )
        trend_axis.fill_between(
            group["normalized_cycle"], group["q25"], group["q75"],
            color=color, alpha=0.24,
        )
        trend_axis.plot(
            group["normalized_cycle"], group["median"],
            color=color, linewidth=2.2,
            label=f"{dataset} median (n={int(group['num_cells'].iloc[0])})",
        )
    trend_axis.set(
        xlabel="Normalized cycle within each cell",
        ylabel="Post-cutoff Q length (Qd / Qnom)",
        title="A. Cell-normalized cutoff-tail trajectories",
    )
    trend_axis.grid(alpha=0.2)
    trend_axis.legend()

    datasets = list(dataset_summary["dataset"])
    positions = np.arange(len(datasets), dtype=np.float64)
    rng = np.random.default_rng(42)
    for position, dataset in zip(positions, datasets):
        values = per_cell.loc[
            (per_cell["dataset"] == dataset) & per_cell["analyzed"],
            "spearman_cycle_vs_tail",
        ].dropna().to_numpy(dtype=np.float64)
        corr_axis.boxplot(
            [values], positions=[position], widths=0.45, showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": COLORS.get(str(dataset), "0.6"), "alpha": 0.45},
            medianprops={"color": "black", "linewidth": 1.5},
        )
        corr_axis.scatter(
            position + rng.uniform(-0.13, 0.13, size=len(values)), values,
            color=COLORS.get(str(dataset), "0.4"), s=16, alpha=0.58,
        )
    corr_axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    corr_axis.axhline(0.5, color="0.4", linestyle=":", linewidth=1)
    corr_axis.set_xticks(positions, datasets)
    corr_axis.set(
        ylabel="Spearman(cycle, post-cutoff Q length)",
        title="B. Per-cell correlation distribution",
    )
    corr_axis.grid(alpha=0.2)

    all_edges = per_cell.loc[per_cell["analyzed"], [
        "dataset", "cell_id", "early_tail_mean", "late_tail_mean"
    ]]
    finite_edges = all_edges[
        np.isfinite(all_edges["early_tail_mean"])
        & np.isfinite(all_edges["late_tail_mean"])
    ]
    if not finite_edges.empty:
        maximum = float(
            max(finite_edges["early_tail_mean"].max(), finite_edges["late_tail_mean"].max())
        )
        edge_axis.plot([0, maximum], [0, maximum], color="black", linestyle="--", lw=1)
    for dataset, group in finite_edges.groupby("dataset", sort=True):
        edge_axis.scatter(
            group["early_tail_mean"], group["late_tail_mean"],
            color=COLORS.get(str(dataset), "0.4"), s=30, alpha=0.65,
            label=str(dataset),
        )
    edge_axis.set(
        xlabel="Mean tail length in early edge cycles",
        ylabel="Mean tail length in late edge cycles",
        title="C. Early vs late tail length (above line = growth)",
    )
    edge_axis.grid(alpha=0.2)
    edge_axis.legend()

    width = 0.34
    positive = dataset_summary["positive_trend_fraction"].to_numpy(dtype=float)
    strong = dataset_summary["strong_positive_trend_fraction"].to_numpy(dtype=float)
    rate_axis.bar(positions - width / 2, positive, width, label="positive", color="#72B7B2")
    rate_axis.bar(positions + width / 2, strong, width, label="strong positive", color="#E45756")
    for x, value in zip(positions - width / 2, positive):
        rate_axis.text(x, value + 0.02, f"{value:.1%}", ha="center", fontsize=9)
    for x, value in zip(positions + width / 2, strong):
        rate_axis.text(x, value + 0.02, f"{value:.1%}", ha="center", fontsize=9)
    rate_axis.set_xticks(positions, datasets)
    rate_axis.set_ylim(0.0, 1.12)
    rate_axis.set(
        ylabel="Fraction of analyzed cells",
        title="D. Cells whose cutoff tail grows with cycle",
    )
    rate_axis.grid(axis="y", alpha=0.2)
    rate_axis.legend()

    figure.suptitle("MATR vs HUST: cycle dependence of the fixed-voltage Q tail", fontsize=15)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def compare_cutoff_tail_correlations(
    matr_config_path: str,
    hust_config_path: str,
    *,
    matr_root: str | None,
    hust_root: str | None,
    output_dir: str | Path,
    matr_cutoff_voltage: float,
    hust_cutoff_voltage: float,
    endpoint_tolerance_v: float,
    minimum_valid_cycles: int,
    edge_fraction: float,
    minimum_edge_cycles: int,
    dpi: int,
) -> Path:
    matr_config = load_config(matr_config_path)
    hust_config = load_config(hust_config_path)
    if matr_config.data.dataset.upper() != "MATR":
        raise ValueError("matr_config must describe MATR")
    if hust_config.data.dataset.upper() != "HUST":
        raise ValueError("hust_config must describe HUST")

    resolved_matr_root = resolve_data_root(matr_config, matr_root)
    resolved_hust_root = resolve_data_root(hust_config, hust_root)
    matr_cells, matr_audit = load_dataset(
        resolved_matr_root, matr_config.data, tolerate_invalid_cells=True
    )
    hust_cells, hust_audit = load_dataset(
        resolved_hust_root, hust_config.data, tolerate_invalid_cells=True
    )
    per_cycle = pd.concat(
        [
            cutoff_tail_cycle_frame(
                matr_cells, "MATR",
                cutoff_voltage=matr_cutoff_voltage,
                endpoint_tolerance_v=endpoint_tolerance_v,
            ),
            cutoff_tail_cycle_frame(
                hust_cells, "HUST",
                cutoff_voltage=hust_cutoff_voltage,
                endpoint_tolerance_v=endpoint_tolerance_v,
            ),
        ],
        ignore_index=True,
    )
    per_cell = cutoff_tail_cell_summary(
        per_cycle,
        minimum_valid_cycles=minimum_valid_cycles,
        edge_fraction=edge_fraction,
        minimum_edge_cycles=minimum_edge_cycles,
    )
    dataset_summary = cutoff_tail_dataset_summary(per_cell)
    normalized = normalized_tail_quantiles(per_cycle, per_cell)

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plot_path = plot_cutoff_tail_comparison(
        per_cell,
        dataset_summary,
        normalized,
        destination / "matr_vs_hust_cutoff_tail_correlations.png",
        dpi=dpi,
    )
    per_cycle.to_csv(destination / "per_cycle_cutoff_tail.csv", index=False)
    per_cell.to_csv(destination / "per_cell_cutoff_tail_correlations.csv", index=False)
    dataset_summary.to_csv(destination / "dataset_summary.csv", index=False)
    normalized.to_csv(destination / "normalized_tail_quantiles.csv", index=False)
    matr_audit.to_csv(destination / "matr_data_audit.csv", index=False)
    hust_audit.to_csv(destination / "hust_data_audit.csv", index=False)
    write_json(
        destination / "analysis_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "matr_root": str(resolved_matr_root),
            "hust_root": str(resolved_hust_root),
            "matr_cells_loaded": len(matr_cells),
            "hust_cells_loaded": len(hust_cells),
            "matr_cutoff_voltage_v": matr_cutoff_voltage,
            "hust_cutoff_voltage_v": hust_cutoff_voltage,
            "endpoint_tolerance_v": endpoint_tolerance_v,
            "minimum_valid_cycles": minimum_valid_cycles,
            "edge_fraction": edge_fraction,
            "minimum_edge_cycles": minimum_edge_cycles,
            "plot": str(plot_path),
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-cell correlations between cycle and the Q accumulated "
            "after reaching a fixed discharge cutoff in MATR and HUST"
        )
    )
    parser.add_argument("--matr-config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--hust-config", default="configs/hust_partial_iv_analysis.yaml")
    parser.add_argument("--matr-root")
    parser.add_argument("--hust-root")
    parser.add_argument(
        "--output-dir",
        default="outputs/data_analysis/matr_vs_hust_cutoff_tail",
    )
    parser.add_argument("--matr-cutoff-voltage", type=float, default=2.0)
    parser.add_argument("--hust-cutoff-voltage", type=float, default=2.0)
    parser.add_argument("--endpoint-tolerance-v", type=float, default=0.01)
    parser.add_argument("--minimum-valid-cycles", type=int, default=20)
    parser.add_argument("--edge-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-edge-cycles", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = compare_cutoff_tail_correlations(
        args.matr_config,
        args.hust_config,
        matr_root=args.matr_root,
        hust_root=args.hust_root,
        output_dir=args.output_dir,
        matr_cutoff_voltage=args.matr_cutoff_voltage,
        hust_cutoff_voltage=args.hust_cutoff_voltage,
        endpoint_tolerance_v=args.endpoint_tolerance_v,
        minimum_valid_cycles=args.minimum_valid_cycles,
        edge_fraction=args.edge_fraction,
        minimum_edge_cycles=args.minimum_edge_cycles,
        dpi=args.dpi,
    )
    summary = pd.read_csv(destination / "dataset_summary.csv")
    print(summary.to_string(index=False))
    print(f"Output: {destination}")


if __name__ == "__main__":
    main()

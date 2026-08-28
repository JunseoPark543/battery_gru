"""Compare MATR and HUST SOH trajectory distributions without training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import load_config, resolve_data_root
from .data import CellData, load_dataset
from .runtime import git_commit, write_json


COLORS = {"MATR": "#4C78A8", "HUST": "#F28E2B"}


def trajectory_frame(cells: list[CellData], dataset: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cell in cells:
        cycles = cell.cycle_numbers.astype(np.int64)
        span = max(float(cycles[-1] - cycles[0]), 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "cell_id": cell.cell_id,
                    "cycle": cycles,
                    "normalized_life": (cycles - cycles[0]) / span,
                    "soh": cell.soh,
                }
            )
        )
    if not frames:
        raise ValueError(f"{dataset} contains no valid cells")
    return pd.concat(frames, ignore_index=True)


def cell_summary(
    trajectories: pd.DataFrame,
    *,
    eol_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, cell_id), group in trajectories.groupby(
        ["dataset", "cell_id"], sort=True
    ):
        ordered = group.sort_values("cycle")
        crossed = ordered[ordered["soh"] <= eol_threshold]
        rows.append(
            {
                "dataset": dataset,
                "cell_id": cell_id,
                "num_observed_cycles": int(len(ordered)),
                "first_cycle": int(ordered["cycle"].iloc[0]),
                "last_cycle": int(ordered["cycle"].iloc[-1]),
                "initial_soh": float(ordered["soh"].iloc[0]),
                "final_soh": float(ordered["soh"].iloc[-1]),
                "first_eol_crossing_cycle": (
                    int(crossed["cycle"].iloc[0]) if not crossed.empty else np.nan
                ),
                "reached_eol_threshold": bool(not crossed.empty),
            }
        )
    return pd.DataFrame(rows)


def normalized_distribution(
    trajectories: pd.DataFrame,
    *,
    grid_points: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if grid_points < 20:
        raise ValueError("grid_points must be at least 20")
    grid = np.linspace(0.0, 1.0, grid_points)
    records: list[dict[str, float | str]] = []
    matrices: dict[str, np.ndarray] = {}
    for dataset, dataset_rows in trajectories.groupby("dataset", sort=True):
        curves = []
        for _, group in dataset_rows.groupby("cell_id", sort=True):
            ordered = group.sort_values("normalized_life")
            curves.append(
                np.interp(
                    grid,
                    ordered["normalized_life"].to_numpy(dtype=np.float64),
                    ordered["soh"].to_numpy(dtype=np.float64),
                )
            )
        matrix = np.stack(curves)
        matrices[str(dataset)] = matrix
        quantiles = np.quantile(matrix, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
        mean = np.mean(matrix, axis=0)
        for index, normalized_life in enumerate(grid):
            records.append(
                {
                    "dataset": str(dataset),
                    "normalized_life": float(normalized_life),
                    "mean": float(mean[index]),
                    "q10": float(quantiles[0, index]),
                    "q25": float(quantiles[1, index]),
                    "median": float(quantiles[2, index]),
                    "q75": float(quantiles[3, index]),
                    "q90": float(quantiles[4, index]),
                    "num_cells": int(matrix.shape[0]),
                }
            )
    return pd.DataFrame(records), matrices


def plot_comparison(
    trajectories: pd.DataFrame,
    summary: pd.DataFrame,
    normalized: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    destination: str | Path,
    *,
    eol_threshold: float,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))

    raw_axis = axes[0, 0]
    for (dataset, _), group in trajectories.groupby(
        ["dataset", "cell_id"], sort=True
    ):
        raw_axis.plot(
            group["cycle"], group["soh"],
            color=COLORS[str(dataset)], linewidth=0.7, alpha=0.13,
        )
    raw_axis.axhline(eol_threshold, color="black", linestyle="--", linewidth=1)
    raw_axis.set(
        xlabel="Cycle number",
        ylabel="SOH",
        title="A. All observed trajectories (absolute cycle)",
    )
    raw_axis.grid(alpha=0.2)

    normalized_axis = axes[0, 1]
    for dataset, rows in normalized.groupby("dataset", sort=True):
        color = COLORS[str(dataset)]
        normalized_axis.fill_between(
            rows["normalized_life"], rows["q10"], rows["q90"],
            color=color, alpha=0.12,
        )
        normalized_axis.fill_between(
            rows["normalized_life"], rows["q25"], rows["q75"],
            color=color, alpha=0.24,
        )
        normalized_axis.plot(
            rows["normalized_life"], rows["median"],
            color=color, linewidth=2.2, label=f"{dataset} median",
        )
    normalized_axis.axhline(
        eol_threshold, color="black", linestyle="--", linewidth=1
    )
    normalized_axis.set(
        xlabel="Normalized lifetime",
        ylabel="SOH",
        title="B. Median and 10-90% / 25-75% trajectory bands",
    )
    normalized_axis.grid(alpha=0.2)
    normalized_axis.legend()

    life_axis = axes[1, 0]
    minimum = float(summary["last_cycle"].min())
    maximum = float(summary["last_cycle"].max())
    bins = np.linspace(minimum, maximum + max(1.0, maximum - minimum) * 1e-6, 26)
    for dataset, rows in summary.groupby("dataset", sort=True):
        life_axis.hist(
            rows["last_cycle"], bins=bins, density=True,
            color=COLORS[str(dataset)], alpha=0.38,
            label=f"{dataset} (n={len(rows)})",
        )
        life_axis.axvline(
            float(rows["last_cycle"].median()),
            color=COLORS[str(dataset)], linestyle="--", linewidth=1.8,
        )
    life_axis.set(
        xlabel="Last observed cycle",
        ylabel="Density",
        title="C. Cell lifetime distribution (dashed = median)",
    )
    life_axis.grid(alpha=0.2)
    life_axis.legend()

    phase_axis = axes[1, 1]
    phases = np.asarray([0.1, 0.25, 0.5, 0.75, 1.0])
    base_positions = np.arange(len(phases), dtype=np.float64)
    offsets = {"MATR": -0.17, "HUST": 0.17}
    handles = []
    for dataset in ("MATR", "HUST"):
        matrix = matrices[dataset]
        indices = np.rint(phases * (matrix.shape[1] - 1)).astype(int)
        boxes = phase_axis.boxplot(
            [matrix[:, index] for index in indices],
            positions=base_positions + offsets[dataset],
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch in boxes["boxes"]:
            patch.set(facecolor=COLORS[dataset], alpha=0.5)
        handles.append(
            Line2D([0], [0], color=COLORS[dataset], linewidth=8, label=dataset)
        )
    phase_axis.axhline(eol_threshold, color="black", linestyle="--", linewidth=1)
    phase_axis.set_xticks(base_positions, [f"{value:g}" for value in phases])
    phase_axis.set(
        xlabel="Normalized lifetime checkpoint",
        ylabel="SOH",
        title="D. SOH distribution at matched lifetime phases",
    )
    phase_axis.grid(alpha=0.2)
    phase_axis.legend(handles=handles)

    raw_axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS[name], linewidth=2, label=name)
            for name in ("MATR", "HUST")
        ]
        + [
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=1,
                label=f"SOH={eol_threshold:g}",
            )
        ]
    )
    figure.suptitle("MATR vs HUST: SOH trajectory distributions")
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def compare_datasets(
    matr_config_path: str,
    hust_config_path: str,
    *,
    matr_root: str | None,
    hust_root: str | None,
    output_dir: str | Path,
    eol_threshold: float,
    grid_points: int,
    dpi: int,
) -> Path:
    if not 0.0 < eol_threshold < 1.5:
        raise ValueError("eol_threshold must lie in (0,1.5)")
    matr_config = load_config(matr_config_path)
    hust_config = load_config(hust_config_path)
    if matr_config.data.dataset.upper() != "MATR":
        raise ValueError("matr_config must describe the MATR dataset")
    if hust_config.data.dataset.upper() != "HUST":
        raise ValueError("hust_config must describe the HUST dataset")
    resolved_matr_root = resolve_data_root(matr_config, matr_root)
    resolved_hust_root = resolve_data_root(hust_config, hust_root)
    matr_cells, matr_audit = load_dataset(
        resolved_matr_root, matr_config.data, tolerate_invalid_cells=True
    )
    hust_cells, hust_audit = load_dataset(
        resolved_hust_root, hust_config.data, tolerate_invalid_cells=True
    )
    trajectories = pd.concat(
        [trajectory_frame(matr_cells, "MATR"), trajectory_frame(hust_cells, "HUST")],
        ignore_index=True,
    )
    summary = cell_summary(trajectories, eol_threshold=eol_threshold)
    normalized, matrices = normalized_distribution(
        trajectories, grid_points=grid_points
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plot_path = plot_comparison(
        trajectories,
        summary,
        normalized,
        matrices,
        destination / "matr_vs_hust_soh_distribution.png",
        eol_threshold=eol_threshold,
        dpi=dpi,
    )
    trajectories.to_csv(destination / "trajectory_points.csv", index=False)
    summary.to_csv(destination / "cell_summary.csv", index=False)
    normalized.to_csv(destination / "normalized_trajectory_quantiles.csv", index=False)
    matr_audit.to_csv(destination / "matr_data_audit.csv", index=False)
    hust_audit.to_csv(destination / "hust_data_audit.csv", index=False)
    write_json(
        destination / "comparison_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "matr_root": str(resolved_matr_root),
            "hust_root": str(resolved_hust_root),
            "matr_cells": len(matr_cells),
            "hust_cells": len(hust_cells),
            "eol_threshold": eol_threshold,
            "normalized_grid_points": grid_points,
            "plot": str(plot_path),
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MATR and HUST SOH trajectory distributions"
    )
    parser.add_argument("--matr-config", default="configs/matr_hs_anp.yaml")
    parser.add_argument("--hust-config", default="configs/hust_partial_iv_analysis.yaml")
    parser.add_argument("--matr-root")
    parser.add_argument("--hust-root")
    parser.add_argument(
        "--output-dir",
        default="outputs/data_analysis/matr_vs_hust_soh",
    )
    parser.add_argument("--eol-threshold", type=float, default=0.8)
    parser.add_argument("--grid-points", type=int, default=201)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = compare_datasets(
        args.matr_config,
        args.hust_config,
        matr_root=args.matr_root,
        hust_root=args.hust_root,
        output_dir=args.output_dir,
        eol_threshold=args.eol_threshold,
        grid_points=args.grid_points,
        dpi=args.dpi,
    )
    print(f"MATR/HUST SOH comparison: {destination}")


if __name__ == "__main__":
    main()

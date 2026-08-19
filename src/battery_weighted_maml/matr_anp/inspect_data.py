"""Audit BatteryLife MATR structure without training a model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, resolve_data_root, save_config
from .data import load_matr_dataset
from .runtime import git_commit, write_json


def inspect(config_path: str, data_root_arg: str | None, output: str | None) -> Path:
    config = load_config(config_path)
    data_root = resolve_data_root(config, data_root_arg)
    destination = (
        Path(output).resolve()
        if output
        else Path(config.paths.output_root).resolve() / "data_audit"
    )
    destination.mkdir(parents=True, exist_ok=True)
    cells, audit = load_matr_dataset(
        data_root, config.data, tolerate_invalid_cells=True
    )
    if not cells:
        raise ValueError("MATR audit found no valid cells")
    audit.to_csv(destination / "data_audit.csv", index=False)
    save_config(config, destination / "resolved_config.yaml")
    write_json(
        destination / "data_audit.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root),
            "dataset": "MATR",
            "git_commit": git_commit(),
            "valid_cell_count": len(cells),
            "invalid_file_count": int((audit["status"] != "valid").sum()),
            "cells": audit.to_dict("records"),
        },
    )
    print(f"Valid MATR cells: {len(cells)}")
    print(f"Audit: {destination / 'data_audit.csv'}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect BatteryLife MATR pickle structure")
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect(args.config, args.data_root, args.output)


if __name__ == "__main__":
    main()

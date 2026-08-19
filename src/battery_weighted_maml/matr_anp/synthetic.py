"""Synthetic MATR-like BatteryLife pickles for local tests and smoke runs."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def write_synthetic_matr_dataset(
    root: str | Path,
    *,
    num_cells: int = 6,
    num_cycles: int = 36,
    signal_points: int = 64,
    seed: int = 42,
) -> Path:
    destination = Path(root) / "MATR_synthetic"
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nominal = 1.1
    for cell_index in range(num_cells):
        records = []
        cell_offset = 0.002 * cell_index
        for cycle in range(1, num_cycles + 1):
            soh = 1.0 - cell_offset - 0.0045 * (cycle - 1)
            capacity_end = nominal * soh
            capacity_trace = np.linspace(0.0, capacity_end, signal_points)
            q = capacity_trace / nominal
            degradation_shift = 0.0018 * cycle + 0.002 * cell_index
            voltage = (
                3.65
                - 0.72 * q
                - degradation_shift
                + 0.002 * np.sin(np.linspace(0, 4 * np.pi, signal_points))
            )
            voltage += rng.normal(0.0, 2.0e-4, signal_points)
            current = -np.full(signal_points, 1.0 + 0.03 * cell_index)
            records.append(
                {
                    "cycle_number": cycle,
                    "time_in_s": np.linspace(0.0, 3600.0, signal_points),
                    "voltage_in_V": voltage,
                    "current_in_A": current,
                    "discharge_capacity_in_Ah": capacity_trace,
                    "stage": ["discharge"] * signal_points,
                }
            )
        payload = {
            "dataset": "MATR",
            "cell_id": f"MATR_SYNTH_{cell_index:02d}",
            "nominal_capacity_in_Ah": nominal,
            "cycle_data": records,
        }
        with (destination / f"MATR_SYNTH_{cell_index:02d}.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
    return destination

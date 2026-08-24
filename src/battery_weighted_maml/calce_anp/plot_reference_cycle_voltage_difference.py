"""CALCE entry point for reference-cycle voltage-Q difference analysis."""

from __future__ import annotations

from battery_weighted_maml.matr_anp.plot_reference_cycle_voltage_difference import main


if __name__ == "__main__":
    main("configs/calce_partial_iv_anp.yaml")

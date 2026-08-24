"""CALCE entry point for online voltage-Q snapshot visualization."""

from __future__ import annotations

from battery_weighted_maml.matr_anp.plot_cycle_time_fraction_voltage import main


if __name__ == "__main__":
    main("configs/calce_partial_iv_anp.yaml")

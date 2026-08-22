"""Run the CALCE SOH-only and partial-I-V ANP experiment suite."""

from __future__ import annotations

from battery_weighted_maml.matr_anp.run_suite import main as _shared_main


def main() -> None:
    _shared_main("configs/calce_partial_iv_anp.yaml")


if __name__ == "__main__":
    main()

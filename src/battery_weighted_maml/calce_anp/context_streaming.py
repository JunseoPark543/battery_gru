"""Evaluate a CALCE ANP checkpoint with progressively expanding context."""

from __future__ import annotations

from battery_weighted_maml.matr_anp.context_streaming import main as _shared_main


def main() -> None:
    _shared_main("configs/calce_partial_iv_anp.yaml")


if __name__ == "__main__":
    main()

"""Evaluate a held-out HUST A0--A3 checkpoint."""

from __future__ import annotations

from battery_weighted_maml.matr_anp.evaluate import main as _shared_main


def main() -> None:
    _shared_main("configs/hust_hs_anp.yaml")


if __name__ == "__main__":
    main()

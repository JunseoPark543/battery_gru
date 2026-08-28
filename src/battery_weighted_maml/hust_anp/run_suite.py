"""Train and evaluate HUST A0--A3 ANP ablations through the shared pipeline."""

from __future__ import annotations

from battery_weighted_maml.matr_anp.run_suite import main as _shared_main


def main() -> None:
    _shared_main("configs/hust_hs_anp.yaml")


if __name__ == "__main__":
    main()

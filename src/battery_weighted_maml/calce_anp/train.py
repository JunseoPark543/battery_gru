"""Train or resume one CALCE ANP fold."""

from __future__ import annotations

from battery_weighted_maml.matr_anp.train import main as _shared_main


def main() -> None:
    _shared_main("configs/calce_partial_iv_anp.yaml")


if __name__ == "__main__":
    main()

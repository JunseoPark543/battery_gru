"""Allow ``python -m calce_direct_rul_boil`` as a short CLI form."""

from .main import parse_args, run


if __name__ == "__main__":
    run(parse_args())


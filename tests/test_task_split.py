from __future__ import annotations

import numpy as np
import pytest


def test_source_support_query_split(parsed_cell):
    task = parsed_cell.source_task(2)
    np.testing.assert_allclose(task.support_soh, [1.05, 1.0])
    np.testing.assert_allclose(task.query_soh, [0.95])


def test_cell_requires_l_plus_one_cycles(parsed_cell):
    with pytest.raises(ValueError, match=r"L\+1"):
        parsed_cell.source_task(3)

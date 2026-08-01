from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from battery_weighted_maml.config import ExperimentConfig
from battery_weighted_maml.data.task_views import TargetSupportView
from battery_weighted_maml.logging_utils import configure_logging
from battery_weighted_maml.models.gru_seq2seq import GRUSeq2Seq
from battery_weighted_maml.training.trainer import WeightedMAMLTrainer


def test_target_support_does_not_expose_future_or_eol(parsed_cell):
    support = parsed_cell.target_support(2)
    assert isinstance(support, TargetSupportView)
    assert not hasattr(support, "future_soh")
    assert not hasattr(support, "true_eol_cycle")
    assert not hasattr(support, "full_trajectory")
    np.testing.assert_allclose(support.soh, parsed_cell.soh[:2])


def test_trainer_signature_accepts_target_support_view_only():
    annotation = inspect.signature(WeightedMAMLTrainer.__init__).parameters["target_support"].annotation
    assert annotation == "TargetSupportView"


def test_trainer_rejects_full_target(parsed_cell, tmp_path):
    source = parsed_cell.source_task(2)
    with pytest.raises(TypeError, match="TargetSupportView"):
        WeightedMAMLTrainer(
            GRUSeq2Seq(hidden_size=4), [source], parsed_cell, ExperimentConfig(),
            torch.device("cpu"), tmp_path, "same_family", configure_logging(None),
        )


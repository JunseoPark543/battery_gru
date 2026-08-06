from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from battery_weighted_maml.config import ExperimentConfig
from battery_weighted_maml.data.task_views import FullCellTrajectory, TargetSupportView
from battery_weighted_maml.logging_utils import configure_logging
from battery_weighted_maml.meta.target_adaptation import adapt_target
from battery_weighted_maml.models.gru_seq2seq import GRUSeq2Seq
from battery_weighted_maml.seed import capture_rng_state, restore_rng_state
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


def test_rng_state_restore_uses_cpu_byte_tensor():
    state = capture_rng_state()
    restore_rng_state(state)
    restored = torch.get_rng_state()
    assert restored.device.type == "cpu"
    assert restored.dtype == torch.uint8


def test_fast_adaptation_snapshots_are_one_continuous_trajectory():
    torch.manual_seed(9)
    model = GRUSeq2Seq(hidden_size=4)
    trajectory = FullCellTrajectory(
        file_name="CALCE_CX2_37.pkl",
        cell_id="CX2_37",
        family="CX2",
        nominal_capacity_ah=1.0,
        cycles=np.arange(1, 8),
        capacities_ah=np.linspace(1.0, 0.8, 7),
        soh=np.linspace(1.0, 0.8, 7),
        is_interpolated=np.zeros(7, dtype=bool),
        true_eol_cycle=7,
        raw_cycle_count=7,
        missing_count_before=0,
        missing_count_after=0,
    )
    support = trajectory.target_support(6)
    result = adapt_target(
        model,
        support,
        max_steps=5,
        learning_rate=0.01,
        batch_size=3,
        teacher_forcing_ratio=0.5,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(123),
        capture_steps=[1, 3, 5],
    )
    assert set(result.snapshots) == {1, 3, 5}
    assert result.history["step"].tolist() == [1, 2, 3, 4, 5]

    for step in (1, 3, 5):
        rerun = adapt_target(
            model,
            support,
            max_steps=step,
            learning_rate=0.01,
            batch_size=3,
            teacher_forcing_ratio=0.5,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(123),
            capture_steps=[step],
        )
        for expected, actual in zip(
            result.snapshots[step].parameters(), rerun.snapshots[step].parameters()
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_boil_target_adaptation_updates_body_and_freezes_head():
    torch.manual_seed(17)
    model = GRUSeq2Seq(hidden_size=4)
    trajectory = FullCellTrajectory(
        file_name="CALCE_CX2_37.pkl",
        cell_id="CX2_37",
        family="CX2",
        nominal_capacity_ah=1.0,
        cycles=np.arange(1, 8),
        capacities_ah=np.linspace(1.0, 0.8, 7),
        soh=np.linspace(1.0, 0.8, 7),
        is_interpolated=np.zeros(7, dtype=bool),
        true_eol_cycle=7,
        raw_cycle_count=7,
        missing_count_before=0,
        missing_count_after=0,
    )
    result = adapt_target(
        model,
        trajectory.target_support(6),
        max_steps=3,
        learning_rate=0.01,
        batch_size=3,
        teacher_forcing_ratio=0.5,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(123),
        capture_steps=[3],
        meta_algorithm="boil",
    )
    snapshot = result.snapshots[3]
    for before, after in zip(model.head_parameters(), snapshot.head_parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(model.body_parameters(), snapshot.body_parameters())
    )

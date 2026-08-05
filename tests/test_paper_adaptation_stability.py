from __future__ import annotations

import logging
import math
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from paper_reproduction.adapt_and_test import (
    _support_validation_mae,
    adapt_and_evaluate_cell,
    model_from_state,
    run_adaptation_trajectory,
)
from paper_reproduction.config import ExperimentConfig
from paper_reproduction.data import CellTask, RecursivePairDataset, sample_support_batch
from paper_reproduction.losses import masked_mse
from paper_reproduction.main import _config_fingerprint, _new_run_dir
from paper_reproduction.model import GRUEncoderDecoder


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.device = "cpu"
    config.data.history_length = 6
    config.model.hidden_size = 4
    config.adaptation.batch_size = 3
    config.adaptation.complete_max_steps = 5
    config.adaptation.complete_patience = 10
    config.adaptation.minimum_validation_length = 2
    config.adaptation.maximum_validation_length = 2
    config.adaptation.gradient_clip_norm = 1.0
    config.validate()
    return config


def _task(query_offset: float = 0.0) -> CellTask:
    support = np.linspace(1.0, 0.88, 6)
    query = np.linspace(0.85, 0.65, 6) + query_offset
    return CellTask("synthetic.pkl", np.arange(1, 13), np.concatenate([support, query]))


def test_run_name_exposes_key_settings_and_fingerprints_full_config(tmp_path):
    config = _config()
    config.maml.experiment_label = "stabilized"
    path = _new_run_dir(
        config, Path(tmp_path), "adapt", "CALCE_CX2_37.pkl"
    )
    name = path.name
    assert "_adapt_stabilized_L6_CX2_37_" in name
    assert "flr0p05" in name
    assert "clr0p005" in name
    assert "al-sb" in name
    assert "sp-ls" in name
    assert "cp1" in name
    assert "sc-const" in name
    assert f"c{_config_fingerprint(config)}" in name

    changed = copy.deepcopy(config)
    changed.adaptation.complete_patience += 1
    assert _config_fingerprint(changed) != _config_fingerprint(config)


def _trajectory(
    model: GRUEncoderDecoder,
    task: CellTask,
    config: ExperimentConfig,
    *,
    query_diagnostics: bool = True,
    validation: bool = False,
):
    support, _ = task.split(config.data.history_length)
    training = support[:-2] if validation else support
    validation_soh = support[-2:] if validation else None
    return run_adaptation_trajectory(
        model,
        task,
        config,
        torch.device("cpu"),
        training_soh=training,
        validation_soh=validation_soh,
        learning_rate=0.05,
        max_steps=5,
        sampling_mode="random",
        seed_offset=321,
        patience=None,
        capture_steps=(1, 3, 5),
        query_diagnostics=query_diagnostics,
    )


def test_sample_balanced_loss_ignores_padding_and_equalizes_samples():
    prediction = torch.tensor(
        [[[1.0], [1.0], [1.0]], [[3.0], [999.0], [999.0]]]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    sample = masked_mse(prediction, target, mask, reduction="sample_balanced")
    point = masked_mse(prediction, target, mask, reduction="point_balanced")
    assert sample == pytest.approx(5.0)  # mean([1, 9])
    assert point == pytest.approx(3.0)  # mean([1, 1, 1, 9])
    changed = prediction.clone()
    changed[1, 1:] = -123456.0
    assert masked_mse(changed, target, mask, reduction="sample_balanced") == sample


def test_length_stratified_and_full_support_sampling_are_auditable():
    dataset = RecursivePairDataset(np.linspace(1.0, 0.7, 21))
    first = sample_support_batch(
        dataset,
        10,
        torch.Generator().manual_seed(7),
        torch.device("cpu"),
        mode="length_stratified",
        length_bins=5,
    )
    second = sample_support_batch(
        dataset,
        10,
        torch.Generator().manual_seed(7),
        torch.device("cpu"),
        mode="length_stratified",
        length_bins=5,
    )
    assert torch.equal(first["split_indices"], second["split_indices"])
    assert len(first["split_indices"]) == 10
    assert len(torch.unique(torch.div(first["sample_indices"], 4, rounding_mode="floor"))) == 5
    full = sample_support_batch(
        dataset,
        1,
        torch.Generator().manual_seed(1),
        torch.device("cpu"),
        mode="full_support",
    )
    assert len(full["sample_indices"]) == len(dataset)


def test_fast_and_complete_same_path_states_match_at_1_3_5():
    torch.manual_seed(11)
    model = GRUEncoderDecoder(hidden_size=4)
    config = _config()
    task = _task()
    fast = _trajectory(model, task, config)
    complete_same_settings = _trajectory(model, task, config)
    for step in (1, 3, 5):
        assert fast.captured_states[step].keys() == complete_same_settings.captured_states[step].keys()
        for name in fast.captured_states[step]:
            torch.testing.assert_close(
                fast.captured_states[step][name],
                complete_same_settings.captured_states[step][name],
                atol=1.0e-7,
                rtol=1.0e-6,
            )


def test_recursive_inference_is_deterministic_and_target_independent():
    torch.manual_seed(3)
    model = GRUEncoderDecoder(hidden_size=4).eval()
    history = torch.tensor([[[1.0], [0.95], [0.9]]])
    first = model.recursive_forecast(history, 5)
    second = model.recursive_forecast(history, 5)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    target_a = torch.zeros(1, 5, 1)
    target_b = torch.full((1, 5, 1), 999.0)
    with torch.no_grad():
        output_a = model(history, target=target_a, predicted_input_probability=0.0)
        output_b = model(history, target=target_b, predicted_input_probability=0.0)
    torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)


def test_gradients_are_finite_and_clipped():
    torch.manual_seed(17)
    model = GRUEncoderDecoder(hidden_size=4)
    config = _config()
    config.adaptation.gradient_clip_norm = 0.01
    trajectory = _trajectory(model, _task(), config, query_diagnostics=False)
    updated = trajectory.diagnostics.iloc[1:]
    assert np.isfinite(updated["gradient_norm_before_clip"]).all()
    assert np.isfinite(updated["gradient_norm_after_clip"]).all()
    assert (updated["gradient_norm_after_clip"] <= 0.010001).all()
    assert np.isfinite(updated["parameter_update_norm"]).all()
    assert (updated["parameter_update_norm"] > 0).all()


def test_deployment_selection_and_updates_do_not_depend_on_query_labels():
    torch.manual_seed(23)
    model = GRUEncoderDecoder(hidden_size=4)
    config = _config()
    original = _trajectory(model, _task(0.0), config, validation=True)
    changed_query = _trajectory(model, _task(0.2), config, validation=True)
    assert original.deployment_best_step == changed_query.deployment_best_step
    assert original.deployment_best_mae == pytest.approx(changed_query.deployment_best_mae)
    for name in original.final_state:
        torch.testing.assert_close(original.final_state[name], changed_query.final_state[name])
    # Oracle diagnostics can be disabled without changing a single update.
    diagnostics_off = _trajectory(
        model, _task(0.0), config, query_diagnostics=False, validation=True
    )
    for name in original.final_state:
        torch.testing.assert_close(original.final_state[name], diagnostics_off.final_state[name])


def test_recorded_best_validation_metric_matches_saved_state():
    torch.manual_seed(29)
    model = GRUEncoderDecoder(hidden_size=4)
    config = _config()
    task = _task()
    trajectory = _trajectory(model, task, config, validation=True)
    selected = trajectory.diagnostics.loc[
        trajectory.diagnostics["step"] == trajectory.deployment_best_step
    ].iloc[0]
    assert trajectory.deployment_best_mae == pytest.approx(
        selected["support_validation_mae_fraction"]
    )
    support, _ = task.split(config.data.history_length)
    best_model = model_from_state(model, trajectory.deployment_best_state, torch.device("cpu"))
    recomputed = _support_validation_mae(best_model, support[:-2], support[-2:])
    assert recomputed == pytest.approx(trajectory.deployment_best_mae, abs=1.0e-12)


def test_cpu_synthetic_end_to_end_writes_required_adaptation_outputs(tmp_path):
    torch.manual_seed(31)
    model = GRUEncoderDecoder(hidden_size=4)
    config = _config()
    config.adaptation.fast_steps = [0, 1, 3, 5]
    logger = logging.getLogger("paper_adaptation_stability_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    rows = adapt_and_evaluate_cell(
        model,
        _task(),
        config,
        torch.device("cpu"),
        tmp_path,
        logger,
        flat_output=True,
    )
    assert len(rows) == 6
    required_columns = {
        "step", "support_train_loss", "support_eval_loss",
        "query_mae_fraction", "query_mae_percent", "query_rmse_percent",
        "query_r2", "gradient_norm", "parameter_update_norm",
        "first_predicted_soh", "last_predicted_soh", "predicted_eol",
        "predicted_rul", "selected_split_indices",
    }
    import pandas as pd

    diagnostics = pd.read_csv(tmp_path / "adaptation/adaptation_diagnostics.csv")
    assert required_columns <= set(diagnostics.columns)
    assert diagnostics["step"].iloc[0] == 0
    assert (tmp_path / "checkpoints/complete_best_model.pt").is_file()
    assert (tmp_path / "checkpoints/complete_final_model.pt").is_file()
    assert (tmp_path / "adaptation/complete_deployment_safe_metrics.json").is_file()
    assert (tmp_path / "adaptation/complete_oracle_diagnostic_metrics.json").is_file()
    for name in (
        "adaptation_step_vs_support_loss.png",
        "adaptation_step_vs_query_mae.png",
        "adaptation_step_vs_gradient_norm.png",
        "adaptation_step_vs_update_norm.png",
        "recursive_forecast_by_step.png",
    ):
        assert (tmp_path / "plots" / name).is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_adaptation_smoke():
    device = torch.device("cuda")
    config = _config()
    model = GRUEncoderDecoder(hidden_size=4).to(device)
    task = _task()
    support, _ = task.split(config.data.history_length)
    trajectory = run_adaptation_trajectory(
        model,
        task,
        config,
        device,
        training_soh=support,
        validation_soh=None,
        learning_rate=0.01,
        max_steps=1,
        sampling_mode="random",
        query_diagnostics=False,
    )
    assert trajectory.final_step == 1
    assert math.isfinite(float(trajectory.diagnostics.iloc[1]["gradient_norm"]))

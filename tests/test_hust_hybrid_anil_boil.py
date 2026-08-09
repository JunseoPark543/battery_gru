from pathlib import Path

from hust_hybrid_anil_boil.config import load_config
from hust_hybrid_anil_boil.verification import run_checks


def test_hybrid_synthetic_verification() -> None:
    result = run_checks()
    assert result["grl_direction"] == "ok"
    assert result["inner_updated_modules"] == ["general_head", "specific_encoder"]
    assert result["inner_trajectory_steps"] == [0, 1]


def test_v3_safety_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "hust_hybrid_anil_boil" / "config_v3.yaml")
    assert config.model.residual_head_initialization_scale > 0
    assert config.train.meta_path_steps == [0, 1, 2, 5]
    assert config.loss.lambda_adaptation_regret > 0
    assert config.evaluation.deployment_step_selection == "support_loo"
    assert 0 in config.evaluation.deployment_candidate_steps

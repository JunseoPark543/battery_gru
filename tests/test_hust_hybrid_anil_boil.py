from hust_hybrid_anil_boil.verification import run_checks


def test_hybrid_synthetic_verification() -> None:
    result = run_checks()
    assert result["grl_direction"] == "ok"
    assert result["inner_updated_modules"] == ["general_head", "specific_encoder"]


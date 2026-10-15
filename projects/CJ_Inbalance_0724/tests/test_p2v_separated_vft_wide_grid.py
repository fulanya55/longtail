from cj0724.p2v_separated_vft_wide_grid import VARIANTS, estimator_kwargs


def test_estimator_kwargs_removes_input_selector() -> None:
    kwargs = estimator_kwargs(VARIANTS["R2_separated_wide"])
    assert "input" not in kwargs
    assert kwargs["doppler_min_hz"] == -30000.0
    assert kwargs["doppler_max_hz"] == 10000.0


def test_all_variants_have_explicit_input_source() -> None:
    assert {value["input"] for value in VARIANTS.values()} == {"raw", "separated"}

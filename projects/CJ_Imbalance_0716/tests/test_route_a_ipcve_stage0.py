from pathlib import Path

from experiments.audit_route_a_ipcve_stage0 import (
    CONFIRMATION_BASE_SEED,
    DEV_BASE_SEED,
    FORMAL_TRIPLES,
    SMOKE_BASE_SEED,
    SMOKE_TRIPLES,
    TARGET_PARAMETER_RANGES,
    _overlap,
    _range,
)


def test_reserved_stage0_ranges_are_disjoint_and_uint32_safe():
    smoke = _range(SMOKE_BASE_SEED, SMOKE_TRIPLES)
    development = _range(DEV_BASE_SEED, FORMAL_TRIPLES)
    confirmation = _range(CONFIRMATION_BASE_SEED, FORMAL_TRIPLES)
    assert not _overlap(smoke, development)
    assert not _overlap(development, confirmation)
    assert confirmation[1] < 2**32


def test_matlab_generator_declares_three_levels_and_no_test_generation():
    root = Path(__file__).resolve().parents[1]
    text = (
        root / "experiments/generate_route_a_ipcve_development_cf_v1.m"
    ).read_text(encoding="utf-8")
    assert "{'absent','low','high'}" in text
    assert "confirmation_generated', false" in text
    assert "test_split_generated', false" in text
    assert "target_offset_dB(column) = -12" in text
    assert "target_offset_dB(column) = 0" in text


def test_target_parameter_ranges_cover_full_generator_schema_blocks():
    assert TARGET_PARAMETER_RANGES[3] == tuple(range(27, 48))
    assert TARGET_PARAMETER_RANGES[4] == tuple(range(48, 52))
    assert TARGET_PARAMETER_RANGES[5] == tuple(range(52, 58))

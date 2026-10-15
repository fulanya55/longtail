import numpy as np

from experiments.run_route_a_frozen_ridge_visibility_conditioned_fusion_v3 import (
    _fit_ridge_visibility,
)
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES


def test_ridge_visibility_uses_only_finite_positive_fit_rows():
    rng = np.random.default_rng(7)
    fit_x = rng.normal(size=(8, 5)).astype(np.float32)
    early_x = rng.normal(size=(6, 5)).astype(np.float32)
    standardized = {"fit": fit_x, "early": early_x}
    roles = {
        "fit": {"sample_ids": np.arange(8)},
        "early": {"sample_ids": np.arange(8, 14)},
    }
    labels = np.zeros((14, 6), dtype=np.uint8)
    gap = np.full((14, 6), np.nan, dtype=np.float32)
    for index in TARGET_INDICES:
        labels[[0, 1, 2, 3, 8, 9, 10, 11], index] = 1
        gap[[0, 1, 2, 3], index] = np.asarray([-20, -12, -5, 0], dtype=np.float32)
        gap[[8, 9, 10, 11], index] = np.asarray([-20, -12, -5, 0], dtype=np.float32)
    predictions, states, diagnostics = _fit_ridge_visibility(
        standardized, roles, labels, gap, alpha=10.0,
        gap_clip_db=40.0, weak_threshold_db=10.0,
    )
    assert predictions["fit"].shape == (8, 3)
    assert predictions["early"].shape == (6, 3)
    assert all(state["fit_count"] == 4 for state in states.values())
    assert set(diagnostics) == {"MFT", "VFT", "ISRJ", "mean_across_labels"}


def test_ridge_predictions_do_not_require_early_gap_for_inference():
    rng = np.random.default_rng(11)
    standardized = {
        "fit": rng.normal(size=(8, 4)).astype(np.float32),
        "early": rng.normal(size=(8, 4)).astype(np.float32),
    }
    roles = {
        "fit": {"sample_ids": np.arange(8)},
        "early": {"sample_ids": np.arange(8, 16)},
    }
    labels = np.ones((16, 6), dtype=np.uint8)
    gap = np.zeros((16, 6), dtype=np.float32)
    for index in TARGET_INDICES:
        gap[:8, index] = np.linspace(-20, 0, 8)
        gap[8:, index] = np.linspace(-20, 0, 8)
    first, _, _ = _fit_ridge_visibility(
        standardized, roles, labels, gap, 10.0, 40.0, 10.0
    )
    changed = gap.copy()
    changed[8:, :] = changed[8:, :] + 3.0
    second, _, _ = _fit_ridge_visibility(
        standardized, roles, labels, changed, 10.0, 40.0, 10.0
    )
    np.testing.assert_allclose(first["early"], second["early"], rtol=0, atol=0)

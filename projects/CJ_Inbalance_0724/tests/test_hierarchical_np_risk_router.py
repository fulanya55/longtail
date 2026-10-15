import numpy as np

from cj0724.hierarchical_np_risk_router import (
    build_risk_benefit_targets,
    fit_hierarchical_models,
    predict_hierarchical_scores,
    selective_blend,
)


def test_risk_and_benefit_targets_have_expected_directions():
    labels = np.zeros((4, 6), dtype=np.uint8)
    labels[0, 3] = 1
    main = np.full((4, 6), 0.1, dtype=np.float32)
    expert = np.full((4, 3), 0.1, dtype=np.float32)
    main[0, 3], expert[0, 0] = 0.2, 0.9
    main[1, 3], expert[1, 0] = 0.2, 0.8
    main[2, 3], expert[2, 0] = 0.8, 0.1
    risk, benefit, weight, audit = build_risk_benefit_targets(main, expert, labels, 0, 0.5)
    assert benefit[0] == 1
    assert risk[1] == 1 and benefit[1] == 0
    assert benefit[2] == 1 and risk[2] == 0
    assert audit["false_positive_risk"] == 1
    assert weight[1] >= 2.0


def test_selective_blend_uses_independent_target_budgets():
    main = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
    expert = np.asarray([[0.9, 0.1, 0.8]], dtype=np.float32)
    score = np.asarray([[2.0, 0.0, 3.0]], dtype=np.float32)
    selections = [
        {"alpha": 0.5, "score_cutoff": 1.0},
        {"alpha": 0.75, "score_cutoff": 1.0},
        {"alpha": 0.25, "score_cutoff": 2.0},
    ]
    final, accept = selective_blend(main, expert, score, selections)
    np.testing.assert_array_equal(final[:, :3], main[:, :3])
    np.testing.assert_array_equal(accept, [[True, False, True]])
    np.testing.assert_allclose(final[:, 3:], [[0.65, 0.5, 0.65]])


def test_hierarchical_models_produce_finite_scores():
    rng = np.random.default_rng(19)
    labels = rng.integers(0, 2, size=(160, 6), dtype=np.uint8)
    main = rng.uniform(0.05, 0.95, size=(160, 6)).astype(np.float32)
    expert = rng.uniform(0.05, 0.95, size=(160, 3)).astype(np.float32)
    thresholds = np.full(6, 0.5, dtype=np.float32)
    models, audits = fit_hierarchical_models(main, expert, labels, thresholds, 2041)
    risk, benefit, score = predict_hierarchical_scores(models, main, expert, thresholds)
    assert risk.shape == benefit.shape == score.shape == (160, 3)
    assert np.isfinite(score).all()
    assert np.all((risk >= 0) & (risk <= 1))
    assert np.all((benefit >= 0) & (benefit <= 1))
    assert len(audits) == 3

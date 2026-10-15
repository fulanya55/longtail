import numpy as np

from experiments.run_route_a_vft_strict_complementary_score_fusion_v3 import (
    _fit_score_stats,
    _fixed_permutation,
    _fused_scores,
    _standardize_scores,
)


def test_fit_only_score_standardization_is_zero_mean_unit_scale():
    scores = np.asarray([-2.0, 0.0, 2.0, 4.0], dtype=np.float32)
    standardized = _standardize_scores(scores, _fit_score_stats(scores))
    assert abs(float(standardized.mean())) < 1e-7
    assert abs(float(standardized.std()) - 1.0) < 1e-7


def test_fixed_permutation_is_reproducible_and_hashed():
    first, first_sha = _fixed_permutation(20, 1234)
    second, second_sha = _fixed_permutation(20, 1234)
    different, different_sha = _fixed_permutation(20, 1235)
    np.testing.assert_array_equal(first, second)
    assert first_sha == second_sha
    assert first_sha != different_sha
    assert sorted(different.tolist()) == list(range(20))


def test_fusion_is_fixed_equal_weight_without_fit():
    final = np.asarray([1.0, -1.0], dtype=np.float32)
    stft = np.asarray([3.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(_fused_scores(final, stft), [2.0, 0.0])


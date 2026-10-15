import numpy as np

from cj0724.integrate_swept_experts_with_six_class import (
    TARGET_INDICES,
    TARGET_NAMES,
    _identity_split,
    bounded_blend,
    fit_routers,
    route_probabilities,
)


def test_legacy_swept_slice_maps_to_mft_vft_isrj():
    assert TARGET_NAMES == ("MFT", "VFT", "ISRJ")
    np.testing.assert_array_equal(TARGET_INDICES, [3, 4, 5])


def test_bounded_blend_preserves_non_targets_and_convex_bounds():
    main = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
    expert = np.asarray([[0.9, 0.1, 0.8]], dtype=np.float32)
    route = np.asarray([[0.9, 0.4, 0.7]], dtype=np.float32)
    final = bounded_blend(main, expert, route, alpha=0.5, route_threshold=0.6)
    np.testing.assert_array_equal(final[:, :3], main[:, :3])
    np.testing.assert_allclose(final[:, 3:], [[0.65, 0.5, 0.7]])
    assert np.all(final[:, 3:] >= np.minimum(main[:, 3:], expert))
    assert np.all(final[:, 3:] <= np.maximum(main[:, 3:], expert))


def test_identity_split_is_disjoint_and_complete():
    indices = np.arange(20, dtype=np.int64)
    identities = np.arange(100, 120, dtype=np.int64)
    fit, safety = _identity_split(indices, identities, 17)
    assert len(fit) + len(safety) == len(indices)
    assert np.intersect1d(identities[fit], identities[safety]).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate((fit, safety))), indices)


def test_router_training_and_probabilities_are_finite():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, size=(80, 6), dtype=np.uint8)
    main = rng.uniform(0.05, 0.95, size=(80, 6)).astype(np.float32)
    expert = main[:, 3:6].copy()
    expert[:40] = labels[:40, 3:6] * 0.8 + 0.1
    routers, prevalence = fit_routers(main, expert, labels, 2041)
    probability = route_probabilities(routers, main, expert)
    assert probability.shape == (80, 3)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert len(prevalence) == 3

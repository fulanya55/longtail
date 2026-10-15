import numpy as np

from experiments.run_route_a_stage2_reliability_identifiability import (
    _domain_class_weights,
    _search_reliability_threshold,
    _temporal_descriptors,
)


def test_temporal_descriptors_are_fixed_shape_and_finite():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(5, 64, 375)).astype(np.float32)
    evidence = rng.normal(size=(5, 375)).astype(np.float32)
    descriptors = _temporal_descriptors(features, evidence)
    assert descriptors.shape == (5, 360)
    assert np.isfinite(descriptors).all()


def test_domain_class_weights_balance_each_domain_and_class():
    domains = np.asarray(["root", "root", "root", "40", "40", "40", "40"])
    targets = np.asarray([0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
    weights = _domain_class_weights(domains, targets)
    totals = {
        (domain, target): weights[(domains == domain) & (targets == target)].sum()
        for domain in np.unique(domains)
        for target in np.unique(targets[domains == domain])
    }
    assert np.allclose(list(totals.values()), list(totals.values())[0])


def test_reliability_threshold_search_respects_coverage_constraints():
    groups = {
        "root": {
            "targets": np.asarray([1, 1, 0, 0], dtype=np.uint8),
            "frozen_predictions": np.asarray([1, 0, 0, 1], dtype=np.uint8),
        },
        "40": {
            "targets": np.asarray([1, 1, 0, 0], dtype=np.uint8),
            "frozen_predictions": np.asarray([1, 0, 0, 1], dtype=np.uint8),
        },
    }
    scores = {
        "root": np.asarray([0.1, 0.9, 0.2, 0.8]),
        "40": np.asarray([0.1, 0.9, 0.2, 0.8]),
    }
    selected, rows = _search_reliability_threshold(groups, scores)
    assert rows
    assert selected is not None
    assert selected["domains"]["root"]["coverage"] >= 0.70
    assert selected["domains"]["40"]["coverage"] >= 0.40

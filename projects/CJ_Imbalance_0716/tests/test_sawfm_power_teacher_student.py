import itertools

import numpy as np

from experiments.run_route_a_sawfm_stage2_power_teacher_student import (
    _development_pairs,
    _stratified_triplets,
)


def _synthetic_visibility_roles():
    labels = []
    for combo in itertools.product((0, 1), repeat=5):
        if not any(combo[index] for index in (2, 3, 4)):
            continue
        label_without_saw = [combo[0], 0, combo[1], combo[2], combo[3], combo[4]]
        for _ in range(2):
            positive = label_without_saw.copy()
            positive[1] = 1
            labels.append(positive)
            labels.append(label_without_saw.copy())
    values = np.asarray(labels, dtype=np.uint8)
    indices = np.arange(len(values), dtype=np.int64)
    identities = 10000 + indices
    return indices, values, identities


def test_stratified_triplets_preserve_28_other_label_combinations():
    indices, labels, identities = _synthetic_visibility_roles()

    selected = _stratified_triplets(
        indices, labels, identities, pairs_per_combo=1, seed=17
    )

    assert len(selected["positive_identities"]) == 28
    assert len(selected["hard_negative_identities"]) == 28
    assert len(selected["combo_counts"]) == 28
    for positive, negative in zip(
        selected["positive_base_indices"], selected["hard_negative_base_indices"]
    ):
        assert labels[positive, 1] == 1
        assert labels[negative, 1] == 0
        assert np.array_equal(labels[positive, [0, 2, 3, 4, 5]], labels[negative, [0, 2, 3, 4, 5]])


def test_development_pairs_are_combo_matched_and_balanced():
    indices, labels, identities = _synthetic_visibility_roles()

    selected = _development_pairs(indices, labels, identities, seed=19)

    assert len(selected["positive_identities"]) == len(
        selected["hard_negative_identities"]
    )
    assert len(selected["positive_identities"]) == 56

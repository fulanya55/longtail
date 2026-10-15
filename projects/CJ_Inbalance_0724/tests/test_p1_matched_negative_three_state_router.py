import numpy as np

from cj0724.p1_matched_negative_three_state_router import (
    ABSTAIN,
    ACCEPT_EXPERT,
    KEEP_MAIN,
    matched_group_mask,
    split_three_ways,
    three_state_targets,
)


def test_matched_groups_require_same_non_target_labels_and_both_target_states():
    labels = np.asarray([
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
    ], dtype=np.uint8)
    powers = np.asarray([10, 10, 10, 10, 10])
    mask = matched_group_mask(labels, powers, target_index=4)
    assert mask.tolist() == [True, True, True, True, False]


def test_three_state_targets_use_loss_margin():
    labels = np.asarray([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0]], dtype=np.uint8)
    main = np.full((3, 6), 0.5, dtype=np.float32)
    expert = np.full((3, 3), 0.5, dtype=np.float32)
    main[:, 4] = [0.8, 0.2, 0.4]
    expert[:, 1] = [0.1, 0.9, 0.41]
    target, _ = three_state_targets(labels, main, expert, target_index=4, local_index=1, margin=0.05)
    assert target.tolist() == [ACCEPT_EXPERT, ACCEPT_EXPERT, ABSTAIN]
    main[0, 4], expert[0, 1] = 0.1, 0.8
    target, _ = three_state_targets(labels, main, expert, target_index=4, local_index=1, margin=0.05)
    assert target[0] == KEEP_MAIN


def test_three_way_identity_split_is_disjoint_and_complete():
    indices = np.arange(30, dtype=np.int64)
    identities = np.arange(30, dtype=np.int64)
    first, second, third = split_three_ways(indices, identities, 7)
    assert len(np.intersect1d(first, second)) == 0
    assert len(np.intersect1d(first, third)) == 0
    assert len(np.intersect1d(second, third)) == 0
    assert np.array_equal(np.sort(np.concatenate((first, second, third))), indices)

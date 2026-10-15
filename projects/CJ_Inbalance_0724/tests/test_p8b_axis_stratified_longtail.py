import json

import numpy as np

from cj0724.p8b_axis_stratified_longtail import AxisStratifiedBatchSampler, ReplacementBatchSampler, _json_safe


def _labels():
    rows = []
    rows += [[1, 0, 0, 0, 1, 0]] * 32
    rows += [[0, 0, 0, 0, 1, 0]] * 16
    rows += [[0, 0, 1, 1, 0, 1]] * 64
    rows += [[1, 1, 0, 0, 0, 0]] * 96
    return np.asarray(rows, dtype=np.uint8)


def test_axis_sampler_has_pre_registered_quotas_and_is_deterministic():
    labels = _labels(); levels = np.full(len(labels), 30, dtype=np.int16); levels[:32] = 10
    sampler = AxisStratifiedBatchSampler(labels, levels, 64, 3, 71)
    first = list(iter(sampler)); sampler.set_epoch(0); second = list(iter(sampler))
    assert first == second and all(len(batch) == 64 for batch in first)
    for batch in first:
        current = labels[np.asarray(batch)]; current_levels = levels[np.asarray(batch)]
        weak = np.logical_and(current[:, 4] == 1, np.logical_and(current[:, :3].any(axis=1), current_levels < 30))
        other = np.logical_and(current[:, 4] == 1, ~weak)
        hard = np.logical_and(current[:, 4] == 0, np.logical_and.reduce((current[:, 2] == 1, current[:, 3] == 1, current[:, 5] == 1)))
        assert weak.sum() == 16 and other.sum() == 4 and hard.sum() == 16


def test_random_sampler_matches_fixed_update_budget():
    sampler = ReplacementBatchSampler(2400, 64, 38, 19)
    assert len(sampler) == 38
    assert all(len(batch) == 64 for batch in sampler)


def test_result_receipt_serializes_numpy_manifest_values():
    value = {"indices": np.asarray([3, 7], dtype=np.int64), "metric": np.float32(0.25)}
    assert json.loads(json.dumps(_json_safe(value))) == {"indices": [3, 7], "metric": 0.25}

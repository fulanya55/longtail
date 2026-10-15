import numpy as np

from cj0724.p8a_multi_axis_longtail_atlas import build_manifest, validate_manifest


def _labels() -> np.ndarray:
    combinations = np.asarray([[int(bit) for bit in format(value, "06b")] for value in range(1, 64)], dtype=np.uint8)
    return np.repeat(combinations, 400, axis=0)


def test_combo_manifest_keeps_exact_marginals_and_all_combinations():
    labels = _labels()
    manifest = build_manifest(labels, np.arange(len(labels)), "combo_lt", [1200] * 6, "combo_lt", 7, False)
    checks = validate_manifest(manifest, labels)
    assert all(checks.values())
    even = [count for combo, count in manifest["combination_counts"].items() if sum(map(int, combo)) % 2 == 0]
    odd = [count for combo, count in manifest["combination_counts"].items() if sum(map(int, combo)) % 2 == 1]
    assert np.mean(even) < np.mean(odd)


def test_mixed_manifest_assigns_weak_power_only_to_target_fixed_rows():
    labels = _labels()
    manifest = build_manifest(labels, np.arange(len(labels)), "mixed_lt", [1200, 1200, 1200, 1200, 400, 1200], "combo_lt", 11, True)
    checks = validate_manifest(manifest, labels)
    assert all(checks.values())
    selected = np.asarray(manifest["selected_indices"], dtype=np.int64)
    levels = np.asarray(manifest["view_level_by_selected_position_db"], dtype=np.int16)
    expected = np.logical_and(labels[selected, 4] == 1, labels[selected, :3].any(axis=1))
    assert np.array_equal(levels < 30, expected)

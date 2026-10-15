import numpy as np

from experiments.run_route_a_vft_capacity_matched_mlp_unified_decision_ablation import _Linear6, _MLP1, _count_parameters, _fixed6, _metric


def test_capacity_is_exactly_matched_at_seven_parameters():
    assert _count_parameters(_Linear6()) == 7
    assert _count_parameters(_MLP1()) == 7


def test_fixed_linear_statistics_are_deterministic():
    base = np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    assert np.array_equal(_fixed6(base), np.array([[2.0, 3.0, 4.0, 5.0, 4.0, 9.0]], dtype=np.float32))


def test_metric_reads_a_variant_record_directly():
    record = {"metrics": {"per_label": {"VFT": {"weak_vs_negative_auroc": 0.6}}}}
    assert _metric(record, "weak_vs_negative_auroc") == 0.6

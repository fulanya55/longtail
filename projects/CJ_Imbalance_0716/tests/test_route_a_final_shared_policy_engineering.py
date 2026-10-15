from experiments.benchmark_route_a_final_shared_policy_engineering import _latency


def test_latency_summary_uses_sample_count():
    summary = _latency([10.0, 20.0, 30.0], 10)
    assert summary["median_ms"] == 20.0
    assert summary["median_ms_per_sample"] == 2.0
    assert summary["median_samples_per_second"] == 500.0

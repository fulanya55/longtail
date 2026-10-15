from experiments.benchmark_route_a_u2_engineering import _summary


def test_latency_summary_reports_per_sample_and_throughput():
    result = _summary([10.0, 20.0, 30.0], 100)
    assert result["median_ms"] == 20.0
    assert result["median_ms_per_sample"] == 0.2
    assert result["median_samples_per_second"] == 5000.0

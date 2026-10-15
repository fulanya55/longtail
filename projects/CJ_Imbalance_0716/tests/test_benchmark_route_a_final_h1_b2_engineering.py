from experiments.benchmark_route_a_final_h1_b2_engineering import _summary


def test_latency_summary_reports_per_sample_and_throughput():
    row = _summary([100.0, 120.0, 110.0], 10)
    assert row["median_ms"] == 110.0
    assert row["median_ms_per_sample"] == 11.0
    assert row["median_samples_per_second"] > 90.0


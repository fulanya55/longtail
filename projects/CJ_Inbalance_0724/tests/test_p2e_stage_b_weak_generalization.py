import numpy as np

from cj0724.p2e_stage_b_weak_generalization import condition_metrics, grouped_bootstrap, target_metric


def make_record():
    labels = np.zeros((4, 6), dtype=bool)
    labels[:, 1] = [1, 1, 0, 0]
    labels[:, 0] = [0, 1, 1, 0]
    probability = np.zeros((4, 6), dtype=np.float32)
    probability[:, 1] = [0.9, 0.4, 0.8, 0.1]
    target = np.zeros((4, 6, 2), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[:, 1, :] = [[0.1, 0.2], [0.2, 0.2], [0.0, 0.0], [0.0, 0.0]]
    mask = np.zeros_like(target, dtype=bool)
    mask[:2, 1, :] = True
    return {
        "probability": probability,
        "thresholds": np.full(6, 0.5, dtype=np.float32),
        "prediction": prediction,
        "labels": labels,
        "target": target,
        "mask": mask,
        "condition_id": np.zeros(4, dtype=np.int16),
        "scale": np.ones((6, 2), dtype=np.float32),
    }


def test_target_metric_uses_predicted_label_gate():
    record = make_record()
    assert target_metric(record, 1, (0,), "recall") == 0.5
    assert target_metric(record, 1, (0,), "fpr") == 0.5
    assert target_metric(record, 1, (0,), "coverage") == 0.5


def test_condition_metrics_reports_sizes_and_worst_combination():
    metrics = condition_metrics(make_record(), 0, 1)
    assert set(metrics["by_combination_size"]) == {"1", "2"}
    assert len(metrics["by_combination"]) == 2
    assert np.isfinite(metrics["worst_combination_predicted_gate_nmae"])


def test_grouped_bootstrap_preserves_paired_identity_multiplicity():
    base = make_record()
    candidate = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in base.items()}
    baseline = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in base.items()}
    candidate["probability"][:, 1] = [0.9, 0.8, 0.4, 0.1]
    identities = np.array([10, 11, 12, 13])
    rows = [{"identity": identities, "records": {"candidate": candidate, "baseline": baseline}}]
    actual = grouped_bootstrap(rows, 1, (0,), "recall", "candidate", "baseline", 25, 123)

    rng = np.random.default_rng(123)
    reference = []
    for _ in range(25):
        sampled = rng.integers(0, len(identities), size=len(identities))
        subset = {}
        for name, record in rows[0]["records"].items():
            subset[name] = {
                key: value[sampled] if isinstance(value, np.ndarray) and value.shape[:1] == (len(identities),) else value
                for key, value in record.items()
            }
        reference.append(
            target_metric(subset["candidate"], 1, (0,), "recall")
            - target_metric(subset["baseline"], 1, (0,), "recall")
        )
    reference = np.asarray(reference)
    assert actual["ci95_low"] == np.nanquantile(reference, 0.025)
    assert actual["ci95_high"] == np.nanquantile(reference, 0.975)

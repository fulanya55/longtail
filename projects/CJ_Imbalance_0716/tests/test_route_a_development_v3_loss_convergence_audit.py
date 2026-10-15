import json

from experiments.run_route_a_development_v3_loss_convergence_audit import (
    LONG_VARIANTS,
    ORIGINAL_VARIANTS,
    _load_existing_loss,
)


def test_extended_and_original_variants_are_one_to_one():
    assert len(LONG_VARIANTS) == len(ORIGINAL_VARIANTS) == 3
    assert list(LONG_VARIANTS.values()) == list(ORIGINAL_VARIANTS.values())


def test_load_existing_loss_preserves_metrics_and_records_hash(tmp_path):
    source = tmp_path / "seed_2039" / "L4_DB_RawSaw"
    source.mkdir(parents=True)
    (source / "metrics.json").write_text(json.dumps({"macro_aupr": 0.5}), encoding="utf-8")
    (source / "calibration.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (source / "thresholds.json").write_text(json.dumps({"b": 2}), encoding="utf-8")
    (source / "predictions.npz").write_bytes(b"prediction-receipt")
    result = _load_existing_loss(tmp_path, 2039, "L4_DB_RawSaw", "O4_DB_12")
    assert result["variant"] == "O4_DB_12"
    assert result["metrics"]["macro_aupr"] == 0.5
    assert len(result["predictions_sha256"]) == 64

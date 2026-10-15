import numpy as np

from experiments.run_route_a_dfms_iq_feature_sufficiency import (
    _bounded_logits,
    _stage2_summary,
    _temporal_summary,
)


def test_temporal_and_stage2_summaries_have_registered_dimensions():
    rng = np.random.default_rng(7)
    temporal = rng.normal(size=(3, 128, 19)).astype(np.float16)
    real = rng.normal(size=(3, 32, 23)).astype(np.float16)
    imag = rng.normal(size=(3, 32, 23)).astype(np.float16)

    assert _temporal_summary(temporal).shape == (3, 512)
    assert _stage2_summary(real, imag).shape == (3, 768)


def test_bounded_residual_obeys_hard_limit_and_confidence_gate():
    base = np.asarray([[0.0, 8.0], [-8.0, 0.0]], dtype=np.float32)
    raw = np.asarray([[100.0, 100.0], [-100.0, -100.0]], dtype=np.float32)
    center = np.zeros(2, dtype=np.float32)
    scale = np.ones(2, dtype=np.float32)

    logits, residual = _bounded_logits(base, raw, center, scale, 0.20)

    assert logits.shape == base.shape
    assert np.max(np.abs(residual)) <= 0.20 + 1e-7
    assert abs(residual[0, 1]) < 1e-3
    assert abs(residual[1, 0]) < 1e-3

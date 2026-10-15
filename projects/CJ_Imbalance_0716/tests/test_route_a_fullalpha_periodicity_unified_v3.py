import numpy as np

from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _replace_isrj


def test_isrj_replacement_preserves_other_five_logits():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(20, 6)).astype(np.float32)
    score = rng.normal(size=20).astype(np.float32)
    changed = _replace_isrj(logits, score)
    assert np.array_equal(changed[:, :5], logits[:, :5])
    assert np.array_equal(changed[:, 5], score)


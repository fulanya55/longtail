import numpy as np

from experiments.run_route_a_isrj_periodicity_oof_hard_negative_expert import (
    _oof_hard_negatives,
)


def test_oof_hard_negative_selector_returns_negative_top_quartile():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(120, 8)).astype(np.float32)
    y = np.r_[np.zeros(80, dtype=np.uint8), np.ones(40, dtype=np.uint8)]
    score, hard, cutoff = _oof_hard_negatives(x, y, 19, folds=4, quantile=0.75)
    assert np.isfinite(score).all()
    assert hard.sum() == 20
    assert np.all(y[hard] == 0)
    assert np.all(score[hard] >= cutoff)


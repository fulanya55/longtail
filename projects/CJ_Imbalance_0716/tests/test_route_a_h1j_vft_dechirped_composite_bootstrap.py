import numpy as np

from experiments.analyze_route_a_h1j_vft_dechirped_composite_bootstrap import bootstrap_paired_delta


def test_paired_bootstrap_reports_positive_rank_delta_for_separated_candidate():
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    candidate = np.asarray([.1, .2, .3, .7, .8, .9])
    control = np.asarray([.4, .9, .1, .8, .2, .6])
    result = bootstrap_paired_delta(y, candidate, control, draws=100, seed=7)
    assert result['auroc']['point'] > 0
    assert result['aupr']['point'] > 0
    assert result['auroc']['draws'] == 100

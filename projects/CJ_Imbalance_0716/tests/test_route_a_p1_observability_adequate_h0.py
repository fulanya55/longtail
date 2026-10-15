from __future__ import annotations

import numpy as np

from experiments.generate_route_a_p1_observability_adequate_pairs import K, _mft, _reference, _scale, _vft
from experiments.run_route_a_p1_observability_adequate_h0 import _mft_score, _vft_scores


def test_p1_mft_known_chirp_score_exceeds_wrong_chirp_on_noiseless_component() -> None:
    signal = _scale(_mft(17), 8.0)
    assert _mft_score(signal, K) > _mft_score(signal, .83 * K)


def test_p1_vft_true_timestamp_score_exceeds_permuted_control_on_noiseless_component() -> None:
    true, permuted = _vft_scores(_scale(_vft(19), 8.0))
    assert np.isfinite(true) and np.isfinite(permuted)
    assert true > permuted

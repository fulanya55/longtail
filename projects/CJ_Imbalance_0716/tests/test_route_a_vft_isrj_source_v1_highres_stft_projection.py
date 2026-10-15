import numpy as np

from experiments.run_route_a_vft_isrj_source_v1_highres_stft_projection import ARMS, NATIVE_DIM, _projector


def test_high_resolution_projection_is_fixed_and_capacity_matched():
    assert ARMS == ("Native16", "High32Projected", "High32PhaseScrambledProjected")
    first = _projector(32 * 32 + 12)
    second = _projector(32 * 32 + 12)
    assert first.shape == (1036, NATIVE_DIM)
    assert np.array_equal(first, second)

import numpy as np

from experiments.run_route_a_vft_isrj_source_v1_stft_raw_complementarity import ARMS, _compose


def test_arms_and_capacity_control_are_fixed():
    assert ARMS == ("RidgeSTFT", "RidgeSTFTPlusRawSource", "RidgeSTFTPlusPhaseScrambled")
    features = {role: {"RidgeSTFT": np.ones((3, 256), dtype=np.float32) * (1 + index),
                       "RawSource": np.ones((3, 32), dtype=np.float32) * (2 + index),
                       "PhaseScrambledControl": np.ones((3, 32), dtype=np.float32) * (3 + index)}
                for index, role in enumerate(("fit", "validation", "early"))}
    ridge, _ = _compose(features, "RidgeSTFT")
    raw, _ = _compose(features, "RidgeSTFTPlusRawSource")
    control, _ = _compose(features, "RidgeSTFTPlusPhaseScrambled")
    assert ridge["fit"].shape == (3, 256)
    assert raw["fit"].shape == control["fit"].shape == (3, 288)

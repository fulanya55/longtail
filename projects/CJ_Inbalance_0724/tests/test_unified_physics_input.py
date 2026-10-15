import numpy as np

from cj0724.b1_physics_sufficiency import physics_features_from_normalized_iq


def test_pre_normalized_physics_features_shapes():
    rng = np.random.default_rng(9)
    iq = rng.normal(size=(2, 2, 3000)).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(iq).sum(axis=1), axis=1, keepdims=True))
    output = physics_features_from_normalized_iq(iq / rms[:, None])
    assert output["mft_delay"].shape == (2, 190)
    assert output["isrj_periodicity"].shape == (2, 310)
    assert output["combined"].shape == (2, 320)

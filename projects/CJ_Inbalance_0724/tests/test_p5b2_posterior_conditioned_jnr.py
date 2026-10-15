import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import FeatureRole
from cj0724.p5b2_posterior_conditioned_jnr import TARGET_NAMES, with_context


def test_p5b2_posterior_context_replaces_but_does_not_append_true_composition():
    role = FeatureRole(
        feature=np.zeros((2, 26), dtype=np.float32), composition=np.ones((2, 6), dtype=np.float32),
        target=np.zeros((2, 3), dtype=np.float32), valid=np.ones((2, 3), dtype=bool),
        identities=np.array([1, 2]), power=np.array([10, 10]), combination=np.array([3, 3]),
    )
    posterior = torch.sigmoid(torch.randn(2, 6)).numpy()
    replaced = with_context(role, posterior)
    assert replaced.composition.shape == (2, 6)
    assert np.allclose(replaced.composition, posterior)
    assert not np.allclose(replaced.composition, role.composition)
    assert len(TARGET_NAMES) == 3

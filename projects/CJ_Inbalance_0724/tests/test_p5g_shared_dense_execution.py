import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import FeatureProbe, FeatureRole
from cj0724.p5b4_bounded_jnr_confirmation import estimate
from cj0724.p5g_shared_dense_execution import _finite_values_exact
from cj0724.p5f_shared_lazy_execution import attach_target_values


def test_dense_head_repeat_and_attachment_are_exact_including_inactive_nans():
    rng = np.random.default_rng(31)
    role = FeatureRole(
        feature=rng.normal(size=(5, 26)).astype(np.float32),
        composition=rng.normal(size=(5, 6)).astype(np.float32),
        target=np.zeros((5, 3), dtype=np.float32), valid=np.ones((5, 3), dtype=bool),
        identities=np.arange(5), power=np.full(5, 30), combination=np.ones(5),
    )
    torch.manual_seed(32)
    head = FeatureProbe(32).eval()
    state = {"feature_mean": np.zeros(32, np.float32), "feature_scale": np.ones(32, np.float32), "target_mean": np.zeros(3, np.float32), "target_scale": np.ones(3, np.float32)}
    first = estimate(role, head, state, torch.device("cpu"))
    second = estimate(role, head, state, torch.device("cpu"))
    assert _finite_values_exact(first, second)

    decision = np.asarray([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1]], dtype=bool)
    base = np.zeros((5, 6, 2), dtype=np.float32); base[0, 0, 0] = np.nan
    candidate_one, allowed_one = attach_target_values(base, first, decision)
    candidate_two, allowed_two = attach_target_values(base, second, decision)
    assert _finite_values_exact(candidate_one, candidate_two)
    np.testing.assert_array_equal(allowed_one, allowed_two)
    assert _finite_values_exact(candidate_one[~allowed_one], candidate_two[~allowed_two])

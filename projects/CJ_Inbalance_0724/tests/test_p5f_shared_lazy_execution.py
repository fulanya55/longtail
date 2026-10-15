import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import FeatureProbe, FeatureRole
from cj0724.p5b4_bounded_jnr_confirmation import estimate
from cj0724.p5f_shared_lazy_execution import attach_target_values, lazy_estimate, union_positive_rows


def test_lazy_head_matches_dense_values_on_selected_union_rows():
    rng = np.random.default_rng(7)
    role = FeatureRole(
        feature=rng.normal(size=(4, 26)).astype(np.float32),
        composition=rng.normal(size=(4, 6)).astype(np.float32),
        target=np.zeros((4, 3), dtype=np.float32),
        valid=np.ones((4, 3), dtype=bool),
        identities=np.arange(4), power=np.full(4, 30), combination=np.ones(4),
    )
    torch.manual_seed(8)
    head = FeatureProbe(32).eval()
    state = {
        "feature_mean": np.zeros(32, dtype=np.float32),
        "feature_scale": np.ones(32, dtype=np.float32),
        "target_mean": np.zeros(3, dtype=np.float32),
        "target_scale": np.ones(3, dtype=np.float32),
    }
    decision = np.asarray([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]], dtype=bool)
    dense = estimate(role, head, state, torch.device("cpu"))
    active = union_positive_rows(decision)
    lazy = lazy_estimate(role, head, state, torch.device("cpu"), active)

    np.testing.assert_allclose(lazy[active], dense[active], rtol=0.0, atol=1.0e-6)
    assert np.isnan(lazy[~active]).all()

    base = np.zeros((4, 6, 2), dtype=np.float32)
    dense_candidate, dense_allowed = attach_target_values(base, dense, decision)
    lazy_candidate, lazy_allowed = attach_target_values(base, lazy, decision)
    np.testing.assert_allclose(lazy_candidate, dense_candidate, rtol=0.0, atol=1.0e-6)
    np.testing.assert_array_equal(lazy_candidate[~lazy_allowed], dense_candidate[~dense_allowed])
    np.testing.assert_array_equal(lazy_allowed, dense_allowed)

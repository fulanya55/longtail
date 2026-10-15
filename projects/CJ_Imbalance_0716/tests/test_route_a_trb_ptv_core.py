import numpy as np
import pytest

from experiments.route_a_trb_ptv_core import activation_gate, solve_target_risk_projection, weak_positive_bce


def _fit_arrays():
    targets = np.zeros((12, 5), dtype=np.uint8)
    weak = np.zeros_like(targets)
    targets[:, 2:5] = 1
    weak[:, 2:5] = 1
    return targets, weak


def test_weak_positive_bce_rejects_missing_target_weak_samples():
    logits = np.zeros((4, 5), dtype=np.float64)
    targets = np.zeros((4, 5), dtype=np.uint8)
    weak = np.zeros_like(targets)
    with pytest.raises(ValueError, match="no positive weak"):
        weak_positive_bce(logits, targets, weak)


def test_solver_projects_full_point_to_target_risk_boundary():
    targets, weak = _fit_arrays()

    def logits(q):
        result = np.full((12, 5), 3.0, dtype=np.float64)
        # Full q=(1.5,1.5) increases weak positive loss; classifier movement
        # offsets shared movement, so the constrained closest point is nonzero.
        result[:, 2:5] = 2.0 - 0.40 * q[0] + 0.20 * q[1]
        return result

    q_star, diagnostics = solve_target_risk_projection(logits, targets, weak)
    assert diagnostics["solver_success"]
    assert diagnostics["max_constraint_violation"] <= 1e-7
    assert 0.0 <= q_star[0] <= q_star[1] <= 1.5
    assert np.linalg.norm(q_star - np.asarray([1.5, 1.5])) > 0.05
    assert max(diagnostics["risk_delta_vs_c6"]) <= 1e-7


def test_activation_gate_is_fixed_and_counts_seeds():
    gate = activation_gate([
        np.asarray([1.5, 1.5]), np.asarray([1.45, 1.5]), np.asarray([1.4, 1.5]),
        np.asarray([1.5, 1.4]), np.asarray([1.5, 1.5]),
    ])
    assert gate["activated_seed_count"] == 3
    assert gate["passed"] is False

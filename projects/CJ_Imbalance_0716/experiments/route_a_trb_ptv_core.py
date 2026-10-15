"""Numerical core for the preregistered Target-Risk-Balanced PTV (TRB-PTV).

The module contains no model, dataset, split, threshold, or test-set access.
It accepts a deterministic fit-only logit callback and solves exactly the
two-dimensional constrained problem frozen in the TRB preregistration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize


TARGET_NAMES = ("MFT", "VFT", "ISRJ")
TARGET_INDICES = (2, 3, 4)


def _as_finite_matrix(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return array


def weak_positive_bce(
    logits: np.ndarray,
    targets: np.ndarray,
    weak_mask: np.ndarray,
    target_indices: tuple[int, ...] = TARGET_INDICES,
) -> np.ndarray:
    """Return the frozen, unweighted BCE risk for each target weak component."""
    score = _as_finite_matrix("logits", logits)
    truth = np.asarray(targets, dtype=np.uint8)
    weak = np.asarray(weak_mask, dtype=np.uint8)
    if truth.shape != score.shape or weak.shape != score.shape:
        raise ValueError("logits, targets, and weak_mask must have identical shapes")
    risks: list[float] = []
    for index in target_indices:
        selected = (truth[:, index] == 1) & (weak[:, index] == 1)
        if not np.any(selected):
            raise ValueError(f"target index {index} has no positive weak fit samples")
        # BCEWithLogits(z, 1) = softplus(-z), evaluated stably in float64.
        risks.append(float(np.logaddexp(0.0, -score[selected, index]).mean()))
    result = np.asarray(risks, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("weak target risks are non-finite")
    return result


def _validate_q(q: np.ndarray, upper: float) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(values)):
        raise ValueError("q is non-finite")
    if values[0] < -1e-12 or values[1] < -1e-12 or values[0] > upper + 1e-12 or values[1] > upper + 1e-12:
        raise ValueError("q is outside frozen bounds")
    if values[0] > values[1] + 1e-12:
        raise ValueError("q_shared must not exceed q_classifier")
    return values


def solve_target_risk_projection(
    fit_logits: Callable[[np.ndarray], np.ndarray],
    targets: np.ndarray,
    weak_mask: np.ndarray,
    *,
    q0: tuple[float, float] = (1.5, 1.5),
    upper: float = 1.5,
    maxiter: int = 200,
    ftol: float = 1e-12,
    constraint_tolerance: float = 1e-7,
    target_indices: tuple[int, ...] = TARGET_INDICES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project q0 onto the fit-only target-risk feasible set.

    ``fit_logits`` must be deterministic and fit-role-only.  The exact C6
    anchor is evaluated at q=(0,0), then every target risk is constrained to
    be no larger than that anchor.  SLSQP starts only from q0 and never uses a
    grid, restart, performance metric, threshold, calibration, or test value.
    """
    q0_array = _validate_q(np.asarray(q0, dtype=np.float64), float(upper))
    cache: dict[tuple[float, float], np.ndarray] = {}
    calls = 0

    def risks(q: np.ndarray) -> np.ndarray:
        nonlocal calls
        values = np.asarray(q, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("solver proposed non-finite q")
        key = (float(values[0]), float(values[1]))
        if key not in cache:
            logits = _as_finite_matrix("fit_logits(q)", fit_logits(values.copy()))
            cache[key] = weak_positive_bce(logits, targets, weak_mask, target_indices)
            calls += 1
        return cache[key]

    baseline = risks(np.zeros(2, dtype=np.float64))
    initial_risks = risks(q0_array)
    constraints = [{"type": "ineq", "fun": lambda q: float(q[1] - q[0])}]
    constraints.extend(
        {"type": "ineq", "fun": lambda q, index=index: float(baseline[index] - risks(q)[index])}
        for index in range(len(target_indices))
    )
    result = minimize(
        lambda q: float(np.sum((np.asarray(q, dtype=np.float64) - q0_array) ** 2)),
        q0_array.copy(),
        method="SLSQP",
        bounds=((0.0, float(upper)), (0.0, float(upper))),
        constraints=constraints,
        options={"maxiter": int(maxiter), "ftol": float(ftol), "disp": False},
    )
    q_star = np.asarray(result.x, dtype=np.float64).reshape(2)
    candidate = risks(q_star)
    bound_violation = max(0.0, -q_star[0], -q_star[1], q_star[0] - upper, q_star[1] - upper, q_star[0] - q_star[1])
    risk_violation = float(max(0.0, np.max(candidate - baseline)))
    max_violation = float(max(bound_violation, risk_violation))
    diagnostics: dict[str, Any] = {
        "solver": "scipy.optimize.minimize(method=SLSQP)",
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "solver_evaluations": int(calls),
        "q0": q0_array.tolist(),
        "q_star": q_star.tolist(),
        "objective": float(np.sum((q_star - q0_array) ** 2)),
        "baseline_target_weak_bce": baseline.tolist(),
        "full_target_weak_bce": initial_risks.tolist(),
        "trb_target_weak_bce": candidate.tolist(),
        "risk_delta_vs_c6": (candidate - baseline).tolist(),
        "max_bound_violation": float(bound_violation),
        "max_risk_violation": risk_violation,
        "max_constraint_violation": max_violation,
        "constraint_tolerance": float(constraint_tolerance),
        "finite": bool(np.all(np.isfinite(q_star)) and np.all(np.isfinite(candidate))),
    }
    if not diagnostics["solver_success"] or max_violation > float(constraint_tolerance) or not diagnostics["finite"]:
        raise RuntimeError(f"TRB-PTV frozen solver contract failed: {diagnostics}")
    return q_star, diagnostics


def activation_gate(q_values: list[np.ndarray], *, minimum_norm: float = 0.05, minimum_seeds: int = 4) -> dict[str, Any]:
    """Evaluate the preregistered q-activation gate without changing q."""
    q0 = np.asarray((1.5, 1.5), dtype=np.float64)
    norms = [float(np.linalg.norm(np.asarray(q, dtype=np.float64).reshape(2) - q0)) for q in q_values]
    if not all(np.isfinite(norm) for norm in norms):
        raise ValueError("activation gate received a non-finite q")
    activated = [norm >= float(minimum_norm) for norm in norms]
    return {
        "minimum_norm": float(minimum_norm),
        "minimum_seeds": int(minimum_seeds),
        "norms": norms,
        "activated_seed_count": int(sum(activated)),
        "passed": bool(sum(activated) >= int(minimum_seeds)),
    }

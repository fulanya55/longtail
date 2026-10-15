"""Pure numerical core for the preregistered FC-PTV projection."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls


GROUP_NAMES = ("sawfm_absent", "sawfm_present")


def robust_logit_scales(base_logits, sawfm_present, min_scale=1e-6):
    logits = np.asarray(base_logits, dtype=np.float64)
    saw = np.asarray(sawfm_present, dtype=bool).reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != saw.size:
        raise ValueError("logits and SawFM strata have incompatible shapes")
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("base logits are non-finite")
    medians = np.empty((2, logits.shape[1]), dtype=np.float64)
    scales = np.empty_like(medians)
    counts = np.empty(2, dtype=np.int64)
    for group, present in enumerate((False, True)):
        mask = saw == present
        counts[group] = int(mask.sum())
        if counts[group] < 2:
            raise ValueError(f"SawFM stratum {GROUP_NAMES[group]} has fewer than two fit samples")
        values = logits[mask]
        medians[group] = np.median(values, axis=0)
        scales[group] = 1.4826 * np.median(np.abs(values - medians[group]), axis=0)
    if not np.all(np.isfinite(scales)) or np.any(scales < float(min_scale)):
        raise ValueError("C6 fit-only robust logit scale is degenerate")
    return {"medians": medians, "scales": scales, "counts": counts}


def functional_gram(base_logits, shared_logits, classifier_logits, sawfm_present, scales):
    base = np.asarray(base_logits, dtype=np.float64)
    shared = np.asarray(shared_logits, dtype=np.float64)
    classifier = np.asarray(classifier_logits, dtype=np.float64)
    saw = np.asarray(sawfm_present, dtype=bool).reshape(-1)
    scale = np.asarray(scales, dtype=np.float64)
    if base.shape != shared.shape or base.shape != classifier.shape or base.shape[0] != saw.size:
        raise ValueError("functional signature logits have incompatible shapes")
    if scale.shape != (2, base.shape[1]):
        raise ValueError("functional scale shape is invalid")
    if not all(np.all(np.isfinite(value)) for value in (base, shared, classifier, scale)):
        raise FloatingPointError("functional signature inputs are non-finite")
    gram = np.empty((2, base.shape[1], 2, 2), dtype=np.float64)
    for group, present in enumerate((False, True)):
        mask = saw == present
        for label in range(base.shape[1]):
            signatures = np.column_stack((
                (shared[mask, label] - base[mask, label]) / scale[group, label],
                (classifier[mask, label] - base[mask, label]) / scale[group, label],
            ))
            matrix = signatures.T @ signatures / float(len(signatures))
            matrix = 0.5 * (matrix + matrix.T)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            gram[group, label] = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return gram


def linear_distances(q, gram):
    coefficients = np.asarray(q, dtype=np.float64).reshape(2)
    matrices = np.asarray(gram, dtype=np.float64)
    values = np.einsum("i,glij,j->gl", coefficients, matrices, coefficients)
    return np.sqrt(np.maximum(values, 0.0))


def _constraint_rows(q, gram, upper):
    q = np.asarray(q, dtype=np.float64).reshape(2)
    rows = [
        ("shared_lower", -q[0], np.asarray([-1.0, 0.0])),
        ("classifier_lower", -q[1], np.asarray([0.0, -1.0])),
        ("shared_upper", q[0] - upper, np.asarray([1.0, 0.0])),
        ("classifier_upper", q[1] - upper, np.asarray([0.0, 1.0])),
        ("classifier_not_below_shared", q[0] - q[1], np.asarray([1.0, -1.0])),
    ]
    for group in range(gram.shape[0]):
        for label in range(gram.shape[1]):
            matrix = gram[group, label]
            rows.append((
                f"functional_{group}_{label}",
                float(q @ matrix @ q - 1.0),
                2.0 * matrix @ q,
            ))
    return rows


def _kkt_diagnostics(q, q0, gram, upper=1.5, active_tolerance=1e-6):
    rows = _constraint_rows(q, gram, upper)
    active = [row for row in rows if row[1] >= -float(active_tolerance)]
    gradient = 2.0 * (np.asarray(q, dtype=np.float64) - np.asarray(q0, dtype=np.float64))
    if active:
        matrix = np.column_stack([row[2] for row in active])
        multipliers, _ = nnls(matrix, -gradient)
        stationarity = gradient + matrix @ multipliers
        complementarity = max(abs(multiplier * row[1]) for multiplier, row in zip(multipliers, active))
    else:
        multipliers = np.empty(0, dtype=np.float64)
        stationarity = gradient
        complementarity = 0.0
    return {
        "max_constraint_violation": float(max(0.0, max(row[1] for row in rows))),
        "stationarity_residual": float(np.linalg.norm(stationarity)),
        "complementarity_residual": float(complementarity),
        "active_constraints": [row[0] for row in active],
        "active_multipliers": multipliers.tolist(),
    }


def solve_two_block_projection(gram, q0=(1.5, 1.5), upper=1.5):
    matrices = np.asarray(gram, dtype=np.float64)
    if matrices.ndim != 4 or matrices.shape[-2:] != (2, 2) or not np.all(np.isfinite(matrices)):
        raise ValueError("functional Gram matrices are invalid")
    q0 = np.asarray(q0, dtype=np.float64).reshape(2)
    initial_scale = max(1.0, float(np.max(linear_distances(q0, matrices))))
    initial = q0 / initial_scale
    constraints = [{"type": "ineq", "fun": lambda q: q[1] - q[0]}]
    for group in range(matrices.shape[0]):
        for label in range(matrices.shape[1]):
            matrix = matrices[group, label].copy()
            constraints.append({"type": "ineq", "fun": lambda q, matrix=matrix: 1.0 - q @ matrix @ q})
    result = minimize(
        lambda q: float(np.sum((q - q0) ** 2)),
        initial,
        method="SLSQP",
        bounds=((0.0, float(upper)), (0.0, float(upper))),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    q = np.asarray(result.x, dtype=np.float64)
    diagnostics = _kkt_diagnostics(q, q0, matrices, upper=float(upper))
    diagnostics.update({
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective": float(np.sum((q - q0) ** 2)),
        "linear_distances": linear_distances(q, matrices).tolist(),
    })
    if (not result.success or diagnostics["max_constraint_violation"] > 1e-7
            or diagnostics["stationarity_residual"] > 1e-5
            or diagnostics["complementarity_residual"] > 1e-5):
        raise RuntimeError(f"FC-PTV projection did not satisfy the frozen solver contract: {diagnostics}")
    return q, diagnostics


def exact_functional_distances(base_logits, candidate_logits, sawfm_present, scales):
    base = np.asarray(base_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    saw = np.asarray(sawfm_present, dtype=bool).reshape(-1)
    scale = np.asarray(scales, dtype=np.float64)
    if base.shape != candidate.shape or base.shape[0] != saw.size or scale.shape != (2, base.shape[1]):
        raise ValueError("exact functional-distance inputs have incompatible shapes")
    if not np.all(np.isfinite(candidate)):
        raise FloatingPointError("candidate fit logits are non-finite")
    distances = np.empty((2, base.shape[1]), dtype=np.float64)
    for group, present in enumerate((False, True)):
        mask = saw == present
        standardized = (candidate[mask] - base[mask]) / scale[group][None, :]
        distances[group] = np.sqrt(np.mean(standardized ** 2, axis=0))
    if not np.all(np.isfinite(distances)):
        raise FloatingPointError("exact functional distance is non-finite")
    return distances


def simultaneous_lower_bounds(per_seed_label_deltas, bootstrap_samples=10000, seed=20261215):
    values = np.asarray(per_seed_label_deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("simultaneous bounds require a finite seed-by-label matrix")
    mean = values.mean(axis=0)
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, values.shape[0], size=(int(bootstrap_samples), values.shape[0]))
    draws = values[indices].mean(axis=1)
    critical = float(np.percentile(np.max(mean[None, :] - draws, axis=1), 95.0))
    return {
        "mean": mean.tolist(),
        "critical_value": critical,
        "familywise_95_lower": (mean - critical).tolist(),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
    }

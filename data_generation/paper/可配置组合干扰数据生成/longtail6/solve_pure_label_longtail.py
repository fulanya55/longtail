"""Plan six-class pure-label long-tail combination counts.

This module only plans integer counts; it never opens or modifies waveform MAT
files.  The six labels are represented by the 63 non-empty bit-mask patterns.
For a count vector ``n`` we report:

* label IF = max(A @ n) / min(A @ n), where A is the 63x6 pattern matrix;
* combination CF = max(n[p]) / min(n[p]) over retained patterns;
* JS divergence between the normalised retained-pattern distribution and the
  uniform 63-pattern anchor (zero mass for omitted patterns).

The MILP maximises retained-pattern coverage first, then seeks a large label
spread while keeping the requested IF and CF upper bounds.  JS is evaluated as
an exact nonlinear post-condition; candidates are generated with an L1
distance tie-breaker so that a configured JS threshold is not silently relaxed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


DEFAULT_LABELS = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
LABELS = DEFAULT_LABELS
PATTERN_MASKS = np.arange(1, 1 << len(LABELS), dtype=np.int64)
PATTERN_MATRIX = ((PATTERN_MASKS[:, None] >> np.arange(len(LABELS))) & 1).astype(float)


def set_label_space(labels: Sequence[str]) -> None:
    """Set the six-class label order before constructing a planning grid."""
    global LABELS, PATTERN_MASKS, PATTERN_MATRIX
    values = tuple(str(value).strip() for value in labels)
    if len(values) != 6 or len(set(values)) != 6 or any(not value for value in values):
        raise ValueError("labels must contain six distinct non-empty names")
    LABELS = values
    PATTERN_MASKS = np.arange(1, 1 << len(LABELS), dtype=np.int64)
    PATTERN_MATRIX = ((PATTERN_MASKS[:, None] >> np.arange(len(LABELS))) & 1).astype(float)


@dataclass
class Candidate:
    status: str
    solver_status: int | None
    solver_message: str
    sample_count: int
    target_if: float
    cf_cap: float
    min_per_pattern: int
    js_threshold: float
    retained_pattern_count: int
    counts: list[int]
    label_counts: list[int]
    achieved_if: float | None
    achieved_cf: float | None
    js_divergence: float | None
    l1_to_uniform: float | None
    active_masks: list[int]
    sha256: str
    note: str


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        keep = a > 0
        return float(np.sum(a[keep] * np.log(a[keep] / b[keep])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _sha256_payload(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_milp(n_total: int, target_if: float, cf_cap: float,
                min_per_pattern: int, coverage_floor: int | None = None,
                l1_weight: float = 0.0, head_label: int | None = None,
                tail_label: int | None = None):
    """Create a MILP with n[63], y[63], label min/max and combo min/max.

    Variable order is n, y, q_label_min, r_label_max, a_combo_min,
    b_combo_max, and optionally d[63] for an L1 tie-breaker.
    """
    P, C = PATTERN_MATRIX.shape
    base = 2 * P + 4
    d0 = base
    size = base + (P if l1_weight else 0)
    n_slice = slice(0, P)
    y_slice = slice(P, 2 * P)
    q, r, a, b = 2 * P, 2 * P + 1, 2 * P + 2, 2 * P + 3
    c = np.zeros(size, dtype=float)
    # The caller runs a coverage-maximisation pass first.  For a fixed
    # coverage, favour a large label spread and then a small L1 deviation.
    c[r] = -1.0
    c[q] = 1.0
    if l1_weight:
        c[d0:] = float(l1_weight)

    integrality = np.zeros(size, dtype=int)
    integrality[:P] = 1
    integrality[P:2 * P] = 1
    lower = np.zeros(size, dtype=float)
    upper = np.full(size, np.inf, dtype=float)
    upper[:P] = n_total
    upper[P:2 * P] = 1
    lower[q:r + 1] = 1
    upper[q:r + 1] = n_total * C
    lower[a:b + 1] = 1
    upper[a:b + 1] = n_total
    if l1_weight:
        upper[d0:] = 1.0
    rows, lo, hi = [], [], []

    def add(row, lb=-np.inf, ub=np.inf):
        rows.append(row); lo.append(lb); hi.append(ub)

    row = np.zeros(size); row[:P] = 1; add(row, n_total, n_total)
    if coverage_floor is not None:
        row = np.zeros(size); row[P:2 * P] = 1
        add(row, coverage_floor, coverage_floor)
    # m*y <= n <= N*y
    for i in range(P):
        row = np.zeros(size); row[i] = 1; row[P + i] = -min_per_pattern
        add(row, 0, np.inf)
        row = np.zeros(size); row[i] = 1; row[P + i] = -n_total
        add(row, -np.inf, 0)
    # q <= marginal_j <= r and r <= target_if*q
    for j in range(C):
        row = np.zeros(size); row[:P] = PATTERN_MATRIX[:, j]; row[q] = -1
        add(row, 0, np.inf)
        row = np.zeros(size); row[:P] = PATTERN_MATRIX[:, j]; row[r] = -1
        add(row, -np.inf, 0)
    # To optimise the achieved IF without a nonlinear max/min objective, the
    # caller enumerates a designated head and tail label.  Equality plus the
    # ordering inequalities make r and q the actual extrema for that pair.
    if head_label is not None and tail_label is not None:
        if head_label == tail_label or not (0 <= head_label < C and 0 <= tail_label < C):
            raise ValueError("head_label and tail_label must be distinct label indices")
        head = PATTERN_MATRIX[:, head_label]
        tail = PATTERN_MATRIX[:, tail_label]
        row = np.zeros(size); row[:P] = head; row[r] = -1
        add(row, 0, 0)
        row = np.zeros(size); row[:P] = tail; row[q] = -1
        add(row, 0, 0)
        for j in range(C):
            row = np.zeros(size); row[:P] = head - PATTERN_MATRIX[:, j]
            add(row, 0, np.inf)
            row = np.zeros(size); row[:P] = PATTERN_MATRIX[:, j] - tail
            add(row, 0, np.inf)
    row = np.zeros(size); row[r] = 1; row[q] = -float(target_if)
    add(row, -np.inf, 0)
    # For active combinations: a <= n_i and n_i <= b.  Big-M terms disable
    # the lower bound when y=0; b is also upper-bounded by N*y.
    for i in range(P):
        row = np.zeros(size); row[a] = 1; row[i] = -1; row[P + i] = n_total
        add(row, -np.inf, n_total)
        row = np.zeros(size); row[i] = 1; row[b] = -1
        add(row, -np.inf, 0)
        row = np.zeros(size); row[b] = 1; row[P + i] = -n_total
        add(row, -np.inf, 0)
    row = np.zeros(size); row[b] = 1; row[a] = -float(cf_cap)
    add(row, -np.inf, 0)
    if l1_weight:
        # d_i >= |n_i/N - 1/P|, with d measured in probability units.
        for i in range(P):
            row = np.zeros(size); row[d0 + i] = 1; row[i] = -1.0 / n_total
            add(row, -1.0 / P, np.inf)
            row = np.zeros(size); row[d0 + i] = 1; row[i] = 1.0 / n_total
            add(row, 1.0 / P, np.inf)
    A = np.vstack(rows)
    return c, integrality, Bounds(lower, upper), LinearConstraint(A, np.asarray(lo), np.asarray(hi)), d0


def solve_one(n_total: int, target_if: float, cf_cap: float,
              min_per_pattern: int = 8, js_threshold: float = 0.05,
              requested_coverage: int = 63) -> Candidate:
    if n_total <= 0 or min_per_pattern <= 0:
        raise ValueError("n_total and min_per_pattern must be positive")
    if target_if < 1 or cf_cap < 1:
        raise ValueError("target_if and cf_cap must be >= 1")
    if requested_coverage < 1 or requested_coverage > len(PATTERN_MASKS):
        raise ValueError("requested_coverage must be in [1,63]")

    # Pass 1: maximise coverage subject to the exact IF/CF upper bounds.
    c, integ, bounds, constraint, _ = _build_milp(
        n_total, target_if, cf_cap, min_per_pattern)
    c[:] = 0
    c[len(PATTERN_MASKS):2 * len(PATTERN_MASKS)] = -1
    result = milp(c, integrality=integ, bounds=bounds, constraints=constraint,
                  options={"presolve": True})
    if not result.success:
        return Candidate("infeasible", int(result.status), str(result.message),
                         n_total, target_if, cf_cap, min_per_pattern,
                         js_threshold, 0, [], [], None, None, None, None,
                         [], "", "MILP has no integer solution under IF/CF bounds")
    x = np.rint(result.x).astype(np.int64)
    P = len(PATTERN_MASKS)
    coverage = int(np.sum(x[P:2 * P]))
    coverage = min(coverage, requested_coverage)

    # Pass 2: enumerate the (six*five) possible marginal head/tail labels.
    # This makes the IF objective exact instead of relying on loose extrema
    # auxiliaries whose values could drift above/below the observed marginals.
    best = None
    for head_label in range(PATTERN_MATRIX.shape[1]):
        for tail_label in range(PATTERN_MATRIX.shape[1]):
            if head_label == tail_label:
                continue
            c, integ, bounds, constraint, d0 = _build_milp(
                n_total, target_if, cf_cap, min_per_pattern,
                coverage_floor=coverage, l1_weight=1e-4,
                head_label=head_label, tail_label=tail_label)
            candidate = milp(c, integrality=integ, bounds=bounds,
                             constraints=constraint, options={"presolve": True})
            if candidate.success and (best is None or candidate.fun < best.fun):
                best = candidate
    if best is None:
        return Candidate("infeasible", int(result.status), str(result.message),
                         n_total, target_if, cf_cap, min_per_pattern,
                         js_threshold, coverage, [], [], None, None, None, None,
                         [], "", "coverage was feasible but spread tie-break MILP failed")
    result = best
    counts = np.rint(result.x[:P]).astype(int)
    active = counts > 0
    label_counts = PATTERN_MATRIX.T.astype(np.int64) @ counts
    min_label, max_label = int(label_counts.min()), int(label_counts.max())
    min_combo, max_combo = int(counts[active].min()), int(counts[active].max())
    achieved_if = max_label / min_label
    achieved_cf = max_combo / min_combo
    p = counts.astype(float) / float(n_total)
    q_anchor = np.full(P, 1.0 / P)
    js = _js_divergence(p, q_anchor)
    l1 = float(np.abs(p - q_anchor).sum())
    payload = {"n": counts.tolist(), "target_if": target_if, "cf_cap": cf_cap,
               "min_per_pattern": min_per_pattern}
    status = "feasible" if js <= js_threshold + 1e-12 else "js_infeasible"
    note = "exact integer counts satisfy sum, coverage, IF and CF; JS is an exact post-condition"
    if status != "feasible":
        note += "; no claim is made that the requested JS threshold is feasible"
    return Candidate(status, int(result.status), str(result.message), n_total,
                     target_if, cf_cap, min_per_pattern, js_threshold,
                     int(np.count_nonzero(active)), counts.tolist(),
                     label_counts.astype(int).tolist(), achieved_if, achieved_cf,
                     js, l1, PATTERN_MASKS[active].astype(int).tolist(),
                     _sha256_payload(payload), note)


def plan_grid(n_values: Sequence[int], target_ifs: Sequence[float],
              cf_caps: Sequence[float], min_per_pattern: int,
              js_threshold: float, output: Path) -> dict:
    records = []
    for n in n_values:
        for target_if in target_ifs:
            for cf_cap in cf_caps:
                records.append(asdict(solve_one(int(n), float(target_if),
                                                 float(cf_cap), min_per_pattern,
                                                 js_threshold)))
    payload = {
        "schema": "six-class-pure-label-longtail-planner-v1",
        "labels": list(LABELS), "pattern_masks": PATTERN_MASKS.tolist(),
        "anchor": {"type": "uniform_63_patterns", "pattern_count": 63},
        "definitions": {
            "IF": "max(A@n)/min(A@n), marginal label counts",
            "CF": "max(n_active)/min(n_active), retained combination counts",
            "JS": "Jensen-Shannon divergence to uniform 63-pattern anchor",
        },
        "constraints": {"min_per_pattern": int(min_per_pattern),
                         "js_threshold": float(js_threshold),
                         "target_ifs": list(map(float, target_ifs)),
                         "cf_caps": list(map(float, cf_caps)),
                         "n_values": list(map(int, n_values))},
        "records": records,
    }
    maxima = {}
    for record in records:
        if record["status"] != "feasible":
            continue
        key = f"N={record['sample_count']},CF<={record['cf_cap']}"
        value = record["achieved_if"]
        if value is not None and (key not in maxima or value > maxima[key]):
            maxima[key] = value
    payload["max_feasible_if_by_n_cf"] = maxima
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _parse_numbers(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", default="8000", help="comma-separated total sample counts")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                        help="six labels in bit order, comma-separated")
    parser.add_argument("--target-if", default="3,5,6,10")
    parser.add_argument("--cf-cap", default="8,16,32")
    parser.add_argument("--min-per-pattern", type=int, default=8)
    parser.add_argument("--js-threshold", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_label_space(_parse_numbers(args.labels, str))
    payload = plan_grid(_parse_numbers(args.n, int),
                        _parse_numbers(args.target_if, float),
                        _parse_numbers(args.cf_cap, float),
                        args.min_per_pattern, args.js_threshold, args.output)
    summary = [{"n": r["sample_count"], "if_target": r["target_if"],
                "cf_cap": r["cf_cap"], "status": r["status"],
                "coverage": r["retained_pattern_count"],
                "if": r["achieved_if"], "cf": r["achieved_cf"],
                "js": r["js_divergence"]} for r in payload["records"]]
    print(json.dumps({"output": str(args.output.resolve()), "records": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

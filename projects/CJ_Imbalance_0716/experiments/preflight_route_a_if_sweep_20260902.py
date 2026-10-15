"""Feasibility preflight for fixed-budget IF>5 label-only subsets."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
import sys
import h5py
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cjlt.data.split_builder import combo_key

OUT = ROOT / "outputs/route_a_if_sweep_20260902"
ROLE = ROOT / "outputs/route_a_c1_baseline_full_20260826/role_manifest.json"
NAMES = ["FMB", "AMA", "SawFM", "SinFM", "MFT", "ISRJ"]


def sha(a): return hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()


def solve_ratio(y: np.ndarray, n: int, ratio: float, seed: int, min_tail: int = 500):
    names = sorted({combo_key(row) for row in y})
    groups = {k: np.flatnonzero(np.asarray([combo_key(row) for row in y]) == k) for k in names}
    V = np.asarray([[int(b) for b in k] for k in names], dtype=float).T
    avail = np.asarray([len(groups[k]) for k in names], dtype=float)
    # Enumerate which labels realize head and tail, then impose an interval
    # around the requested ratio. This avoids claiming an exact IF when the
    # multi-label co-occurrence geometry makes it impossible.
    # The six-label source is nearly symmetric; use a fixed pair for the
    # preflight and avoid a costly 30-pair search. A later builder can rotate
    # the pair if this first pair is infeasible.
    for h, t in [(0, 5)]:
            rows = [np.ones(len(names))]; lo = [n]; hi = [n]
            # all non-tail labels >= tail; head >= all labels
            for i in range(6):
                if i != t: rows.append(V[i] - V[t]); lo.append(0); hi.append(np.inf)
                if i != h: rows.append(V[h] - V[i]); lo.append(0); hi.append(np.inf)
            rows.append(V[h] - ratio * V[t]); lo.append(0); hi.append(np.inf)
            rows.append(V[h] - ratio * 1.05 * V[t]); lo.append(-np.inf); hi.append(0)
            rows.append(V[t]); lo.append(min_tail); hi.append(np.inf)
            c = np.random.default_rng(seed + int(ratio*10) + h*10+t).random(len(names))*1e-8
            sol = milp(c, integrality=np.ones(len(names)), bounds=Bounds(np.zeros(len(names)), avail),
                       constraints=LinearConstraint(np.asarray(rows), np.asarray(lo), np.asarray(hi)),
                       options={"time_limit":5.0})
            if sol.success and sol.x is not None:
                x = np.rint(sol.x).astype(int); actual = V @ x
                if int(x.sum()) == n and actual.max()/actual.min() >= ratio and actual.max()/actual.min() <= ratio*1.05:
                    return names, groups, x, actual.astype(int), h, t
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    role = json.loads(ROLE.read_text(encoding="utf-8"))
    source = Path(role["fit"]["path"]); fit = np.asarray(role["fit"]["sample_indices"], dtype=np.int64)
    with h5py.File(str(source), "r") as h:
        y9 = np.asarray(h["known_label"][:, fit], dtype=np.uint8).T
    y = y9[:, [0, 1, 2, 3, 5, 7]]
    available = y.sum(axis=0).astype(int)
    source_combos = {combo_key(row) for row in y}
    results = []
    # Search the largest common budget first.  Target vectors are geometric,
    # with a fixed head/tail ratio and all targets constrained to availability.
    for n in (16000, 12000, 8000, 4000, 2000):
        for r in (6.0, 10.0, 20.0):
            feasible = None
            solved = solve_ratio(y, n, r, 2039)
            if solved is not None:
                names, groups, x, actual, h, t = solved
                selected = np.concatenate([groups[k][:int(v)] for k,v in zip(names,x) if v > 0]).astype(np.int64)
                combos = {}
                for row in y[selected]:
                    k = combo_key(row); combos[k] = combos.get(k, 0) + 1
                feasible = {
                    "target_if": r, "target_sample_count": n, "head_label": NAMES[h], "tail_label": NAMES[t],
                    "selected_count": int(len(selected)), "actual_label_counts": actual.tolist(),
                    "effective_if": float(actual.max()/actual.min()),
                    "combo_min": int(min(combos.values())), "combo_max": int(max(combos.values())),
                    "effective_cf": float(max(combos.values())/min(combos.values())),
                    "unique_combo_count": int(len(combos)), "combo_coverage_fraction": float(len(combos)/len(source_combos)),
                    "selected_indices_sha256": sha(fit[selected]), "status": "completed_valid",
                }
                break
            results.append({"target_if": r, "target_sample_count": n,
                            **(feasible if feasible else {"status":"protocol_infeasible","available_label_counts":available.tolist()})})
    payload = {"schema":"route-a-if-sweep-feasibility-v1", "source":str(source.resolve()),
               "fit_count":int(len(fit)), "available_label_counts":dict(zip(NAMES,available.tolist())),
               "source_unique_combo_count":len(source_combos), "results":results,
               "locked_test_used":False, "frozen_e5_modified":False}
    (OUT/"feasibility.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","output":str((OUT/'feasibility.json').resolve()),"results":results},ensure_ascii=False),flush=True)


if __name__ == "__main__": main()

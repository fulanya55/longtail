"""Run the preregistered no-fit H0v natural-visibility-graph diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {3: "MFT", 4: "VFT", 5: "ISRJ"}
NODE_COUNT = 128
PERMUTATION = np.random.default_rng(20260803).permutation(NODE_COUNT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def degree_entropy(values: np.ndarray) -> float:
    """Natural-visibility-graph degree-distribution entropy, with no fit."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size != NODE_COUNT or not np.all(np.isfinite(x)):
        raise ValueError(f"H0v expects exactly {NODE_COUNT} finite samples")
    degree = np.zeros(x.size, dtype=np.int32)
    for left in range(x.size - 1):
        distance = np.arange(1, x.size - left, dtype=np.float64)
        slope = (x[left + 1 :] - x[left]) / distance
        preceding = np.empty_like(slope)
        preceding[0] = -np.inf
        if slope.size > 1:
            preceding[1:] = np.maximum.accumulate(slope[:-1])
        visible = slope > preceding
        right = np.flatnonzero(visible) + left + 1
        degree[left] += int(right.size)
        degree[right] += 1
    counts = np.bincount(degree, minlength=x.size).astype(np.float64)
    probability = counts[counts > 0] / x.size
    return float(-(probability * np.log(probability)).sum() / np.log(x.size - 1))


def _read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        raw = np.asarray(handle["iq_complex"])
        if raw.dtype.fields is None or not {"real", "imag"}.issubset(raw.dtype.fields):
            raise ValueError("H0v requires MATLAB v7.3 compound real/imag I/Q storage")
        iq = raw["real"].astype(np.float64, copy=False) + 1j * raw["imag"].astype(np.float64, copy=False)
        labels = np.asarray(handle["label_selected"], dtype=np.uint8)
        pair = np.asarray(handle["pair_id"]).reshape(-1).astype(np.int64)
        target = np.asarray(handle["pair_target_index"]).reshape(-1).astype(np.int64) - 1
        side = np.asarray(handle["pair_side"]).reshape(-1).astype(np.int64)
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
        sample = np.asarray(handle["sample_seed"]).reshape(-1).astype(np.int64)
    if iq.ndim != 2 or iq.shape[0] != labels.shape[0] or labels.shape[1] != 6:
        raise ValueError("H0v I/Q/label shape contract failed")
    return iq, labels, pair, target, side, receiver, sample


def _scores(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    take = np.linspace(0, iq.shape[1] - 1, NODE_COUNT, dtype=np.int64)
    true = np.asarray([degree_entropy(row.real[take]) for row in iq], dtype=np.float64)
    permuted = np.asarray([degree_entropy(row.real[take][PERMUTATION]) for row in iq], dtype=np.float64)
    return true, permuted


def _paired_rows(pair: np.ndarray, target: np.ndarray, side: np.ndarray, labels: np.ndarray, receiver: np.ndarray, sample: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    output: dict[int, list[tuple[int, int]]] = {key: [] for key in TARGETS}
    for pair_id in np.unique(pair):
        rows = np.flatnonzero(pair == pair_id)
        if rows.size != 2 or set(side[rows]) != {0, 1} or np.unique(target[rows]).size != 1:
            raise ValueError(f"pair {pair_id} is not an exact two-view counterfactual")
        if np.unique(receiver[rows]).size != 1 or np.unique(sample[rows]).size != 1:
            raise ValueError(f"pair {pair_id} receiver/sample identity contract failed")
        negative, positive = rows[side[rows] == 0][0], rows[side[rows] == 1][0]
        current = int(target[positive])
        if current not in TARGETS or labels[negative, current] != 0 or labels[positive, current] != 1:
            raise ValueError(f"pair {pair_id} target/label contract failed")
        if not np.array_equal(np.delete(labels[negative], current), np.delete(labels[positive], current)):
            raise ValueError(f"pair {pair_id} nuisance labels differ")
        output[current].append((int(negative), int(positive)))
    return {key: (np.asarray([item[0] for item in items]), np.asarray([item[1] for item in items])) for key, items in output.items()}


def _metrics(absent: np.ndarray, weak: np.ndarray, absent_perm: np.ndarray, weak_perm: np.ndarray) -> dict[str, float]:
    y = np.r_[np.zeros(absent.size, dtype=np.uint8), np.ones(weak.size, dtype=np.uint8)]
    true = np.r_[absent, weak]
    permuted = np.r_[absent_perm, weak_perm]
    return {
        "true_auroc": float(roc_auc_score(y, true)),
        "permuted_auroc": float(roc_auc_score(y, permuted)),
        "true_minus_permuted_auroc": float(roc_auc_score(y, true) - roc_auc_score(y, permuted)),
        "true_aupr": float(average_precision_score(y, true)),
        "permuted_aupr": float(average_precision_score(y, permuted)),
        "paired_weak_gt_absent_fraction": float(np.mean(weak > absent)),
    }


def _bootstrap(absent: np.ndarray, weak: np.ndarray, absent_perm: np.ndarray, weak_perm: np.ndarray, draws: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(20260803)
    auroc, delta = [], []
    for _ in range(draws):
        selected = rng.integers(0, absent.size, absent.size)
        values = _metrics(absent[selected], weak[selected], absent_perm[selected], weak_perm[selected])
        auroc.append(values["true_auroc"]); delta.append(values["true_minus_permuted_auroc"])
    interval = lambda values: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {"true_auroc_95_ci": interval(auroc), "true_minus_permuted_auroc_95_ci": interval(delta)}


def run(args: argparse.Namespace) -> dict:
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 must remain sealed")
    iq, labels, pair, target, side, receiver, sample = _read(args.data_path)
    paired = _paired_rows(pair, target, side, labels, receiver, sample)
    # The smoke generator emits 30 total pairs, cycling equally over three
    # targets; the formal source emits 1,800 total pairs (600 per target).
    expected = 10 if args.smoke else 600
    if any(rows[0].size != expected for rows in paired.values()):
        raise ValueError(f"H0v expected {expected} pairs per target")
    true, permuted = _scores(iq)
    finite = bool(np.all(np.isfinite(true)) and np.all(np.isfinite(permuted)))
    if args.smoke:
        result = {
            "status": "completed_smoke_layout_only",
            "experiment": "RouteA_H0v_NaturalVisibilityGraph",
            "data_path": str(args.data_path), "source_sha256": _sha256(args.data_path),
            "view_count": int(iq.shape[0]), "pair_count": int(np.unique(pair).size),
            "iq_shape": list(iq.shape), "node_count": NODE_COUNT, "scores_finite": finite,
            "metrics_reported": False, "gate_evaluated": False,
            "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
        }
    else:
        by_target = {}
        all_checks = []
        for index, name in TARGETS.items():
            absent_rows, weak_rows = paired[index]
            values = _metrics(true[absent_rows], true[weak_rows], permuted[absent_rows], permuted[weak_rows])
            bootstrap = _bootstrap(true[absent_rows], true[weak_rows], permuted[absent_rows], permuted[weak_rows], args.bootstrap_draws)
            checks = {
                "true_auroc_ge_0p58": values["true_auroc"] >= 0.58,
                "true_minus_permuted_auroc_ge_0p040": values["true_minus_permuted_auroc"] >= 0.040,
                "delta_95_ci_lower_gt_zero": bootstrap["true_minus_permuted_auroc_95_ci"][0] > 0.0,
                "paired_weak_gt_absent_fraction_ge_0p56": values["paired_weak_gt_absent_fraction"] >= 0.56,
            }
            by_target[name] = {"pair_count": int(absent_rows.size), **values, "bootstrap": {"draws": args.bootstrap_draws, **bootstrap}, "checks": checks, "passed": bool(all(checks.values()))}
            all_checks.extend(checks.values())
        result = {
            "status": "completed_no_fit_observable",
            "experiment": "RouteA_H0v_NaturalVisibilityGraph",
            "evidence_role": "fresh_development_only_counterfactual_pair_identifiability",
            "data_path": str(args.data_path), "source_sha256": _sha256(args.data_path),
            "node_count": NODE_COUNT, "input": "128 evenly spaced real-I samples", "score": "natural_visibility_graph_normalized_degree_entropy",
            "fixed_permutation_seed": 20260803, "trained_parameters": 0, "metrics_by_target": by_target,
            "gate": {"passed": bool(finite and all(all_checks)), "all_scores_finite": finite},
            "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-v3-manifest", type=Path, default=ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "gate": result.get("gate")}, ensure_ascii=False, indent=2))

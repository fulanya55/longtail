"""Read-only co-label audit of completed frozen H0g/H0h ISRJ predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("FMB", "SawFM", "SinFM", "MFT", "VFT", "ISRJ")
MFT, VFT, ISRJ = 3, 4, 5
VARIANTS = ("B0_H1_Frozen", "B2_H1AlignedPeriodicity_Frozen")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(targets: np.ndarray, weak: np.ndarray, probability: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float | int | bool]:
    weak_positive = mask & (targets[:, ISRJ] == 1) & weak[:, ISRJ]
    negative = mask & (targets[:, ISRJ] == 0)
    strong_positive = mask & (targets[:, ISRJ] == 1) & ~weak[:, ISRJ]
    supported = int(weak_positive.sum()) >= 10 and int(negative.sum()) >= 10
    result: dict[str, float | int | bool] = {
        "count": int(mask.sum()),
        "weak_positive_count": int(weak_positive.sum()),
        "strong_positive_count": int(strong_positive.sum()),
        "negative_count": int(negative.sum()),
        "supported": supported,
        "weak_recall": float(prediction[weak_positive, ISRJ].mean()) if weak_positive.any() else float("nan"),
        "negative_fpr": float(prediction[negative, ISRJ].mean()) if negative.any() else float("nan"),
    }
    result["weak_vs_negative_auroc"] = (
        float(roc_auc_score(
            np.concatenate((np.ones(weak_positive.sum(), dtype=np.uint8), np.zeros(negative.sum(), dtype=np.uint8))),
            np.concatenate((probability[weak_positive, ISRJ], probability[negative, ISRJ])),
        ))
        if supported else float("nan")
    )
    return result


def _strata(targets: np.ndarray) -> dict[str, np.ndarray]:
    mft = targets[:, MFT].astype(bool)
    vft = targets[:, VFT].astype(bool)
    masks = {"all": np.ones(len(targets), dtype=bool), "mft_present": mft, "mft_absent": ~mft, "vft_present": vft, "vft_absent": ~vft}
    for mft_value in (False, True):
        for vft_value in (False, True):
            masks[f"mft_{int(mft_value)}__vft_{int(vft_value)}"] = (mft == mft_value) & (vft == vft_value)
    return masks


def _mean_ci(values: list[float], seed: int) -> dict[str, float | list[float]] | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return None
    rng = np.random.default_rng(seed)
    draws = finite[rng.integers(0, len(finite), size=(10000, len(finite)))].mean(axis=1)
    return {"mean": float(finite.mean()), "bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))], "seed_count": int(len(finite))}


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    required = {"targets", "weak_mask", "probabilities", "predictions"}
    if not required.issubset(payload.files):
        raise ValueError(f"prediction fields missing: {path}")
    return {name: np.asarray(payload[name]) for name in required}


def _validate_summary(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("external_refit") is not False:
        raise ValueError(f"not a completed frozen external summary: {path}")
    if summary.get("test_v3_used") is not False or summary.get("locked_test_used") is not False:
        raise ValueError(f"test boundary unsafe: {path}")
    return summary


def _audit_source(name: str, root: Path, seeds: list[int]) -> dict[str, object]:
    summary_path = next(root.glob("*summary.json"), None)
    if summary_path is None:
        raise FileNotFoundError(f"summary missing in {root}")
    _validate_summary(summary_path)
    per_seed: list[dict[str, object]] = []
    hashes: dict[str, str] = {str(summary_path): _sha256(summary_path)}
    for seed in seeds:
        views = {variant: _load_prediction(root / f"seed_{seed}" / variant / "predictions.npz") for variant in VARIANTS}
        for variant in VARIANTS:
            path = root / f"seed_{seed}" / variant / "predictions.npz"
            hashes[str(path)] = _sha256(path)
        base = views[VARIANTS[0]]
        candidate = views[VARIANTS[1]]
        if not np.array_equal(base["targets"], candidate["targets"]) or not np.array_equal(base["weak_mask"], candidate["weak_mask"]):
            raise ValueError(f"targets/weak masks differ at seed {seed}")
        targets = base["targets"].astype(np.uint8)
        weak = base["weak_mask"].astype(bool)
        if targets.shape[1] != len(CLASS_NAMES):
            raise ValueError(f"unexpected label count at seed {seed}")
        groups: dict[str, object] = {}
        for group_index, (group, mask) in enumerate(_strata(targets).items()):
            b0 = _metric(targets, weak, base["probabilities"], base["predictions"].astype(bool), mask)
            b2 = _metric(targets, weak, candidate["probabilities"], candidate["predictions"].astype(bool), mask)
            delta = {
                name: float(b2[name]) - float(b0[name]) if np.isfinite(float(b2[name])) and np.isfinite(float(b0[name])) else float("nan")
                for name in ("weak_vs_negative_auroc", "weak_recall", "negative_fpr")
            }
            groups[group] = {"b0": b0, "b2": b2, "b2_minus_b0": delta}
        per_seed.append({"seed": seed, "groups": groups})
    aggregate: dict[str, object] = {}
    first_targets = _load_prediction(root / f"seed_{seeds[0]}" / VARIANTS[0] / "predictions.npz")["targets"]
    for group_index, group in enumerate(_strata(first_targets).keys()):
        aggregate[group] = {
            "b2_minus_b0": {
                metric: _mean_ci([float(row["groups"][group]["b2_minus_b0"][metric]) for row in per_seed], 20_261_000 + 100 * group_index + metric_index)
                for metric_index, metric in enumerate(("weak_vs_negative_auroc", "weak_recall", "negative_fpr"))
            },
            "supported_seed_count": int(sum(bool(row["groups"][group]["b0"]["supported"]) for row in per_seed)),
        }
    return {"source": name, "summary_path": str(summary_path), "prediction_sha256": hashes, "per_seed": per_seed, "aggregate": aggregate}


def run(args: argparse.Namespace) -> dict[str, object]:
    results = [_audit_source(name, root, args.seeds) for name, root in (("H0g_high_replica", args.h0g_root), ("H0h_ar1_noise", args.h0h_root))]
    result = {
        "status": "completed_nonselection_frozen_prediction_audit",
        "experiment": "RouteA_ISRJFrozenCoLabelAudit",
        "evidence_role": "post_completed_nonselection_subgroup_description",
        "class_names": list(CLASS_NAMES),
        "target": "ISRJ weak-positive versus ISRJ-absent ranking",
        "strata": ["all", "mft_present", "mft_absent", "vft_present", "vft_absent", "mft_0__vft_0", "mft_0__vft_1", "mft_1__vft_0", "mft_1__vft_1"],
        "inputs": results,
        "selection_or_refit_performed": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    root = ROOT / "outputs" / "tapc"
    parser = argparse.ArgumentParser()
    parser.add_argument("--h0g-root", type=Path, default=root / "route_a_h0g_isrj_parameter_ood_v2_frozen")
    parser.add_argument("--h0h-root", type=Path, default=root / "route_a_h0h_ar1_noise_frozen")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--output", type=Path, default=root / "route_a_isrj_frozen_colabel_audit/route_a_isrj_frozen_colabel_audit.json")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, allow_nan=True))

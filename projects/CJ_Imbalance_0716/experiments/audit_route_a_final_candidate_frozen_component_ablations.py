"""Frozen single-component lesions for the Route-A v2 candidate on development-v12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import _hard_metrics, _subgroups  # noqa: E402
from experiments.run_route_a_mft_shared_decision_policy_audit import hybrid_policy  # noqa: E402


MFT, VFT, ISRJ = 3, 4, 5
COMPLETE = "A0_CompleteSharedPolicy"
VARIANTS = (COMPLETE, "A1_RemoveVFTRidge_SameCapacity", "A2_RemoveVFTRidge_Shuffled", "A3_RemoveMFTGuard", "A4_RemoveISRJB2")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _calibration(root: Path, seed: int, variant: str):
    directory = root / f"seed_{seed}" / variant
    return _read_json(directory / "calibration.json"), _read_json(directory / "thresholds.json")["thresholds"]


def _policy(logits, q4_cal, q4_thr, q1_cal, q1_thr):
    return hybrid_policy(np.asarray(logits, dtype=np.float32), q4_cal, q4_thr, q1_cal, q1_thr)


def _record(labels, weak, probability, prediction, logits, thresholds):
    metrics = _metrics(labels, weak, probability, prediction)
    _subgroups(metrics, labels, prediction)
    return {
        "metrics": metrics,
        "mft_hard": _hard_metrics(labels, weak, probability, prediction),
        "logits": logits,
        "probabilities": probability,
        "predictions": prediction,
        "thresholds": thresholds,
    }


def _save(root: Path, name: str, receiver, labels, weak, item):
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz", receiver_noise_seed=receiver, targets=labels,
        weak_mask=weak.astype(np.uint8), logits=item["logits"], probabilities=item["probabilities"],
        predictions=item["predictions"].astype(np.uint8), thresholds=item["thresholds"],
    )
    payload = {"variant": name, "metrics": item["metrics"], "mft_hard": item["mft_hard"]}
    _write_json(output / "metrics.json", payload)
    return payload


def _delta(candidate, lesion):
    a, b = candidate["metrics"], lesion["metrics"]
    values = {
        "macro_aupr": a["macro_aupr"] - b["macro_aupr"],
        "macro_f1": a["macro_f1"] - b["macro_f1"],
        "target_weak_recall": a["target_mean_weak_recall"] - b["target_mean_weak_recall"],
        "target_weak_auroc": a["target_weak_vs_negative_auroc"] - b["target_weak_vs_negative_auroc"],
        "target_fpr": a["target_mean_negative_fpr"] - b["target_mean_negative_fpr"],
        "mft_hard_fpr": candidate["mft_hard"]["hard_fpr"] - lesion["mft_hard"]["hard_fpr"],
        "mft_order5_fpr": candidate["mft_hard"]["order5_fpr"] - lesion["mft_hard"]["order5_fpr"],
    }
    for label in ("MFT", "VFT", "ISRJ"):
        for key in ("aupr", "weak_recall", "weak_vs_negative_auroc", "negative_fpr"):
            values[f"{label.lower()}_{key}"] = a["per_label"][label][key] - b["per_label"][label][key]
    return values


def _run_seed(args, seed: int):
    source = args.source_root / f"seed_{seed}"
    target_root = args.output_root / f"seed_{seed}"
    q0 = np.load(source / "Q0_H1" / "predictions.npz")
    q1 = np.load(source / "Q1_H1B2" / "predictions.npz")
    q3 = np.load(source / "Q3_VFTBaseOnlyCapacity" / "predictions.npz")
    q5 = np.load(source / "Q5_VFTShuffledRidgeFusion" / "predictions.npz")
    q6 = np.load(source / "Q6_SharedMFTDecision_VFTAligned" / "predictions.npz")
    labels = np.asarray(q6["targets"], dtype=np.uint8)
    weak = np.asarray(q6["weak_mask"], dtype=bool)
    receiver = np.asarray(q6["receiver_noise_seed"], dtype=np.int64)
    q4_cal, q4_thr = _calibration(args.construction_root, seed, "Q4_VFTAlignedRidgeFusion")
    q3_cal, q3_thr = _calibration(args.construction_root, seed, "Q3_VFTBaseOnlyCapacity")
    q5_cal, q5_thr = _calibration(args.construction_root, seed, "Q5_VFTShuffledRidgeFusion")
    q1_cal, q1_thr = _calibration(args.construction_root, seed, "Q1_H1B2")

    complete = _record(labels, weak, np.asarray(q6["probabilities"], dtype=np.float32), np.asarray(q6["predictions"], dtype=bool), np.asarray(q6["logits"], dtype=np.float32), np.asarray(q6["thresholds"], dtype=np.float32))
    items = {COMPLETE: complete}
    for name, donor, cal, thr in (("A1_RemoveVFTRidge_SameCapacity", q3, q3_cal, q3_thr), ("A2_RemoveVFTRidge_Shuffled", q5, q5_cal, q5_thr)):
        logits = complete["logits"].copy()
        logits[:, VFT] = np.asarray(donor["logits"], dtype=np.float32)[:, VFT]
        probability, prediction, thresholds = _policy(logits, cal, thr, q1_cal, q1_thr)
        items[name] = _record(labels, weak, probability, prediction, logits, thresholds)
    logits = complete["logits"].copy()
    logits[:, MFT] = np.asarray(q1["logits"], dtype=np.float32)[:, MFT]
    probability, prediction, thresholds = _policy(logits, q4_cal, q4_thr, q1_cal, q1_thr)
    items["A3_RemoveMFTGuard"] = _record(labels, weak, probability, prediction, logits, thresholds)
    logits = complete["logits"].copy()
    logits[:, ISRJ] = np.asarray(q0["logits"], dtype=np.float32)[:, ISRJ]
    probability, prediction, thresholds = _policy(logits, q4_cal, q4_thr, q1_cal, q1_thr)
    probability[:, ISRJ] = np.asarray(q0["probabilities"], dtype=np.float32)[:, ISRJ]
    prediction[:, ISRJ] = np.asarray(q0["predictions"], dtype=bool)[:, ISRJ]
    thresholds[ISRJ] = np.asarray(q0["thresholds"], dtype=np.float32)[ISRJ]
    items["A4_RemoveISRJB2"] = _record(labels, weak, probability, prediction, logits, thresholds)
    summaries = {name: _save(target_root, name, receiver, labels, weak, item) for name, item in items.items()}
    return {"seed": int(seed), "variants": summaries, "complete_minus_lesion": {name: _delta(complete, item) for name, item in items.items() if name != COMPLETE}}


def run(args):
    seal = _read_json(args.test_v3_manifest)["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    frozen = _read_json(args.frozen_manifest)
    if frozen["candidate_name"] != "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2":
        raise ValueError("wrong frozen candidate")
    rows = [_run_seed(args, seed) for seed in args.seeds]
    aggregate = {}
    for name in VARIANTS[1:]:
        keys = tuple(rows[0]["complete_minus_lesion"][name])
        aggregate[name] = {key: _mean_ci([row["complete_minus_lesion"][name][key] for row in rows], 20441000 + ni * 100 + ki) for ki, key in enumerate(keys) for ni in [VARIANTS.index(name)]}
    checks = {
        "all_five_seeds": len(rows) == 5,
        "test_v3_unused": True,
        "complete_sawfm_exact": all(np.max(np.abs(np.load(args.source_root / f"seed_{seed}" / "Q0_H1" / "predictions.npz")["probabilities"][:, 1] - np.load(args.source_root / f"seed_{seed}" / "Q6_SharedMFTDecision_VFTAligned" / "predictions.npz")["probabilities"][:, 1])) == 0.0 for seed in args.seeds),
    }
    payload = {"status": "completed", "experiment": "RouteA_FinalCandidate_FrozenComponentAblations", "evidence_role": "development_v12_posthoc_frozen_diagnostic", "rows": rows, "aggregate": aggregate, "checks": checks, "passed": bool(all(checks.values())), "locked_test_used": False, "test_v3_used": False, "source_summary_sha256": _sha256(args.source_root / "route_a_h1_b2_mft_vft_development_v12_frozen_summary.json")}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_final_candidate_frozen_component_ablations_summary.json", payload)
    print(json.dumps({"output": str(args.output_root), "checks": checks, "test_v3_used": False}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_development_v12_frozen")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_final_candidate_frozen_component_ablations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

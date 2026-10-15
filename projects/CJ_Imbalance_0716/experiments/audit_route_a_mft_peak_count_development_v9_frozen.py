"""Frozen zero-parameter MFT peak-count confirmation on fresh development-v9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import (  # noqa: E402
    SCORE_NAMES,
    extract_scores,
)
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals  # noqa: E402


MFT_INDEX = 3
CANDIDATE_SCORE = "nms_count_gt_bg_q99"
SCORE_INDEX = SCORE_NAMES.index(CANDIDATE_SCORE)
EXPECTED_SOURCE_SHA = "4406aa67d57123df55c45a297d1da57414810a4f92bb32f9273f38ede21eb33c"
V3_REFERENCE_AUROC = 0.6368878959059323


def _metric(target, score):
    return {
        "auroc": float(roc_auc_score(target, score)),
        "aupr": float(average_precision_score(target, score)),
    }


def _view(labels, weak, mode):
    order = labels.sum(axis=1)
    positive = (labels[:, MFT_INDEX] == 1) & weak[:, MFT_INDEX] & (order >= 4)
    negative = labels[:, MFT_INDEX] == 0
    if mode == "high_order":
        negative &= order >= 4
    elif mode == "order5_negative":
        negative &= order == 5
    else:
        raise ValueError(mode)
    selected = positive | negative
    return selected, positive[selected].astype(np.uint8), {
        "weak_positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
    }


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["evidence_role"] != "development_v9":
        raise ValueError("development-v9 identity audit failed")
    manifest = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    if manifest["candidate_name"] != "MFT_PulseCompression_NMSPeakCount_v1":
        raise ValueError("unexpected frozen candidate")
    source = PROJECT_ROOT / "experiments/audit_route_a_mft_pulse_compression_morphology_v3.py"
    if _sha256(source).lower() != EXPECTED_SOURCE_SHA:
        raise ValueError("frozen feature implementation hash mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    data_path = Path(audit["development_v9_path"])
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    signals = _load_signals(data_path, np.arange(len(labels)), args.io_batch_size)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_scores = extract_scores(signals, device, args.feature_batch_size)
    scores = {
        variant: values[:, SCORE_INDEX]
        for variant, values in all_scores.items()
        if variant in ("MF_AlignedLFM", "MF_WrongChirp")
    }
    results = {}
    for mode in ("high_order", "order5_negative"):
        selected, target, receipt = _view(labels, weak, mode)
        results[mode] = {"receipt": receipt, "overall": {}, "receiver_blocks": {}}
        for variant in scores:
            results[mode]["overall"][variant] = _metric(target, scores[variant][selected])
        for block in range(5):
            block_selected = selected & ((receiver % 5) == block)
            block_target = ((labels[:, MFT_INDEX] == 1) & weak[:, MFT_INDEX] & (labels.sum(axis=1) >= 4))[block_selected].astype(np.uint8)
            results[mode]["receiver_blocks"][str(block)] = {
                "count": int(block_selected.sum()),
                "positive_count": int(block_target.sum()),
                "negative_count": int(len(block_target) - block_target.sum()),
                **{
                    variant: _metric(block_target, scores[variant][block_selected])
                    for variant in scores
                },
            }
    high = results["high_order"]
    aligned = high["overall"]["MF_AlignedLFM"]["auroc"]
    wrong = high["overall"]["MF_WrongChirp"]["auroc"]
    block_aligned = [high["receiver_blocks"][str(block)]["MF_AlignedLFM"]["auroc"] for block in range(5)]
    block_wrong = [high["receiver_blocks"][str(block)]["MF_WrongChirp"]["auroc"] for block in range(5)]
    checks = {
        "overall_high_order_auroc_ge_0p60": aligned >= 0.60,
        "four_of_five_blocks_above_chance": sum(value > 0.50 for value in block_aligned) >= 4,
        "aligned_minus_wrong_ge_0p08": aligned - wrong >= 0.08,
        "four_of_five_blocks_favor_aligned": sum(a > b for a, b in zip(block_aligned, block_wrong)) >= 4,
        "overall_order5_negative_auroc_ge_0p58": results["order5_negative"]["overall"]["MF_AlignedLFM"]["auroc"] >= 0.58,
        "v3_to_v9_drop_le_0p06": V3_REFERENCE_AUROC - aligned <= 0.06,
    }
    result = {
        "status": "completed",
        "experiment": "RouteA_MFT_PeakCount_DevelopmentV9_Frozen",
        "evidence_role": "fresh_identity_frozen_feature_confirmation",
        "candidate": CANDIDATE_SCORE,
        "data_sha256": audit["sha256"],
        "frozen_manifest_sha256": _sha256(args.frozen_manifest),
        "feature_source_sha256": _sha256(source),
        "results": results,
        "comparison": {
            "development_v3_high_order_auroc": V3_REFERENCE_AUROC,
            "development_v9_high_order_auroc": aligned,
            "v3_to_v9_drop": V3_REFERENCE_AUROC - aligned,
            "aligned_minus_wrong_chirp": aligned - wrong,
        },
        "confirmation_gate": {"checks": checks, "passed": bool(all(checks.values()))},
        "next_action": "design_u2_bounded_fusion_on_development_v3" if all(checks.values()) else "stop_peak_count_mechanism",
        "external_refit": False,
        "learned_parameters": 0,
        "locked_test_used": False,
        "test_v3_used": False,
        "test_v9_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_mft_peak_count_development_v9_frozen.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "gate": result["confirmation_gate"], "comparison": result["comparison"], "summary": str(output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v9_generation/identity_audit.json")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_mft_peak_count_v1/mft_peak_count_frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_peak_count_development_v9_frozen")
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


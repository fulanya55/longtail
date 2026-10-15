"""Apply the frozen strict MFT expert to development-v8 for hard-negative identifiability."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _load_signals
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _scores
from experiments.run_route_a_vft_strict_complementary_score_fusion_v3 import _extract_final_temporal, _load_probe, _normalize_features, _standardize_scores

EXPECTED_V8_SHA256 = "985b6892e7c989f76dd08539b0555fe406232dcea5aaf8dfd2d76dec4f85c332"
MFT_INDEX = 3


def _auc(weak_positive, negative, score):
    mask = weak_positive | negative
    return float(roc_auc_score(weak_positive[mask].astype(np.uint8), np.asarray(score)[mask]))


def _run_seed(args, signals, labels, weak, seed, device):
    c6_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    state = torch.load(c6_path, map_location="cpu", weights_only=False)["model_state_dict"]
    model = _load_model_from_state(args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}", device, state)
    features = _extract_final_temporal(model, signals, device, args.batch_size)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    probe_path = args.layerwise_root / f"seed_{seed}" / "T_FinalTemporal" / "MFT" / "best_probe.pth"
    probe, payload = _load_probe(probe_path, device)
    raw_score = _scores(probe, _normalize_features(features, payload), device)
    old_summary = json.loads((args.mft_expert_root / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))
    expert_z = _standardize_scores(raw_score, old_summary["mft_expert_score_stats"])
    prediction_path = args.v8_prediction_root / f"seed_{seed}" / "B1_U2_FullAlphaAligned" / "predictions.npz"
    prediction = np.load(prediction_path)
    u2_score = prediction["probabilities"][:, MFT_INDEX]
    if not np.array_equal(prediction["targets"], labels):
        raise AssertionError("v8 prediction targets differ")
    positive = labels[:, MFT_INDEX] == 1
    weak_positive = positive & weak[:, MFT_INDEX]
    order = labels.sum(axis=1)
    negatives = {
        "all_negative": ~positive,
        "order_ge4_negative": (~positive) & (order >= 4),
        "order5_negative": (~positive) & (order == 5),
    }
    metrics = {}
    for name, negative in negatives.items():
        metrics[name] = {
            "weak_positive_count": int(weak_positive.sum()), "negative_count": int(negative.sum()),
            "expert_auroc": _auc(weak_positive, negative, expert_z),
            "u2_auroc": _auc(weak_positive, negative, u2_score),
        }
    return {
        "seed": seed, "metrics": metrics, "c6_checkpoint_sha256": _sha256(c6_path),
        "probe_sha256": _sha256(probe_path), "prediction_sha256": _sha256(prediction_path),
        "external_refit": False, "test_v3_used": False,
    }


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["identity_audit_passed"] or audit["sha256"].lower() != EXPECTED_V8_SHA256:
        raise ValueError("development-v8 audit failed")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    data_path = Path(audit["development_v8_path"])
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    signals = _load_signals(data_path, np.arange(len(labels)), args.io_batch_size)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = [_run_seed(args, signals, labels, weak, seed, device) for seed in args.seeds]
    aggregate = {}
    for group_index, group in enumerate(("all_negative", "order_ge4_negative", "order5_negative")):
        aggregate[group] = {}
        for metric_index, metric in enumerate(("expert_auroc", "u2_auroc")):
            aggregate[group][metric] = _mean_ci([row["metrics"][group][metric] for row in rows], 20359000+group_index*10+metric_index)
        aggregate[group]["expert_minus_u2"] = _mean_ci([row["metrics"][group]["expert_auroc"]-row["metrics"][group]["u2_auroc"] for row in rows],20359100+group_index)
    hard = aggregate["order_ge4_negative"]["expert_auroc"]["mean"]
    v3_reference = 0.65217
    interpretation = {
        "v3_strict_mft_reference_auroc": v3_reference,
        "v8_order_ge4_expert_auroc": hard,
        "v3_to_v8_drop": v3_reference-hard,
        "feature_generalization_passed": bool(hard >= 0.62 and v3_reference-hard <= 0.10),
        "next_action": "fusion_or_training_support" if hard >= 0.62 and v3_reference-hard <= 0.10 else "new_feature_extraction_required",
    }
    result = {
        "status":"completed","experiment":"RouteA_MFT_Expert_DevelopmentV8_Identifiability",
        "evidence_role":"read_only_frozen_fresh_identity_diagnostic","rows":rows,"aggregate":aggregate,
        "interpretation":interpretation,"external_refit":False,"locked_test_used":False,"test_v3_used":False,
    }
    args.output_root.mkdir(parents=True,exist_ok=True)
    output=args.output_root/"route_a_mft_expert_development_v8_identifiability.json"
    _write_json(output,result)
    print(json.dumps({"status":"completed","interpretation":interpretation,"summary":str(output)},ensure_ascii=False,indent=2))
    return result


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v8_generation/identity_audit.json")
    parser.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--layerwise-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_layerwise_matched_negative_separability_v3_strict")
    parser.add_argument("--mft-expert-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_mft_frozen_expert_bounded_residual_v3")
    parser.add_argument("--v8-prediction-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_m5_periodicity_development_v8_frozen")
    parser.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_mft_expert_development_v8_identifiability")
    parser.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043])
    parser.add_argument("--io-batch-size",type=int,default=128)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--weak-threshold-db",type=float,default=10.0)
    parser.add_argument("--device",default=None)
    return parser.parse_args()


if __name__=="__main__": run(parse_args())

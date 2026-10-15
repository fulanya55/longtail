"""Frozen U2 MFT peak-count guard evaluation on paired parameter-OOD views."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import h5py
import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import SCORE_NAMES, extract_scores  # noqa: E402
from experiments.audit_route_a_mft_peak_guard_ood_dataset import LEVELS, REGIMES  # noqa: E402
from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _component_power, _sha256, _write_json  # noqa: E402
from experiments.run_route_a_fullalpha_periodicity_development_v7_frozen import _run_seed as _run_u2_seed  # noqa: E402
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS as U_VARIANTS  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration, _load_signals  # noqa: E402
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import expert_logit, guarded_logits  # noqa: E402


MFT_INDEX = 3
PEAK_INDEX = SCORE_NAMES.index("nms_count_gt_bg_q99")
SUPPORTED_OOD = ("close_spacing", "ultra_close", "weak_target")


def _target_metrics(labels, weak, probabilities, predictions):
    order = labels.sum(axis=1)
    positive = labels[:, MFT_INDEX] == 1
    hard_negative = (~positive) & (order >= 4)
    selected = positive | hard_negative
    weak_positive = positive & weak[:, MFT_INDEX]
    return {
        "mft_recall": float(predictions[positive, MFT_INDEX].mean()),
        "mft_weak_recall": float(predictions[weak_positive, MFT_INDEX].mean()) if weak_positive.any() else float("nan"),
        "hard_fpr": float(predictions[hard_negative, MFT_INDEX].mean()),
        "positive_vs_hard_negative_auroc": float(roc_auc_score(positive[selected].astype(np.uint8), probabilities[selected, MFT_INDEX])),
        "positive_count": int(positive.sum()), "weak_positive_count": int(weak_positive.sum()),
        "hard_negative_count": int(hard_negative.sum()),
    }


def _evaluate(logits, labels, weak, calibration, thresholds):
    probability = _apply_frozen_calibration(logits, calibration)
    prediction = probability >= thresholds[None, :]
    return probability, prediction, {"metrics": _metrics(labels, weak, probability, prediction), **_target_metrics(labels, weak, probability, prediction)}


def _run_view(args, audit, regime, level, device, manifest):
    entry = audit["regimes"][regime][str(level)]
    data_path = Path(entry["path"])
    if _sha256(data_path).lower() != entry["sha256"].lower():
        raise ValueError("OOD data hash changed")
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    signals = _load_signals(data_path, np.arange(len(labels)), args.io_batch_size)
    with h5py.File(data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    raw_scores = extract_scores(signals, device, args.peak_batch_size)
    aligned_count = raw_scores["MF_AlignedLFM"][:, PEAK_INDEX]
    view_key = f"{regime}__{level}dB"
    view_root = args.output_root / "views" / view_key
    u2_root = args.output_root / "u2_reconstruction" / view_key
    u2_args = copy.copy(args)
    u2_args.output_root = u2_root
    rows = []
    for seed in args.seeds:
        u2_row = _run_u2_seed(u2_args, signals, labels, weak, receiver, seed, device, entry["sha256"])
        base_prediction = np.load(u2_root / f"seed_{seed}" / U_VARIANTS[2] / "predictions.npz")
        base_logits = np.asarray(base_prediction["logits"], dtype=np.float32)
        source_root = args.source_root / f"seed_{seed}"
        calibration = json.loads((source_root / U_VARIANTS[2] / "calibration.json").read_text(encoding="utf-8"))
        thresholds = np.asarray(json.loads((source_root / U_VARIANTS[2] / "thresholds.json").read_text(encoding="utf-8"))["thresholds"], dtype=np.float32)
        expert_root = args.guard_root / f"seed_{seed}" / "experts"
        model_path = expert_root / "aligned_model.joblib"
        normalization_path = expert_root / "normalization.npz"
        frozen = manifest["seed_artifacts"][str(seed)]
        if _sha256(model_path).lower() != frozen["aligned_model_sha256"] or _sha256(normalization_path).lower() != frozen["normalization_sha256"]:
            raise ValueError("frozen guard artifact changed")
        model = joblib.load(model_path)
        normalization = np.load(normalization_path)
        evidence = expert_logit(model, aligned_count, normalization["aligned_mean"], normalization["aligned_std"])
        candidate_logits, residual = guarded_logits(base_logits, evidence, 0.75)
        base_probability, base_binary, base = _evaluate(base_logits, labels, weak, calibration, thresholds)
        candidate_probability, candidate_binary, candidate = _evaluate(candidate_logits, labels, weak, calibration, thresholds)
        non_mft = [0, 1, 2, 4, 5]
        preservation = float(np.max(np.abs(candidate_logits[:, non_mft] - base_logits[:, non_mft])))
        seed_root = view_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(seed_root / "predictions.npz", receiver_noise_seed=receiver, targets=labels,
                            weak_mask=weak.astype(np.uint8), base_probabilities=base_probability,
                            candidate_probabilities=candidate_probability, base_predictions=base_binary.astype(np.uint8),
                            candidate_predictions=candidate_binary.astype(np.uint8), residual=residual)
        row = {"seed": int(seed), "base": base, "candidate": candidate, "non_mft_logit_max_abs": preservation,
               "u2_source": u2_row, "external_refit": False, "test_v3_used": False}
        _write_json(seed_root / "seed_summary.json", row)
        rows.append(row)
    keys = ("macro_aupr", "mft_recall", "hard_fpr", "positive_vs_hard_negative_auroc")
    deltas = {}
    for key in keys:
        values = []
        for row in rows:
            if key == "macro_aupr":
                values.append(row["candidate"]["metrics"][key] - row["base"]["metrics"][key])
            else:
                values.append(row["candidate"][key] - row["base"][key])
        deltas[key] = {"values": values, "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}
    result = {"regime": regime, "level_dB": int(level), "rows": rows, "deltas": deltas,
              "non_mft_logits_exact": all(row["non_mft_logit_max_abs"] == 0.0 for row in rows)}
    _write_json(view_root / "view_summary.json", result)
    return result


def run(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["paired_receiver_identities_across_all_views"] or audit["test_files_opened"]:
        raise ValueError("OOD dataset audit failed")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    manifest = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    views = [_run_view(args, audit, regime, level, device, manifest) for regime in REGIMES for level in LEVELS]
    regimes = {}
    for regime in REGIMES:
        selected = [view for view in views if view["regime"] == regime]
        regimes[regime] = {
            key: {"mean": float(np.mean([view["deltas"][key]["mean"] for view in selected])),
                  "worst_level": float(min(view["deltas"][key]["mean"] for view in selected))}
            for key in ("macro_aupr", "mft_recall", "hard_fpr", "positive_vs_hard_negative_auroc")
        }
        regimes[regime]["non_mft_logits_exact"] = all(view["non_mft_logits_exact"] for view in selected)
    idc = regimes["ID_control"]
    supported_checks = {
        regime: {
            "macro_aupr_delta_ge_minus0p005": regimes[regime]["macro_aupr"]["mean"] >= -0.005,
            "mft_recall_delta_ge_minus0p050": regimes[regime]["mft_recall"]["mean"] >= -0.050,
            "hard_fpr_nonincrease": regimes[regime]["hard_fpr"]["mean"] <= 0.0,
        }
        for regime in SUPPORTED_OOD
    }
    checks = {
        "id_macro_aupr_delta_ge_minus0p002": idc["macro_aupr"]["mean"] >= -0.002,
        "id_mft_recall_delta_ge_minus0p030": idc["mft_recall"]["mean"] >= -0.030,
        "id_hard_fpr_delta_le_minus0p020": idc["hard_fpr"]["mean"] <= -0.020,
        "all_supported_regime_safety_checks": all(all(value.values()) for value in supported_checks.values()),
        "two_of_three_supported_hard_auroc_improve": sum(regimes[regime]["positive_vs_hard_negative_auroc"]["mean"] > 0 for regime in SUPPORTED_OOD) >= 2,
        "worst_supported_level_recall_delta_ge_minus0p080": min(view["deltas"]["mft_recall"]["mean"] for view in views if view["regime"] in SUPPORTED_OOD) >= -0.080,
        "non_mft_logits_exact": all(view["non_mft_logits_exact"] for view in views),
    }
    result = {
        "status": "completed", "experiment": "RouteA_U2_MFT_PeakGuard_ParameterOOD_Frozen",
        "evidence_role": "external_generator_parameter_ood_frozen_evaluation", "views": views,
        "regimes": regimes, "supported_regime_checks": supported_checks,
        "external_evidence_gate": {"checks": checks, "passed": bool(all(checks.values()))},
        "low_count_claim_boundary": "diagnostic_only_dense_MFT_claim_requires_at_least_3_replicas",
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_u2_mft_peak_guard_parameter_ood_frozen_summary.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "gate": result["external_evidence_gate"], "regimes": regimes, "summary": str(output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_peak_guard_parameter_ood/dataset_audit.json")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_u2_mft_peak_guard_v1/u2_mft_peak_guard_frozen_manifest.json")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--score-stats-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_frozen_score_fusion_head_v3")
    parser.add_argument("--guard-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_mft_peak_guard_parameter_ood")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--io-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--peak-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


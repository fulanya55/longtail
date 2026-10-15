"""Bounded U2 MFT negative guard using the confirmed matched-filter peak count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch  # Load Windows DLLs before sklearn.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import (  # noqa: E402
    SCORE_NAMES,
    extract_scores,
)
from experiments.run_route_a_cooccurrence_safety_head_v3 import _metrics  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_fullalpha_periodicity_unified_v3 import VARIANTS as U_VARIANTS  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    _load_roles,
    _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


MFT_INDEX = 3
ALL_ROLES = ("fit", "validation", "early", "calibration", "threshold")
PEAK_INDEX = SCORE_NAMES.index("nms_count_gt_bg_q99")
VARIANTS = (
    "K0_U2Frozen",
    "K1_AlignedPeakNegativeGuard",
    "K2_WrongChirpNegativeGuard",
    "K3_ShuffledAlignedGuard",
)


def fit_expert(count, labels, weak, c_value, seed):
    positive = (labels[:, MFT_INDEX] == 1) & weak[:, MFT_INDEX]
    negative = (labels[:, MFT_INDEX] == 0) & (labels.sum(axis=1) >= 4)
    selected = positive | negative
    x = np.asarray(count[selected], dtype=np.float32)[:, None]
    y = positive[selected].astype(np.uint8)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), 1e-5).astype(np.float32)
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=int(seed),
    )
    model.fit((x - mean) / std, y)
    return model, mean, std, {
        "weak_positive_count": int(positive.sum()),
        "hard_negative_count": int(negative.sum()),
    }


def expert_logit(model, count, mean, std):
    x = np.asarray(count, dtype=np.float32)[:, None]
    return model.decision_function((x - mean) / np.maximum(std, 1e-5)).astype(np.float32)


def guarded_logits(base_logits, evidence, cap):
    output = np.asarray(base_logits, dtype=np.float32).copy()
    base = output[:, MFT_INDEX]
    probability = 1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))
    uncertainty = np.sqrt(np.maximum(4.0 * probability * (1.0 - probability), 0.0))
    negative_direction = np.minimum(np.tanh(np.asarray(evidence, dtype=np.float32)), 0.0)
    residual = float(cap) * uncertainty * negative_direction
    output[:, MFT_INDEX] = base + residual
    return output, residual.astype(np.float32)


def run_seed(args, config, labels, weak, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    receipt_path = root / "seed_summary.json"
    if args.resume and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(
        args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size
    )
    role_ids = np.load(role_path)
    source_root = args.source_root / f"seed_{seed}"
    logits_path = source_root / "all_role_logits.npz"
    loaded = np.load(logits_path)
    source = {
        role: np.asarray(loaded[f"{U_VARIANTS[2]}__{role}"], dtype=np.float32)
        for role in ALL_ROLES
    }
    score_sets = {
        role: extract_scores(roles[role]["signals"], device, args.feature_batch_size)
        for role in ALL_ROLES
    }
    aligned_count = {role: score_sets[role]["MF_AlignedLFM"][:, PEAK_INDEX] for role in ALL_ROLES}
    wrong_count = {role: score_sets[role]["MF_WrongChirp"][:, PEAK_INDEX] for role in ALL_ROLES}
    fit_labels = roles["fit"]["targets6"]
    fit_weak = weak[roles["fit"]["sample_ids"]]
    aligned_model, aligned_mean, aligned_std, fit_receipt = fit_expert(
        aligned_count["fit"], fit_labels, fit_weak, args.c_value, 20407000 + seed
    )
    wrong_model, wrong_mean, wrong_std, wrong_receipt = fit_expert(
        wrong_count["fit"], fit_labels, fit_weak, args.c_value, 20407100 + seed
    )
    expert_root = root / "experts"
    expert_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(aligned_model, expert_root / "aligned_model.joblib")
    joblib.dump(wrong_model, expert_root / "wrong_model.joblib")
    np.savez_compressed(
        expert_root / "normalization.npz",
        aligned_mean=aligned_mean,
        aligned_std=aligned_std,
        wrong_mean=wrong_mean,
        wrong_std=wrong_std,
    )
    evidence = {VARIANTS[1]: {}, VARIANTS[2]: {}, VARIANTS[3]: {}}
    permutations = {}
    for role_index, role in enumerate(ALL_ROLES):
        aligned = expert_logit(aligned_model, aligned_count[role], aligned_mean, aligned_std)
        wrong = expert_logit(wrong_model, wrong_count[role], wrong_mean, wrong_std)
        permutation = np.random.default_rng(20407200 + seed * 10 + role_index).permutation(len(aligned))
        evidence[VARIANTS[1]][role] = aligned
        evidence[VARIANTS[2]][role] = wrong
        evidence[VARIANTS[3]][role] = aligned[permutation]
        permutations[role] = permutation.astype(np.int64)
    logits = {VARIANTS[0]: {role: source[role].copy() for role in ALL_ROLES}}
    residuals = {VARIANTS[0]: {role: np.zeros(len(source[role]), dtype=np.float32) for role in ALL_ROLES}}
    for variant in VARIANTS[1:]:
        logits[variant], residuals[variant] = {}, {}
        for role in ALL_ROLES:
            logits[variant][role], residuals[variant][role] = guarded_logits(
                source[role], evidence[variant][role], args.residual_cap
            )
    calibration = json.loads((source_root / U_VARIANTS[2] / "calibration.json").read_text(encoding="utf-8"))
    thresholds_payload = json.loads((source_root / U_VARIANTS[2] / "thresholds.json").read_text(encoding="utf-8"))
    thresholds = np.asarray(thresholds_payload["thresholds"], dtype=np.float32)
    early_labels = roles["early"]["targets6"]
    early_weak = weak[roles["early"]["sample_ids"]]
    order = early_labels.sum(axis=1)
    results = []
    for variant in VARIANTS:
        probability = _apply_frozen_calibration(logits[variant]["early"], calibration)
        prediction = probability >= thresholds[None, :]
        metrics = _metrics(early_labels, early_weak, probability, prediction)
        positive = early_labels[:, MFT_INDEX] == 1
        weak_positive = positive & early_weak[:, MFT_INDEX]
        hard_negative = (~positive) & (order >= 4)
        rank = weak_positive | hard_negative
        hard_auroc = float(roc_auc_score(weak_positive[rank].astype(np.uint8), probability[rank, MFT_INDEX]))
        hard_fpr = float(prediction[hard_negative, MFT_INDEX].mean())
        out = root / variant
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "metrics.json", metrics)
        np.savez_compressed(
            out / "predictions.npz",
            sample_ids=roles["early"]["sample_ids"],
            targets=early_labels,
            weak_mask=early_weak.astype(np.uint8),
            logits=logits[variant]["early"],
            probabilities=probability,
            predictions=prediction.astype(np.uint8),
            thresholds=thresholds,
            expert_evidence=evidence.get(variant, {}).get("early", np.zeros(len(early_labels))),
            residual=residuals[variant]["early"],
        )
        results.append(
            {
                "variant": variant,
                "metrics": metrics,
                "mft_order_ge4_fpr": hard_fpr,
                "mft_weak_vs_order_ge4_negative_auroc": hard_auroc,
            }
        )
    non_mft = [0, 1, 2, 4, 5]
    preservation = {
        variant: float(np.max(np.abs(logits[variant]["early"][:, non_mft] - source["early"][:, non_mft])))
        for variant in VARIANTS[1:]
    }
    np.savez_compressed(
        expert_root / "all_role_expert_evidence.npz",
        **{
            f"{variant}__{role}": evidence[variant][role]
            for variant in VARIANTS[1:]
            for role in ALL_ROLES
        },
        **{f"permutation__{role}": permutations[role] for role in ALL_ROLES},
    )
    row = {
        "seed": int(seed),
        "status": "completed",
        "data_sha256": data_sha,
        "source_logits_sha256": _sha256(logits_path),
        "role_indices_sha256": _sha256(role_path),
        "fit_receipt": {"aligned": fit_receipt, "wrong": wrong_receipt},
        "artifacts": {
            "aligned_model_sha256": _sha256(expert_root / "aligned_model.joblib"),
            "wrong_model_sha256": _sha256(expert_root / "wrong_model.joblib"),
            "normalization_sha256": _sha256(expert_root / "normalization.npz"),
            "evidence_sha256": _sha256(expert_root / "all_role_expert_evidence.npz"),
        },
        "variants": results,
        "non_mft_logit_max_abs": preservation,
        "residual_cap": float(args.residual_cap),
        "calibration_reused": True,
        "thresholds_reused": True,
        "locked_test_used": False,
        "test_v3_used": False,
    }
    _write_json(receipt_path, row)
    return row


def construction_guard(rows):
    per_seed = []
    for row in rows:
        variants = {item["variant"]: item for item in row["variants"]}
        base = variants[VARIANTS[0]]
        aligned = variants[VARIANTS[1]]
        wrong = variants[VARIANTS[2]]
        shuffled = variants[VARIANTS[3]]
        per_seed.append(
            {
                "seed": row["seed"],
                "macro_aupr": aligned["metrics"]["macro_aupr"] - base["metrics"]["macro_aupr"],
                "target_weak_recall": aligned["metrics"]["target_mean_weak_recall"] - base["metrics"]["target_mean_weak_recall"],
                "target_weak_auroc": aligned["metrics"]["target_weak_vs_negative_auroc"] - base["metrics"]["target_weak_vs_negative_auroc"],
                "hard_fpr": aligned["mft_order_ge4_fpr"] - base["mft_order_ge4_fpr"],
                "hard_auroc": aligned["mft_weak_vs_order_ge4_negative_auroc"] - base["mft_weak_vs_order_ge4_negative_auroc"],
                "vs_wrong_hard_auroc": aligned["mft_weak_vs_order_ge4_negative_auroc"] - wrong["mft_weak_vs_order_ge4_negative_auroc"],
                "vs_shuffled_hard_auroc": aligned["mft_weak_vs_order_ge4_negative_auroc"] - shuffled["mft_weak_vs_order_ge4_negative_auroc"],
            }
        )
    mean = {key: float(np.mean([row[key] for row in per_seed])) for key in per_seed[0] if key != "seed"}
    preservation = all(row["non_mft_logit_max_abs"][VARIANTS[1]] == 0.0 for row in rows)
    checks = {
        "macro_aupr_decline_le_0p002": mean["macro_aupr"] >= -0.002,
        "target_weak_recall_decline_le_0p010": mean["target_weak_recall"] >= -0.010,
        "target_weak_auroc_decline_le_0p002": mean["target_weak_auroc"] >= -0.002,
        "hard_fpr_decrease_ge_0p020": mean["hard_fpr"] <= -0.020,
        "hard_auroc_gain_ge_0p010": mean["hard_auroc"] >= 0.010,
        "aligned_vs_wrong_hard_auroc_ge_0p005": mean["vs_wrong_hard_auroc"] >= 0.005,
        "aligned_vs_shuffled_hard_auroc_ge_0p005": mean["vs_shuffled_hard_auroc"] >= 0.005,
        "non_mft_logits_exact": preservation,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "mean_delta": mean,
        "per_seed": per_seed,
    }


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if any(seal["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    manifest = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    if manifest["candidate_name"] != "MFT_PulseCompression_NMSPeakCount_v1":
        raise ValueError("wrong frozen peak-count candidate")
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = [run_seed(args, config, labels, weak, seed, device, data_sha) for seed in args.seeds]
    guard = construction_guard(rows)
    result = {
        "status": "completed",
        "experiment": "RouteA_U2_MFT_PeakCountNegativeGuard_v3",
        "evidence_role": "development_v3_construction_guard",
        "candidate": VARIANTS[1],
        "rows": rows,
        "construction_guard": guard,
        "next_action": "freeze_and_generate_development_v10" if guard["passed"] else "stop_peak_count_guard",
        "locked_test_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_u2_mft_peak_count_negative_guard_v3_summary.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "guard": guard, "summary": str(output)}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_fullalpha_periodicity_unified_v3")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_mft_peak_count_v1/mft_peak_count_frozen_manifest.json")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--residual-cap", type=float, default=0.75)
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())


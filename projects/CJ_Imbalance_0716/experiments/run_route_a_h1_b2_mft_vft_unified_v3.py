"""Strong H1+B2 model with frozen MFT guard and conditional VFT-STFT fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import _fit_score_stats, _standardize_scores  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES  # noqa: E402
from experiments.run_route_a_raw_iq_physics_separability_v3 import _vft_features  # noqa: E402
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import guarded_logits  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _evaluate, _load_roles, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


MFT_INDEX = 3
VFT_INDEX = 4
NON_VFT = (0, 1, 2, 3, 5)
VARIANTS = (
    "Q0_H1",
    "Q1_H1B2",
    "Q2_H1B2_MFTGuard",
    "Q3_VFTBaseOnlyCapacity",
    "Q4_VFTAlignedRidgeFusion",
    "Q5_VFTShuffledRidgeFusion",
)


def _extract_vft(signals, device, batch_size):
    rows = []
    with torch.no_grad():
        for start in range(0, len(signals), int(batch_size)):
            batch = torch.from_numpy(signals[start:start + int(batch_size)]).to(device)
            rows.append(_vft_features(batch, grid_size=16).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def _ridge_scores(features, payload):
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(payload["std"], dtype=np.float32), 1e-5)
    coef = np.asarray(payload["coef"], dtype=np.float32).reshape(-1)
    intercept = float(np.asarray(payload["intercept"]).reshape(-1)[0])
    normalized = ((features - mean[None, :]) / std[None, :]).astype(np.float32)
    return (normalized @ coef + intercept).astype(np.float32)


def _fusion_features(logits, expert_z=None):
    base = np.asarray(logits, dtype=np.float32)[:, VFT_INDEX]
    if expert_z is None:
        return base[:, None]
    probability = 1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))
    uncertainty = np.sqrt(np.maximum(4.0 * probability * (1.0 - probability), 0.0))
    z = np.asarray(expert_z, dtype=np.float32)
    return np.stack([base, z, uncertainty * z, np.tanh(base) * z], axis=1).astype(np.float32)


def _fit_head(features, labels, weights, seed, c_value):
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-5).astype(np.float32)
    x = ((features - mean[None, :]) / std[None, :]).astype(np.float32)
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=1000, random_state=int(seed))
    model.fit(x, labels, sample_weight=weights)
    return model, mean, std


def _head_scores(model, features, mean, std):
    x = ((features - mean[None, :]) / np.maximum(std[None, :], 1e-5)).astype(np.float32)
    return model.decision_function(x).astype(np.float32)


def _replace_vft(logits, scores):
    output = np.asarray(logits, dtype=np.float32).copy()
    output[:, VFT_INDEX] = np.asarray(scores, dtype=np.float32)
    return output


def _permutation(length, seed):
    values = np.random.default_rng(int(seed)).permutation(int(length)).astype(np.int64)
    return values, hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    h1_path = args.h1_b2_root / f"seed_{seed}" / "all_role_logits.npz"
    h1_loaded = np.load(h1_path)
    q0 = {role: np.asarray(h1_loaded[f"B0_H1__{role}"], dtype=np.float32) for role in ALL_ROLES}
    q1 = {role: np.asarray(h1_loaded[f"B2_H1AlignedPeriodicity__{role}"], dtype=np.float32) for role in ALL_ROLES}

    evidence_path = args.mft_evidence_root / f"seed_{seed}" / "experts" / "all_role_expert_evidence.npz"
    mft_evidence = np.load(evidence_path)
    q2 = {}
    for role in ALL_ROLES:
        q2[role], _ = guarded_logits(
            q1[role], mft_evidence[f"K1_AlignedPeakNegativeGuard__{role}"], args.mft_residual_cap
        )

    ridge_path = args.vft_ridge_root / f"seed_{seed}" / "ridge_model.npz"
    ridge_payload = np.load(ridge_path)
    features = {role: _extract_vft(roles[role]["signals"], device, args.feature_batch_size) for role in ALL_ROLES}
    raw_score = {role: _ridge_scores(features[role], ridge_payload) for role in ALL_ROLES}
    fit_target = roles["fit"]["targets6"][:, VFT_INDEX] == 1
    fit_weak = weak6[roles["fit"]["sample_ids"]][:, VFT_INDEX]
    strict_positive = fit_target & fit_weak
    strict_negative = ~fit_target
    stats = _fit_score_stats(raw_score["fit"][strict_positive | strict_negative])
    aligned_z = {role: _standardize_scores(raw_score[role], stats) for role in ALL_ROLES}
    shuffled_z, permutation_receipt = {}, {}
    for offset, role in enumerate(ALL_ROLES):
        permutation, digest = _permutation(len(aligned_z[role]), 20421000 + seed * 10 + offset)
        shuffled_z[role] = aligned_z[role][permutation]
        permutation_receipt[role] = {"sha256": digest, "count": int(len(permutation))}

    feature_sets = {
        "base": {role: _fusion_features(q2[role]) for role in ALL_ROLES},
        "aligned": {role: _fusion_features(q2[role], aligned_z[role]) for role in ALL_ROLES},
        "shuffled": {role: _fusion_features(q2[role], shuffled_z[role]) for role in ALL_ROLES},
    }
    labels = roles["fit"]["targets6"][:, VFT_INDEX].astype(np.uint8)
    weights = np.where(labels.astype(bool) & fit_weak, float(args.weak_positive_weight), 1.0).astype(np.float64)
    heads = {}
    for offset, name in enumerate(("base", "aligned", "shuffled")):
        model, mean, std = _fit_head(feature_sets[name]["fit"], labels, weights, 20422000 + seed + offset, args.c_value)
        out = root / f"head_{name}"
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
        heads[name] = (model, mean, std)

    logits = {
        VARIANTS[0]: q0,
        VARIANTS[1]: q1,
        VARIANTS[2]: q2,
        VARIANTS[3]: {role: _replace_vft(q2[role], _head_scores(heads["base"][0], feature_sets["base"][role], heads["base"][1], heads["base"][2])) for role in ALL_ROLES},
        VARIANTS[4]: {role: _replace_vft(q2[role], _head_scores(heads["aligned"][0], feature_sets["aligned"][role], heads["aligned"][1], heads["aligned"][2])) for role in ALL_ROLES},
        VARIANTS[5]: {role: _replace_vft(q2[role], _head_scores(heads["shuffled"][0], feature_sets["shuffled"][role], heads["shuffled"][1], heads["shuffled"][2])) for role in ALL_ROLES},
    }
    for role in ALL_ROLES:
        if np.max(np.abs(logits[VARIANTS[4]][role][:, NON_VFT] - q2[role][:, NON_VFT])) != 0.0:
            raise AssertionError("aligned VFT fusion changed a non-VFT logit")
    results = [_evaluate(name, logits[name], roles, weak6, root / name, args) for name in VARIANTS]
    np.savez_compressed(root / "all_role_logits.npz", **{
        f"{variant}__{role}": logits[variant][role] for variant in VARIANTS for role in ALL_ROLES
    })
    row = {
        "seed": int(seed), "status": "completed", "data_sha256": data_sha,
        "role_indices_sha256": _sha256(role_path), "h1_b2_logits_sha256": _sha256(h1_path),
        "mft_evidence_sha256": _sha256(evidence_path), "vft_ridge_sha256": _sha256(ridge_path),
        "vft_score_stats": stats, "vft_permutations": permutation_receipt,
        "variants": results, "non_vft_logit_max_abs": 0.0,
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    _write_json(root / "seed_summary.json", row)
    return row


def _mean_delta(rows, candidate, baseline, path):
    values = []
    for row in rows:
        variants = _variant_map(row)
        a, b = variants[candidate]["metrics"], variants[baseline]["metrics"]
        av, bv = a, b
        for key in path:
            av, bv = av[key], bv[key]
        values.append(float(av - bv))
    return float(np.mean(values)), values


def _vft_gate(rows):
    paths = {
        "weak_auroc": ("per_label", "VFT", "weak_vs_negative_auroc"),
        "weak_recall": ("per_label", "VFT", "weak_recall"),
        "aupr": ("per_label", "VFT", "aupr"),
    }
    delta_q2 = {name: _mean_delta(rows, VARIANTS[4], VARIANTS[2], path) for name, path in paths.items()}
    delta_q3 = _mean_delta(rows, VARIANTS[4], VARIANTS[3], paths["weak_auroc"])
    delta_q5 = _mean_delta(rows, VARIANTS[4], VARIANTS[5], paths["weak_auroc"])
    fpr = float(np.mean([_variant_map(row)[VARIANTS[4]]["metrics"]["per_label"]["VFT"]["negative_fpr"] for row in rows]))
    checks = {
        "weak_auroc_vs_q2_ge_0p010": delta_q2["weak_auroc"][0] >= 0.010,
        "weak_recall_vs_q2_ge_0p010": delta_q2["weak_recall"][0] >= 0.010,
        "aupr_vs_q2_nonnegative": delta_q2["aupr"][0] >= 0.0,
        "weak_auroc_vs_capacity_ge_0p005": delta_q3[0] >= 0.005,
        "weak_auroc_vs_shuffled_ge_0p005": delta_q5[0] >= 0.005,
        "four_of_five_auroc_vs_q2": sum(value > 0 for value in delta_q2["weak_auroc"][1]) >= 4,
        "four_of_five_auroc_vs_shuffled": sum(value > 0 for value in delta_q5[1]) >= 4,
        "vft_fpr_le_0p20": fpr <= 0.20,
        "non_vft_logits_exact": all(row["non_vft_logit_max_abs"] == 0.0 for row in rows),
    }
    return {"passed": bool(all(checks.values())), "checks": checks,
            "delta_q4_vs_q2": {key: value[0] for key, value in delta_q2.items()},
            "weak_auroc_delta_vs_capacity": delta_q3[0], "weak_auroc_delta_vs_shuffled": delta_q5[0],
            "vft_fpr": fpr}


def _goal_rows(rows):
    per_seed = []
    for row in rows:
        variants = _variant_map(row)
        base = variants[VARIANTS[0]]["metrics"]
        candidate = variants[VARIANTS[4]]["metrics"]
        per_seed.append({
            "seed": row["seed"],
            "macro_aupr": candidate["macro_aupr"] - base["macro_aupr"],
            "macro_f1": candidate["macro_f1"] - base["macro_f1"],
            "target_weak_recall": candidate["target_mean_weak_recall"] - base["target_mean_weak_recall"],
            "target_weak_auroc": candidate["target_weak_vs_negative_auroc"] - base["target_weak_vs_negative_auroc"],
            "target_fpr": candidate["target_mean_negative_fpr"],
            "target_strong_recall": candidate["target_mean_strong_recall"] - base["target_mean_strong_recall"],
            "sawfm_aupr": candidate["per_label"]["SawFM"]["aupr"] - base["per_label"]["SawFM"]["aupr"],
            "sawfm_weak_recall": candidate["per_label"]["SawFM"]["weak_recall"] - base["per_label"]["SawFM"]["weak_recall"],
            "sawfm_present_macro_f1": candidate["subgroups"]["sawfm_present"]["macro_f1"] - base["subgroups"]["sawfm_present"]["macro_f1"],
            "sawfm_absent_macro_f1": candidate["subgroups"]["sawfm_absent"]["macro_f1"] - base["subgroups"]["sawfm_absent"]["macro_f1"],
            "target_aupr_delta": {
                label: candidate["per_label"][label]["aupr"] - base["per_label"][label]["aupr"]
                for label in ("MFT", "VFT", "ISRJ")
            },
        })
    mean = {
        key: float(np.mean([item[key] for item in per_seed]))
        for key in per_seed[0]
        if key not in ("seed", "target_aupr_delta")
    }
    target_aupr = {
        label: float(np.mean([item["target_aupr_delta"][label] for item in per_seed]))
        for label in ("MFT", "VFT", "ISRJ")
    }
    checks = {
        "macro_aupr_gain_ge_0p005": mean["macro_aupr"] >= 0.005,
        "macro_f1_gain_ge_0p010": mean["macro_f1"] >= 0.010,
        "target_weak_recall_gain_ge_0p030": mean["target_weak_recall"] >= 0.030,
        "target_weak_auroc_gain_ge_0p015": mean["target_weak_auroc"] >= 0.015,
        "sawfm_aupr_preserved": mean["sawfm_aupr"] >= -1e-12,
        "sawfm_weak_recall_preserved": mean["sawfm_weak_recall"] >= -1e-12,
        "target_aupr_decline_le_0p010": min(target_aupr.values()) >= -0.010,
        "target_mean_fpr_le_0p20": mean["target_fpr"] <= 0.20,
        "strong_recall_decline_le_0p010": mean["target_strong_recall"] >= -0.010,
        "positive_weak_recall_seeds_ge_4": sum(item["target_weak_recall"] > 0 for item in per_seed) >= 4,
        "sawfm_present_macro_f1_not_regressed": mean["sawfm_present_macro_f1"] >= -0.010,
        "sawfm_absent_macro_f1_not_regressed": mean["sawfm_absent_macro_f1"] >= -0.010,
        "non_target_logits_exact": all(row["non_vft_logit_max_abs"] == 0.0 for row in rows),
    }
    return {"passed": bool(all(checks.values())), "checks": checks,
            "mean_q4_vs_q0": mean, "target_aupr_delta": target_aupr, "per_seed": per_seed}


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
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in seeds]
    vft_gate = _vft_gate(rows) if not args.smoke else {"passed": False, "checks": {}, "evidence_level": "smoke"}
    unified_gate = _goal_rows(rows) if not args.smoke else {"passed": False, "checks": {}, "evidence_level": "smoke"}
    result = {
        "status": "completed", "experiment": "RouteA_H1B2_MFT_VFT_Unified_v3",
        "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
        "rows": rows, "vft_mechanism_gate": vft_gate, "unified_promotion_gate": unified_gate,
        "promotion_passed": bool(vft_gate["passed"] and unified_gate["passed"]),
        "next_action": "freeze_and_fresh_confirmation" if vft_gate["passed"] and unified_gate["passed"] else "stop_vft_integration",
        "locked_test_used": False, "test_v2_used": False, "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "route_a_h1_b2_mft_vft_unified_v3_summary.json"
    _write_json(output, result)
    print(json.dumps({"summary": str(output), "vft_gate": vft_gate, "unified_gate": unified_gate}, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--h1-b2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_h1_periodicity_increment")
    parser.add_argument("--mft-evidence-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3")
    parser.add_argument("--vft-ridge-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_ridge_stft_separability_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--mft-residual-cap", type=float, default=0.75)
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--weak-positive-weight", type=float, default=4.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.2)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

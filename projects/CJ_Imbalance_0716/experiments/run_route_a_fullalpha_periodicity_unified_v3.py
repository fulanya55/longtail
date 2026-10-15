"""Combine frozen FullAlpha1p5/RawSaw with the aligned ISRJ periodicity expert."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask
from cjlt.data.v73_dataset import read_labels, read_parameters
from experiments.run_route_a_controlled_lt import _mean_ci
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power, _predict
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state
from experiments.run_route_a_development_v3_mainstream_loss_baselines import _load_rawsaw, _route_rawsaw
from experiments.run_route_a_isrj_frozen_score_fusion_head_v3 import _fusion_features, _fit_head, _head_scores, _replace_isrj, _sample_weights
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _block_permutation, _block_shuffle, _extract
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import ISRJ_INDEX, _clipped_logit, _fit_score_stats, _load_expert, _standardize_scores
from experiments.run_route_a_layerwise_matched_negative_separability_v3 import _matched_view
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _evaluate, _load_roles, _raw_scores, _validate_config
from experiments.run_shared import _data_path, load_config

VARIANTS = ("U0_C6RawSaw", "U1_FullAlphaRawSaw", "U2_FullAlphaAlignedPeriodicity", "U3_FullAlphaBlockShuffled")
TARGET_LABELS = ("MFT", "VFT", "ISRJ")


def _metric(result, key):
    metrics = result["metrics"]
    if key in metrics:
        return float(metrics[key])
    label, field = key.split("__", 1)
    return float(metrics["per_label"][label][field])


def _compare(rows, candidate, baseline, seed_offset):
    keys = ("macro_aupr", "macro_f1", "target_mean_weak_recall", "target_mean_strong_recall", "target_mean_negative_fpr", "target_weak_vs_negative_auroc", "ISRJ__aupr", "ISRJ__weak_recall", "ISRJ__strong_recall", "ISRJ__negative_fpr", "ISRJ__weak_vs_negative_auroc", "SawFM__aupr", "SawFM__weak_recall")
    per_seed = []
    for row in rows:
        variants = {item["variant"]: item for item in row["variants"]}
        per_seed.append({"seed": row["seed"], **{key: _metric(variants[candidate], key) - _metric(variants[baseline], key) for key in keys}})
    aggregate = {key: _mean_ci([row[key] for row in per_seed], seed_offset + i) for i, key in enumerate(keys)}
    aggregate["positive_target_weak_seeds"] = int(sum(row["target_mean_weak_recall"] > 0 for row in per_seed))
    return {"candidate": candidate, "baseline": baseline, "per_seed": per_seed, "aggregate": aggregate}


def _run_seed(args, config, weak, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    states = {}
    for name in ("C6_Joint", "GradientContrastFixed1p5"):
        path = args.c6_root / f"seed_{seed}" / name / "best_model.pth"
        states[name] = (path, torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    base = {}
    for name, (_, state) in states.items():
        model = _load_model_from_state(f2_dir, device, state)
        base[name] = {role: _predict(model, roles[role]["signals"], device, args.eval_batch_size) for role in ALL_ROLES}
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    raw, raw_path = _load_rawsaw(args, seed, device)
    raw_scores = {role: _raw_scores(raw, roles[role]["signals"], device, args.raw_batch_size) for role in ALL_ROLES}
    del raw
    c6 = _route_rawsaw(base["C6_Joint"], raw_scores)
    full = _route_rawsaw(base["GradientContrastFixed1p5"], raw_scores)

    aligned_model, aligned_mean, aligned_std, aligned_hashes = _load_expert(args.periodicity_root, seed, "H_AlignedHGB")
    shuffled_model, shuffled_mean, shuffled_std, shuffled_hashes = _load_expert(args.periodicity_root, seed, "S_BlockShuffledHGB")
    permutation = _block_permutation(120, 20301000 + seed)[0]
    aligned_features = {role: _extract(roles[role]["signals"], device, args.feature_batch_size) for role in ALL_ROLES}
    shuffled_features = {role: _extract(_block_shuffle(roles[role]["signals"], permutation, args.block_size), device, args.feature_batch_size) for role in ALL_ROLES}
    aligned_raw, shuffled_raw = {}, {}
    for role in ALL_ROLES:
        aligned_raw[role] = _clipped_logit(aligned_model.predict_proba(((aligned_features[role] - aligned_mean) / np.maximum(aligned_std, 1e-5)).astype(np.float32))[:, 1])
        shuffled_raw[role] = _clipped_logit(shuffled_model.predict_proba(((shuffled_features[role] - shuffled_mean) / np.maximum(shuffled_std, 1e-5)).astype(np.float32))[:, 1])
    selected, _, matched = _matched_view(roles["fit"]["targets6"], weak[roles["fit"]["sample_ids"]], ISRJ_INDEX)
    aligned_stats = _fit_score_stats(aligned_raw["fit"][selected]); shuffled_stats = _fit_score_stats(shuffled_raw["fit"][selected])
    aligned_z = {role: _standardize_scores(aligned_raw[role], aligned_stats) for role in ALL_ROLES}
    shuffled_z = {role: _standardize_scores(shuffled_raw[role], shuffled_stats) for role in ALL_ROLES}
    sets = {"aligned": {role: _fusion_features(full[role], aligned_z[role]) for role in ALL_ROLES}, "shuffled": {role: _fusion_features(full[role], shuffled_z[role]) for role in ALL_ROLES}}
    labels = roles["fit"]["targets6"][:, ISRJ_INDEX].astype(np.uint8)
    weights = _sample_weights(roles["fit"]["targets6"], weak[roles["fit"]["sample_ids"]], args.weak_positive_weight)
    heads = {}
    for offset, name in enumerate(("aligned", "shuffled")):
        model, mean, std = _fit_head(sets[name]["fit"], labels, weights, 20346000 + seed + offset, args.c_value)
        out = root / f"head_{name}"; out.mkdir(parents=True, exist_ok=True); joblib.dump(model, out / "model.joblib"); np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
        heads[name] = (model, mean, std)
    logits = {
        VARIANTS[0]: c6,
        VARIANTS[1]: full,
        VARIANTS[2]: {role: _replace_isrj(full[role], _head_scores(heads["aligned"][0], sets["aligned"][role], heads["aligned"][1], heads["aligned"][2])) for role in ALL_ROLES},
        VARIANTS[3]: {role: _replace_isrj(full[role], _head_scores(heads["shuffled"][0], sets["shuffled"][role], heads["shuffled"][1], heads["shuffled"][2])) for role in ALL_ROLES},
    }
    non_isrj_error = max(float(np.max(np.abs(logits[VARIANTS[2]][role][:, :ISRJ_INDEX] - full[role][:, :ISRJ_INDEX]))) for role in ALL_ROLES)
    np.savez_compressed(root / "all_role_logits.npz", **{f"{variant}__{role}": logits[variant][role] for variant in VARIANTS for role in ALL_ROLES})
    variants = [_evaluate(name, logits[name], roles, weak, root / name, args) for name in VARIANTS]
    row = {"seed": seed, "status": "completed", "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path), "c6_checkpoint_sha256": _sha256(states["C6_Joint"][0]), "fullalpha_checkpoint_sha256": _sha256(states["GradientContrastFixed1p5"][0]), "rawsaw_checkpoint_sha256": _sha256(raw_path), "periodicity_expert_sha256": {"aligned": aligned_hashes, "shuffled": shuffled_hashes}, "matched_receipt": matched, "non_isrj_max_abs_error": non_isrj_error, "variants": variants, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(root / "seed_summary.json", row)
    return row


def run(args):
    config = load_config(args.config); _validate_config(config); data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]: raise ValueError("test-v3 is not sealed")
    labels = read_labels(data_path); values, _ = read_parameters(data_path); weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in args.seeds]
    comparisons = {"U2_vs_U0": _compare(rows, VARIANTS[2], VARIANTS[0], 20347000), "U2_vs_U1": _compare(rows, VARIANTS[2], VARIANTS[1], 20347100), "U2_vs_U3": _compare(rows, VARIANTS[2], VARIANTS[3], 20347200)}
    u20 = comparisons["U2_vs_U0"]["aggregate"]; u21 = comparisons["U2_vs_U1"]["aggregate"]; u23 = comparisons["U2_vs_U3"]["aggregate"]
    candidate_results = [{item["variant"]: item for item in row["variants"]}[VARIANTS[2]] for row in rows]
    target_aupr_ok = all(np.mean([_metric(result, f"{label}__aupr") - _metric({item["variant"]: item for item in row["variants"]}[VARIANTS[0]], f"{label}__aupr") for row, result in zip(rows, candidate_results)]) >= -0.010 for label in TARGET_LABELS)
    candidate_target_fpr = float(np.mean([_metric(result, "target_mean_negative_fpr") for result in candidate_results]))
    checks = {"macro_aupr_gain_ge_0p005": u20["macro_aupr"]["mean"] >= .005, "macro_f1_gain_ge_0p010": u20["macro_f1"]["mean"] >= .010, "target_weak_recall_gain_ge_0p030": u20["target_mean_weak_recall"]["mean"] >= .030, "target_weak_auroc_gain_ge_0p015": u20["target_weak_vs_negative_auroc"]["mean"] >= .015, "candidate_target_fpr_le_0p20": candidate_target_fpr <= .20, "target_aupr_noninferior": target_aupr_ok, "target_strong_recall_noninferior": u20["target_mean_strong_recall"]["mean"] >= -.010, "four_of_five_positive_target_weak": u20["positive_target_weak_seeds"] >= 4, "isrj_weak_auroc_vs_u1_ge_0p005": u21["ISRJ__weak_vs_negative_auroc"]["mean"] >= .005, "isrj_weak_recall_vs_u1_ge_0p010": u21["ISRJ__weak_recall"]["mean"] >= .010, "isrj_weak_auroc_vs_shuffle_ge_0p005": u23["ISRJ__weak_vs_negative_auroc"]["mean"] >= .005, "sawfm_exact": abs(u20["SawFM__aupr"]["mean"]) < 1e-12 and abs(u20["SawFM__weak_recall"]["mean"]) < 1e-12, "non_isrj_and_test_boundary_safe": all(row["non_isrj_max_abs_error"] == 0.0 and not row["test_v3_used"] for row in rows)}; checks = {key: bool(value) for key, value in checks.items()}
    result = {"status": "completed", "experiment": "RouteA_FullAlpha_Periodicity_Unified_v3", "seeds": args.seeds, "data_sha256": data_sha, "rows": rows, "comparisons": comparisons, "candidate_target_mean_fpr": candidate_target_fpr, "promotion_gate": {"passed": bool(all(checks.values())), "checks": checks}, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    summary = args.output_root / "route_a_fullalpha_periodicity_unified_v3_summary.json"; _write_json(summary, result); print(json.dumps({"status": "completed", "gate": result["promotion_gate"], "comparisons": {key: value["aggregate"] for key, value in comparisons.items()}, "summary": str(summary)}, ensure_ascii=False, indent=2)); return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/student_a/six_class_development_v3.yaml"); p.add_argument("--f2-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_dfms_iq_structure_screen"); p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection"); p.add_argument("--unified-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3"); p.add_argument("--periodicity-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_isrj_joint_periodicity_expert_v3"); p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_fullalpha_periodicity_unified_v3"); p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]); p.add_argument("--device",default=None); p.add_argument("--io-batch-size",type=int,default=256); p.add_argument("--eval-batch-size",type=int,default=128); p.add_argument("--raw-batch-size",type=int,default=64); p.add_argument("--feature-batch-size",type=int,default=64); p.add_argument("--raw-hidden-channels",type=int,default=16); p.add_argument("--raw-topk",type=int,default=16); p.add_argument("--raw-dropout",type=float,default=.1); p.add_argument("--block-size",type=int,default=25); p.add_argument("--c-value",type=float,default=.1); p.add_argument("--weak-positive-weight",type=float,default=4.); p.add_argument("--max-negative-fpr",type=float,default=.2); p.add_argument("--weak-threshold-db",type=float,default=10.); run(p.parse_args())


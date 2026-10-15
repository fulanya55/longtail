"""Fixed-capacity MLP control for aligned VFT physical evidence."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import ALL_ROLES, NON_VFT, VFT_INDEX, _extract_vft, _fit_score_stats, _fusion_features, _ridge_scores, _standardize_scores, _replace_vft  # noqa: E402
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import guarded_logits  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _evaluate, _load_roles, _validate_config  # noqa: E402
from experiments.run_route_a_vft_capacity_matched_mlp_unified_decision_ablation import _MLP1, _count_parameters, _fit_torch, _normalize, _scores  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402

ARMS = ("MLP1_NoVTSECapacity", "MLP1_AlignedVTSE", "MLP1_ShuffledVTSE")

def _base_only(logits):
    base = np.asarray(logits, dtype=np.float32)[:, VFT_INDEX]
    return np.stack([base, np.tanh(base), base ** 2, np.abs(base)], axis=1).astype(np.float32)

def _permute(x, seed):
    perm = np.random.default_rng(int(seed)).permutation(len(x))
    return x[perm], hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()

def _metric(variant, key): return float(variant["metrics"]["per_label"]["VFT"][key])
def _variant(row, arm): return next(item for item in row["variants"] if item["variant"] == arm)

def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; receipt = root / "seed_summary.json"
    if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    h1 = np.load(args.h1_b2_root / f"seed_{seed}" / "all_role_logits.npz"); mft = np.load(args.mft_evidence_root / f"seed_{seed}" / "experts" / "all_role_expert_evidence.npz")
    q1 = {role: np.asarray(h1[f"B2_H1AlignedPeriodicity__{role}"], dtype=np.float32) for role in ALL_ROLES}
    q2 = {role: guarded_logits(q1[role], mft[f"K1_AlignedPeakNegativeGuard__{role}"], args.mft_residual_cap)[0] for role in ALL_ROLES}
    ridge = np.load(args.vft_ridge_root / f"seed_{seed}" / "ridge_model.npz")
    raw = {role: _ridge_scores(_extract_vft(roles[role]["signals"], device, args.feature_batch_size), ridge) for role in ALL_ROLES}
    labels = roles["fit"]["targets6"][:, VFT_INDEX].astype(np.uint8); weak_fit = weak6[roles["fit"]["sample_ids"]][:, VFT_INDEX]
    stats = _fit_score_stats(raw["fit"][(labels == 0) | ((labels == 1) & weak_fit)])
    z = {role: _standardize_scores(raw[role], stats) for role in ALL_ROLES}; z_shuffled, digests = {}, {}
    for offset, role in enumerate(ALL_ROLES): z_shuffled[role], digests[role] = _permute(z[role], 2026084300 + seed * 10 + offset)
    sets = {"base": {role: _base_only(q2[role]) for role in ALL_ROLES}, "aligned": {role: _fusion_features(q2[role], z[role]) for role in ALL_ROLES}, "shuffled": {role: _fusion_features(q2[role], z_shuffled[role]) for role in ALL_ROLES}}
    weights = np.where((labels == 1) & weak_fit, args.weak_positive_weight, 1.0).astype(np.float32)
    logits, training = {}, {}
    for offset, (arm, key) in enumerate(zip(ARMS, ("base", "aligned", "shuffled"))):
        xfit, mean, std = _normalize(sets[key]["fit"], sets[key]["fit"]); xval, _, _ = _normalize(sets[key]["fit"], sets[key]["validation"])
        model, history = _fit_torch(xfit, labels, weights, xval, roles["validation"]["targets6"][:, VFT_INDEX], "mlp", 2026084400 + seed * 10 + offset, args, device)
        if _count_parameters(model) != 7: raise AssertionError("each MLP control must have exactly 7 parameters")
        logits[arm] = {role: _replace_vft(q2[role], _scores(model, _normalize(sets[key]["fit"], sets[key][role])[0], device)) for role in ALL_ROLES}
        training[arm] = {**history, "normalization_mean": mean.tolist(), "normalization_std": std.tolist()}
    if any(np.max(np.abs(logits[arm][role][:, NON_VFT] - q2[role][:, NON_VFT])) != 0 for arm in ARMS for role in ALL_ROLES): raise AssertionError("VFT-only arm changed non-VFT logits")
    variants = [_evaluate(arm, logits[arm], roles, weak6, root / arm, args) for arm in ARMS]
    row = {"status":"completed", "seed":int(seed), "data_sha256":data_sha, "role_indices_sha256":_sha256(role_path), "capacity_per_arm":7, "training":training, "permutation_sha256":digests, "non_vft_logits_exact":True, "variants":variants, "locked_test_used":False, "test_v2_used":False, "test_v3_used":False}
    _write_json(receipt, row); return row

def _compare(rows, baseline, seed_base):
    out = {}
    for offset, metric in enumerate(("weak_vs_negative_auroc", "aupr", "weak_recall", "strong_recall", "negative_fpr")):
        vals = [_metric(_variant(row, ARMS[1]), metric) - _metric(_variant(row, baseline), metric) for row in rows]
        out[metric] = {"summary":_mean_ci(vals, seed_base+offset), "values":vals, "positive_seeds":int(sum(v > 0 for v in vals))}
    return out

def run(args):
    config=load_config(args.config); _validate_config(config); path=_data_path(config); data_sha=_sha256(path)
    if data_sha.lower()!=EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal=json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[key] for key in ("access_authorized","analysis_allowed","used")): raise ValueError("test-v3 is not sealed")
    labels=read_labels(path); values,_=read_parameters(path); weak6=compute_weak_component_mask(labels,_component_power(values),args.weak_threshold_db)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); seeds=args.seeds[:1] if args.smoke else args.seeds
    rows=[_run_seed(args,config,weak6,seed,device,data_sha) for seed in seeds]
    if args.smoke: comparisons={}; checks={}
    else:
        comparisons={"aligned_minus_base":_compare(rows,ARMS[0],2026084500),"aligned_minus_shuffled":_compare(rows,ARMS[2],2026084600)}
        checks={"seven_parameters_each":all(row["capacity_per_arm"]==7 for row in rows),"non_vft_logits_exact":all(row["non_vft_logits_exact"] for row in rows)}
        for name, data in comparisons.items():
            checks[f"{name}_weak_auroc_ge_0p005"]=data["weak_vs_negative_auroc"]["summary"]["mean"]>=.005
            checks[f"{name}_aupr_nonnegative"]=data["aupr"]["summary"]["mean"]>=0
            checks[f"{name}_four_positive"]=data["weak_vs_negative_auroc"]["positive_seeds"]>=4
    payload={"status":"completed","experiment":"RouteA_VFTMLPCapacityPhysicsControlUnifiedDecisionAblation","smoke":bool(args.smoke),"seeds":list(seeds),"data_sha256":data_sha,"rows":rows,"comparisons":comparisons,"mechanism_gate":{"checks":checks,"passed":bool(checks and all(checks.values()))},"scope":"fixed_capacity_vft_physics_control_no_candidate_selection","locked_test_used":False,"test_v2_used":False,"test_v3_used":False}
    args.output_root.mkdir(parents=True,exist_ok=True); _write_json(args.output_root/"route_a_vft_mlp_capacity_physics_control_unified_decision_ablation_summary.json",payload); print(json.dumps(payload,ensure_ascii=False,indent=2)); return payload

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/student_a/six_class_development_v3.yaml"); p.add_argument("--c6-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_label_selective_projection"); p.add_argument("--h1-b2-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_development_v3_h1_periodicity_increment"); p.add_argument("--mft-evidence-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3"); p.add_argument("--vft-ridge-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_vft_ridge_stft_separability_v3"); p.add_argument("--test-v3-manifest",type=Path,default=PROJECT_ROOT/"frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); p.add_argument("--output-root",type=Path,default=PROJECT_ROOT/"outputs/tapc/route_a_vft_mlp_capacity_physics_control_unified_decision_ablation"); p.add_argument("--seeds",type=int,nargs="+",default=[2039,2040,2041,2042,2043]); p.add_argument("--resume",action="store_true"); p.add_argument("--smoke",action="store_true"); p.add_argument("--device",default=None); p.add_argument("--io-batch-size",type=int,default=256); p.add_argument("--feature-batch-size",type=int,default=32); p.add_argument("--mft-residual-cap",type=float,default=.75); p.add_argument("--weak-positive-weight",type=float,default=4.0); p.add_argument("--learning-rate",type=float,default=.01); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--max-epochs",type=int,default=200); p.add_argument("--patience",type=int,default=20); p.add_argument("--max-negative-fpr",type=float,default=.2); p.add_argument("--weak-threshold-db",type=float,default=10.0); return p.parse_args()
if __name__=="__main__": run(parse_args())

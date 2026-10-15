"""Construct frozen M5 plus aligned/shuffled periodicity variants on development-v3."""
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

VARIANTS = ("Q0_C6RawSaw", "Q1_M5RawSaw", "Q2_M5AlignedPeriodicity", "Q3_M5BlockShuffled")


def _run_seed(args, config, weak, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    receipt_path = root / "seed_summary.json"
    if args.resume and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    f2_dir = args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}"
    states = {}
    base = {}
    for name in ("C6_Joint", "M5_GradientIsolated"):
        path = args.c6_root / f"seed_{seed}" / name / "best_model.pth"
        state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        states[name] = (path, state)
        model = _load_model_from_state(f2_dir, device, state)
        base[name] = {role: _predict(model, roles[role]["signals"], device, args.eval_batch_size) for role in ALL_ROLES}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    raw, raw_path = _load_rawsaw(args, seed, device)
    raw_scores = {role: _raw_scores(raw, roles[role]["signals"], device, args.raw_batch_size) for role in ALL_ROLES}
    del raw
    c6 = _route_rawsaw(base["C6_Joint"], raw_scores)
    m5 = _route_rawsaw(base["M5_GradientIsolated"], raw_scores)
    aligned_model, aligned_mean, aligned_std, aligned_hashes = _load_expert(args.periodicity_root, seed, "H_AlignedHGB")
    shuffled_model, shuffled_mean, shuffled_std, shuffled_hashes = _load_expert(args.periodicity_root, seed, "S_BlockShuffledHGB")
    permutation = _block_permutation(120, 20301000 + seed)[0]
    aligned_features = {role: _extract(roles[role]["signals"], device, args.feature_batch_size) for role in ALL_ROLES}
    shuffled_features = {role: _extract(_block_shuffle(roles[role]["signals"], permutation, args.block_size), device, args.feature_batch_size) for role in ALL_ROLES}
    aligned_raw = {}
    shuffled_raw = {}
    for role in ALL_ROLES:
        aligned_raw[role] = _clipped_logit(aligned_model.predict_proba(((aligned_features[role]-aligned_mean)/np.maximum(aligned_std,1e-5)).astype(np.float32))[:,1])
        shuffled_raw[role] = _clipped_logit(shuffled_model.predict_proba(((shuffled_features[role]-shuffled_mean)/np.maximum(shuffled_std,1e-5)).astype(np.float32))[:,1])
    selected, _, matched = _matched_view(roles["fit"]["targets6"], weak[roles["fit"]["sample_ids"]], ISRJ_INDEX)
    aligned_stats = _fit_score_stats(aligned_raw["fit"][selected])
    shuffled_stats = _fit_score_stats(shuffled_raw["fit"][selected])
    aligned_z = {role: _standardize_scores(aligned_raw[role], aligned_stats) for role in ALL_ROLES}
    shuffled_z = {role: _standardize_scores(shuffled_raw[role], shuffled_stats) for role in ALL_ROLES}
    sets = {
        "aligned": {role: _fusion_features(m5[role], aligned_z[role]) for role in ALL_ROLES},
        "shuffled": {role: _fusion_features(m5[role], shuffled_z[role]) for role in ALL_ROLES},
    }
    labels = roles["fit"]["targets6"][:, ISRJ_INDEX].astype(np.uint8)
    weights = _sample_weights(roles["fit"]["targets6"], weak[roles["fit"]["sample_ids"]], args.weak_positive_weight)
    heads = {}
    for offset, name in enumerate(("aligned", "shuffled")):
        model, mean, std = _fit_head(sets[name]["fit"], labels, weights, 20356000 + seed + offset, args.c_value)
        head_root = root / f"head_{name}"
        head_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, head_root / "model.joblib")
        np.savez_compressed(head_root / "normalization.npz", mean=mean, std=std)
        heads[name] = (model, mean, std)
    logits = {
        VARIANTS[0]: c6,
        VARIANTS[1]: m5,
        VARIANTS[2]: {role: _replace_isrj(m5[role], _head_scores(heads["aligned"][0], sets["aligned"][role], heads["aligned"][1], heads["aligned"][2])) for role in ALL_ROLES},
        VARIANTS[3]: {role: _replace_isrj(m5[role], _head_scores(heads["shuffled"][0], sets["shuffled"][role], heads["shuffled"][1], heads["shuffled"][2])) for role in ALL_ROLES},
    }
    np.savez_compressed(root / "all_role_logits.npz", **{f"{variant}__{role}": logits[variant][role] for variant in VARIANTS for role in ALL_ROLES})
    variants = [_evaluate(name, logits[name], roles, weak, root / name, args) for name in VARIANTS]
    row = {
        "seed": seed, "status": "completed", "data_sha256": data_sha,
        "role_indices_sha256": _sha256(role_path),
        "c6_checkpoint_sha256": _sha256(states["C6_Joint"][0]),
        "m5_checkpoint_sha256": _sha256(states["M5_GradientIsolated"][0]),
        "rawsaw_checkpoint_sha256": _sha256(raw_path),
        "periodicity_expert_sha256": {"aligned": aligned_hashes, "shuffled": shuffled_hashes},
        "score_stats": {"aligned": aligned_stats, "block_shuffled": shuffled_stats},
        "matched_receipt": matched, "variants": variants,
        "candidate_fixed_for_development_v8": VARIANTS[2],
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    _write_json(receipt_path, row)
    return row


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["analysis_allowed"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 is not sealed")
    labels = read_labels(data_path)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in args.seeds]
    result = {
        "status": "completed", "experiment": "RouteA_M5_Periodicity_Unified_v3",
        "evidence_role": "candidate_construction_only_development_v8_fixed_before_confirmation",
        "candidate": VARIANTS[2], "rows": rows,
        "external_refit": False, "locked_test_used": False, "test_v3_used": False,
    }
    output = args.output_root / "route_a_m5_periodicity_unified_v3_summary.json"
    _write_json(output, result)
    print(json.dumps({"status": "completed", "candidate": result["candidate"], "summary": str(output)}, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_m5_periodicity_unified_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--c-value", type=float, default=0.1)
    parser.add_argument("--weak-positive-weight", type=float, default=4.0)
    parser.add_argument("--max-negative-fpr", type=float, default=0.2)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

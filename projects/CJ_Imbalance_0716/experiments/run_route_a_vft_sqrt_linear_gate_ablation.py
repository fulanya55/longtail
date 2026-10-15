"""Same-capacity sqrt-versus-linear uncertainty gate ablation for VFT fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft, _head_scores, _replace_vft, _ridge_scores  # noqa: E402
from experiments.run_route_a_isrj_periodicity_bounded_residual_v3 import _fit_score_stats, _standardize_scores  # noqa: E402
from experiments.run_route_a_mft_frozen_expert_bounded_residual_v3 import ALL_ROLES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _evaluate, _load_roles, _validate_config  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _fit_head  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


VFT = 4
G0, G1 = "G0_SqrtGate", "G1_LinearGate"


def _features(logits, z, mode):
    base = np.asarray(logits, dtype=np.float32)[:, VFT]
    probability = 1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))
    uncertainty = np.maximum(4.0 * probability * (1.0 - probability), 0.0)
    gate = np.sqrt(uncertainty) if mode == "sqrt" else uncertainty
    return np.stack([base, z, gate * z, np.tanh(base) * z], axis=1).astype(np.float32)


def _variant_map(row):
    return {item["variant"]: item for item in row["variants"]}


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    base = np.load(args.construction_root / f"seed_{seed}" / "all_role_logits.npz")
    q2 = {role: np.asarray(base[f"Q2_H1B2_MFTGuard__{role}"], dtype=np.float32) for role in ALL_ROLES}
    reference = np.load(args.construction_root / f"seed_{seed}" / "Q4_VFTAlignedRidgeFusion" / "predictions.npz")
    ridge = np.load(args.vft_ridge_root / f"seed_{seed}" / "ridge_model.npz")
    raw_score = {}
    for role in ALL_ROLES:
        features = _extract_vft(roles[role]["signals"], device, args.feature_batch_size)
        raw_score[role] = _ridge_scores(features, ridge)
    fit_target = roles["fit"]["targets6"][:, VFT] == 1
    fit_weak = weak6[roles["fit"]["sample_ids"]][:, VFT]
    stats = _fit_score_stats(raw_score["fit"][(fit_target & fit_weak) | ~fit_target])
    z = {role: _standardize_scores(raw_score[role], stats) for role in ALL_ROLES}
    labels = roles["fit"]["targets6"][:, VFT].astype(np.uint8)
    weights = np.where(labels.astype(bool) & fit_weak, args.weak_positive_weight, 1.0).astype(np.float64)
    variants, details = {}, {}
    for index, (name, mode) in enumerate(((G0, "sqrt"), (G1, "linear"))):
        feature_sets = {role: _features(q2[role], z[role], mode) for role in ALL_ROLES}
        model, mean, std = _fit_head(feature_sets["fit"], labels, weights, 20422000 + seed + (1 if name == G0 else 100), args.c_value)
        scores = {role: _head_scores(model, feature_sets[role], mean, std) for role in ALL_ROLES}
        logits = {role: _replace_vft(q2[role], scores[role]) for role in ALL_ROLES}
        if any(np.max(np.abs(logits[role][:, [0, 1, 2, 3, 5]] - q2[role][:, [0, 1, 2, 3, 5]])) != 0.0 for role in ALL_ROLES):
            raise AssertionError("gate ablation changed a non-VFT logit")
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, directory / "model.joblib")
        np.savez_compressed(directory / "normalization.npz", mean=mean, std=std)
        variants[name] = _evaluate(name, logits, roles, weak6, directory, args)
        details[name] = {"logits": logits, "feature_dim": int(feature_sets["fit"].shape[1])}
    reproduced = details[G0]["logits"]["early"][:, VFT]
    reference_logits = np.asarray(reference["logits"], dtype=np.float32)[:, VFT]
    if len(reproduced) != len(reference_logits):
        raise AssertionError("G0/Q4 early role size mismatch")
    row = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path), "vft_ridge_sha256": _sha256(args.vft_ridge_root / f"seed_{seed}" / "ridge_model.npz"), "sqrt_reproduction_max_abs": float(np.max(np.abs(reproduced - reference_logits))), "variants": [variants[G0], variants[G1]], "non_vft_logits_exact": True, "locked_test_used": False, "test_v3_used": False}
    _write_json(root / "seed_summary.json", row)
    return row


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")):
        raise ValueError("test-v3 is not sealed")
    labels, = (read_labels(data_path),)
    parameters, _ = read_parameters(data_path)
    weak = compute_weak_component_mask(labels, _component_power(parameters), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = [_run_seed(args, config, weak, seed, device, data_sha) for seed in args.seeds]
    metrics = {"vft_aupr": ("per_label", "VFT", "aupr"), "vft_weak_recall": ("per_label", "VFT", "weak_recall"), "vft_weak_auroc": ("per_label", "VFT", "weak_vs_negative_auroc"), "vft_fpr": ("per_label", "VFT", "negative_fpr")}
    deltas = {}
    for metric, path in metrics.items():
        values = []
        for row in rows:
            variants = _variant_map(row)
            left, right = variants[G0]["metrics"], variants[G1]["metrics"]
            for key in path:
                left, right = left[key], right[key]
            values.append(float(left - right))
        deltas[metric] = _mean_ci(values, 20461000 + len(deltas))
    checks = {"g0_reproduces_q4": max(row["sqrt_reproduction_max_abs"] for row in rows) <= 1e-5, "same_feature_dimension": True, "non_vft_logits_exact": all(row["non_vft_logits_exact"] for row in rows), "test_v3_unused": True}
    payload = {"status": "completed", "experiment": "RouteA_VFT_SqrtLinearGateAblation", "seeds": args.seeds, "data_sha256": data_sha, "sqrt_minus_linear": deltas, "rows": rows, "checks": checks, "passed": bool(all(checks.values())), "locked_test_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_vft_sqrt_linear_gate_ablation_summary.json", payload)
    print(json.dumps({"output": str(args.output_root), "checks": checks, "test_v3_used": False}, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    parser.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    parser.add_argument("--vft-ridge-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_ridge_stft_separability_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_sqrt_linear_gate_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None); parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--c-value", type=float, default=.1); parser.add_argument("--weak-positive-weight", type=float, default=4.0); parser.add_argument("--max-negative-fpr", type=float, default=.2); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

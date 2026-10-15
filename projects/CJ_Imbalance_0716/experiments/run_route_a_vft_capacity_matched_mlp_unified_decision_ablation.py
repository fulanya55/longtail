"""Capacity-matched VFT fusion MLP control under the frozen unified decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import (  # noqa: E402
    ALL_ROLES, NON_VFT, VFT_INDEX, _extract_vft, _fit_score_stats, _fusion_features,
    _ridge_scores, _standardize_scores, _replace_vft,
)
from experiments.run_route_a_u2_mft_peak_count_negative_guard_v3 import guarded_logits  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import EXPECTED_V3_SHA256, _evaluate, _load_roles, _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


ARMS = ("Linear6FixedCapacity", "MLP1CapacityControl")


class _MLP1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(4, 1)
        self.output = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.output(torch.tanh(self.hidden(x))).squeeze(-1)


class _Linear6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Linear(6, 1)

    def forward(self, x):
        return self.output(x).squeeze(-1)


def _count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _fixed6(base4):
    x = np.asarray(base4, dtype=np.float32)
    return np.column_stack([x, x[:, 0] ** 2, x[:, 1] ** 2]).astype(np.float32)


def _normalize(fit, role):
    mean = fit.mean(axis=0).astype(np.float32)
    std = np.maximum(fit.std(axis=0), 1e-5).astype(np.float32)
    return ((role - mean[None, :]) / std[None, :]).astype(np.float32), mean, std


def _fit_torch(features, labels, weights, validation_features, validation_labels, kind, seed, args, device):
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    model = _Linear6() if kind == "linear" else _MLP1()
    if _count_parameters(model) != 7:
        raise AssertionError("capacity match must be exactly seven trainable parameters")
    model = model.to(device)
    x = torch.from_numpy(features).to(device); y = torch.from_numpy(labels.astype(np.float32)).to(device)
    w = torch.from_numpy(weights.astype(np.float32)).to(device)
    vx = torch.from_numpy(validation_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state, best_ap, stale = None, -np.inf, 0
    for epoch in range(args.max_epochs):
        model.train(); optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none") * w).mean()
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): score = model(vx).detach().cpu().numpy()
        ap = float(average_precision_score(validation_labels, score))
        if ap > best_ap + 1e-12:
            best_ap, stale = ap, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state); model.eval()
    return model, {"best_validation_aupr": best_ap, "selected_epoch": epoch - stale + 1, "epochs_ran": epoch + 1, "trainable_parameters": _count_parameters(model)}


def _scores(model, features, device):
    with torch.no_grad(): return model(torch.from_numpy(features).to(device)).cpu().numpy().astype(np.float32)


def _metric(item, name):
    return float(item["metrics"]["per_label"]["VFT"][name])


def _run_seed(args, config, weak6, seed, device, data_sha):
    root = args.output_root / f"seed_{seed}"; receipt = root / "seed_summary.json"
    if args.resume and receipt.is_file(): return json.loads(receipt.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    roles, role_path = _load_roles(args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size)
    construction = np.load(args.construction_root / f"seed_{seed}" / "all_role_logits.npz")
    h1 = np.load(args.h1_b2_root / f"seed_{seed}" / "all_role_logits.npz")
    mft = np.load(args.mft_evidence_root / f"seed_{seed}" / "experts" / "all_role_expert_evidence.npz")
    q1 = {role: np.asarray(h1[f"B2_H1AlignedPeriodicity__{role}"], dtype=np.float32) for role in ALL_ROLES}
    q2 = {role: guarded_logits(q1[role], mft[f"K1_AlignedPeakNegativeGuard__{role}"], args.mft_residual_cap)[0] for role in ALL_ROLES}
    reference = {role: np.asarray(construction[f"Q4_VFTAlignedRidgeFusion__{role}"], dtype=np.float32) for role in ALL_ROLES}
    if any(np.max(np.abs(reference[role][:, NON_VFT] - q2[role][:, NON_VFT])) != 0.0 for role in ALL_ROLES):
        raise AssertionError("frozen sources disagree on non-VFT logits")
    ridge = np.load(args.vft_ridge_root / f"seed_{seed}" / "ridge_model.npz")
    feature = {role: _extract_vft(roles[role]["signals"], device, args.feature_batch_size) for role in ALL_ROLES}
    raw = {role: _ridge_scores(feature[role], ridge) for role in ALL_ROLES}
    labels = roles["fit"]["targets6"][:, VFT_INDEX].astype(np.uint8)
    fit_weak = weak6[roles["fit"]["sample_ids"]][:, VFT_INDEX]
    stats = _fit_score_stats(raw["fit"][(labels == 0) | ((labels == 1) & fit_weak)])
    z = {role: _standardize_scores(raw[role], stats) for role in ALL_ROLES}
    base4 = {role: _fusion_features(q2[role], z[role]) for role in ALL_ROLES}
    sets = {"linear": {role: _fixed6(base4[role]) for role in ALL_ROLES}, "mlp": base4}
    weights = np.where((labels == 1) & fit_weak, args.weak_positive_weight, 1.0).astype(np.float32)
    logits, train = {}, {}
    for offset, (arm, kind) in enumerate(((ARMS[0], "linear"), (ARMS[1], "mlp"))):
        xfit, mean, std = _normalize(sets[kind]["fit"], sets[kind]["fit"])
        xval, _, _ = _normalize(sets[kind]["fit"], sets[kind]["validation"])
        model, history = _fit_torch(xfit, labels, weights, xval, roles["validation"]["targets6"][:, VFT_INDEX], kind, 2026084100 + seed * 10 + offset, args, device)
        standardized = {role: _normalize(sets[kind]["fit"], sets[kind][role])[0] for role in ALL_ROLES}
        logits[arm] = {role: _replace_vft(q2[role], _scores(model, standardized[role], device)) for role in ALL_ROLES}
        train[arm] = {**history, "normalization_mean": mean.tolist(), "normalization_std": std.tolist()}
    for arm in ARMS:
        if any(np.max(np.abs(logits[arm][role][:, NON_VFT] - q2[role][:, NON_VFT])) != 0.0 for role in ALL_ROLES):
            raise AssertionError("capacity control changed a non-VFT logit")
    variants = [_evaluate(arm, logits[arm], roles, weak6, root / arm, args) for arm in ARMS]
    payload = {"status": "completed", "seed": int(seed), "data_sha256": data_sha, "role_indices_sha256": _sha256(role_path),
               "construction_logits_sha256": _sha256(args.construction_root / f"seed_{seed}" / "all_role_logits.npz"),
               "capacity": {"linear6": 7, "mlp1": 7}, "training": train, "non_vft_logits_exact": True, "variants": variants,
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(receipt, payload); return payload


def _summary(rows):
    measures = ("weak_vs_negative_auroc", "aupr", "weak_recall", "strong_recall", "negative_fpr")
    comparison = {}
    for offset, name in enumerate(measures):
        values = [_metric(next(x for x in row["variants"] if x["variant"] == ARMS[1]), name) - _metric(next(x for x in row["variants"] if x["variant"] == ARMS[0]), name) for row in rows]
        comparison[name] = {"summary": _mean_ci(values, 2026084200 + offset), "values": values, "positive_seeds": int(sum(v > 0 for v in values))}
    checks = {"exact_seven_parameter_match": all(row["capacity"] == {"linear6": 7, "mlp1": 7} for row in rows),
              "non_vft_logits_exact": all(row["non_vft_logits_exact"] for row in rows),
              "mlp_weak_auroc_gain_lt_0p005": comparison["weak_vs_negative_auroc"]["summary"]["mean"] < .005,
              "mlp_aupr_not_higher": comparison["aupr"]["summary"]["mean"] <= 0.0,
              "mlp_positive_weak_auroc_seeds_lt_4": comparison["weak_vs_negative_auroc"]["positive_seeds"] < 4}
    return comparison, {"checks": checks, "capacity_explanation_rejected": bool(all(checks.values()))}


def run(args):
    config = load_config(args.config); _validate_config(config)
    path = _data_path(config); data_sha = _sha256(path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))["test_v3"]
    if any(seal[key] for key in ("access_authorized", "analysis_allowed", "used")): raise ValueError("test-v3 is not sealed")
    labels = read_labels(path); values, _ = read_parameters(path)
    weak6 = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, weak6, seed, device, data_sha) for seed in seeds]
    comparison, gate = ({}, {"capacity_explanation_rejected": False, "checks": {}}) if args.smoke else _summary(rows)
    payload = {"status": "completed", "experiment": "RouteA_VFTCapacityMatchedMLPUnifiedDecisionAblation", "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "rows": rows, "mlp_minus_linear6": comparison, "mechanism_gate": gate,
               "scope": "current_frozen_decision_capacity_ablation_no_candidate_selection", "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True); _write_json(args.output_root / "route_a_vft_capacity_matched_mlp_unified_decision_ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml")
    p.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection")
    p.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3")
    p.add_argument("--h1-b2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_h1_periodicity_increment")
    p.add_argument("--mft-evidence-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_u2_mft_peak_count_negative_guard_v3")
    p.add_argument("--vft-ridge-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_ridge_stft_separability_v3")
    p.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_vft_capacity_matched_mlp_unified_decision_ablation")
    p.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043]); p.add_argument("--resume", action="store_true"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None); p.add_argument("--io-batch-size", type=int, default=256); p.add_argument("--feature-batch-size", type=int, default=32)
    p.add_argument("--mft-residual-cap", type=float, default=.75); p.add_argument("--weak-positive-weight", type=float, default=4.0); p.add_argument("--learning-rate", type=float, default=.01); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--max-epochs", type=int, default=200); p.add_argument("--patience", type=int, default=20); p.add_argument("--max-negative-fpr", type=float, default=.2); p.add_argument("--weak-threshold-db", type=float, default=10.0)
    return p.parse_args()


if __name__ == "__main__": run(parse_args())

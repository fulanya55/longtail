"""Same-capacity three-stratum balanced linear head on development-v3."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import F2_MODEL_NAME, _component_power  # noqa: E402
from experiments.run_route_a_development_v3_label_selective_projection import _load_model_from_state  # noqa: E402
from experiments.run_route_a_stage23_identifiability_v2 import TARGET_INDICES, TARGET_NAMES  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    _apply_fusion,
    _evaluate,
    _extract_backbone,
    _load_roles,
    _probe_scores,
    _raw_scores,
    _standardize,
    _train_generic_fusion,
    _validate_config,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


LINEAR_VARIANT = "C4_SameCapacity_NoVisibility"
CANDIDATE = "T1_ThreeStratumBalancedHead"
VARIANTS = (LINEAR_VARIANT, CANDIDATE)


def _stratum_weights(targets3, weak3):
    targets = np.asarray(targets3, dtype=np.float32)
    weak = np.asarray(weak3, dtype=bool)
    result = np.zeros_like(targets, dtype=np.float32)
    counts = {}
    for local, name in enumerate(TARGET_NAMES):
        positive = targets[:, local] > .5
        masks = {
            "weak": positive & weak[:, local],
            "strong": positive & ~weak[:, local],
            "negative": ~positive,
        }
        counts[name] = {key: int(mask.sum()) for key, mask in masks.items()}
        if any(value <= 0 for value in counts[name].values()):
            raise ValueError(f"empty visibility stratum for {name}")
        for mask in masks.values():
            result[mask, local] = 1.0 / (3.0 * float(mask.sum()))
    return result, counts


def _validation_key(logits, targets6):
    score = logits.detach().cpu().numpy()
    target = targets6[:, TARGET_INDICES]
    return (
        float(np.mean([roc_auc_score(target[:, i], score[:, i]) for i in range(3)])),
        float(np.mean([average_precision_score(target[:, i], score[:, i]) for i in range(3)])),
    )


def _train_head(fit_x, validation_x, fit_targets6, validation_targets6,
                fit_weak6, seed, device, args):
    train_seed = 971000 + int(seed)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    y_fit = fit_targets6[:, TARGET_INDICES].astype(np.float32)
    weights, counts = _stratum_weights(y_fit, fit_weak6[:, TARGET_INDICES])
    model = nn.Linear(fit_x.shape[1], 3).to(device)
    dataset = TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(y_fit),
                            torch.from_numpy(weights))
    loader = DataLoader(dataset, batch_size=args.probe_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_learning_rate,
                                  weight_decay=args.weight_decay)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.probe_epochs
    for epoch in range(1, int(epochs) + 1):
        model.train(); losses = []
        for x, y, weight in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device)); y = y.to(device); weight = weight.to(device)
            raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (raw * weight).sum() / weight.sum().clamp_min(1e-12)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_tensor)
        key = _validation_key(validation_logits, validation_targets6)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_target_macro_auroc": key[0],
                        "validation_target_macro_aupr": key[1]})
        if best_key is None or key[0] > best_key[0] + 1e-8:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(args.probe_patience):
                break
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "train_seed": train_seed,
                            "validation_target_macro_auroc": best_key[0],
                            "validation_target_macro_aupr": best_key[1],
                            "stratum_counts": counts}


def _read_c4(seed, root):
    path = root / f"seed_{seed}" / LINEAR_VARIANT / "metrics.json"
    return {"variant": LINEAR_VARIANT, "metrics": json.loads(path.read_text(encoding="utf-8"))}


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _run_seed(args, config, labels6, weak6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(args.c6_root, seed, _data_path(config),
                                           config["normalization"], args.io_batch_size)
    checkpoint_path = args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
    model = _load_model_from_state(args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}",
                                   device, state)
    extracted = _extract_backbone(model, roles, device, args.feature_batch_size, args.topk)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    stats = {role: row["temporal_stats"] for role, row in extracted.items()}
    standardized, mean, std = _standardize(stats["fit"], stats)
    head, history, head_info = _train_head(
        standardized["fit"], standardized["validation"],
        roles["fit"]["targets6"], roles["validation"]["targets6"],
        weak6[roles["fit"]["sample_ids"]], seed, device, args,
    )
    scores = {role: _probe_scores(head, value, device) for role, value in standardized.items()}
    center = scores["fit"].mean(axis=0).astype(np.float32)
    scale = np.maximum(scores["fit"].std(axis=0), 1e-6).astype(np.float32)
    normalized = {role: ((value-center[None, :])/scale[None, :]).astype(np.float32)
                  for role, value in scores.items()}
    combined = ("fit", "validation")
    base_fitval = np.concatenate([extracted[role]["base_logits"] for role in combined])
    expert_fitval = np.concatenate([normalized[role] for role in combined])
    targets_fitval = np.concatenate([roles[role]["targets6"] for role in combined])
    fit = np.arange(len(roles["fit"]["sample_ids"]), dtype=np.int64)
    validation = np.arange(len(fit), len(targets_fitval), dtype=np.int64)
    fusion, fusion_history, fusion_info = _train_generic_fusion(
        base_fitval, expert_fitval, targets_fitval, fit, validation, seed, device, args
    )
    fused = {role: _apply_fusion(fusion, extracted[role]["base_logits"], value, device)[0]
             for role, value in normalized.items()}
    raw_state = torch.load(args.unified_root/f"seed_{seed}"/"checkpoints"/"rawsaw.pth",
                           map_location="cpu", weights_only=False)
    raw = RawStage1SawFMFrontend(mode="raw", hidden_channels=args.raw_hidden_channels,
                                 topk=args.raw_topk, dropout=args.raw_dropout).to(device)
    raw.load_state_dict(raw_state["model_state_dict"], strict=True)
    raw_scores = {role: _raw_scores(raw, row["signals"], device, args.raw_batch_size)
                  for role, row in roles.items()}
    logits = {}
    for role, value in fused.items():
        combined_logits = value.copy(); combined_logits[:, 1] = raw_scores[role]
        logits[role] = combined_logits
    result = _evaluate(CANDIDATE, logits, roles, weak6, seed_root/CANDIDATE, args)
    torch.save({"model_state_dict": head.state_dict(), "mean": mean, "std": std,
                **head_info}, seed_root/"three_stratum_head.pth")
    torch.save({"model_state_dict": fusion.state_dict(), "center": center, "scale": scale,
                **fusion_info}, seed_root/"fusion.pth")
    _write_history(seed_root/"head_history.csv", history)
    _write_history(seed_root/"fusion_history.csv", fusion_history)
    payload = {"seed": int(seed), "status": "completed", "smoke": bool(args.smoke),
               "data_sha256": data_sha, "role_indices_sha256": _sha256(role_indices_path),
               "c6_checkpoint_sha256": _sha256(checkpoint_path),
               "parameter_count": int(sum(p.numel() for p in head.parameters())) +
                                  int(sum(p.numel() for p in fusion.parameters())),
               "head_info": head_info, "fusion_info": fusion_info,
               "variants": [_read_c4(seed, args.unified_root), result],
               "same_capacity_as_c4": True, "locked_test_used": False,
               "test_v2_used": False, "test_v3_used": False}
    _write_json(summary_path, payload); return payload


def _flat(metrics):
    return {"macro_aupr": metrics["macro_aupr"], "macro_f1": metrics["macro_f1"],
            "weak_recall": metrics["target_mean_weak_recall"],
            "weak_auroc": metrics["target_weak_vs_negative_auroc"],
            "strong_recall": metrics["target_mean_strong_recall"],
            "fpr": metrics["target_mean_negative_fpr"],
            "sawfm_aupr": metrics["per_label"]["SawFM"]["aupr"],
            "MFT_aupr": metrics["per_label"]["MFT"]["aupr"],
            "VFT_aupr": metrics["per_label"]["VFT"]["aupr"],
            "ISRJ_aupr": metrics["per_label"]["ISRJ"]["aupr"]}


def _vmap(row): return {item["variant"]: item["metrics"] for item in row["variants"]}


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(labels6, _component_power(values), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, labels6, weak6, seed, device, data_sha) for seed in seeds]
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        values = [_flat(_vmap(row)[variant]) for row in rows]
        aggregate[variant] = {name: _mean_ci([value[name] for value in values], 20271100+vi*100+i)
                              for i, name in enumerate(values[0])}
    paired = []
    for row in rows:
        c, b = _flat(_vmap(row)[CANDIDATE]), _flat(_vmap(row)[LINEAR_VARIANT])
        paired.append({"seed": row["seed"], **{name: c[name]-b[name] for name in c}})
    comparison = {name: _mean_ci([row[name] for row in paired], 20271400+i)
                  for i, name in enumerate(_flat(_vmap(rows[0])[CANDIDATE]))}
    comparison["positive_weak_recall_seeds"] = int(sum(row["weak_recall"] > 0 for row in paired))
    if args.smoke: gate = {"checks": {}, "passed": False}
    else:
        checks = {"macro_aupr_gain_0p005": comparison["macro_aupr"]["mean"] >= .005,
                  "macro_f1_gain_0p010": comparison["macro_f1"]["mean"] >= .010,
                  "weak_recall_gain_0p030": comparison["weak_recall"]["mean"] >= .030,
                  "weak_auroc_gain_0p015": comparison["weak_auroc"]["mean"] >= .015,
                  "strong_recall_noninferior": comparison["strong_recall"]["mean"] >= -.010,
                  "target_fpr_cap": aggregate[CANDIDATE]["fpr"]["mean"] <= .20,
                  "per_target_aupr_safe": all(comparison[f"{name}_aupr"]["mean"] >= -.010 for name in TARGET_NAMES),
                  "four_of_five_positive": comparison["positive_weak_recall_seeds"] >= 4,
                  "sawfm_preserved": abs(comparison["sawfm_aupr"]["mean"]) <= 1e-8,
                  "capacity_and_test_safe": all(row["same_capacity_as_c4"] and not row["test_v3_used"] for row in rows)}
        gate = {"checks": checks, "passed": bool(all(checks.values()))}
    payload = {"status": "completed", "experiment": "RouteA_ThreeStratumBalancedHead_v3",
               "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "aggregate": aggregate, "candidate_minus_c4": {"per_seed": paired, "aggregate": comparison},
               "promotion_gate": gate, "promoted": bool(gate["passed"]),
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root/"route_a_three_stratum_balanced_head_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_dfms_iq_structure_screen")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_three_stratum_balanced_head_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043])
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64); parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16); parser.add_argument("--raw-dropout", type=float, default=.1)
    parser.add_argument("--topk", type=int, default=8); parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3); parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--probe-patience", type=int, default=10); parser.add_argument("--fusion-learning-rate", type=float, default=1e-2)
    parser.add_argument("--fusion-epochs", type=int, default=200); parser.add_argument("--fusion-patience", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--gradient-clip", type=float, default=5.)
    parser.add_argument("--residual-cap", type=float, default=.50); parser.add_argument("--residual-l2-weight", type=float, default=.02)
    parser.add_argument("--bias-l2-weight", type=float, default=.05); parser.add_argument("--max-negative-fpr", type=float, default=.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())

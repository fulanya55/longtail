"""Same-capacity presence/visibility auxiliary-head validation on development-v3."""

from __future__ import annotations

import argparse
import copy
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
    EXPECTED_V3_SHA256, _apply_fusion, _evaluate, _extract_backbone, _load_roles,
    _probe_scores, _raw_scores, _standardize, _validate_config, _write_history,
    _train_generic_fusion,
)
from experiments.run_route_a_visibility_gap_frozen_probe_v3 import _relative_power_gap  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402


BASE_VARIANT = "C0_F2_C6"
LINEAR_VARIANT = "C4_SameCapacity_NoVisibility"
NO_AUX_VARIANT = "D1_LowRank_NoAux"
AUX_VARIANT = "D2_LowRank_VisibilityAux"


class PresenceVisibilityHead(nn.Module):
    def __init__(self, input_dim=512, rank=32):
        super().__init__()
        self.adapter = nn.Linear(int(input_dim), int(rank))
        self.presence = nn.Linear(int(rank), 3)
        self.visibility = nn.Linear(int(rank), 3)

    def forward(self, values):
        shared = F.gelu(self.adapter(values))
        return self.presence(shared), self.visibility(shared)


def _dual_loss(presence_logits, visibility_prediction, targets3, gap3, gap_mask3,
               pos_weight, auxiliary_weight):
    presence = F.binary_cross_entropy_with_logits(
        presence_logits, targets3, pos_weight=pos_weight
    )
    if float(auxiliary_weight) <= 0 or not bool(gap_mask3.any()):
        return presence, presence.detach().new_zeros(())
    auxiliary = F.smooth_l1_loss(
        visibility_prediction[gap_mask3], gap3[gap_mask3], beta=0.10
    )
    return presence + float(auxiliary_weight) * auxiliary, auxiliary


def _validation_key(logits, targets6, weak6):
    score = logits.detach().cpu().numpy()
    target = targets6[:, TARGET_INDICES]
    weak = weak6[:, TARGET_INDICES]
    weak_auroc, aupr = [], []
    for local in range(3):
        weak_positive = (target[:, local] == 1) & weak[:, local]
        negative = target[:, local] == 0
        y = np.concatenate([np.ones(weak_positive.sum()), np.zeros(negative.sum())])
        p = np.concatenate([score[weak_positive, local], score[negative, local]])
        weak_auroc.append(roc_auc_score(y, p))
        aupr.append(average_precision_score(target[:, local], score[:, local]))
    return float(np.mean(weak_auroc)), float(np.mean(aupr))


def _train_head(fit_x, validation_x, fit_targets6, validation_targets6,
                fit_gap6, validation_weak6, seed, auxiliary_weight, device, args):
    train_seed = 931000 + int(seed)
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(train_seed)
    model = PresenceVisibilityHead(fit_x.shape[1], args.adapter_rank).to(device)
    y_fit = fit_targets6[:, TARGET_INDICES].astype(np.float32)
    gap_fit = (fit_gap6[:, TARGET_INDICES] / float(args.gap_clip_db)).astype(np.float32)
    mask_fit = np.isfinite(gap_fit) & (y_fit > 0.5)
    gap_fit = np.nan_to_num(gap_fit, nan=0.0)
    positives = y_fit.sum(axis=0); negatives = len(y_fit) - positives
    pos_weight = torch.as_tensor(negatives / np.maximum(positives, 1.0),
                                 dtype=torch.float32, device=device)
    dataset = TensorDataset(torch.from_numpy(fit_x), torch.from_numpy(y_fit),
                            torch.from_numpy(gap_fit), torch.from_numpy(mask_fit))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
    epochs = args.smoke_epochs if args.smoke else args.epochs
    for epoch in range(1, int(epochs) + 1):
        model.train(); total_losses = []; auxiliary_losses = []
        for x, y, gap, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            presence, visibility = model(x.to(device)); y = y.to(device)
            loss, auxiliary = _dual_loss(presence, visibility, y, gap.to(device),
                                         mask.to(device), pos_weight, auxiliary_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); total_losses.append(float(loss.detach().cpu()))
            auxiliary_losses.append(float(auxiliary.detach().cpu()))
        model.eval()
        with torch.no_grad(): validation_presence, _ = model(validation_tensor)
        key = _validation_key(validation_presence, validation_targets6, validation_weak6)
        history.append({"epoch": epoch, "train_loss": float(np.mean(total_losses)),
                        "train_auxiliary_loss": float(np.mean(auxiliary_losses)),
                        "validation_weak_auroc": key[0], "validation_target_aupr": key[1]})
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(args.patience): break
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "train_seed": train_seed,
                            "validation_weak_auroc": best_key[0],
                            "validation_target_aupr": best_key[1],
                            "auxiliary_weight": float(auxiliary_weight)}


def _head_scores(model, values, device):
    model.eval()
    with torch.no_grad():
        presence, visibility = model(torch.from_numpy(values).to(device))
    return (presence.cpu().numpy().astype(np.float32),
            visibility.cpu().numpy().astype(np.float32))


def _read_prior(seed, variant, root):
    path = root / f"seed_{seed}" / variant / "metrics.json"
    return {"variant": variant, "metrics": json.loads(path.read_text(encoding="utf-8"))}


def _run_seed(args, config, labels6, weak6, gap6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(args.c6_root, seed, _data_path(config),
                                           config["normalization"], args.io_batch_size)
    state = torch.load(args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth",
                       map_location="cpu", weights_only=False)["model_state_dict"]
    base = _load_model_from_state(args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}",
                                  device, state)
    extracted = _extract_backbone(base, roles, device, args.feature_batch_size, args.topk)
    del base
    if device.type == "cuda": torch.cuda.empty_cache()
    stats = {role: row["temporal_stats"] for role, row in extracted.items()}
    standardized, mean, std = _standardize(stats["fit"], stats)
    heads, head_scores, head_info, histories = {}, {}, {}, {}
    for variant, weight in ((NO_AUX_VARIANT, 0.0), (AUX_VARIANT, args.auxiliary_weight)):
        head, history, info = _train_head(
            standardized["fit"], standardized["validation"],
            roles["fit"]["targets6"], roles["validation"]["targets6"],
            gap6[roles["fit"]["sample_ids"]], weak6[roles["validation"]["sample_ids"]],
            seed, weight, device, args,
        )
        heads[variant] = head; histories[variant] = history; head_info[variant] = info
        head_scores[variant] = {role: _head_scores(head, value, device)[0]
                                for role, value in standardized.items()}
    combined = ("fit", "validation")
    base_fitval = np.concatenate([extracted[role]["base_logits"] for role in combined])
    targets_fitval = np.concatenate([roles[role]["targets6"] for role in combined])
    fit = np.arange(len(roles["fit"]["sample_ids"]), dtype=np.int64)
    validation = np.arange(len(fit), len(targets_fitval), dtype=np.int64)
    fused_logits, fusion_info, fusion_history = {}, {}, {}
    for variant in (NO_AUX_VARIANT, AUX_VARIANT):
        scores_fitval = np.concatenate([head_scores[variant][role] for role in combined])
        center = scores_fitval[fit].mean(axis=0).astype(np.float32)
        scale = np.maximum(scores_fitval[fit].std(axis=0), 1e-6).astype(np.float32)
        normalized = {role: ((score - center[None, :]) / scale[None, :]).astype(np.float32)
                      for role, score in head_scores[variant].items()}
        fusion, history, info = _train_generic_fusion(
            base_fitval, np.concatenate([normalized[role] for role in combined]),
            targets_fitval, fit, validation, seed, device, args,
        )
        fusion_info[variant] = {**info, "center": center.tolist(), "scale": scale.tolist()}
        fusion_history[variant] = history
        fused_logits[variant] = {
            role: _apply_fusion(fusion, extracted[role]["base_logits"], value, device)[0]
            for role, value in normalized.items()
        }
        torch.save({"head_state_dict": heads[variant].state_dict(), "fusion_state_dict": fusion.state_dict(),
                    "mean": mean, "std": std, **head_info[variant], **fusion_info[variant]},
                   seed_root / f"{variant}.pth")
    raw_state = torch.load(args.prior_root / f"seed_{seed}" / "checkpoints" / "rawsaw.pth",
                           map_location="cpu", weights_only=False)
    raw = RawStage1SawFMFrontend(mode="raw", hidden_channels=args.raw_hidden_channels,
                                 topk=args.raw_topk, dropout=args.raw_dropout).to(device)
    raw.load_state_dict(raw_state["model_state_dict"], strict=True)
    raw_scores = {role: _raw_scores(raw, row["signals"], device, args.raw_batch_size)
                  for role, row in roles.items()}
    results = [_read_prior(seed, BASE_VARIANT, args.prior_root),
               _read_prior(seed, LINEAR_VARIANT, args.prior_root)]
    for variant in (NO_AUX_VARIANT, AUX_VARIANT):
        logits = {}
        for role, value in fused_logits[variant].items():
            combined_logits = value.copy(); combined_logits[:, 1] = raw_scores[role]
            logits[role] = combined_logits
        results.append(_evaluate(variant, logits, roles, weak6, seed_root / variant, args))
        _write_history(seed_root / f"{variant}_head_history.csv", histories[variant])
        _write_history(seed_root / f"{variant}_fusion_history.csv", fusion_history[variant])
    payload = {"seed": int(seed), "status": "completed", "data_sha256": data_sha,
               "role_indices_sha256": _sha256(role_indices_path), "variants": results,
               "head_info": head_info, "fusion_info": fusion_info,
               "head_parameter_count": int(sum(p.numel() for p in heads[AUX_VARIANT].parameters())),
               "same_capacity": int(sum(p.numel() for p in heads[AUX_VARIANT].parameters())) ==
                                int(sum(p.numel() for p in heads[NO_AUX_VARIANT].parameters())),
               "locked_test_used": False, "test_v2_used": False, "test_v3_used": False}
    _write_json(summary_path, payload); return payload


def _vmap(row): return {item["variant"]: item["metrics"] for item in row["variants"]}


def _flat(metrics):
    return {"macro_aupr": metrics["macro_aupr"], "macro_f1": metrics["macro_f1"],
            "weak_recall": metrics["target_mean_weak_recall"],
            "weak_auroc": metrics["target_weak_vs_negative_auroc"],
            "strong_recall": metrics["target_mean_strong_recall"],
            "fpr": metrics["target_mean_negative_fpr"],
            "sawfm_aupr": metrics["per_label"]["SawFM"]["aupr"]}


def _aggregate(rows):
    result = {}
    for vi, variant in enumerate((BASE_VARIANT, LINEAR_VARIANT, NO_AUX_VARIANT, AUX_VARIANT)):
        values = [_flat(_vmap(row)[variant]) for row in rows]
        result[variant] = {name: _mean_ci([item[name] for item in values], 20267100+vi*100+i)
                           for i, name in enumerate(values[0])}
    return result


def _compare(rows, baseline):
    paired = []
    for row in rows:
        candidate = _flat(_vmap(row)[AUX_VARIANT]); base = _flat(_vmap(row)[baseline])
        paired.append({"seed": row["seed"], **{key: candidate[key]-base[key] for key in candidate}})
    aggregate = {key: _mean_ci([row[key] for row in paired], 20267600+i)
                 for i, key in enumerate(_flat(_vmap(rows[0])[AUX_VARIANT]))}
    aggregate["positive_weak_recall_seeds"] = int(sum(row["weak_recall"] > 0 for row in paired))
    return {"baseline": baseline, "candidate": AUX_VARIANT, "per_seed": paired,
            "aggregate": aggregate}


def run(args):
    config = load_config(args.config); _validate_config(config)
    data_path = _data_path(config); data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256: raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path); values, _ = read_parameters(data_path)
    power6 = _component_power(values)
    weak6 = compute_weak_component_mask(labels6, power6, args.weak_threshold_db)
    gap6 = _relative_power_gap(labels6, power6, -args.gap_clip_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [_run_seed(args, config, labels6, weak6, gap6, seed, device, data_sha) for seed in seeds]
    aggregate = _aggregate(rows)
    comparisons = {"vs_no_aux": _compare(rows, NO_AUX_VARIANT),
                   "vs_c4": _compare(rows, LINEAR_VARIANT), "vs_c0": _compare(rows, BASE_VARIANT)}
    if args.smoke: gate = {"checks": {}, "passed": False}
    else:
        d = comparisons["vs_no_aux"]["aggregate"]; c0 = comparisons["vs_c0"]["aggregate"]
        checks = {"weak_recall_gain_0p010_vs_no_aux": d["weak_recall"]["mean"] >= .010,
                  "weak_auroc_gain_0p010_vs_no_aux": d["weak_auroc"]["mean"] >= .010,
                  "macro_aupr_noninferior_vs_no_aux": d["macro_aupr"]["mean"] >= -.003,
                  "strong_recall_noninferior_vs_no_aux": d["strong_recall"]["mean"] >= -.010,
                  "target_fpr_cap": aggregate[AUX_VARIANT]["fpr"]["mean"] <= .20,
                  "weak_recall_gain_0p030_vs_c0": c0["weak_recall"]["mean"] >= .030,
                  "weak_auroc_gain_0p015_vs_c0": c0["weak_auroc"]["mean"] >= .015,
                  "four_of_five_positive_vs_c0": c0["positive_weak_recall_seeds"] >= 4,
                  "same_capacity_and_test_safe": all(row["same_capacity"] and not row["test_v3_used"] for row in rows)}
        gate = {"checks": checks, "passed": bool(all(checks.values()))}
    payload = {"status": "completed", "experiment": "RouteA_PresenceVisibilityAux_v3",
               "smoke": bool(args.smoke), "seeds": list(seeds), "data_sha256": data_sha,
               "auxiliary_weight": args.auxiliary_weight, "adapter_rank": args.adapter_rank,
               "aggregate": aggregate, "comparisons": comparisons, "promotion_gate": gate,
               "promoted": bool(gate["passed"]), "locked_test_used": False,
               "test_v2_used": False, "test_v3_used": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "route_a_presence_visibility_aux_v3_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT/"configs"/"student_a"/"six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_dfms_iq_structure_screen")
    parser.add_argument("--prior-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT/"frozen"/"20260731_route_a_test_v3_sealed"/"test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT/"outputs"/"tapc"/"route_a_presence_visibility_aux_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043])
    parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=256); parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64); parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16); parser.add_argument("--raw-dropout", type=float, default=.1)
    parser.add_argument("--topk", type=int, default=8); parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--adapter-rank", type=int, default=32); parser.add_argument("--auxiliary-weight", type=float, default=.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--fusion-learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8); parser.add_argument("--fusion-epochs", type=int, default=200)
    parser.add_argument("--fusion-patience", type=int, default=20); parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--residual-cap", type=float, default=.50); parser.add_argument("--residual-l2-weight", type=float, default=.02)
    parser.add_argument("--bias-l2-weight", type=float, default=.05); parser.add_argument("--max-negative-fpr", type=float, default=.20)
    parser.add_argument("--gap-clip-db", type=float, default=40.0); parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())

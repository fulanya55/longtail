"""Frozen full-dimensional Ridge visibility conditioned fusion on development-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from cjlt.models.evidence_readout import RawStage1SawFMFrontend  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256, _write_json  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import (  # noqa: E402
    F2_MODEL_NAME,
    _component_power,
)
from experiments.run_route_a_development_v3_label_selective_projection import (  # noqa: E402
    _load_model_from_state,
)
from experiments.run_route_a_presence_visibility_aux_v3 import (  # noqa: E402
    AUX_VARIANT,
    BASE_VARIANT,
    LINEAR_VARIANT,
    PresenceVisibilityHead,
    _head_scores,
)
from experiments.run_route_a_predicted_visibility_conditioned_fusion_v3 import (  # noqa: E402
    ALIGNED_VARIANT as PRIOR_ALIGNED_VARIANT,
    _apply_conditioned,
    _flat,
    _predicted_visibility_severity,
    _shuffle_by_role,
    _standardize_triplet,
    _train_fusion,
    _write_history,
)
from experiments.run_route_a_stage23_identifiability_v2 import (  # noqa: E402
    TARGET_INDICES,
    TARGET_NAMES,
)
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import (  # noqa: E402
    EXPECTED_V3_SHA256,
    _evaluate,
    _extract_backbone,
    _load_roles,
    _raw_scores,
    _standardize,
    _validate_config,
)
from experiments.run_route_a_visibility_gap_frozen_probe_v3 import (  # noqa: E402
    _relative_power_gap,
    _role_metrics,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402


SHUFFLED_VARIANT = "E2_RidgeVisibilityShuffled"
ALIGNED_VARIANT = "E3_RidgeVisibilityAligned"
VARIANTS = (
    BASE_VARIANT,
    LINEAR_VARIANT,
    AUX_VARIANT,
    PRIOR_ALIGNED_VARIANT,
    SHUFFLED_VARIANT,
    ALIGNED_VARIANT,
)


def _fit_ridge_visibility(
    standardized_by_role,
    roles,
    labels6,
    gap6,
    alpha,
    gap_clip_db,
    weak_threshold_db,
):
    """Fit positive-only per-label Ridge probes and predict every role."""
    fit_ids = np.asarray(roles["fit"]["sample_ids"], dtype=np.int64)
    predictions = {role: np.zeros((len(values), 3), dtype=np.float32)
                   for role, values in standardized_by_role.items()}
    probe_states = {}
    diagnostics = {}
    for local, (index, name) in enumerate(zip(TARGET_INDICES, TARGET_NAMES)):
        valid = (labels6[fit_ids, index] == 1) & np.isfinite(gap6[fit_ids, index])
        if int(valid.sum()) < 3:
            raise ValueError(f"insufficient finite positive gaps for {name}")
        probe = Ridge(alpha=float(alpha), fit_intercept=True)
        probe.fit(
            standardized_by_role["fit"][valid],
            gap6[fit_ids[valid], index] / float(gap_clip_db),
        )
        for role, values in standardized_by_role.items():
            predictions[role][:, local] = probe.predict(values).astype(np.float32)
        probe_states[name] = {
            "coef": np.asarray(probe.coef_, dtype=np.float32),
            "intercept": np.float32(probe.intercept_),
            "fit_count": int(valid.sum()),
        }
        early_ids = np.asarray(roles["early"]["sample_ids"], dtype=np.int64)
        early_valid = ((labels6[early_ids, index] == 1)
                       & np.isfinite(gap6[early_ids, index]))
        diagnostics[name] = _role_metrics(
            gap6[early_ids[early_valid], index].astype(np.float64),
            (predictions["early"][early_valid, local] * float(gap_clip_db)).astype(np.float64),
            weak_threshold_db,
        )
    diagnostics["mean_across_labels"] = {
        "spearman": float(np.nanmean([diagnostics[name]["spearman"] for name in TARGET_NAMES])),
        "weak_vs_strong_auroc": float(np.nanmean([
            diagnostics[name]["weak_vs_strong_auroc"] for name in TARGET_NAMES
        ])),
    }
    return predictions, probe_states, diagnostics


def _save_ridge(path, states):
    payload = {}
    for name, state in states.items():
        payload[f"{name}_coef"] = state["coef"]
        payload[f"{name}_intercept"] = np.asarray([state["intercept"]], dtype=np.float32)
        payload[f"{name}_fit_count"] = np.asarray([state["fit_count"]], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _read_prior(seed, variant, root):
    path = root / f"seed_{seed}" / variant / "metrics.json"
    return {"variant": variant, "metrics": json.loads(path.read_text(encoding="utf-8"))}


def _run_seed(args, config, labels6, weak6, gap6, seed, device, data_sha):
    seed_root = args.output_root / f"seed_{seed}"
    summary_path = seed_root / "seed_summary.json"
    if args.resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_root.mkdir(parents=True, exist_ok=True)
    roles, role_indices_path = _load_roles(
        args.c6_root, seed, _data_path(config), config["normalization"], args.io_batch_size
    )
    state = torch.load(
        args.c6_root / f"seed_{seed}" / "C6_Joint" / "best_model.pth",
        map_location="cpu", weights_only=False,
    )["model_state_dict"]
    base_model = _load_model_from_state(
        args.f2_root / "runs" / F2_MODEL_NAME / f"seed_{seed}", device, state
    )
    extracted = _extract_backbone(base_model, roles, device, args.feature_batch_size, args.topk)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stats = {role: row["temporal_stats"] for role, row in extracted.items()}
    standardized, mean, std = _standardize(stats["fit"], stats)
    predicted_gap, ridge_states, ridge_diagnostics = _fit_ridge_visibility(
        standardized, roles, labels6, gap6, args.ridge_alpha,
        args.gap_clip_db, args.weak_threshold_db,
    )
    ridge_path = seed_root / "frozen_ridge_visibility.npz"
    _save_ridge(ridge_path, ridge_states)

    d2_checkpoint_path = args.d2_root / f"seed_{seed}" / f"{AUX_VARIANT}.pth"
    d2 = torch.load(d2_checkpoint_path, map_location="cpu", weights_only=False)
    head = PresenceVisibilityHead(stats["fit"].shape[1], args.adapter_rank).to(device)
    head.load_state_dict(d2["head_state_dict"], strict=True)
    presence_raw = {
        role: _head_scores(head, values, device)[0]
        for role, values in standardized.items()
    }
    severity_raw = {
        role: _predicted_visibility_severity(
            values, args.visibility_center_db, args.visibility_temperature_db,
            args.gap_clip_db,
        )
        for role, values in predicted_gap.items()
    }
    presence, severity_aligned, normalization = _standardize_triplet(
        presence_raw["fit"], severity_raw["fit"], presence_raw, severity_raw
    )
    severity_shuffled, shuffle_receipt = _shuffle_by_role(severity_aligned, seed)

    combined_roles = ("fit", "validation")
    base_fitval = np.concatenate([extracted[role]["base_logits"] for role in combined_roles])
    presence_fitval = np.concatenate([presence[role] for role in combined_roles])
    targets_fitval = np.concatenate([roles[role]["targets6"] for role in combined_roles])
    fit = np.arange(len(roles["fit"]["sample_ids"]), dtype=np.int64)
    validation = np.arange(len(fit), len(targets_fitval), dtype=np.int64)

    fused_logits = {}
    histories = {}
    fusion_info = {}
    residual_diagnostics = {}
    for variant, severity in (
        (SHUFFLED_VARIANT, severity_shuffled),
        (ALIGNED_VARIANT, severity_aligned),
    ):
        severity_fitval = np.concatenate([severity[role] for role in combined_roles])
        fusion, history, info = _train_fusion(
            base_fitval, presence_fitval, severity_fitval, targets_fitval,
            fit, validation, seed, device, args,
        )
        histories[variant] = history
        fusion_info[variant] = info
        fused_logits[variant] = {}
        residuals = {}
        for role in roles:
            fused_logits[variant][role], residuals[role] = _apply_conditioned(
                fusion, extracted[role]["base_logits"], presence[role], severity[role], device
            )
        residual_diagnostics[variant] = {
            role: {
                "mean_abs_target_residual": float(np.abs(value[:, TARGET_INDICES]).mean()),
                "max_abs_target_residual": float(np.abs(value[:, TARGET_INDICES]).max()),
            }
            for role, value in residuals.items()
        }
        torch.save(
            {
                "model_state_dict": fusion.state_dict(),
                "normalization": normalization,
                "ridge_sha256": _sha256(ridge_path),
                "variant": variant,
                **info,
            },
            seed_root / f"{variant}.pth",
        )
        _write_history(seed_root / f"{variant}_history.csv", history)

    raw_state = torch.load(
        args.unified_root / f"seed_{seed}" / "checkpoints" / "rawsaw.pth",
        map_location="cpu", weights_only=False,
    )
    raw = RawStage1SawFMFrontend(
        mode="raw", hidden_channels=args.raw_hidden_channels,
        topk=args.raw_topk, dropout=args.raw_dropout,
    ).to(device)
    raw.load_state_dict(raw_state["model_state_dict"], strict=True)
    raw_scores = {
        role: _raw_scores(raw, row["signals"], device, args.raw_batch_size)
        for role, row in roles.items()
    }

    results = [
        _read_prior(seed, BASE_VARIANT, args.unified_root),
        _read_prior(seed, LINEAR_VARIANT, args.unified_root),
        _read_prior(seed, AUX_VARIANT, args.d2_root),
        _read_prior(seed, PRIOR_ALIGNED_VARIANT, args.e1_root),
    ]
    for variant in (SHUFFLED_VARIANT, ALIGNED_VARIANT):
        logits = {}
        for role, values in fused_logits[variant].items():
            combined = values.copy()
            combined[:, 1] = raw_scores[role]
            logits[role] = combined
        results.append(_evaluate(variant, logits, roles, weak6, seed_root / variant, args))

    active_ridge_parameters = int(sum(
        state["coef"].size + 1 for state in ridge_states.values()
    ))
    payload = {
        "seed": int(seed),
        "status": "completed",
        "smoke": bool(args.smoke),
        "data_sha256": data_sha,
        "role_indices_sha256": _sha256(role_indices_path),
        "d2_checkpoint_sha256": _sha256(d2_checkpoint_path),
        "ridge_sha256": _sha256(ridge_path),
        "ridge_alpha": float(args.ridge_alpha),
        "ridge_parameter_count": active_ridge_parameters,
        "fusion_parameter_count": 12,
        "ridge_diagnostics": ridge_diagnostics,
        "variants": results,
        "fusion_info": fusion_info,
        "normalization": normalization,
        "shuffle_receipt": shuffle_receipt,
        "residual_diagnostics": residual_diagnostics,
        "same_capacity_aligned_vs_shuffled": True,
        "inference_uses_true_visibility": False,
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    _write_json(summary_path, payload)
    return payload


def _variant_map(row):
    return {item["variant"]: item["metrics"] for item in row["variants"]}


def _aggregate(rows):
    result = {}
    for vi, variant in enumerate(VARIANTS):
        values = [_flat(_variant_map(row)[variant]) for row in rows]
        result[variant] = {
            name: _mean_ci([value[name] for value in values], 20270100 + vi * 100 + i)
            for i, name in enumerate(values[0])
        }
    result["ridge_visibility"] = {
        metric: _mean_ci([
            row["ridge_diagnostics"]["mean_across_labels"][metric] for row in rows
        ], 20270800 + i)
        for i, metric in enumerate(("spearman", "weak_vs_strong_auroc"))
    }
    return result


def _compare(rows, baseline):
    paired = []
    for row in rows:
        candidate = _flat(_variant_map(row)[ALIGNED_VARIANT])
        base = _flat(_variant_map(row)[baseline])
        paired.append({
            "seed": row["seed"],
            **{name: candidate[name] - base[name] for name in candidate},
        })
    aggregate = {
        name: _mean_ci([row[name] for row in paired], 20270900 + i)
        for i, name in enumerate(_flat(_variant_map(rows[0])[ALIGNED_VARIANT]))
    }
    aggregate["positive_weak_recall_seeds"] = int(sum(
        row["weak_recall"] > 0 for row in paired
    ))
    return {"baseline": baseline, "candidate": ALIGNED_VARIANT,
            "per_seed": paired, "aggregate": aggregate}


def run(args):
    config = load_config(args.config)
    _validate_config(config)
    data_path = _data_path(config)
    data_sha = _sha256(data_path)
    if data_sha.lower() != EXPECTED_V3_SHA256:
        raise ValueError("development-v3 SHA mismatch")
    seal = json.loads(args.test_v3_manifest.read_text(encoding="utf-8"))
    if seal["test_v3"]["access_authorized"] or seal["test_v3"]["used"]:
        raise ValueError("test-v3 boundary is not sealed")
    labels6 = read_labels(data_path)
    parameter_values, _ = read_parameters(data_path)
    weak6 = compute_weak_component_mask(
        labels6, _component_power(parameter_values), args.weak_threshold_db
    )
    gap6 = _relative_power_gap(labels6, _component_power(parameter_values), -args.gap_clip_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = args.seeds[:1] if args.smoke else args.seeds
    rows = [
        _run_seed(args, config, labels6, weak6, gap6, seed, device, data_sha)
        for seed in seeds
    ]
    aggregate = _aggregate(rows)
    comparisons = {
        "vs_shuffled": _compare(rows, SHUFFLED_VARIANT),
        "vs_e1": _compare(rows, PRIOR_ALIGNED_VARIANT),
        "vs_d2": _compare(rows, AUX_VARIANT),
        "vs_c4": _compare(rows, LINEAR_VARIANT),
        "vs_c0": _compare(rows, BASE_VARIANT),
    }
    if args.smoke:
        gate = {"checks": {}, "passed": False}
    else:
        aligned = aggregate[ALIGNED_VARIANT]
        shuffled = comparisons["vs_shuffled"]["aggregate"]
        c0 = comparisons["vs_c0"]["aggregate"]
        checks = {
            "weak_recall_gain_0p010_vs_shuffled": shuffled["weak_recall"]["mean"] >= .010,
            "weak_auroc_gain_0p010_vs_shuffled": shuffled["weak_auroc"]["mean"] >= .010,
            "macro_aupr_noninferior_vs_shuffled": shuffled["macro_aupr"]["mean"] >= -.003,
            "strong_recall_noninferior_vs_shuffled": shuffled["strong_recall"]["mean"] >= -.010,
            "weak_recall_gain_0p030_vs_c0": c0["weak_recall"]["mean"] >= .030,
            "weak_auroc_gain_0p015_vs_c0": c0["weak_auroc"]["mean"] >= .015,
            "target_fpr_cap": aligned["fpr"]["mean"] <= .20,
            "four_of_five_positive_vs_c0": c0["positive_weak_recall_seeds"] >= 4,
            "sawfm_preserved": abs(comparisons["vs_d2"]["aggregate"]["sawfm_aupr"]["mean"]) <= 1e-8,
            "same_capacity_no_truth_and_test_safe": all(
                row["same_capacity_aligned_vs_shuffled"]
                and not row["inference_uses_true_visibility"]
                and not row["test_v3_used"] for row in rows
            ),
        }
        gate = {"checks": checks, "passed": bool(all(checks.values()))}
    payload = {
        "status": "completed",
        "experiment": "RouteA_FrozenRidgeVisibilityConditionedFusion_v3",
        "smoke": bool(args.smoke),
        "seeds": [int(seed) for seed in seeds],
        "data_sha256": data_sha,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "promotion_gate": gate,
        "promoted": bool(gate["passed"]),
        "locked_test_used": False,
        "test_v2_used": False,
        "test_v3_used": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_root / "route_a_frozen_ridge_visibility_conditioned_fusion_v3_summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class_development_v3.yaml")
    parser.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_development_v3_label_selective_projection")
    parser.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_unified_rawsaw_vtse_vbf_v3")
    parser.add_argument("--d2-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_presence_visibility_aux_v3")
    parser.add_argument("--e1-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_predicted_visibility_conditioned_fusion_v3")
    parser.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen" / "20260731_route_a_test_v3_sealed" / "test_v3_sealed_manifest.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_frozen_ridge_visibility_conditioned_fusion_v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2039, 2040, 2041, 2042, 2043])
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--io-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--raw-hidden-channels", type=int, default=16)
    parser.add_argument("--raw-topk", type=int, default=16)
    parser.add_argument("--raw-dropout", type=float, default=.1)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fusion-epochs", type=int, default=200)
    parser.add_argument("--fusion-patience", type=int, default=20)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--residual-cap", type=float, default=.50)
    parser.add_argument("--residual-l2-weight", type=float, default=.02)
    parser.add_argument("--bias-l2-weight", type=float, default=.05)
    parser.add_argument("--max-negative-fpr", type=float, default=.20)
    parser.add_argument("--weak-threshold-db", type=float, default=10.0)
    parser.add_argument("--visibility-center-db", type=float, default=-10.0)
    parser.add_argument("--visibility-temperature-db", type=float, default=2.5)
    parser.add_argument("--gap-clip-db", type=float, default=40.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

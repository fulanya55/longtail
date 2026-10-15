"""Screen early complex evidence sources for weak SawFM identifiability."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.power_pairs import load_level_identity  # noqa: E402
from cjlt.data.registry import select_labels  # noqa: E402
from cjlt.data.v73_dataset import V73SignalDataset, read_labels  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read, _write  # noqa: E402
from experiments.run_route_a_dfms_iq_frozen_heads import _sha256  # noqa: E402
from experiments.run_route_a_sawfm_stage2_power_teacher_student import _indices_for_identities  # noqa: E402
from experiments.run_route_a_stage2_reliability_identifiability import (  # noqa: E402
    _domain_class_weights,
    _fit_logistic_variant,
    _predict_bundle,
    _safe_metrics,
)
from experiments.run_shared import _data_path, load_config  # noqa: E402
import experiments.run_route_a_selective_unknown_decision as selective  # noqa: E402
import experiments.run_route_a_stable_expert_visibility_calibration as stable  # noqa: E402


VARIANTS = (
    "P0_H2Logit",
    "P1_VisibilityStats",
    "P2_RawIQPhysics",
    "P3_Stage1Complex",
    "P4_Stage2Complex",
    "P5_Stage3Complex",
    "P6_RawPlusStage1",
)


def _complex_physics_descriptor(real, imag, bins=25):
    real = np.asarray(real, dtype=np.float32)
    imag = np.asarray(imag, dtype=np.float32)
    if real.shape != imag.shape or real.ndim != 3:
        raise ValueError("real and imag must be aligned [N,C,T] arrays")
    if real.shape[-1] % int(bins):
        raise ValueError("temporal length must be divisible by bins")
    magnitude = np.sqrt(np.square(real) + np.square(imag) + 1e-8)
    previous_magnitude = np.concatenate([magnitude[..., :1], magnitude[..., :-1]], axis=2)
    normalized_real = real / magnitude
    normalized_imag = imag / magnitude
    previous_real = np.concatenate([normalized_real[..., :1], normalized_real[..., :-1]], axis=2)
    previous_imag = np.concatenate([normalized_imag[..., :1], normalized_imag[..., :-1]], axis=2)
    phase_cos = normalized_real * previous_real + normalized_imag * previous_imag
    phase_sin = normalized_imag * previous_real - normalized_real * previous_imag
    traces = np.stack(
        [
            magnitude.mean(axis=1),
            magnitude.max(axis=1),
            magnitude.std(axis=1),
            np.abs(magnitude - previous_magnitude).mean(axis=1),
            phase_cos.mean(axis=1),
            phase_sin.mean(axis=1),
        ],
        axis=1,
    )
    n, channels, length = traces.shape
    blocked = traces.reshape(n, channels, int(bins), length // int(bins))
    parts = [
        blocked.mean(axis=3).reshape(n, -1),
        blocked.max(axis=3).reshape(n, -1),
        blocked.std(axis=3).reshape(n, -1),
    ]
    top_count = max(1, length // 20)
    topk = np.partition(traces, -top_count, axis=2)[:, :, -top_count:].mean(axis=2)
    parts.append(
        np.stack(
            [traces.mean(axis=2), traces.std(axis=2), traces.max(axis=2), topk],
            axis=2,
        ).reshape(n, -1)
    )
    result = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite complex descriptor")
    return result


def _extract_descriptors(backbone, path, indices, normalization, device, batch_size):
    dataset = V73SignalDataset(
        path=path, indices=np.asarray(indices, dtype=np.int64),
        class_mode="six", normalization=normalization,
    )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output = {"raw_physics": [], "stage1_physics": [], "stage2_physics": [], "stage3_physics": []}
    try:
        with torch.no_grad():
            for batch in loader:
                signal = batch["signal"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    maps = backbone.forward_structure_maps(signal)
                pairs = {
                    "raw_physics": (signal[:, 0:1], signal[:, 1:2]),
                    "stage1_physics": (maps["stage1_real"], maps["stage1_imag"]),
                    "stage2_physics": (maps["stage2_real"], maps["stage2_imag"]),
                    "stage3_physics": (maps["stage3_real"], maps["stage3_imag"]),
                }
                for key, (real, imag) in pairs.items():
                    output[key].append(
                        _complex_physics_descriptor(
                            real.detach().cpu().float().numpy(),
                            imag.detach().cpu().float().numpy(),
                        )
                    )
    finally:
        dataset.close()
    return {key: np.concatenate(values, axis=0) for key, values in output.items()}


def _wrap_base_extract(original):
    def augmented(args, config, backbone, base_head, h2_checkpoint, indices, device):
        result = original(args, config, backbone, base_head, h2_checkpoint, indices, device)
        result.update(_extract_descriptors(
            backbone, _data_path(config), indices, config["normalization"],
            device, args.feature_batch_size,
        ))
        return result
    return augmented


def _wrap_power_extract(original):
    def augmented(args, config, backbone, base_head, h2_checkpoint, power_path, level_info, identities, device):
        result = original(
            args, config, backbone, base_head, h2_checkpoint,
            power_path, level_info, identities, device,
        )
        indices = _indices_for_identities(level_info, identities)
        result.update(_extract_descriptors(
            backbone, power_path, indices, config["normalization"],
            device, args.feature_batch_size,
        ))
        return result
    return augmented


def _binary_metrics(targets, scores):
    return _safe_metrics(np.asarray(targets, dtype=np.uint8), np.asarray(scores, dtype=np.float64))


def _role_groups(payloads, role, levels):
    groups = []
    for level in ["root"] + [str(value) for value in levels]:
        positive = payloads[(role, level, "positive")]
        negative = payloads[(role, level, "negative")]
        group = {"name": level}
        keys = (
            "base_logits", "visibility_features", "raw_physics",
            "stage1_physics", "stage2_physics", "stage3_physics",
        )
        for key in keys:
            group[key] = np.concatenate([positive[key], negative[key]], axis=0).astype(np.float32)
        group["targets"] = np.concatenate([
            np.ones(len(positive["base_logits"]), dtype=np.uint8),
            np.zeros(len(negative["base_logits"]), dtype=np.uint8),
        ])
        groups.append(group)
    return groups


def _fit_fixed_probe(fit_groups, variant, c_value, shuffle_seed=None):
    fit_x, fit_y, fit_domains = _stack(fit_groups, variant)
    mean = fit_x.mean(axis=0)
    std = fit_x.std(axis=0)
    std[std < 1e-6] = 1.0
    targets = fit_y.copy()
    if shuffle_seed is not None:
        targets = np.random.default_rng(int(shuffle_seed)).permutation(targets)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=float(c_value), max_iter=3000, solver="lbfgs")
    model.fit(
        (fit_x - mean) / std,
        targets,
        sample_weight=_domain_class_weights(fit_domains, targets),
    )
    return {"model": model, "mean": mean, "std": std}


def _fixed_probe_scores(bundle, groups, variant):
    return {
        group["name"]: _predict_bundle(bundle, _features(group, variant))
        for group in groups if group["name"] != "root"
    }


def _development_groups(payloads, levels):
    groups = []
    for level in levels:
        positive = payloads[("development", str(level), "positive")]
        negative = payloads[("development", str(level), "negative")]
        group = {"name": str(level)}
        for key in (
            "base_logits", "visibility_features", "raw_physics",
            "stage1_physics", "stage2_physics", "stage3_physics",
        ):
            group[key] = np.concatenate([positive[key], negative[key]], axis=0).astype(np.float32)
        group["targets"] = np.concatenate([
            np.ones(len(positive["base_logits"]), dtype=np.uint8),
            np.zeros(len(negative["base_logits"]), dtype=np.uint8),
        ])
        groups.append(group)
    return groups


def _feature_key(variant):
    return {
        "P1_VisibilityStats": "visibility_features",
        "P2_RawIQPhysics": "raw_physics",
        "P3_Stage1Complex": "stage1_physics",
        "P4_Stage2Complex": "stage2_physics",
        "P5_Stage3Complex": "stage3_physics",
    }.get(variant)


def _features(group, variant):
    if variant == "P6_RawPlusStage1":
        return np.concatenate([group["raw_physics"], group["stage1_physics"]], axis=1)
    return group[_feature_key(variant)]


def _stack(groups, variant):
    return (
        np.concatenate([_features(group, variant) for group in groups], axis=0),
        np.concatenate([group["targets"] for group in groups]),
        np.concatenate([np.full(len(group["targets"]), group["name"], dtype=object) for group in groups]),
    )


def _evaluate(groups, scores):
    by_level = {
        group["name"]: _binary_metrics(group["targets"], scores[group["name"]])
        for group in groups
    }
    aurocs = [item["auroc"] for item in by_level.values()]
    auprs = [item["aupr"] for item in by_level.values()]
    return {
        "by_level": by_level,
        "mean_power_auroc": float(np.mean(aurocs)),
        "worst_power_auroc": float(np.min(aurocs)),
        "mean_power_aupr": float(np.mean(auprs)),
    }


def _run_seed(args, config, labels, identities, seed, device, prior_summary):
    replay_args = copy.copy(args)
    replay_args.output_root = args.output_root / "frozen_replay"
    original_base = stable._extract_base_indices
    original_power = stable._extract_power_identities
    try:
        stable._extract_base_indices = _wrap_base_extract(original_base)
        stable._extract_power_identities = _wrap_power_extract(original_power)
        _, audit = selective._run_seed(
            replay_args, config, labels, identities, seed, device, prior_summary
        )
    finally:
        stable._extract_base_indices = original_base
        stable._extract_power_identities = original_power
    payloads = selective._PAYLOADS
    fit = _role_groups(payloads, "fit", args.levels)
    validation = _role_groups(payloads, "threshold", args.levels)
    development = _development_groups(payloads, args.levels)
    rows = []
    selection = {}
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        if variant == "P0_H2Logit":
            validation_scores = {group["name"]: group["base_logits"][:, 1] for group in validation}
            development_scores = {group["name"]: group["base_logits"][:, 1] for group in development}
            selected_c = None
            feature_dimension = 1
        else:
            fit_x, fit_y, fit_domains = _stack(fit, variant)
            validation_x, validation_y, validation_domains = _stack(validation, variant)
            bundle = _fit_logistic_variant(
                fit_x, fit_y, fit_domains,
                validation_x, validation_y, validation_domains,
                args.c_values,
            )
            validation_scores = {
                group["name"]: _predict_bundle(bundle, _features(group, variant))
                for group in validation if group["name"] != "root"
            }
            development_scores = {
                group["name"]: _predict_bundle(bundle, _features(group, variant))
                for group in development
            }
            selected_c = bundle["selected_c"]
            feature_dimension = int(fit_x.shape[1])
            selection[variant] = {
                "selected_c": selected_c, "feature_dimension": feature_dimension,
                "candidates": bundle["candidates"],
            }
        validation_power = [group for group in validation if group["name"] != "root"]
        validation_metrics = _evaluate(validation_power, validation_scores)
        development_metrics = _evaluate(development, development_scores)
        row = {
            "seed": int(seed), "variant": variant,
            "selected_c": selected_c, "feature_dimension": feature_dimension,
            "validation": validation_metrics,
            "development": development_metrics,
            "validation_development_auroc_gap": float(
                validation_metrics["mean_power_auroc"] - development_metrics["mean_power_auroc"]
            ),
            "identity_overlap": int(audit["identity_overlap"]),
            "locked_test_used": False,
        }
        rows.append(row)
        variant_root = seed_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        _write(variant_root / "metrics.json", row)
        np.savez_compressed(
            str(variant_root / "development_predictions.npz"),
            **{f"positive_negative_score_{level}": score for level, score in development_scores.items()},
            **{f"target_{group['name']}": group["targets"] for group in development},
        )
        _write(variant_root / "run_receipt.json", {
            "status": "completed", "seed": int(seed), "variant": variant,
            "selected_expert_sha256": audit["selected_expert_sha256"],
            "f2_checkpoint_sha256": audit["f2_checkpoint_sha256"],
            "h2_checkpoint_sha256": audit["h2_checkpoint_sha256"],
            "identity_overlap": int(audit["identity_overlap"]), "locked_test_used": False,
        })
    _write(seed_root / "probe_selection.json", selection)
    _write(seed_root / "source_and_identity_audit.json", audit)
    return rows, audit


def _aggregate(rows):
    aggregate = {}
    for variant_index, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            role: {
                metric: _mean_ci(
                    [row[role][metric] for row in selected],
                    421000 + variant_index * 100 + (0 if role == "validation" else 20) + metric_index,
                )
                for metric_index, metric in enumerate((
                    "mean_power_auroc", "worst_power_auroc", "mean_power_aupr"
                ))
            }
            for role in ("validation", "development")
        }
        aggregate[variant]["validation_development_auroc_gap"] = _mean_ci(
            [row["validation_development_auroc_gap"] for row in selected],
            422000 + variant_index,
        )
    return aggregate


def _promotion_gate(rows, audits):
    p1 = {row["seed"]: row for row in rows if row["variant"] == "P1_VisibilityStats"}
    p6 = {row["seed"]: row for row in rows if row["variant"] == "P6_RawPlusStage1"}
    per_seed = []
    for seed, row in sorted(p6.items()):
        development = row["development"]
        checks = {
            "mean_power_auroc_0p65": development["mean_power_auroc"] >= 0.65,
            "worst_power_auroc_0p60": development["worst_power_auroc"] >= 0.60,
            "delta_vs_p1_0p03": development["mean_power_auroc"] >= p1[seed]["development"]["mean_power_auroc"] + 0.03,
            "generalization_gap_0p15": row["validation_development_auroc_gap"] <= 0.15,
            "50db_auroc_0p60": development["by_level"]["50"]["auroc"] >= 0.60,
        }
        per_seed.append({"seed": int(seed), "checks": checks, "passed": all(checks.values())})
    audit_pass = all(audit["identity_overlap"] == 0 and not audit["locked_test_used"] for audit in audits)
    passed_count = int(sum(item["passed"] for item in per_seed))
    return {
        "per_seed": per_seed, "passed_seed_count": passed_count,
        "identity_and_locked_test_audit": audit_pass,
        "passed": passed_count >= 2 and audit_pass,
    }


def run(args):
    config = load_config(args.config)
    labels = select_labels(read_labels(_data_path(config)), "six")
    identities, identity_labels = load_level_identity(_data_path(config))
    if not np.array_equal(labels, identity_labels):
        raise ValueError("Base labels and identities are misaligned")
    prior_summary = _read(args.prior_summary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, audits = [], []
    for seed in args.seeds:
        seed_rows, audit = _run_seed(args, config, labels, identities, int(seed), device, prior_summary)
        rows.extend(seed_rows)
        audits.append(audit)
    aggregate = _aggregate(rows)
    gate = _promotion_gate(rows, audits)
    best_variant = max(
        VARIANTS,
        key=lambda variant: aggregate[variant]["development"]["mean_power_auroc"]["mean"],
    )
    p1 = aggregate["P1_VisibilityStats"]["development"]["mean_power_auroc"]["mean"]
    best = aggregate[best_variant]["development"]["mean_power_auroc"]["mean"]
    if gate["passed"]:
        next_decision = "confirm_fixed_raw_plus_stage1_sawfm_frontend_on_new_seeds"
    elif best_variant in ("P3_Stage1Complex", "P6_RawPlusStage1") and best >= p1 + 0.03:
        next_decision = "confirm_best_early_complex_candidate_before_branch_training"
    else:
        next_decision = "stop_frozen_f2_sawfm_readouts_require_new_supervision_or_independent_frontend"
    payload = {
        "status": "completed", "device": str(device), "locked_test_used": False,
        "protocol": {
            "seeds": list(args.seeds), "levels": list(args.levels),
            "variants": list(VARIANTS), "c_values": list(args.c_values),
            "target": "weak_sawfm_positive_vs_combo_matched_hard_negative",
            "frozen_sources": True,
        },
        "audits": audits, "per_run": rows, "aggregate": aggregate,
        "promotion_gate": gate, "best_variant": best_variant,
        "next_decision": next_decision,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "route_a_early_complex_sawfm_identifiability_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "student_a" / "six_class.yaml")
    parser.add_argument("--screen-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_structure_screen")
    parser.add_argument("--frozen-head-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_dfms_iq_frozen_heads")
    parser.add_argument("--expert-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_stage2_power_teacher_student")
    parser.add_argument("--stability-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_sawfm_s2_initialization_stability")
    parser.add_argument("--frozen-visibility-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration")
    parser.add_argument("--prior-summary", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_stable_expert_visibility_calibration" / "route_a_stable_expert_visibility_calibration_summary.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "tapc" / "route_a_early_complex_sawfm_identifiability")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2041, 2042, 2043])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[274301, 274302, 274303])
    parser.add_argument("--levels", type=int, nargs="+", default=[40, 45, 50])
    parser.add_argument("--pairs-per-combo", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--max-positive-fpr", type=float, default=0.20)
    parser.add_argument("--max-positive-fn", type=float, default=0.10)
    parser.add_argument("--min-ordinary-coverage", type=float, default=0.70)
    parser.add_argument("--min-power-coverage", type=float, default=0.40)
    parser.add_argument("--max-negative-fpr", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

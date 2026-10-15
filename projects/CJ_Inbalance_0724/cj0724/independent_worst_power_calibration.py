"""P1d independent-data worst-power buffered calibration for frozen P1c."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta as beta_distribution

from cj0724.dual_axis import POWER_LEVELS, discover_power_train_files, read_identity_metadata
from cj0724.hierarchical_np_risk_router import (
    fit_hierarchical_models,
    predict_hierarchical_scores,
    selective_blend,
)
from cj0724.identity_safe_power_parameter_eval import _level_metrics, _weak_mask
from cj0724.integrate_swept_experts_with_six_class import (
    TARGET_INDICES,
    TARGET_NAMES,
    _identity_split,
    _jsonable,
    _load_main,
    _load_unet,
    align_payloads,
    classification_metrics,
    collect_main,
    expert_probabilities,
    extract_unet_role,
    fit_frozen_experts,
)
from cj0724.oracle_swept_isolation_experiment import set_seed
from cj0724.paper_core_evidence import load_normalizer


def discover_split_files(root: str | Path, split: str) -> dict[int, Path]:
    root = Path(root)
    filename = f"closedset_6class_{split}.mat"
    files = {}
    for index, level in enumerate(POWER_LEVELS, 1):
        path = root / f"level_{index:02d}__jnr_{level}dB" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        files[level] = path
    return files


@functools.lru_cache(maxsize=None)
def clopper_pearson_upper(false_positives: int, negatives: int, confidence: float = 0.95) -> float:
    if negatives <= 0:
        return float("nan")
    if false_positives >= negatives:
        return 1.0
    return float(beta_distribution.ppf(confidence, false_positives + 1, negatives - false_positives))


def buffered_threshold(records, target_index: int, probability_key: str) -> dict[str, object]:
    candidates = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    rows = []
    for threshold in candidates:
        power_rows, feasible = [], True
        for record in records:
            truth = record["labels"][:, target_index].astype(bool)
            score = record[probability_key][:, target_index]
            negative = ~truth
            predicted = score >= threshold
            false_positives = int(predicted[negative].sum())
            negatives = int(negative.sum())
            fpr = false_positives / max(negatives, 1)
            upper = clopper_pearson_upper(false_positives, negatives)
            if fpr > 0.08 + 1e-12 or upper > 0.10 + 1e-12:
                feasible = False
            power_rows.append({
                "power_level_db": int(record["power"][0]), "false_positives": false_positives,
                "negatives": negatives, "observed_fpr": fpr, "cp95_upper": upper,
                "recall": float(predicted[truth].mean()),
            })
        if feasible:
            extreme_recall = float(np.mean([
                row["recall"] for row in power_rows if row["power_level_db"] in (10, 50)
            ]))
            rows.append({"threshold": float(threshold), "extreme_recall": extreme_recall, "by_power": power_rows})
    if not rows:
        return {"threshold": 1.0, "extreme_recall": 0.0, "feasible": False, "reason": "no buffered threshold"}
    selected = max(rows, key=lambda row: (row["extreme_recall"], -row["threshold"]))
    selected["feasible"] = True
    selected["feasible_candidate_count"] = len(rows)
    return selected


def calibrate(records, original_thresholds, probability_key):
    thresholds = np.asarray(original_thresholds, dtype=np.float32).copy()
    audit = {}
    for local, index in enumerate(TARGET_INDICES):
        selected = buffered_threshold(records, int(index), probability_key)
        thresholds[index] = selected["threshold"]
        audit[TARGET_NAMES[local]] = selected
    return thresholds, audit


def prepare_records(
    main_model, unet, experts, route_models, selections, files, labels,
    batch_size, seed, device, normalizer, main_thresholds,
):
    indices = np.arange(len(labels), dtype=np.int64)
    records = []
    for level in POWER_LEVELS:
        main = collect_main(main_model, files[level], indices, batch_size, device, normalizer)
        unet_record = extract_unet_role(unet, files[level], files[50], labels, indices, batch_size, seed + level, device)
        align_payloads(main, unet_record)
        expert = expert_probabilities(unet_record, experts)
        risk, benefit, score = predict_hierarchical_scores(
            route_models, main["probabilities"], expert, main_thresholds
        )
        final, accept = selective_blend(main["probabilities"], expert, score, selections)
        records.append({
            **{name: main[name] for name in ("sample_indices", "labels", "probabilities", "prediction", "target", "mask")},
            "expert": expert, "risk_probability": risk, "benefit_probability": benefit,
            "route_score": score, "final_probability": final, "accept": accept,
            "power": np.full(len(indices), level, dtype=np.int16),
        })
    return records


def evaluate_test(records, base_thresholds, final_thresholds, scale):
    rows = []
    for record in records:
        level = int(record["power"][0])
        weak = _weak_mask(record["labels"].astype(bool), level)
        common = {
            "sample_indices": record["sample_indices"], "labels": record["labels"],
            "prediction": record["prediction"], "target": record["target"],
            "mask": record["mask"], "scale": scale,
        }
        base_class = classification_metrics(record["labels"], record["probabilities"], base_thresholds)
        final_class = classification_metrics(record["labels"], record["final_probability"], final_thresholds)
        cp_upper = {}
        for local, index in enumerate(TARGET_INDICES):
            truth = record["labels"][:, index].astype(bool)
            predicted = record["final_probability"][:, index] >= final_thresholds[index]
            fp, negatives = int(predicted[~truth].sum()), int((~truth).sum())
            cp_upper[TARGET_NAMES[local]] = clopper_pearson_upper(fp, negatives)
        rows.append({
            "power_level_db": level, "base": base_class, "final": final_class,
            "final_cp95_fpr_upper": cp_upper,
            "base_parameter_gate": _level_metrics({**common, "probabilities": record["probabilities"], "thresholds": base_thresholds}, weak),
            "final_parameter_gate": _level_metrics({**common, "probabilities": record["final_probability"], "thresholds": final_thresholds}, weak),
            "accept_coverage": dict(zip(TARGET_NAMES, record["accept"].mean(axis=0))),
        })
    gain = float(np.mean([
        row["final"]["per_class"][index]["recall"] - row["base"]["per_class"][index]["recall"]
        for row in rows if row["power_level_db"] in (10, 50) for index in TARGET_INDICES
    ]))
    macro_delta = float(np.mean([row["final"]["macro_aupr"] - row["base"]["macro_aupr"] for row in rows]))
    all_fpr = {
        f"{row['power_level_db']}dB_{TARGET_NAMES[local]}": row["final"]["per_class"][index]["fpr"]
        for row in rows for local, index in enumerate(TARGET_INDICES)
    }
    all_upper = {
        f"{row['power_level_db']}dB_{name}": value
        for row in rows for name, value in row["final_cp95_fpr_upper"].items()
    }
    return rows, {
        "extreme_three_class_recall_gain": gain,
        "mean_nine_power_macro_aupr_delta": macro_delta,
        "maximum_target_power_observed_fpr": float(max(all_fpr.values())),
        "maximum_target_power_cp95_upper": float(max(all_upper.values())),
        "worst_observed_fpr_cell": max(all_fpr, key=all_fpr.get),
        "worst_cp95_cell": max(all_upper, key=all_upper.get),
        "mean_parameter_coverage_delta": float(np.mean([
            row["final_parameter_gate"]["predicted_gate_parameter_coverage"]
            - row["base_parameter_gate"]["predicted_gate_parameter_coverage"] for row in rows
        ])),
        "mean_miss_penalty_parameter_nmae_delta": float(np.mean([
            row["final_parameter_gate"]["miss_penalty_parameter_nmae"]
            - row["base_parameter_gate"]["miss_penalty_parameter_nmae"] for row in rows
        ])),
    }


def _identity_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p1c-root", default="outputs/p1c_hierarchical_np_router_20260727_run1")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--original-power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--new-power-root", default="generated_data/p1d_independent_power_sweep_seed100000000")
    parser.add_argument("--output-root", default="outputs/p1d_independent_worst_power_calibration_20260727")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()

    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root, p1c_root = project / args.run_root, project / args.p1c_root
    separator_root, new_root, output_root = project / args.separator_root, project / args.new_power_root, project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    role_file = np.load(run_root / "role_indices.npz")
    original_files = discover_power_train_files(args.original_power_root)
    original_identities, original_labels = read_identity_metadata(protocol["data_file"])
    normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {int(row["seed"]): row for row in summary if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}
    new_train_files, new_test_files = discover_split_files(new_root, "train"), discover_split_files(new_root, "test")
    train_identities, train_labels = read_identity_metadata(new_train_files[30])
    test_identities, test_labels = read_identity_metadata(new_test_files[30])
    identity_audit = {
        "new_train_count": len(train_identities), "new_test_count": len(test_identities),
        "new_train_hash": _identity_hash(train_identities), "new_test_hash": _identity_hash(test_identities),
        "new_train_test_overlap": int(np.intersect1d(train_identities, test_identities).size),
        "new_train_original_overlap": int(np.intersect1d(train_identities, original_identities).size),
        "new_test_original_overlap": int(np.intersect1d(test_identities, original_identities).size),
    }
    if any(identity_audit[key] for key in ("new_train_test_overlap", "new_train_original_overlap", "new_test_original_overlap")):
        raise RuntimeError("independent identity audit failed")

    results = []
    for seed in args.seeds:
        set_seed(seed)
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        row = selected_rows[seed]
        raw_thresholds = row["thresholds"]
        main_thresholds = np.asarray(json.loads(raw_thresholds) if isinstance(raw_thresholds, str) else raw_thresholds, dtype=np.float32)
        main_model, main_checkpoint = _load_main(row, protocol, project, device)
        unet_checkpoint = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(unet_checkpoint, device)

        train_unet = extract_unet_role(unet, original_files[10], original_files[50], original_labels, role_file["train"].astype(np.int64), args.batch_size, seed, device)
        early_unet = extract_unet_role(unet, original_files[10], original_files[50], original_labels, role_file["early"].astype(np.int64), args.batch_size, seed, device)
        experts = fit_frozen_experts(train_unet, early_unet, seed)
        fit_indices, _ = _identity_split(role_file["calibration"].astype(np.int64), original_identities, seed + 7000)
        fit_main = []
        fit_expert = []
        fit_labels = []
        for level in POWER_LEVELS:
            main = collect_main(main_model, original_files[level], fit_indices, args.batch_size, device, normalizer)
            main["thresholds"] = main_thresholds
            unet_fit = extract_unet_role(unet, original_files[level], original_files[50], original_labels, fit_indices, args.batch_size, seed + level, device)
            align_payloads(main, unet_fit)
            fit_main.append(main["probabilities"])
            fit_expert.append(expert_probabilities(unet_fit, experts))
            fit_labels.append(main["labels"])
        route_models, _ = fit_hierarchical_models(
            np.concatenate(fit_main), np.concatenate(fit_expert), np.concatenate(fit_labels), main_thresholds, seed
        )
        p1c_result = json.loads((p1c_root / f"seed_{seed}" / "result.json").read_text(encoding="utf-8"))
        selections = [p1c_result["selected_per_target"][name] for name in TARGET_NAMES]

        def prepare(files, labels):
            indices = np.arange(len(labels), dtype=np.int64)
            records = []
            for level in POWER_LEVELS:
                main = collect_main(main_model, files[level], indices, args.batch_size, device, normalizer)
                main["thresholds"] = main_thresholds
                unet_record = extract_unet_role(unet, files[level], files[50], labels, indices, args.batch_size, seed + level, device)
                align_payloads(main, unet_record)
                expert = expert_probabilities(unet_record, experts)
                risk, benefit, score = predict_hierarchical_scores(route_models, main["probabilities"], expert, main_thresholds)
                final, accept = selective_blend(main["probabilities"], expert, score, selections)
                records.append({
                    **{name: main[name] for name in ("sample_indices", "labels", "probabilities", "prediction", "target", "mask")},
                    "expert": expert, "risk_probability": risk, "benefit_probability": benefit,
                    "route_score": score, "final_probability": final, "accept": accept,
                    "power": np.full(len(indices), level, dtype=np.int16),
                })
            return records

        calibration_records = prepare(new_train_files, train_labels)
        base_thresholds, base_calibration = calibrate(calibration_records, main_thresholds, "probabilities")
        final_thresholds, final_calibration = calibrate(calibration_records, main_thresholds, "final_probability")
        test_records = prepare(new_test_files, test_labels)
        by_power, aggregate = evaluate_test(test_records, base_thresholds, final_thresholds, normalizer.scale.numpy())

        merged = {name: np.concatenate([record[name] for record in test_records]) for name in (
            "sample_indices", "labels", "probabilities", "expert", "risk_probability", "benefit_probability",
            "route_score", "final_probability", "accept", "power", "prediction", "target", "mask"
        )}
        np.savez_compressed(
            seed_dir / "independent_test_per_sample.npz",
            sample_index=merged["sample_indices"], receiver_noise_seed=test_identities[merged["sample_indices"]],
            power_db=merged["power"], labels=merged["labels"].astype(np.uint8), main_probability=merged["probabilities"],
            expert_probability=merged["expert"], risk_probability=merged["risk_probability"],
            benefit_probability=merged["benefit_probability"], route_score=merged["route_score"],
            final_probability=merged["final_probability"], accept_mask=merged["accept"].astype(np.uint8),
            base_thresholds=base_thresholds, final_thresholds=final_thresholds,
            parameter_prediction=merged["prediction"], parameter_target=merged["target"], parameter_mask=merged["mask"].astype(np.uint8),
        )
        result = {
            "seed": seed, "frozen_main_checkpoint": str(main_checkpoint), "frozen_unet_checkpoint": str(unet_checkpoint),
            "frozen_p1c_selections": dict(zip(TARGET_NAMES, selections)),
            "base_buffered_calibration": base_calibration, "final_buffered_calibration": final_calibration,
            "base_thresholds": base_thresholds, "final_thresholds": final_thresholds,
            "independent_test_by_power": by_power, "independent_test_aggregate": aggregate,
            "identity_audit": identity_audit, "existing_confirm_used": False,
            "existing_power_test_used": False, "locked_test_used": False,
        }
        (seed_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        results.append(result)
        del main_model, unet, experts, route_models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gains = [row["independent_test_aggregate"]["extreme_three_class_recall_gain"] for row in results]
    macro = [row["independent_test_aggregate"]["mean_nine_power_macro_aupr_delta"] for row in results]
    fpr = [row["independent_test_aggregate"]["maximum_target_power_observed_fpr"] for row in results]
    miss = [row["independent_test_aggregate"]["mean_miss_penalty_parameter_nmae_delta"] for row in results]
    passed = bool(np.mean(gains) >= 0.10 and min(gains) > 0 and max(fpr) <= 0.10 + 1e-12 and np.mean(macro) >= -0.005 and np.mean(miss) <= 0.0)
    payload = {
        "experiment": "P1d independent worst-power buffered calibration",
        "mean_extreme_three_class_recall_gain": float(np.mean(gains)), "per_seed_gain": gains,
        "same_direction_seeds": int(sum(value > 0 for value in gains)),
        "mean_macro_aupr_delta": float(np.mean(macro)),
        "maximum_seed_target_power_observed_fpr": float(max(fpr)),
        "mean_miss_penalty_parameter_nmae_delta": float(np.mean(miss)),
        "promotion_passed": passed,
        "next_gate": "P1c deployment gate passed; enter P2" if passed else "P1c deployment gate failed; do not enter P2",
        "identity_audit": identity_audit, "wall_seconds": float(time.perf_counter() - started),
        "existing_confirm_used": False, "existing_power_test_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()

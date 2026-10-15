"""P1f: ISRJ confidence-bound safe routing on fresh transition data.

All model fitting, route selection and threshold calibration use the fresh
P1f train identities.  A method-freeze receipt is written before the P1f test
files are discovered or opened.  The experiment modifies only the ISRJ output
and compares an ensemble-mean route with a 10th-percentile confidence-lower-
bound route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from cj0724.dual_axis import read_identity_metadata
from cj0724.identity_safe_power_parameter_eval import _level_metrics, _weak_mask
from cj0724.independent_worst_power_calibration import clopper_pearson_upper
from cj0724.integrate_swept_experts_with_six_class import (
    TARGET_INDICES,
    TARGET_NAMES,
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
from cj0724.isrj_separator_physics_head import clipped_logit
from cj0724.isrj_separator_physics_head import threshold_at_fpr
from cj0724.oracle_swept_isolation_experiment import set_seed
from cj0724.p1e_transition_physical_sufficiency import (
    _fit_binary,
    _targets,
    feature_matrix,
    route_metrics,
)
from cj0724.paper_core_evidence import load_normalizer


POWERS = (20, 25)
ISRJ_LOCAL = 2
ISRJ_GLOBAL = int(TARGET_INDICES[ISRJ_LOCAL])
ENSEMBLE_SIZE = 15
LOWER_QUANTILE = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def transition_files(
    root: Path, split: str, powers: tuple[int, ...] = POWERS,
) -> dict[int, Path]:
    files = {}
    canonical = list(range(10, 51, 5))
    for power in powers:
        index = canonical.index(power) + 1
        path = root / f"level_{index:02d}__jnr_{power}dB" / f"closedset_6class_{split}.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        files[power] = path
    return files


def stratified_train_roles(labels: np.ndarray, seed: int = 20260731) -> dict[str, np.ndarray]:
    """Split each of 63 combinations 20/10/10 into fit/select/calibration."""
    labels = np.asarray(labels, dtype=np.uint8)
    _, inverse = np.unique(labels, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    roles = {"fit": [], "selection": [], "calibration": []}
    for group in range(int(inverse.max()) + 1):
        indices = np.flatnonzero(inverse == group)
        if len(indices) != 40:
            raise ValueError(f"Expected 40 identities per combination, got {len(indices)}")
        indices = rng.permutation(indices)
        roles["fit"].extend(indices[:20])
        roles["selection"].extend(indices[20:30])
        roles["calibration"].extend(indices[30:40])
    return {name: np.sort(np.asarray(values, dtype=np.int64)) for name, values in roles.items()}


def _role_payload(records: list[dict[str, np.ndarray]], indices: np.ndarray) -> dict[str, np.ndarray]:
    names = (
        "sample_indices", "power", "labels", "main", "expert", "mft", "trajectory", "isrj",
        "prediction", "target", "mask",
    )
    output = {name: [] for name in names}
    selected = set(np.asarray(indices, dtype=np.int64).tolist())
    for record in records:
        keep = np.asarray([int(value) in selected for value in record["sample_indices"]], dtype=bool)
        for name in names:
            output[name].append(record[name][keep])
    return {name: np.concatenate(parts) for name, parts in output.items()}


def grouped_bootstrap_rows(sample_indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    identities = np.unique(sample_indices)
    sampled = rng.choice(identities, size=len(identities), replace=True)
    rows = [np.flatnonzero(sample_indices == identity) for identity in sampled]
    return np.concatenate(rows).astype(np.int64)


def fit_score_ensemble(payload: dict[str, np.ndarray], threshold: float, seed: int):
    features = feature_matrix(payload, ISRJ_LOCAL, threshold, "probability_plus_physical")
    risk, benefit, audit = _targets(payload, ISRJ_LOCAL, threshold)
    rng = np.random.default_rng(seed)
    models = []
    for member in range(ENSEMBLE_SIZE):
        rows = grouped_bootstrap_rows(payload["sample_indices"], rng)
        models.append({
            "risk": _fit_binary(features[rows], risk[rows], seed + member * 29),
            "benefit": _fit_binary(features[rows], benefit[rows], seed + member * 29 + 1),
        })
    return models, audit


def ensemble_scores(models, payload: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    features = feature_matrix(payload, ISRJ_LOCAL, threshold, "probability_plus_physical")
    member_scores = []
    for model in models:
        risk = model["risk"].predict_proba(features)[:, 1]
        benefit = model["benefit"].predict_proba(features)[:, 1]
        member_scores.append(clipped_logit(benefit) - clipped_logit(risk))
    matrix = np.column_stack(member_scores).astype(np.float32)
    return {
        "member": matrix,
        "mean": matrix.mean(axis=1).astype(np.float32),
        "lower": np.quantile(matrix, LOWER_QUANTILE, axis=1).astype(np.float32),
        "std": matrix.std(axis=1).astype(np.float32),
    }


def select_coverage_route(
    payload: dict[str, np.ndarray], score: np.ndarray, threshold: float, alpha: float = 0.75,
) -> dict[str, object]:
    rows = []
    for coverage in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        cutoff = float(np.quantile(score, 1.0 - coverage))
        accept = score >= cutoff
        metrics = route_metrics(
            payload["labels"], payload["main"], payload["expert"],
            accept, ISRJ_LOCAL, threshold, alpha=float(alpha),
        )
        row = {"coverage_budget": coverage, "score_cutoff": cutoff, **metrics}
        row["eligible"] = bool(metrics["fpr_delta"] <= 0.005 + 1e-12 and metrics["aupr_delta"] >= -0.005)
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return {
            "coverage_budget": 0.0, "score_cutoff": float("inf"), "recall_gain": 0.0,
            "fpr_delta": 0.0, "aupr_delta": 0.0, "accept_coverage": 0.0,
            "eligible": False, "candidate_rows": rows,
        }
    selected = max(eligible, key=lambda row: (row["recall_gain"], row["aupr_delta"], -row["accept_coverage"]))
    return {**selected, "candidate_rows": rows}


def apply_route(
    payload: dict[str, np.ndarray], score: np.ndarray, selection: dict[str, object], alpha: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    accept = score >= float(selection["score_cutoff"])
    final = payload["main"].copy()
    base = final[:, ISRJ_GLOBAL]
    final[:, ISRJ_GLOBAL] = np.clip(
        base + float(alpha) * accept * (payload["expert"][:, ISRJ_LOCAL] - base), 0.0, 1.0
    )
    return final.astype(np.float32), accept.astype(bool)


def buffered_isrj_threshold(
    payload: dict[str, np.ndarray], probabilities: np.ndarray,
    powers: tuple[int, ...] = POWERS,
) -> dict[str, object]:
    rows = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        by_power, feasible = [], True
        for power in powers:
            selected = payload["power"] == power
            truth = payload["labels"][selected, ISRJ_GLOBAL].astype(bool)
            predicted = probabilities[selected, ISRJ_GLOBAL] >= threshold
            false_positives = int(predicted[~truth].sum())
            negatives = int((~truth).sum())
            fpr = false_positives / max(negatives, 1)
            upper = clopper_pearson_upper(false_positives, negatives)
            if fpr > 0.06 + 1e-12 or upper > 0.08 + 1e-12:
                feasible = False
            by_power.append({
                "power_db": power, "false_positives": false_positives, "negatives": negatives,
                "observed_fpr": fpr, "cp95_upper": upper,
                "recall": float(predicted[truth].mean()),
            })
        if feasible:
            rows.append({
                "threshold": float(threshold),
                "mean_recall": float(np.mean([row["recall"] for row in by_power])),
                "by_power": by_power,
            })
    if not rows:
        return {"threshold": 1.0, "mean_recall": 0.0, "feasible": False, "reason": "no CP-buffered threshold"}
    selected = max(rows, key=lambda row: (row["mean_recall"], -row["threshold"]))
    selected["feasible"] = True
    selected["feasible_candidate_count"] = len(rows)
    return selected


def pooled_isrj_threshold(payload: dict[str, np.ndarray], probabilities: np.ndarray) -> dict[str, object]:
    threshold = float(threshold_at_fpr(
        payload["labels"][:, ISRJ_GLOBAL], probabilities[:, ISRJ_GLOBAL], 0.10
    ))
    by_power = []
    for power in POWERS:
        selected = payload["power"] == power
        truth = payload["labels"][selected, ISRJ_GLOBAL].astype(bool)
        predicted = probabilities[selected, ISRJ_GLOBAL] >= threshold
        false_positives, negatives = int(predicted[~truth].sum()), int((~truth).sum())
        by_power.append({
            "power_db": power,
            "observed_fpr": false_positives / max(negatives, 1),
            "cp95_upper": clopper_pearson_upper(false_positives, negatives),
            "recall": float(predicted[truth].mean()),
        })
    return {"threshold": threshold, "by_power": by_power, "pooled_fpr_cap": 0.10}


def collect_records(main_model, unet, experts, files, labels, indices, batch_size, seed, device, normalizer):
    records = []
    for power in POWERS:
        main = collect_main(main_model, files[power], indices, batch_size, device, normalizer)
        unet_record = extract_unet_role(
            unet, files[power], files[power], labels, indices, batch_size, seed + power, device
        )
        align_payloads(main, unet_record)
        records.append({
            "sample_indices": main["sample_indices"],
            "power": np.full(len(indices), power, dtype=np.int16),
            "labels": main["labels"],
            "main": main["probabilities"],
            "expert": expert_probabilities(unet_record, experts),
            "mft": unet_record["mft"],
            "trajectory": unet_record["trajectory"],
            "isrj": unet_record["isrj"],
            "prediction": main["prediction"],
            "target": main["target"],
            "mask": main["mask"],
        })
    return records


def evaluate_variant(payload, probabilities, thresholds, base_probabilities, base_thresholds, scale):
    rows = []
    for power in POWERS:
        selected = payload["power"] == power
        weak = _weak_mask(payload["labels"][selected].astype(bool), power)
        common = {
            "sample_indices": payload["sample_indices"][selected],
            "labels": payload["labels"][selected],
            "prediction": payload["prediction"][selected],
            "target": payload["target"][selected],
            "mask": payload["mask"][selected],
            "scale": scale,
        }
        base_class = classification_metrics(common["labels"], base_probabilities[selected], base_thresholds)
        final_class = classification_metrics(common["labels"], probabilities[selected], thresholds)
        truth = common["labels"][:, ISRJ_GLOBAL].astype(bool)
        final_pred = probabilities[selected, ISRJ_GLOBAL] >= thresholds[ISRJ_GLOBAL]
        false_positives, negatives = int(final_pred[~truth].sum()), int((~truth).sum())
        rows.append({
            "power_db": power,
            "base": base_class,
            "final": final_class,
            "isrj_cp95_upper": clopper_pearson_upper(false_positives, negatives),
            "base_parameter": _level_metrics({**common, "probabilities": base_probabilities[selected], "thresholds": base_thresholds}, weak),
            "final_parameter": _level_metrics({**common, "probabilities": probabilities[selected], "thresholds": thresholds}, weak),
        })
    return rows, {
        "mean_isrj_recall_gain": float(np.mean([
            row["final"]["per_class"][ISRJ_GLOBAL]["recall"] - row["base"]["per_class"][ISRJ_GLOBAL]["recall"] for row in rows
        ])),
        "mean_macro_aupr_delta": float(np.mean([row["final"]["macro_aupr"] - row["base"]["macro_aupr"] for row in rows])),
        "maximum_isrj_fpr": float(max(row["final"]["per_class"][ISRJ_GLOBAL]["fpr"] for row in rows)),
        "maximum_isrj_cp95_upper": float(max(row["isrj_cp95_upper"] for row in rows)),
        "mean_parameter_coverage_delta": float(np.mean([
            row["final_parameter"]["predicted_gate_parameter_coverage"] - row["base_parameter"]["predicted_gate_parameter_coverage"] for row in rows
        ])),
        "mean_miss_penalty_nmae_delta": float(np.mean([
            row["final_parameter"]["miss_penalty_parameter_nmae"] - row["base_parameter"]["miss_penalty_parameter_nmae"] for row in rows
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--separator-root", default="outputs/conditional_unet_separator_full_20260725")
    parser.add_argument("--original-power-root", default=r"C:\LJQ_code\干扰数据\复合干扰数据0716\power_sweep_9level_v1")
    parser.add_argument("--new-power-root", default="generated_data/p1f_isrj_transition_seed300000000_rechunked")
    parser.add_argument("--output-root", default="outputs/p1f_isrj_confidence_bound_router_20260731")
    parser.add_argument("--experiment-name", default="P1f ISRJ confidence-bound safe routing")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2041, 2042, 2043])
    parser.add_argument("--batch-size", type=int, default=126)
    args = parser.parse_args()

    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    run_root, separator_root = project / args.run_root, project / args.separator_root
    new_root, output_root = project / args.new_power_root, project / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    role_file = np.load(run_root / "role_indices.npz")
    original_train, original_early = role_file["train"].astype(np.int64), role_file["early"].astype(np.int64)
    original_root = Path(args.original_power_root)
    original_files = {
        power: original_root / f"level_{list(range(10, 51, 5)).index(power)+1:02d}__jnr_{power}dB" / "closedset_6class_train.mat"
        for power in (10, 50)
    }
    original_identities, original_labels = read_identity_metadata(protocol["data_file"])
    train_files = transition_files(new_root, "train")
    train_identities, train_labels = read_identity_metadata(train_files[20])
    roles = stratified_train_roles(train_labels)
    normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {
        int(row["seed"]): row for row in summary
        if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"
    }
    if np.intersect1d(train_identities, original_identities).size:
        raise RuntimeError("P1f train identities overlap original identities")

    frozen_models, seed_freezes = {}, []
    for seed in args.seeds:
        set_seed(seed)
        row = selected_rows[seed]
        raw_thresholds = row["thresholds"]
        main_thresholds = np.asarray(json.loads(raw_thresholds) if isinstance(raw_thresholds, str) else raw_thresholds, dtype=np.float32)
        main_model, main_checkpoint = _load_main(row, protocol, project, device)
        unet_checkpoint = separator_root / "unet_wave" / f"seed_{seed}" / "best.pt"
        unet = _load_unet(unet_checkpoint, device)
        expert_train = extract_unet_role(unet, original_files[10], original_files[50], original_labels, original_train, args.batch_size, seed, device)
        expert_early = extract_unet_role(unet, original_files[10], original_files[50], original_labels, original_early, args.batch_size, seed + 1, device)
        experts = fit_frozen_experts(expert_train, expert_early, seed)
        records = collect_records(
            main_model, unet, experts, train_files, train_labels,
            np.arange(len(train_labels), dtype=np.int64), args.batch_size, seed, device, normalizer,
        )
        payloads = {name: _role_payload(records, indices) for name, indices in roles.items()}
        ensemble, target_audit = fit_score_ensemble(payloads["fit"], float(main_thresholds[ISRJ_GLOBAL]), seed + 9000)
        selection_scores = ensemble_scores(ensemble, payloads["selection"], float(main_thresholds[ISRJ_GLOBAL]))
        selections = {
            name: select_coverage_route(payloads["selection"], selection_scores[name], float(main_thresholds[ISRJ_GLOBAL]))
            for name in ("mean", "lower")
        }
        calibration_scores = ensemble_scores(ensemble, payloads["calibration"], float(main_thresholds[ISRJ_GLOBAL]))
        base_calibration = payloads["calibration"]["main"]
        routed_calibration, accepts = {}, {}
        for name in ("mean", "lower"):
            routed_calibration[name], accepts[name] = apply_route(payloads["calibration"], calibration_scores[name], selections[name])
        thresholds = {"base": buffered_isrj_threshold(payloads["calibration"], base_calibration)}
        thresholds.update({name: buffered_isrj_threshold(payloads["calibration"], routed_calibration[name]) for name in ("mean", "lower")})
        pooled_thresholds = {"base": pooled_isrj_threshold(payloads["calibration"], base_calibration)}
        pooled_thresholds.update({name: pooled_isrj_threshold(payloads["calibration"], routed_calibration[name]) for name in ("mean", "lower")})
        seed_freeze = {
            "seed": seed,
            "main_checkpoint": str(main_checkpoint),
            "unet_checkpoint": str(unet_checkpoint),
            "main_thresholds": main_thresholds,
            "fit_target_audit": target_audit,
            "selections": selections,
            "buffered_thresholds": thresholds,
            "pooled_thresholds": pooled_thresholds,
            "role_hashes": {name: {"index": _hash_int64(values), "identity": _hash_int64(train_identities[values])} for name, values in roles.items()},
        }
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir()
        (seed_dir / "seed_freeze.json").write_text(json.dumps(_jsonable(seed_freeze), indent=2), encoding="utf-8")
        joblib.dump({"experts": experts, "ensemble": ensemble}, seed_dir / "frozen_models.joblib")
        frozen_models[seed] = {"experts": experts, "ensemble": ensemble}
        seed_freezes.append(seed_freeze)
        del main_model, unet
        if device.type == "cuda":
            torch.cuda.empty_cache()

    method_freeze = {
        "experiment": args.experiment_name,
        "recent_evidence": {
            "P1d": "fresh independent test worst FPR 0.11129 at 20-25 dB",
            "P1e": "ISRJ oracle recall ceiling 0.15417; MFT/VFT ceilings insufficient",
            "P1f": "confidence-lower route gained 0.07435 recall with worst FPR 0.04274, but did not improve safety versus the mean route",
        },
        "fresh_data_root": str(new_root),
        "train_identity_count": int(len(train_identities)),
        "train_identity_hash": _hash_int64(train_identities),
        "powers_db": list(POWERS),
        "target": "ISRJ",
        "feature_rule": "probability_plus_ISRJ_periodicity",
        "ensemble_size": ENSEMBLE_SIZE,
        "confidence_lower_quantile": LOWER_QUANTILE,
        "route_alpha": 0.75,
        "coverage_grid": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "threshold_buffer": {"observed_fpr_max": 0.06, "cp95_upper_max": 0.08, "every_power": True},
        "promotion_gates": {
            "mean_test_recall_gain_min": 0.10,
            "three_of_three_positive": True,
            "maximum_test_power_fpr": 0.10,
            "mean_macro_aupr_delta_min": -0.005,
            "miss_penalty_nmae_delta_max": 0.0,
            "lower_retains_mean_recall_fraction_min": 0.80,
            "confidence_safety_advantage": "lower worst FPR <= mean worst FPR - 0.005, or mean fails 0.10 while lower passes",
        },
        "risk_calibration_confirmation_gates": {
            "buffered_mean_recall_gain_min": 0.05,
            "three_of_three_positive": True,
            "buffered_maximum_test_power_fpr": 0.10,
            "macro_aupr_delta_min": -0.005,
            "miss_penalty_nmae_delta_max": 0.0,
            "retain_pooled_recall_fraction_min": 0.70,
            "safety_advantage": "buffered worst FPR <= pooled worst FPR - 0.02, or pooled fails 0.10 while buffered passes",
        },
        "seed_freezes": seed_freezes,
        "test_used_at_freeze": False,
        "confirm_used": False,
        "locked_test_used": False,
    }
    freeze_path = output_root / "method_freeze.json"
    freeze_path.write_text(json.dumps(_jsonable(method_freeze), indent=2), encoding="utf-8")
    freeze_hash = _sha256(freeze_path)
    (output_root / "test_access_started.json").write_text(json.dumps({
        "method_freeze_sha256": freeze_hash,
        "purpose": f"one-time fresh test evaluation for {args.experiment_name} after complete method freeze",
        "started_unix": time.time(),
    }, indent=2), encoding="utf-8")

    test_files = transition_files(new_root, "test")
    test_identities, test_labels = read_identity_metadata(test_files[20])
    identity_audit = {
        "train_count": int(len(train_identities)), "test_count": int(len(test_identities)),
        "train_hash": _hash_int64(train_identities), "test_hash": _hash_int64(test_identities),
        "train_test_overlap": int(np.intersect1d(train_identities, test_identities).size),
        "test_original_overlap": int(np.intersect1d(test_identities, original_identities).size),
    }
    if identity_audit["train_test_overlap"] or identity_audit["test_original_overlap"]:
        raise RuntimeError("P1f test identity isolation failed")

    results = []
    for seed, seed_freeze in zip(args.seeds, seed_freezes):
        set_seed(seed)
        row = selected_rows[seed]
        main_thresholds = np.asarray(seed_freeze["main_thresholds"], dtype=np.float32)
        main_model, _ = _load_main(row, protocol, project, device)
        unet = _load_unet(Path(seed_freeze["unet_checkpoint"]), device)
        models = frozen_models[seed]
        records = collect_records(
            main_model, unet, models["experts"], test_files, test_labels,
            np.arange(len(test_labels), dtype=np.int64), args.batch_size, seed + 100000, device, normalizer,
        )
        test_payload = _role_payload(records, np.arange(len(test_labels), dtype=np.int64))
        scores = ensemble_scores(models["ensemble"], test_payload, float(main_thresholds[ISRJ_GLOBAL]))
        base_probabilities = test_payload["main"]
        base_thresholds = main_thresholds.copy()
        base_thresholds[ISRJ_GLOBAL] = float(seed_freeze["buffered_thresholds"]["base"]["threshold"])
        pooled_base_thresholds = main_thresholds.copy()
        pooled_base_thresholds[ISRJ_GLOBAL] = float(seed_freeze["pooled_thresholds"]["base"]["threshold"])
        variants = {}
        saved = {
            "sample_index": test_payload["sample_indices"], "receiver_noise_seed": test_identities[test_payload["sample_indices"]],
            "power_db": test_payload["power"], "labels": test_payload["labels"].astype(np.uint8),
            "main_probability": base_probabilities, "expert_probability": test_payload["expert"],
        }
        for name in ("mean", "lower"):
            final, accept = apply_route(test_payload, scores[name], seed_freeze["selections"][name])
            thresholds = main_thresholds.copy()
            thresholds[ISRJ_GLOBAL] = float(seed_freeze["buffered_thresholds"][name]["threshold"])
            by_power, aggregate = evaluate_variant(
                test_payload, final, thresholds, base_probabilities, base_thresholds, normalizer.scale.numpy()
            )
            variants[name] = {"by_power": by_power, "aggregate": aggregate, "accept_coverage": float(accept.mean())}
            saved[f"{name}_score"] = scores[name]
            saved[f"{name}_accept"] = accept.astype(np.uint8)
            saved[f"{name}_final_probability"] = final
            saved[f"{name}_thresholds"] = thresholds
        lower_final = saved["lower_final_probability"]
        pooled_lower_thresholds = main_thresholds.copy()
        pooled_lower_thresholds[ISRJ_GLOBAL] = float(seed_freeze["pooled_thresholds"]["lower"]["threshold"])
        pooled_by_power, pooled_aggregate = evaluate_variant(
            test_payload, lower_final, pooled_lower_thresholds,
            base_probabilities, pooled_base_thresholds, normalizer.scale.numpy()
        )
        variants["lower_pooled"] = {
            "by_power": pooled_by_power, "aggregate": pooled_aggregate,
            "accept_coverage": variants["lower"]["accept_coverage"],
        }
        saved["lower_pooled_thresholds"] = pooled_lower_thresholds
        seed_result = {
            "seed": seed, "variants": variants,
            "method_freeze_sha256": freeze_hash,
            "identity_audit": identity_audit,
            "test_fit_or_selection": False,
            "confirm_used": False, "locked_test_used": False,
        }
        seed_dir = output_root / f"seed_{seed}"
        np.savez_compressed(seed_dir / "test_per_sample.npz", **saved)
        (seed_dir / "test_result.json").write_text(json.dumps(_jsonable(seed_result), indent=2), encoding="utf-8")
        results.append(seed_result)
        del main_model, unet
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_gain = [row["variants"]["mean"]["aggregate"]["mean_isrj_recall_gain"] for row in results]
    lower_gain = [row["variants"]["lower"]["aggregate"]["mean_isrj_recall_gain"] for row in results]
    mean_fpr = [row["variants"]["mean"]["aggregate"]["maximum_isrj_fpr"] for row in results]
    lower_fpr = [row["variants"]["lower"]["aggregate"]["maximum_isrj_fpr"] for row in results]
    lower_macro = [row["variants"]["lower"]["aggregate"]["mean_macro_aupr_delta"] for row in results]
    lower_miss = [row["variants"]["lower"]["aggregate"]["mean_miss_penalty_nmae_delta"] for row in results]
    pooled_gain = [row["variants"]["lower_pooled"]["aggregate"]["mean_isrj_recall_gain"] for row in results]
    pooled_fpr = [row["variants"]["lower_pooled"]["aggregate"]["maximum_isrj_fpr"] for row in results]
    retention = float(np.mean(lower_gain) / max(np.mean(mean_gain), 1e-12))
    safety_advantage = bool(
        max(lower_fpr) <= max(mean_fpr) - 0.005
        or (max(mean_fpr) > 0.10 and max(lower_fpr) <= 0.10)
    )
    gates = {
        "lower_mean_recall_gain_at_least_0p10": bool(np.mean(lower_gain) >= 0.10),
        "three_of_three_positive": bool(min(lower_gain) > 0.0),
        "maximum_power_fpr_at_most_0p10": bool(max(lower_fpr) <= 0.10 + 1e-12),
        "macro_aupr_noninferior": bool(np.mean(lower_macro) >= -0.005),
        "miss_penalty_nonworse": bool(np.mean(lower_miss) <= 0.0),
        "retains_at_least_80pct_mean_recall": bool(retention >= 0.80),
        "confidence_safety_advantage": safety_advantage,
    }
    risk_retention = float(np.mean(lower_gain) / max(np.mean(pooled_gain), 1e-12))
    risk_safety_advantage = bool(
        max(lower_fpr) <= max(pooled_fpr) - 0.02
        or (max(pooled_fpr) > 0.10 and max(lower_fpr) <= 0.10)
    )
    risk_gates = {
        "buffered_mean_recall_gain_at_least_0p05": bool(np.mean(lower_gain) >= 0.05),
        "three_of_three_positive": bool(min(lower_gain) > 0.0),
        "buffered_maximum_power_fpr_at_most_0p10": bool(max(lower_fpr) <= 0.10 + 1e-12),
        "macro_aupr_noninferior": bool(np.mean(lower_macro) >= -0.005),
        "miss_penalty_nonworse": bool(np.mean(lower_miss) <= 0.0),
        "retains_at_least_70pct_pooled_recall": bool(risk_retention >= 0.70),
        "buffered_safety_advantage": risk_safety_advantage,
    }
    summary_payload = {
        "experiment": args.experiment_name,
        "mean_route": {"mean_recall_gain": float(np.mean(mean_gain)), "per_seed_gain": mean_gain, "maximum_fpr": float(max(mean_fpr))},
        "confidence_lower_route": {
            "mean_recall_gain": float(np.mean(lower_gain)), "per_seed_gain": lower_gain,
            "maximum_fpr": float(max(lower_fpr)), "mean_macro_aupr_delta": float(np.mean(lower_macro)),
            "mean_miss_penalty_nmae_delta": float(np.mean(lower_miss)), "recall_retention_vs_mean": retention,
        },
        "gates": gates,
        "innovation_validated": bool(all(gates.values())),
        "pooled_fpr_ablation": {
            "mean_recall_gain": float(np.mean(pooled_gain)), "per_seed_gain": pooled_gain,
            "maximum_fpr": float(max(pooled_fpr)),
        },
        "risk_calibration_confirmation": {
            "recall_retention_vs_pooled": risk_retention,
            "gates": risk_gates,
            "innovation_validated": bool(all(risk_gates.values())),
        },
        "method_freeze_sha256": freeze_hash,
        "identity_audit": identity_audit,
        "elapsed_seconds": float(time.perf_counter() - started),
        "test_used_once": True, "test_fit_or_selection": False,
        "confirm_used": False, "locked_test_used": False,
    }
    (output_root / "results.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(_jsonable(summary_payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary_payload), indent=2))


if __name__ == "__main__":
    main()

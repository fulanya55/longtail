"""Evaluate a cheap raw-physics ISRJ router against the frozen safe MLCNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from cj0724.b1_physics_sufficiency import extract_physics_features
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_main, collect_main
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.paper_core_evidence import load_normalizer


MODEL_SEEDS = (2041, 2042, 2043)
ISRJ_INDEX = 5


def cp_upper(false_positives: int, negatives: int, alpha: float = 0.05) -> float:
    if negatives <= 0: return 1.0
    if false_positives >= negatives: return 1.0
    return float(beta.ppf(1.0 - alpha, false_positives + 1, negatives - false_positives))


def choose_threshold(probability: np.ndarray, labels: np.ndarray, regime: np.ndarray, power: np.ndarray) -> dict[str, object]:
    feasible = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        rows, valid = [], True
        for regime_index in range(len(REGIMES)):
            for power_value in POWERS:
                chosen = (regime == regime_index) & (power == power_value)
                truth = labels[chosen].astype(bool); pred = probability[chosen] >= threshold
                fp, neg = int(pred[~truth].sum()), int((~truth).sum())
                observed, upper = fp / max(neg, 1), cp_upper(fp, neg)
                recall = float(pred[truth].mean())
                if observed > 0.08 + 1e-12 or upper > 0.15 + 1e-12: valid = False
                rows.append({"regime": REGIMES[regime_index], "power": power_value, "fpr": observed, "cp95_upper": upper, "recall": recall})
        if valid: feasible.append({"threshold": float(threshold), "mean_recall": float(np.mean([row["recall"] for row in rows])), "cells": rows})
    if not feasible: return {"threshold": 1.0, "mean_recall": 0.0, "feasible": False, "cells": []}
    selected = max(feasible, key=lambda row: (row["mean_recall"], -row["threshold"])); selected["feasible"] = True; selected["feasible_count"] = len(feasible)
    return selected


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, regime: np.ndarray, power: np.ndarray) -> dict[str, object]:
    prediction = probabilities >= thresholds[None, :]
    per_aupr = [float(average_precision_score(labels[:, index], probabilities[:, index])) for index in range(6)]
    cells = []
    for regime_index in range(len(REGIMES)):
        for power_value in POWERS:
            chosen = (regime == regime_index) & (power == power_value)
            truth = labels[chosen, ISRJ_INDEX].astype(bool); pred = prediction[chosen, ISRJ_INDEX]
            cells.append({"regime": REGIMES[regime_index], "power": power_value, "recall": float(pred[truth].mean()), "fpr": float(pred[~truth].mean()), "aupr": float(average_precision_score(truth, probabilities[chosen, ISRJ_INDEX]))})
    return {"macro_aupr": float(np.mean(per_aupr)), "isrj_aupr": per_aupr[ISRJ_INDEX], "cells": cells}


def collect_features(files_by_regime, metadata, role_indices: np.ndarray | None) -> dict[str, np.ndarray]:
    parts = {name: [] for name in ("features", "labels", "identity", "regime", "power", "sample_index")}
    for regime_index, regime in enumerate(REGIMES):
        record = metadata[regime]
        indices = np.arange(len(record["identities"])) if role_indices is None else role_indices
        for power in POWERS:
            parts["features"].append(extract_physics_features(files_by_regime[regime][power], indices, 128)["isrj_periodicity"])
            parts["labels"].append(record["labels"][indices])
            parts["identity"].append(record["identities"][indices])
            parts["regime"].append(np.full(len(indices), regime_index, dtype=np.int8))
            parts["power"].append(np.full(len(indices), power, dtype=np.int16))
            parts["sample_index"].append(indices.astype(np.int64))
    return {name: np.concatenate(values) for name, values in parts.items()}


def collect_main_probabilities(model, normalizer, files_by_regime, metadata, role_indices: np.ndarray | None, device: torch.device) -> np.ndarray:
    parts = []
    for regime in REGIMES:
        indices = np.arange(len(metadata[regime]["identities"])) if role_indices is None else role_indices
        for power in POWERS:
            parts.append(collect_main(model, files_by_regime[regime][power], indices, 128, device, normalizer)["probabilities"])
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3f_shared_isrj_train_seed480000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3f_shared_isrj_dev_seed490000000_{regime}_rechunked")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--preregister", default="configs/p3f_shared_isrj_router_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3f_shared_isrj_router_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]; output = project / args.output_root; output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((project / args.run_root / "protocol.json").read_text(encoding="utf-8")); summary_rows = json.loads((project / args.run_root / "summary.json").read_text(encoding="utf-8"))
    p2c = json.loads((project / args.p2c_root / "method_freeze.json").read_text(encoding="utf-8")); normalizer = load_normalizer(protocol["normalizer"])
    excluded = [Path(protocol["data_file"])] + [Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files]
    roots = {role: {regime: transition_files(project / template.format(regime=regime), "train", POWERS) for regime in REGIMES} for role, template in (("train", args.train_root_template), ("development", args.development_root_template))}
    metadata, references = {"train": {}, "development": {}}, {}
    for role in ("train", "development"):
        reference = None
        for regime in REGIMES:
            audit_power_alignment(roots[role][regime]); identities, labels = read_identity_metadata(roots[role][regime][POWERS[0]])
            if len(identities) != 630: raise RuntimeError("P3f identity count mismatch")
            if reference is None: reference = identities
            elif not np.array_equal(reference, identities): raise RuntimeError("P3f regimes not identity paired")
            overlaps = []
            for path in excluded:
                previous, _ = read_identity_metadata(path); overlaps.append({"path": str(path.resolve()), "overlap_count": int(np.intersect1d(identities, previous).size)})
            if any(row["overlap_count"] for row in overlaps): raise RuntimeError("P3f identity overlap")
            metadata[role][regime] = {"identities": identities, "labels": labels, "exclusions": overlaps}
        references[role] = reference
    if np.intersect1d(references["train"], references["development"]).size: raise RuntimeError("P3f cross-role overlap")
    roles = stratified_roles_10(metadata["train"]["ID_control"]["labels"], 20260801)
    fit = collect_features(roots["train"], metadata["train"], roles["fit"])
    selection = collect_features(roots["train"], metadata["train"], roles["selection"])
    development = collect_features(roots["development"], metadata["development"], None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); runs = []
    selected_rows = {int(row["seed"]): row for row in summary_rows if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}
    freezes = {int(row["seed"]): row for row in p2c["seed_freezes"]}
    for seed in MODEL_SEEDS:
        model, _ = _load_main(selected_rows[seed], protocol, project, device)
        selection_main = collect_main_probabilities(model, normalizer, roots["train"], metadata["train"], roles["selection"], device)
        development_main = collect_main_probabilities(model, normalizer, roots["development"], metadata["development"], None, device)
        classifier = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, random_state=seed)
        classifier.fit(fit["features"], fit["labels"][:, ISRJ_INDEX])
        selection_probability = classifier.predict_proba(selection["features"])[:, 1]
        threshold = choose_threshold(selection_probability, selection["labels"][:, ISRJ_INDEX], selection["regime"], selection["power"])
        development_probability = classifier.predict_proba(development["features"])[:, 1]
        baseline_thresholds = np.asarray(freezes[seed]["routed_thresholds"], dtype=np.float64)
        candidate_thresholds = baseline_thresholds.copy(); candidate_thresholds[ISRJ_INDEX] = threshold["threshold"]
        candidate_probability = development_main.copy(); candidate_probability[:, ISRJ_INDEX] = development_probability
        baseline = classification_metrics(development_main, development["labels"], baseline_thresholds, development["regime"], development["power"])
        candidate = classification_metrics(candidate_probability, development["labels"], candidate_thresholds, development["regime"], development["power"])
        cell_gain = [candidate["cells"][index]["recall"] - baseline["cells"][index]["recall"] for index in range(len(candidate["cells"]))]
        regime_gain = {regime: float(np.mean([cell_gain[index] for index, row in enumerate(candidate["cells"]) if row["regime"] == regime])) for regime in REGIMES}
        run = {"seed": seed, "threshold_selection": threshold, "baseline": baseline, "candidate": candidate, "mean_recall_gain": float(np.mean(cell_gain)), "maximum_fpr": float(max(row["fpr"] for row in candidate["cells"])), "macro_aupr_delta": float(candidate["macro_aupr"] - baseline["macro_aupr"]), "regime_mean_recall_gain": regime_gain}
        run_dir = output / f"seed_{seed}"; run_dir.mkdir(); joblib.dump(classifier, run_dir / "isrj_shared_physics.joblib")
        np.savez_compressed(run_dir / "development_predictions.npz", identity=development["identity"], regime=development["regime"], power=development["power"], labels=development["labels"], baseline_probability=development_main, candidate_probability=candidate_probability, baseline_thresholds=baseline_thresholds, candidate_thresholds=candidate_thresholds)
        (run_dir / "result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8"); runs.append(run); print(json.dumps({"seed": seed, "mean_recall_gain": run["mean_recall_gain"], "maximum_fpr": run["maximum_fpr"], "macro_aupr_delta": run["macro_aupr_delta"], "worst_regime_gain": min(regime_gain.values())}), flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    gates = {"identity_overlap_zero": True, "each_seed_mean_recall_gain_at_least_0p05": bool(all(run["mean_recall_gain"] >= 0.05 for run in runs)), "three_of_three_recall_gain_positive": bool(all(run["mean_recall_gain"] > 0 for run in runs)), "maximum_fpr_at_most_0p10": bool(all(run["maximum_fpr"] <= 0.10 for run in runs)), "each_seed_macro_aupr_delta_at_least_minus_0p005": bool(all(run["macro_aupr_delta"] >= -0.005 for run in runs)), "each_seed_worst_regime_gain_nonnegative": bool(all(min(run["regime_mean_recall_gain"].values()) >= 0 for run in runs))}
    passed = bool(all(gates.values()))
    summary = {"experiment": "P3f shared raw-physics ISRJ router", "preregister_sha256": sha256_file(project / args.preregister), "identity_audit": {"train_sha256": _hash_int64(references["train"]), "development_sha256": _hash_int64(references["development"]), "cross_role_overlap": 0}, "promotion_gates": gates, "development_passed": passed, "runs": runs, "interpretation": "Freeze for fresh end-to-end and efficiency confirmation." if passed else "Shared raw-physics replacement failed; retain P1n UNet and unresolved cost limitation.", "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8"); print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__": main()

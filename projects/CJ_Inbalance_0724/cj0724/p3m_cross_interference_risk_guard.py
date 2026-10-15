"""Learn a bounded veto for P1n false positives caused by cross-interference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3f_shared_isrj_router import classification_metrics
from cj0724.p3g_cascaded_isrj_prefilter import collect_shared
from cj0724.p3h_latent_physics_prefilter import collect_route_and_latent
from cj0724.p3i_distilled_isrj_student import mean_recall
from cj0724.paper_core_evidence import load_normalizer


MODEL_SEEDS = (2041, 2042, 2043)
ISRJ_INDEX = 5


def apply_veto(probability: np.ndarray, score: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    veto = score >= threshold
    result = probability.copy()
    result[veto, ISRJ_INDEX] = 0.0
    return result, veto


def select_veto(score, full_probability, main_probability, labels, thresholds, regime, power):
    main_metric = classification_metrics(main_probability, labels, thresholds, regime, power)
    full_metric = classification_metrics(full_probability, labels, thresholds, regime, power)
    full_gain = mean_recall(full_metric) - mean_recall(main_metric)
    candidates = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        guarded, veto = apply_veto(full_probability, score, float(threshold))
        metric = classification_metrics(guarded, labels, thresholds, regime, power)
        gain = mean_recall(metric) - mean_recall(main_metric)
        retention = gain / max(full_gain, 1e-12)
        aupr_delta = metric["macro_aupr"] - full_metric["macro_aupr"]
        if retention >= 0.90 and aupr_delta >= -0.005:
            candidates.append({"threshold": float(threshold), "veto_coverage": float(veto.mean()), "recall_gain_retention": float(retention), "maximum_cell_fpr": float(max(row["fpr"] for row in metric["cells"])), "macro_aupr_delta_vs_p1n": float(aupr_delta)})
    if not candidates:
        return {"threshold": 1.1, "veto_coverage": 0.0, "recall_gain_retention": 1.0, "maximum_cell_fpr": float(max(row["fpr"] for row in full_metric["cells"])), "macro_aupr_delta_vs_p1n": 0.0, "feasible": False}
    selected = min(candidates, key=lambda row: (row["maximum_cell_fpr"], row["veto_coverage"], -row["recall_gain_retention"], -row["threshold"]))
    selected["feasible"] = True; selected["candidate_count"] = len(candidates)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3m_risk_guard_train_seed660000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3m_risk_guard_dev_seed670000000_{regime}_rechunked")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156"); parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--preregister", default="configs/p3m_cross_interference_risk_guard_preregister_20260801.json"); parser.add_argument("--output-root", default="outputs/p3m_cross_interference_risk_guard_development_20260801"); parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; output = project / args.output_root; output.mkdir(parents=True, exist_ok=False); run_root, p2c_root = project / args.run_root, project / args.p2c_root
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8")); rows = json.loads((run_root / "summary.json").read_text(encoding="utf-8")); p2c = json.loads((p2c_root / "method_freeze.json").read_text(encoding="utf-8")); normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {int(row["seed"]): row for row in rows if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}; freezes = {int(row["seed"]): row for row in p2c["seed_freezes"]}; excluded = [Path(protocol["data_file"])] + [Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files]
    roots = {role: {regime: transition_files(project / template.format(regime=regime), "train", POWERS) for regime in REGIMES} for role, template in (("train", args.train_root_template), ("development", args.development_root_template))}; metadata, references = {"train": {}, "development": {}}, {}
    for role in ("train", "development"):
        reference = None
        for regime in REGIMES:
            audit_power_alignment(roots[role][regime]); identities, labels = read_identity_metadata(roots[role][regime][POWERS[0]])
            if reference is None: reference = identities
            elif not np.array_equal(reference, identities): raise RuntimeError("P3m regimes not paired")
            for path in excluded:
                previous, _ = read_identity_metadata(path)
                if np.intersect1d(identities, previous).size: raise RuntimeError("P3m identity overlap")
            metadata[role][regime] = {"identities": identities, "labels": labels}
        references[role] = reference
    if np.intersect1d(references["train"], references["development"]).size: raise RuntimeError("P3m cross-role overlap")
    roles = stratified_roles_10(metadata["train"]["ID_control"]["labels"], 20260801); shared_train = collect_shared(roots["train"], metadata["train"]); shared_dev = collect_shared(roots["development"], metadata["development"]); fit_mask = np.isin(shared_train["sample_index"], roles["fit"]); selection_mask = np.isin(shared_train["sample_index"], roles["selection"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); results = []
    for seed in MODEL_SEEDS:
        train_route = collect_route_and_latent(project, roots["train"], metadata["train"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root, protocol); dev_route = collect_route_and_latent(project, roots["development"], metadata["development"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root, protocol); thresholds = np.asarray(freezes[seed]["routed_thresholds"], dtype=np.float64)
        x_train = np.concatenate((train_route["latent"], shared_train["features"], train_route["main"]), axis=1); x_dev = np.concatenate((dev_route["latent"], shared_dev["features"], dev_route["main"]), axis=1); truth = train_route["labels"][:, ISRJ_INDEX].astype(bool); full_prediction = train_route["full"][:, ISRJ_INDEX] >= thresholds[ISRJ_INDEX]; false_positive = ((~truth) & full_prediction).astype(np.uint8)
        guard = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, class_weight="balanced", random_state=seed); guard.fit(x_train[fit_mask], false_positive[fit_mask]); selection_score = guard.predict_proba(x_train[selection_mask])[:, 1]; selection = select_veto(selection_score, train_route["full"][selection_mask], train_route["main"][selection_mask], train_route["labels"][selection_mask], thresholds, shared_train["regime"][selection_mask], shared_train["power"][selection_mask])
        dev_score = guard.predict_proba(x_dev)[:, 1]; guarded_probability, veto = apply_veto(dev_route["full"], dev_score, selection["threshold"]); main_metric = classification_metrics(dev_route["main"], dev_route["labels"], thresholds, shared_dev["regime"], shared_dev["power"]); full_metric = classification_metrics(dev_route["full"], dev_route["labels"], thresholds, shared_dev["regime"], shared_dev["power"]); guarded_metric = classification_metrics(guarded_probability, dev_route["labels"], thresholds, shared_dev["regime"], shared_dev["power"]); full_gain, guarded_gain = mean_recall(full_metric) - mean_recall(main_metric), mean_recall(guarded_metric) - mean_recall(main_metric); retention = guarded_gain / max(full_gain, 1e-12)
        run = {"seed": seed, "selection": selection, "veto_coverage": float(veto.mean()), "full_recall_gain": full_gain, "guarded_recall_gain": guarded_gain, "recall_gain_retention": retention, "maximum_cell_fpr": float(max(row["fpr"] for row in guarded_metric["cells"])), "macro_aupr_delta_vs_p1n": float(guarded_metric["macro_aupr"] - full_metric["macro_aupr"]), "main": main_metric, "p1n": full_metric, "guarded": guarded_metric}; run_dir = output / f"seed_{seed}"; run_dir.mkdir(); joblib.dump(guard, run_dir / "false_positive_guard.joblib"); np.savez_compressed(run_dir / "development_predictions.npz", identity=shared_dev["identity"], regime=shared_dev["regime"], power=shared_dev["power"], labels=dev_route["labels"], main_probability=dev_route["main"], p1n_probability=dev_route["full"], guarded_probability=guarded_probability, risk_score=dev_score, veto=veto.astype(np.uint8), thresholds=thresholds); (run_dir / "result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8"); results.append(run); print(json.dumps({key: run[key] for key in ("seed", "veto_coverage", "full_recall_gain", "guarded_recall_gain", "recall_gain_retention", "maximum_cell_fpr", "macro_aupr_delta_vs_p1n")}), flush=True)
    gates = {"identity_overlap_zero": True, "each_seed_retain_90pct_p1n_recall_gain": bool(all(run["recall_gain_retention"] >= 0.90 for run in results)), "maximum_cell_fpr_at_most_0p10": bool(all(run["maximum_cell_fpr"] <= 0.10 for run in results)), "each_seed_macro_aupr_delta_vs_p1n_at_least_minus_0p005": bool(all(run["macro_aupr_delta_vs_p1n"] >= -0.005 for run in results)), "three_of_three_positive_gain": bool(all(run["guarded_recall_gain"] > 0 for run in results))}; passed = bool(all(gates.values())); summary = {"experiment": "P3m cross-interference consistency risk guard", "preregister_sha256": sha256_file(project / args.preregister), "identity_audit": {"train_sha256": _hash_int64(references["train"]), "development_sha256": _hash_int64(references["development"]), "cross_role_overlap": 0}, "runs": results, "promotion_gates": gates, "development_passed": passed, "interpretation": "Freeze guard for two-identity confirmation." if passed else "Latent false-positive guard failed; freeze branch and redesign main representation with explicit cross-interference supervision.", "test_generated": False, "test_opened": False, "locked_test_used": False}; (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8"); print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__": main()

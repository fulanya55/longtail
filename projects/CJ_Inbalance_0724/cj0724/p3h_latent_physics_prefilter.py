"""Test whether frozen MLCNN embeddings make P1n decision changes cheaply predictable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import (
    _jsonable, _load_main, _load_unet, align_payloads, collect_main,
    expert_probabilities, extract_unet_role,
)
from cj0724.p1f_isrj_confidence_bound_router import apply_route, ensemble_scores, transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3g_cascaded_isrj_prefilter import collect_shared, metrics, select_gate_threshold
from cj0724.paper_core_evidence import load_normalizer


MODEL_SEEDS = (2041, 2042, 2043)
ISRJ_INDEX = 5


def collect_route_and_latent(project: Path, files_by_regime, metadata, seed: int, selected_row, seed_freeze, normalizer, device: torch.device, p2c_root: Path, protocol: dict) -> dict[str, np.ndarray]:
    model, _ = _load_main(selected_row, protocol, project, device); unet = _load_unet(Path(seed_freeze["unet_checkpoint"]), device)
    frozen = joblib.load(p2c_root / f"seed_{seed}" / "frozen_models.joblib"); original_threshold = float(seed_freeze["original_thresholds"][ISRJ_INDEX]); indices = np.arange(630, dtype=np.int64)
    parts = {name: [] for name in ("main", "full", "latent", "labels", "regime", "power")}
    for regime_index, regime in enumerate(REGIMES):
        labels = metadata[regime]["labels"]
        for power in POWERS:
            captured = []
            hook = model.classification_adapter.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().cpu().numpy()))
            main = collect_main(model, files_by_regime[regime][power], indices, 128, device, normalizer); hook.remove()
            latent = np.concatenate(captured)
            if len(latent) != len(indices): raise RuntimeError("P3h latent capture mismatch")
            unet_record = extract_unet_role(unet, files_by_regime[regime][power], files_by_regime[regime][power], labels, indices, 128, seed + power + regime_index * 1000, device); align_payloads(main, unet_record)
            record = {"sample_indices": main["sample_indices"], "power": np.full(630, power, dtype=np.int16), "labels": main["labels"], "main": main["probabilities"], "expert": expert_probabilities(unet_record, frozen["experts"]), "mft": unet_record["mft"], "trajectory": unet_record["trajectory"], "isrj": unet_record["isrj"]}
            score = ensemble_scores(frozen["ensemble"], record, original_threshold)["lower"]; full, _ = apply_route(record, score, seed_freeze["isrj_selection"], 1.0)
            parts["main"].append(main["probabilities"]); parts["full"].append(full); parts["latent"].append(latent); parts["labels"].append(main["labels"]); parts["regime"].append(np.full(630, regime_index, dtype=np.int8)); parts["power"].append(np.full(630, power, dtype=np.int16))
    del model, unet
    if device.type == "cuda": torch.cuda.empty_cache()
    return {name: np.concatenate(values) for name, values in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3h_latent_prefilter_train_seed550000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3h_latent_prefilter_dev_seed560000000_{regime}_rechunked")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--preregister", default="configs/p3h_latent_physics_prefilter_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3h_latent_physics_prefilter_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; output = project / args.output_root; output.mkdir(parents=True, exist_ok=False)
    run_root, p2c_root = project / args.run_root, project / args.p2c_root; protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8")); summary_rows = json.loads((run_root / "summary.json").read_text(encoding="utf-8")); p2c = json.loads((p2c_root / "method_freeze.json").read_text(encoding="utf-8")); normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {int(row["seed"]): row for row in summary_rows if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}; freezes = {int(row["seed"]): row for row in p2c["seed_freezes"]}
    excluded = [Path(protocol["data_file"])] + [Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files]
    roots = {role: {regime: transition_files(project / template.format(regime=regime), "train", POWERS) for regime in REGIMES} for role, template in (("train", args.train_root_template), ("development", args.development_root_template))}
    metadata, references = {"train": {}, "development": {}}, {}
    for role in ("train", "development"):
        reference = None
        for regime in REGIMES:
            audit_power_alignment(roots[role][regime]); identities, labels = read_identity_metadata(roots[role][regime][POWERS[0]])
            if reference is None: reference = identities
            elif not np.array_equal(reference, identities): raise RuntimeError("P3h regimes not paired")
            for path in excluded:
                previous, _ = read_identity_metadata(path)
                if np.intersect1d(identities, previous).size: raise RuntimeError("P3h identity overlap")
            metadata[role][regime] = {"identities": identities, "labels": labels}
        references[role] = reference
    if np.intersect1d(references["train"], references["development"]).size: raise RuntimeError("P3h cross-role overlap")
    roles = stratified_roles_10(metadata["train"]["ID_control"]["labels"], 20260801); shared_train = collect_shared(roots["train"], metadata["train"]); shared_dev = collect_shared(roots["development"], metadata["development"]); fit_mask = np.isin(shared_train["sample_index"], roles["fit"]); selection_mask = np.isin(shared_train["sample_index"], roles["selection"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); results = []
    for seed in MODEL_SEEDS:
        train_route = collect_route_and_latent(project, roots["train"], metadata["train"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root, protocol); dev_route = collect_route_and_latent(project, roots["development"], metadata["development"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root, protocol)
        threshold = np.asarray(freezes[seed]["routed_thresholds"], dtype=np.float64); changed = (train_route["full"][:, ISRJ_INDEX] >= threshold[ISRJ_INDEX]) != (train_route["main"][:, ISRJ_INDEX] >= threshold[ISRJ_INDEX])
        x_train = np.concatenate((train_route["latent"], shared_train["features"], train_route["main"]), axis=1); x_dev = np.concatenate((dev_route["latent"], shared_dev["features"], dev_route["main"]), axis=1)
        gate = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, class_weight="balanced", random_state=seed); gate.fit(x_train[fit_mask], changed[fit_mask]); selection_score = gate.predict_proba(x_train[selection_mask])[:, 1]; selection = select_gate_threshold(selection_score, changed[selection_mask], shared_train["regime"][selection_mask], shared_train["power"][selection_mask])
        dev_score = gate.predict_proba(x_dev)[:, 1]; call = dev_score >= selection["threshold"]; cascade = dev_route["main"].copy(); cascade[call, ISRJ_INDEX] = dev_route["full"][call, ISRJ_INDEX]
        safe, full, candidate = metrics(dev_route["main"], dev_route["labels"], threshold, shared_dev["regime"], shared_dev["power"]), metrics(dev_route["full"], dev_route["labels"], threshold, shared_dev["regime"], shared_dev["power"]), metrics(cascade, dev_route["labels"], threshold, shared_dev["regime"], shared_dev["power"])
        full_gain, candidate_gain = full["isrj_recall"] - safe["isrj_recall"], candidate["isrj_recall"] - safe["isrj_recall"]; retention = candidate_gain / max(full_gain, 1e-12)
        run = {"seed": seed, "selection": selection, "unet_call_coverage": float(call.mean()), "full_recall_gain": full_gain, "cascade_recall_gain": candidate_gain, "recall_gain_retention": retention, "maximum_cell_fpr": float(max(row["fpr"] for row in candidate["cells"])), "macro_aupr_delta_vs_full": float(candidate["macro_aupr"] - full["macro_aupr"]), "safe": safe, "full_p1n": full, "cascade": candidate}
        run_dir = output / f"seed_{seed}"; run_dir.mkdir(); joblib.dump(gate, run_dir / "latent_physics_prefilter.joblib"); np.savez_compressed(run_dir / "development_predictions.npz", identity=shared_dev["identity"], regime=shared_dev["regime"], power=shared_dev["power"], labels=dev_route["labels"], safe_probability=dev_route["main"], full_probability=dev_route["full"], cascade_probability=cascade, gate_score=dev_score, gate_call=call.astype(np.uint8), thresholds=threshold); (run_dir / "result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8"); results.append(run); print(json.dumps({key: run[key] for key in ("seed", "unet_call_coverage", "full_recall_gain", "cascade_recall_gain", "recall_gain_retention", "maximum_cell_fpr", "macro_aupr_delta_vs_full")}), flush=True)
    gates = {"identity_overlap_zero": True, "each_seed_retain_90pct_recall_gain": bool(all(run["recall_gain_retention"] >= 0.90 for run in results)), "each_seed_unet_coverage_at_most_0p30": bool(all(run["unet_call_coverage"] <= 0.30 for run in results)), "maximum_cell_fpr_at_most_0p10": bool(all(run["maximum_cell_fpr"] <= 0.10 for run in results)), "each_seed_macro_aupr_delta_vs_full_at_least_minus_0p005": bool(all(run["macro_aupr_delta_vs_full"] >= -0.005 for run in results)), "three_of_three_positive_gain": bool(all(run["cascade_recall_gain"] > 0 for run in results))}; passed = bool(all(gates.values()))
    summary = {"experiment": "P3h latent plus shared-physics prefilter sufficiency", "preregister_sha256": sha256_file(project / args.preregister), "identity_audit": {"train_sha256": _hash_int64(references["train"]), "development_sha256": _hash_int64(references["development"]), "cross_role_overlap": 0}, "promotion_gates": gates, "development_passed": passed, "runs": results, "interpretation": "Frozen latent sufficiency passed; proceed to a compact trainable student and efficiency confirmation." if passed else "Frozen latent sufficiency failed; a new jointly learned/distilled representation is required.", "test_generated": False, "test_opened": False, "locked_test_used": False}; (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8"); print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__": main()

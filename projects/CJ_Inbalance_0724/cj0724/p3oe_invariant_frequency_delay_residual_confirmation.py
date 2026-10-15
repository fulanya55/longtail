"""P3o-E confirmation on 1130M/1140M, with the development model completely frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3n_cross_interference_backbone import ISRJ_INDEX, sha256_file
from cj0724.p3o_low_power_attribution import audit_identities, build_concat_dataset, close_concat_dataset, evaluate_outputs, fit_multi_file_normalizer, ordered_paths, role_metadata, save_prediction_payload, select_cp_safe_isrj_threshold
from cj0724.p3ob_dualview_representation import collect_outputs
from cj0724.p3oc_parameter_matched_multiview import ParameterMatchedMultiView
from cj0724.p3oe_invariant_frequency_delay_parameter_residual import InvariantFrozenFrequencyDelayResidual
from cj0724.train import get_device
from cj0724.tuned_multitask_sweep import optimize_class_thresholds


def _paths(project: Path, template: str) -> dict[str, dict[int, Path]]:
    return {regime: transition_files(project / template.format(regime=regime), "train", POWERS) for regime in REGIMES}


def _class_nmae(record: dict, name: str) -> float:
    return float(np.mean([row["normalized_mae"] for row in record["parameter"]["per_target"] if row["class_name"] == name]))


def _load_development(project: Path, preregistration: dict) -> dict:
    path = project / preregistration["source_development"]["output"]
    if sha256_file(path) != preregistration["source_development"]["sha256"]:
        raise RuntimeError("P3o-E development summary hash mismatch")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary["development_passed"] or not all(summary["promotion_gates"].values()):
        raise RuntimeError("P3o-E development was not eligible for confirmation")
    expected = preregistration["source_development"]["frozen_model_seeds"]
    if [int(row["seed"]) for row in summary["results"]] != expected:
        raise RuntimeError("P3o-E frozen model seed mismatch")
    return summary


def _load_metadata(roots: dict[str, dict[int, Path]]) -> tuple[dict, np.ndarray]:
    metadata, reference = {}, None
    for regime, paths in roots.items():
        audit_power_alignment(paths)
        identities, labels = read_identity_metadata(paths[25])
        if reference is None:
            reference = identities
        elif not np.array_equal(reference, identities):
            raise RuntimeError("confirmation regimes are not identity paired")
        metadata[regime] = {"identities": identities, "labels": labels}
    return metadata, reference


def _thresholds(base, calibration_loader, calibration_meta, normalizer, args) -> tuple[np.ndarray, dict]:
    calibration = collect_outputs(base, calibration_loader, args.device_resolved, normalizer, args.parameter_weight)
    thresholds = optimize_class_thresholds(torch.sigmoid(calibration["logits"]).numpy(), calibration["labels"].numpy())
    safe = select_cp_safe_isrj_threshold(torch.sigmoid(calibration["logits"]).numpy(), calibration["labels"].numpy(), calibration_meta["regime"], calibration_meta["power"], args.cp_cap, args.cp_confidence)
    if not safe["feasible"]:
        raise RuntimeError("frozen base calibration became infeasible")
    thresholds[ISRJ_INDEX] = safe["threshold"]
    return np.asarray(thresholds, dtype=np.float64), safe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", default="configs/p3oe_invariant_frequency_delay_residual_confirmation_preregister_20260802.json")
    parser.add_argument("--method-freeze", default="configs/p3oe_invariant_frequency_delay_residual_confirmation_method_freeze_20260802.json")
    parser.add_argument("--development-root", default="outputs/p3oe_invariant_frequency_delay_parameter_residual_development_20260802")
    parser.add_argument("--train-root-template", default="generated_data/p3oe_invariant_residual_train_seed1110000000_{regime}_rechunked")
    parser.add_argument("--confirmation-root-113", default="generated_data/p3oe_invariant_residual_confirmation_seed1130000000_{regime}_rechunked")
    parser.add_argument("--confirmation-root-114", default="generated_data/p3oe_invariant_residual_confirmation_seed1140000000_{regime}_rechunked")
    parser.add_argument("--output-root", default="outputs/p3oe_invariant_frequency_delay_residual_confirmation_20260802")
    parser.add_argument("--batch-size", type=int, default=63); parser.add_argument("--parameter-weight", type=float, default=.1); parser.add_argument("--cp-cap", type=float, default=.1); parser.add_argument("--cp-confidence", type=float, default=.95); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    freeze = json.loads((project / args.method_freeze).read_text(encoding="utf-8"))
    if freeze["preregister_sha256"] != sha256_file(project / args.preregister) or freeze["implementation_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("P3o-E confirmation frozen hash mismatch")
    preregistration = json.loads((project / args.preregister).read_text(encoding="utf-8")); development = _load_development(project, preregistration)
    output = project / args.output_root; output.mkdir(parents=True, exist_ok=False); args.device_resolved = get_device(args.device)
    train_roots = _paths(project, args.train_root_template); train_metadata, train_ids = _load_metadata(train_roots)
    labels = train_metadata["ID_control"]["labels"]; roles_file = project / args.development_root / "identity_roles.npz"; roles = {key: value for key, value in np.load(roles_file).items()}
    if not np.array_equal(labels, train_metadata["weak_target"]["labels"]): raise RuntimeError("training label contract drift")
    confirmations = {1130000000: _paths(project, args.confirmation_root_113), 1140000000: _paths(project, args.confirmation_root_114)}
    audits, confirmation_metadata, confirmation_ids = {}, {}, {}
    original = json.loads((project / "outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json").read_text(encoding="utf-8"))
    for identity_seed, roots in confirmations.items():
        metadata, identities = _load_metadata(roots)
        if not np.array_equal(labels, metadata["ID_control"]["labels"]): raise RuntimeError("confirmation label contract drift")
        audits[str(identity_seed)] = audit_identities(project, train_ids, identities, {1110000000, 1120000000, 1130000000, 1140000000}, Path(original["data_file"]))
        confirmation_metadata[identity_seed], confirmation_ids[identity_seed] = metadata, identities
    mutual_overlap = int(np.intersect1d(confirmation_ids[1130000000], confirmation_ids[1140000000]).size)
    if mutual_overlap: raise RuntimeError("confirmation identities overlap")
    (output / "identity_audit.json").write_text(json.dumps({"per_confirmation": audits, "mutual_confirmation_overlap": mutual_overlap}, indent=2), encoding="utf-8")
    normalizer = fit_multi_file_normalizer(ordered_paths(train_roots), roles["full_fit"], labels)
    calibration_dataset = build_concat_dataset(ordered_paths(train_roots), roles["calibration"]); calibration_loader = DataLoader(calibration_dataset, batch_size=args.batch_size)
    calibration_meta = role_metadata(train_metadata, roles["calibration"]); records = []
    try:
        for source in development["results"]:
            seed, base_epoch, side_epoch = int(source["seed"]), int(source["base_epoch"]), int(source["side_epoch"])
            base = ParameterMatchedMultiView("F_time_frequency_matched").to(args.device_resolved)
            base.load_state_dict(torch.load(project / args.development_root / f"base_{seed}_{base_epoch}.pt", map_location=args.device_resolved, weights_only=True)); thresholds, safe = _thresholds(base, calibration_loader, calibration_meta, normalizer, args)
            side = InvariantFrozenFrequencyDelayResidual(base).to(args.device_resolved)
            side.load_state_dict(torch.load(project / args.development_root / f"side_{seed}_{side_epoch}.pt", map_location=args.device_resolved, weights_only=True))
            for identity_seed, roots in confirmations.items():
                dataset = build_concat_dataset(ordered_paths(roots), np.arange(len(confirmation_ids[identity_seed]), dtype=np.int64)); loader = DataLoader(dataset, batch_size=args.batch_size); metadata = role_metadata(confirmation_metadata[identity_seed], np.arange(len(confirmation_ids[identity_seed]), dtype=np.int64))
                try:
                    base_outputs = collect_outputs(base, loader, args.device_resolved, normalizer, args.parameter_weight); side_outputs = collect_outputs(side, loader, args.device_resolved, normalizer, args.parameter_weight)
                finally:
                    close_concat_dataset(dataset)
                logit_equal = bool(torch.equal(base_outputs["logits"], side_outputs["logits"])); base_result, side_result = evaluate_outputs(base_outputs, thresholds, metadata, normalizer), evaluate_outputs(side_outputs, thresholds, metadata, normalizer)
                records.append({"seed": seed, "confirmation_identity_seed": identity_seed, "logits_bitwise_equal": logit_equal, "safe_audit": safe, "base": base_result, "side": side_result})
                save_prediction_payload(output / f"seed_{seed}_confirmation_{identity_seed}_base.npz", base_outputs, metadata); save_prediction_payload(output / f"seed_{seed}_confirmation_{identity_seed}_side.npz", side_outputs, metadata)
    finally:
        close_concat_dataset(calibration_dataset)
    gates = {"each_confirmation_identity_and_history_zero_overlap": all(audit["cross_role_overlap"] == 0 and audit["maximum_prior_overlap"] == 0 for audit in audits.values()), "confirmation_identities_mutually_zero_overlap": mutual_overlap == 0, "all_6_frozen_model_identity_cells_logits_bitwise_equal": all(row["logits_bitwise_equal"] for row in records), "all_6_cells_base_safe_calibration_feasible_and_maximum_cell_fpr_at_most_0p10": all(row["safe_audit"]["feasible"] and row["base"]["maximum_cell_fpr"] <= .1 for row in records), "all_6_cells_base_overall_isrj_recall_at_least_0p85_and_25db_recall_at_least_0p78": all(row["base"]["isrj_recall"] >= .85 and row["base"]["per_power"]["25"]["isrj_recall"] >= .78 for row in records), "all_6_cells_oracle_and_predicted_gate_nmae_ratio_at_most_0p98": all(row["side"]["parameter"]["oracle_macro_target_nmae"] / row["base"]["parameter"]["oracle_macro_target_nmae"] <= .98 and row["side"]["predicted_gate"]["predicted_gate_parameter_nmae"] / row["base"]["predicted_gate"]["predicted_gate_parameter_nmae"] <= .98 for row in records), "all_6_cells_miss_penalty_and_worst_combination_nonworse": all(row["side"]["predicted_gate"]["miss_penalty_parameter_nmae"] <= row["base"]["predicted_gate"]["miss_penalty_parameter_nmae"] and row["side"]["predicted_gate"]["worst_combo_predicted_gate_nmae"] <= row["base"]["predicted_gate"]["worst_combo_predicted_gate_nmae"] for row in records), "all_6_cells_mft_and_isrj_oracle_nmae_ratio_at_most_0p99": all(_class_nmae(row["side"], "MFT") / _class_nmae(row["base"], "MFT") <= .99 and _class_nmae(row["side"], "ISRJ") / _class_nmae(row["base"], "ISRJ") <= .99 for row in records), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    payload = {"confirmation_passed": all(gates.values()), "promotion_gates": gates, "records": records, "identity_audit": {"per_confirmation": audits, "mutual_confirmation_overlap": mutual_overlap}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(payload, default=lambda value: value.item() if hasattr(value, "item") else str(value), indent=2), encoding="utf-8")


if __name__ == "__main__": main()

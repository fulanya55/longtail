"""Fresh-data development screen for a bounded predicted-width VFT residual."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cjlt.data.longtail_manifest import compute_weak_component_mask  # noqa: E402
from cjlt.data.v73_dataset import read_labels, read_parameters  # noqa: E402
from experiments.run_route_a_controlled_lt import _mean_ci, _read as _load_json, _write  # noqa: E402
from experiments.run_route_a_development_v16_isrj_width_probe import WidthProbe, _load_signals as _load_width_signals  # noqa: E402
from experiments.run_route_a_development_v2_h5 import _sha256  # noqa: E402
from experiments.run_route_a_development_v2_shared_gradient import _component_power  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import _run_seed  # noqa: E402
from experiments.run_route_a_isrj_parameter_ood_v1_frozen import _apply_frozen_calibration  # noqa: E402
from experiments.run_route_a_isrj_joint_periodicity_expert_v3 import _extract as _extract_periodicity  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_unified_v3 import _extract_vft  # noqa: E402
from experiments.audit_route_a_mft_pulse_compression_morphology_v3 import extract_scores  # noqa: E402
from experiments.run_route_a_h1_b2_mft_vft_development_v11_frozen import PEAK_INDEX  # noqa: E402
from experiments.run_route_a_unified_rawsaw_vtse_vbf_v3 import _validate_config  # noqa: E402
from experiments.run_shared import _data_path, load_config  # noqa: E402

Q4 = "Q4_VFTAlignedRidgeFusion"
SAW, VFT, ISRJ = 1, 4, 5
GRID = np.round(np.arange(0.0, 1.01, 0.1), 3)

def _sealed(payload: dict) -> bool:
    return not any(payload["test_v3"][key] for key in ("access_authorized", "analysis_allowed", "used"))

def _width_predict(model, signals, mean, std, device, batch_size=128):
    model.eval(); rows = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            rows.append(model(torch.from_numpy(signals[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(rows).astype(np.float64) * std + mean

def _load_q4(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as item:
        return {key: np.asarray(item[key]) for key in ("receiver_noise_seed", "targets", "logits", "probabilities", "predictions", "thresholds")}

def _prediction(logits, calibration, thresholds):
    probability = _apply_frozen_calibration(logits, calibration)
    return probability, probability >= thresholds[None, :]

def _endpoint(labels, weak, probability, prediction, hard):
    positive = labels[:, VFT] == 1
    vft_negative = ~positive
    saw_positive = labels[:, SAW] == 1
    isrj_positive = labels[:, ISRJ] == 1
    return {
        "hard_fpr": float(prediction[hard, VFT].mean()),
        "vft_positive_recall": float(prediction[positive, VFT].mean()),
        "vft_aupr": float(average_precision_score(labels[:, VFT], probability[:, VFT])),
        "vft_weak_recall": float(prediction[positive & weak[:, VFT], VFT].mean()),
        "vft_negative_fpr": float(prediction[vft_negative, VFT].mean()),
        "isrj_aupr": float(average_precision_score(labels[:, ISRJ], probability[:, ISRJ])),
        "sawfm_aupr": float(average_precision_score(labels[:, SAW], probability[:, SAW])),
        "sawfm_weak_recall": float(prediction[saw_positive & weak[:, SAW], SAW].mean()),
        "isrj_positive_recall": float(prediction[isrj_positive, ISRJ].mean()),
    }

def _select_c(labels, weak, logits, calibration, thresholds, evidence, fit_ids):
    hard = (labels[:, VFT] == 0) & (labels[:, ISRJ] == 1)
    base_probability, base_prediction = _prediction(logits, calibration, thresholds)
    base = _endpoint(labels[fit_ids], weak[fit_ids], base_probability[fit_ids], base_prediction[fit_ids], hard[fit_ids])
    candidates = []
    for c in GRID:
        modified = logits.copy(); modified[:, VFT] -= float(c) * evidence
        probability, prediction = _prediction(modified, calibration, thresholds)
        row = _endpoint(labels[fit_ids], weak[fit_ids], probability[fit_ids], prediction[fit_ids], hard[fit_ids])
        row.update({"c": float(c), "fpr_reduction": base["hard_fpr"] - row["hard_fpr"], "positive_recall_loss": base["vft_positive_recall"] - row["vft_positive_recall"]})
        candidates.append(row)
    feasible = [row for row in candidates if row["positive_recall_loss"] <= .010 + 1e-12]
    if not feasible:
        raise ValueError("c=0 must satisfy the fit safety constraint")
    return max(feasible, key=lambda row: (row["fpr_reduction"], -row["c"])), base, candidates

def _variant(logits, calibration, thresholds, evidence, c, labels, weak, role_ids):
    adjusted = logits.copy(); adjusted[:, VFT] -= float(c) * evidence
    probability, prediction = _prediction(adjusted, calibration, thresholds)
    hard = (labels[:, VFT] == 0) & (labels[:, ISRJ] == 1)
    return adjusted, probability, prediction, _endpoint(labels[role_ids], weak[role_ids], probability[role_ids], prediction[role_ids], hard[role_ids])

def _load_inputs(args):
    audit = _load_json(args.audit); roles = np.load(args.roles, allow_pickle=False)
    if not audit.get("identity_audit_passed") or audit.get("test_files_opened"):
        raise ValueError("v18 data audit failed")
    if _sha256(args.data_path).lower() != audit["sha256"].lower():
        raise ValueError("v18 data hash changed")
    labels = read_labels(args.data_path); values, _ = read_parameters(args.data_path)
    weak = compute_weak_component_mask(labels, _component_power(values), args.weak_threshold_db)
    with h5py.File(args.data_path, "r") as handle:
        receiver = np.asarray(handle["receiver_noise_seed"]).reshape(-1).astype(np.int64)
    role_ids = {name: np.asarray(roles[name], dtype=np.int64) for name in ("fit", "selection", "evaluation")}
    if any(np.intersect1d(role_ids[a], role_ids[b]).size for a in role_ids for b in role_ids if a < b):
        raise ValueError("v18 role index leakage")
    return audit, labels, weak, receiver, role_ids

def run(args):
    audit, labels, weak, receiver, role_ids = _load_inputs(args)
    seal = _load_json(args.test_v3_manifest); manifest = _load_json(args.frozen_manifest); normalizer = _load_json(args.normalizer); v16 = _load_json(args.v16_summary)
    if not _sealed(seal): raise ValueError("test-v3 seal is not intact")
    if manifest.get("candidate_name") != "H1_B2_MFTSharedDecision_VFTRidgeFusion_v2": raise ValueError("wrong frozen Q4 candidate")
    if not v16.get("gate", {}).get("passed") or not normalizer.get("fit_only"): raise ValueError("frozen v16 width contract invalid")
    config = load_config(args.config); _validate_config(config)
    labels3 = read_labels(_data_path(config)); params3, _ = read_parameters(_data_path(config)); weak3 = compute_weak_component_mask(labels3, _component_power(params3), args.weak_threshold_db)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    signals = _load_width_signals(args.data_path, np.arange(len(labels), dtype=np.int64))
    periodic = _extract_periodicity(signals, device, args.periodicity_batch_size); vft = _extract_vft(signals, device, args.vft_batch_size); mft = extract_scores(signals, device, args.mft_batch_size)["MF_AlignedLFM"][:, PEAK_INDEX]
    output_root = args.output_root; q4_root = output_root / "q4_replay"; rows = []
    for seed in args.seeds:
        args.output_root = q4_root
        _run_seed(args, config, labels3, weak3, signals, labels, weak, receiver, periodic, vft, mft, seed, device, manifest)
        q4 = _load_q4(q4_root / f"seed_{seed}" / Q4 / "predictions.npz")
        if not np.array_equal(q4["receiver_noise_seed"], receiver) or not np.array_equal(q4["targets"], labels): raise ValueError("frozen Q4 source identity/label mismatch")
        checkpoint = args.v16_root / "runs" / "W_real" / f"seed_{seed}" / "best_model.pth"
        model = WidthProbe(width=16).to(device); model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
        width = _width_predict(model, signals, normalizer["mean_seconds"], normalizer["std_seconds"], device, args.width_batch_size); del model
        width_evidence = np.clip((width - 20e-6) / 10e-6, 0.0, 1.0) * q4["probabilities"][:, ISRJ]
        control_evidence = q4["probabilities"][:, ISRJ]
        calibration = _load_json(args.construction_root / f"seed_{seed}" / Q4 / "calibration.json"); thresholds = q4["thresholds"]
        width_fit, base_fit, width_grid = _select_c(labels, weak, q4["logits"], calibration, thresholds, width_evidence, role_ids["fit"])
        control_fit, _, control_grid = _select_c(labels, weak, q4["logits"], calibration, thresholds, control_evidence, role_ids["fit"])
        variants = {"Q4_Frozen": (np.zeros(len(labels)), 0.0), "Control_ISRJOnly": (control_evidence, control_fit["c"]), "Candidate_BoundedPredictedWidth": (width_evidence, width_fit["c"])}
        seed_root = output_root / f"seed_{seed}"; seed_root.mkdir(parents=True, exist_ok=True); seed_rows = {}
        for name, (evidence, c) in variants.items():
            logits, probability, prediction, selection = _variant(q4["logits"], calibration, thresholds, evidence, c, labels, weak, role_ids["selection"])
            _, _, _, evaluation = _variant(q4["logits"], calibration, thresholds, evidence, c, labels, weak, role_ids["evaluation"])
            if np.max(np.abs(logits[:, [0, 1, 2, 3, 5]] - q4["logits"][:, [0, 1, 2, 3, 5]])) > 0: raise ValueError("non-VFT Q4 logits changed")
            out = seed_root / name; out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "predictions.npz", receiver_noise_seed=receiver, targets=labels, weak_mask=weak.astype(np.uint8), logits=logits, probabilities=probability, predictions=prediction.astype(np.uint8), thresholds=thresholds, width_prediction_seconds=width, evidence=evidence, c=np.asarray([c], dtype=np.float64))
            _write(out / "metrics.json", {"selection": selection, "evaluation": evaluation, "thresholds_exact": bool(np.array_equal(thresholds, q4["thresholds"])), "non_vft_logit_max_abs": 0.0})
            seed_rows[name] = {"c": float(c), "selection": selection, "evaluation": evaluation}
        row = {"seed": int(seed), "width_checkpoint_sha256": _sha256(checkpoint), "q4_thresholds_exact": True, "q4_non_vft_logits_exact": True, "width_fit": width_fit, "control_fit": control_fit, "width_grid": width_grid, "control_grid": control_grid, "variants": seed_rows, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v18_used": False}
        _write(seed_root / "seed_summary.json", row); rows.append(row)
    args.output_root = output_root
    metric_names = ("hard_fpr", "vft_positive_recall", "vft_aupr", "vft_weak_recall", "isrj_aupr", "sawfm_aupr", "sawfm_weak_recall")
    deltas = {"candidate_minus_q4": [], "candidate_minus_control": []}
    for row in rows:
        candidate = row["variants"]["Candidate_BoundedPredictedWidth"]["evaluation"]; base = row["variants"]["Q4_Frozen"]["evaluation"]; control = row["variants"]["Control_ISRJOnly"]["evaluation"]
        deltas["candidate_minus_q4"].append({"seed": row["seed"], **{name: candidate[name] - base[name] for name in metric_names}})
        deltas["candidate_minus_control"].append({"seed": row["seed"], **{name: candidate[name] - control[name] for name in metric_names}})
    aggregate = {group: {name: _mean_ci([row[name] for row in values], 2026080600 + i + 100 * j) for i, name in enumerate(metric_names)} for j, (group, values) in enumerate(deltas.items())}
    fpr_q4 = [row["hard_fpr"] for row in deltas["candidate_minus_q4"]]; fpr_control = [row["hard_fpr"] for row in deltas["candidate_minus_control"]]; recall = [row["vft_positive_recall"] for row in deltas["candidate_minus_q4"]]
    checks = {
        "candidate_q4_fpr_delta_le_minus0p030_4of5": aggregate["candidate_minus_q4"]["hard_fpr"]["mean"] <= -.030 and sum(value <= 0 for value in fpr_q4) >= 4,
        "candidate_control_fpr_delta_le_minus0p010_4of5": aggregate["candidate_minus_control"]["hard_fpr"]["mean"] <= -.010 and sum(value <= 0 for value in fpr_control) >= 4,
        "vft_positive_recall_loss_ge_minus0p010_every_seed": aggregate["candidate_minus_q4"]["vft_positive_recall"]["mean"] >= -.010 and all(value >= -.010 for value in recall),
        "vft_aupr_drop_ge_minus0p010": aggregate["candidate_minus_q4"]["vft_aupr"]["mean"] >= -.010,
        "isrj_aupr_nonregression": aggregate["candidate_minus_q4"]["isrj_aupr"]["mean"] >= -.010,
        "sawfm_aupr_nonregression": aggregate["candidate_minus_q4"]["sawfm_aupr"]["mean"] >= -.010,
        "sawfm_weak_recall_nonregression": aggregate["candidate_minus_q4"]["sawfm_weak_recall"]["mean"] >= -.010,
        "all_frozen_contracts_exact": all(row["q4_thresholds_exact"] and row["q4_non_vft_logits_exact"] for row in rows),
        "test_v3_sealed": True,
    }
    result = {"status": "completed", "experiment": "route_a_v18_bounded_predicted_width_residual", "evidence_role": "fresh_identity_isolated_development_candidate", "data_audit_sha256": _sha256(args.audit), "frozen_manifest_sha256": _sha256(args.frozen_manifest), "seeds": [int(x) for x in args.seeds], "grid": [float(x) for x in GRID], "roles": {name: int(len(ids)) for name, ids in role_ids.items()}, "rows": rows, "deltas": deltas, "aggregate": aggregate, "gate": {"checks": checks, "passed": bool(all(checks.values())), "decision": "permit_fresh_confirmation_design" if all(checks.values()) else "stop_bounded_width_residual"}, "external_refit": False, "calibration_reused": True, "thresholds_reused": True, "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v18_used": False}
    output_root.mkdir(parents=True, exist_ok=True); _write(output_root / "route_a_v18_bounded_predicted_width_residual_summary.json", result); _write(output_root / "run_receipt.json", {"status": "completed", "gate_passed": result["gate"]["passed"], "decision": result["gate"]["decision"], "locked_test_used": False, "test_v2_used": False, "test_v3_used": False, "test_v18_used": False})
    print(json.dumps({"passed": result["gate"]["passed"], "checks": checks, "output": str(output_root)}, ensure_ascii=False, indent=2)); return result

def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--data-path", type=Path, default=Path(r"C:\LJQ_code\干扰数据\复合干扰数据0802_v18_bounded_width_residual\closedset_6class_train.mat")); p.add_argument("--audit", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v18_bounded_width_residual_generation/identity_audit.json"); p.add_argument("--roles", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v18_bounded_width_residual_generation/role_indices.npz"); p.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260801_route_a_h1_b2_mft_vft_shared_policy_v2/frozen_manifest.json"); p.add_argument("--test-v3-manifest", type=Path, default=PROJECT_ROOT / "frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json"); p.add_argument("--v16-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v16_isrj_width_probe"); p.add_argument("--v16-summary", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v16_isrj_width_probe/route_a_development_v16_isrj_width_probe_summary.json"); p.add_argument("--normalizer", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v16_isrj_width_probe/width_normalizer.json"); p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/student_a/six_class_development_v3.yaml"); p.add_argument("--f2-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_dfms_iq_structure_screen"); p.add_argument("--c6-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_label_selective_projection"); p.add_argument("--head-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_development_v3_frozen_head_capacity_baselines"); p.add_argument("--unified-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_unified_rawsaw_vtse_vbf_v3"); p.add_argument("--periodicity-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_isrj_joint_periodicity_expert_v3"); p.add_argument("--construction-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_h1_b2_mft_vft_unified_v3"); p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/tapc/route_a_v18_bounded_predicted_width_residual"); p.add_argument("--seeds", type=int, nargs="+", default=[2039,2040,2041,2042,2043]); p.add_argument("--device", default=None); p.add_argument("--io-batch-size", type=int, default=128); p.add_argument("--width-batch-size", type=int, default=128); p.add_argument("--backbone-batch-size", type=int, default=128); p.add_argument("--eval-batch-size", type=int, default=128); p.add_argument("--raw-batch-size", type=int, default=64); p.add_argument("--periodicity-batch-size", type=int, default=64); p.add_argument("--vft-batch-size", type=int, default=32); p.add_argument("--mft-batch-size", type=int, default=64); p.add_argument("--raw-hidden-channels", type=int, default=16); p.add_argument("--raw-topk", type=int, default=16); p.add_argument("--raw-dropout", type=float, default=.1); p.add_argument("--weak-threshold-db", type=float, default=10.); return p.parse_args()

if __name__ == "__main__": run(parse_args())

"""Formal P3o-U development runner: local phase dynamics versus strong controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cj0724.data import V73IQDataset
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable, _load_main
from cj0724.multitask import PARAMETER_COLUMNS, reshape_selected_parameters
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import worst_combo_predicted_gate_nmae
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p3os_mft_oracle_subtraction import paired_bootstrap_source_fpr_reduction
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3ot_mft_conditional_residual import (
    BASE_MODEL_SEEDS, _base_receipts, _close, _dataset, _difference, _freeze,
    _metadata, _regime_direction, _apply_selection_threshold,
    choose_selection_isrj_threshold, classification_summary, frozen_base_outputs,
)
from cj0724.paper_core_evidence import gated_metrics, load_normalizer
from cj0724.p3ou_local_phase_dynamics import BoundedISRJResidual, LocalPhaseDynamics, conditional_residual_loss, corrected_logits
from cj0724.train import set_seed
from cj0724.tuned_multitask_sweep import _set_warmup_cosine_lr


RESIDUAL_SEEDS = (2501, 2502, 2503)
ROLES = {"train": 2010000000, "selection": 2020000000, "development": 2030000000}
CONDITIONS = (("U1_posterior_only_capacity_control", "zero"), ("U2_local_phase_dynamics", "phase"), ("U3_timebin_shuffled_phase_control", "shuffled"))


def role_paths(root: Path, role: str, seed: int, suffix: str = "") -> dict[str, dict[int, Path]]:
    stem = f"p3ou_local_phase_dynamics_{role}_seed{int(seed)}"
    return {regime: transition_files(root / f"{stem}_{regime}{suffix}", "train", POWERS) for regime in ("ID_control", "low_count", "close_spacing", "ultra_close", "weak_target")}


def _ordered(paths: dict[str, dict[int, Path]]) -> list[Path]:
    return [paths[regime][power] for regime in paths for power in POWERS]


def audit_roles(project: Path, root: Path, suffix: str, prior_roots: list[Path]) -> dict[str, object]:
    paths = {role: role_paths(root, role, seed, suffix) for role, seed in ROLES.items()}
    identities: dict[str, np.ndarray] = {}; contracts = {}; labels_reference = None
    for role, values in paths.items():
        reference_identity = reference_labels = None; rows = []
        for regime, files in values.items():
            alignment = audit_power_alignment(files)
            identity, labels = read_identity_metadata(files[25])
            if len(identity) != 2079 or len(np.unique(identity)) != 2079:
                raise RuntimeError(f"P3o-U invalid identity contract for {role}/{regime}")
            if reference_identity is None: reference_identity, reference_labels = identity, labels
            elif not np.array_equal(reference_identity, identity) or not np.array_equal(reference_labels, labels):
                raise RuntimeError(f"P3o-U view alignment mismatch for {role}/{regime}")
            rows.append({"regime": regime, "power_alignment": alignment})
        assert reference_identity is not None and reference_labels is not None
        identities[role] = np.asarray(reference_identity, dtype=np.int64); contracts[role] = rows
        if labels_reference is None: labels_reference = reference_labels
        elif not np.array_equal(labels_reference, reference_labels): raise RuntimeError("P3o-U labels differ across roles")
    names = list(identities); cross = {f"{a}_{b}": int(np.intersect1d(identities[a], identities[b]).size) for i, a in enumerate(names) for b in names[i + 1:]}
    if any(cross.values()): raise RuntimeError(f"P3o-U cross-role identity overlap: {cross}")
    current = {path.resolve() for role in paths.values() for view in role.values() for path in view.values()}
    roots = [project / "generated_data", Path("D:/CJ0724_generated_data"), Path("D:/CJ0724_rechunked_data")] + prior_roots
    prior = set()
    for candidate in roots:
        if candidate.is_dir(): prior.update(path.resolve() for path in candidate.rglob("closedset_6class_train.mat"))
    protocol = json.loads((project / "outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156/protocol.json").read_text(encoding="utf-8"))
    if Path(protocol["data_file"]).is_file(): prior.add(Path(protocol["data_file"]).resolve())
    history, maximum = [], 0
    for path in sorted(prior):
        if path in current: continue
        old, _ = read_identity_metadata(path); overlaps = {role: int(np.intersect1d(value, old).size) for role, value in identities.items()}
        maximum = max(maximum, *overlaps.values()); history.append({"path": str(path), "overlaps": overlaps})
    if maximum: raise RuntimeError("P3o-U identity overlap with historical data")
    return {"roles": {role: {"seed": ROLES[role], "identity_count": len(value), "identity_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(), "contract": contracts[role]} for role, value in identities.items()}, "cross_role_overlap": cross, "maximum_prior_overlap": maximum, "prior_files_checked": len(history), "prior_rows": history, "power_alignment_verified": True, "label_contract_verified": True, "test_generated": False, "test_opened": False, "locked_test_used": False}


def _forward(base, branch, frontend, iq, mode):
    logits, _, prediction = frozen_base_outputs(base, iq)
    if branch is None: return logits, prediction
    return corrected_logits(logits, branch(logits, frontend(iq, mode))), prediction


def _selection_loss(base, branch, frontend, mode, loader, device):
    branch.eval(); values = []
    with torch.inference_mode():
        for batch in loader:
            logits, _ = _forward(base, branch, frontend, batch["iq"].to(device), mode)
            values.append(float(F.binary_cross_entropy_with_logits(logits[:, 5], batch["label"].to(device)[:, 5])))
    return float(np.mean(values))


def train_branch(base, frontend, mode, fit, selection, seed, output, args, device):
    set_seed(seed); branch = BoundedISRJResidual().to(device); optimizer = torch.optim.AdamW(branch.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best, best_epoch, history = float("inf"), 0, []; checkpoint = output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        branch.train(); values = []; lr = _set_warmup_cosine_lr(optimizer, args.learning_rate, epoch, args.epochs, args.warmup_epochs)
        for batch in fit:
            iq, labels = batch["iq"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); base_logits, _, _ = frozen_base_outputs(base, iq)
            logits = corrected_logits(base_logits, branch(base_logits, frontend(iq, mode)))
            loss, bce, source = conditional_residual_loss(logits, base_logits, labels, args.conditional_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(branch.parameters(), args.gradient_clip_norm); optimizer.step()
            values.append((float(loss.detach()), float(bce.detach()), float(source.detach())))
        selection_bce = _selection_loss(base, branch, frontend, mode, selection, device)
        history.append({"epoch": epoch, "lr": float(lr), "train_loss": float(np.mean([x[0] for x in values])), "train_bce": float(np.mean([x[1] for x in values])), "train_conditional": float(np.mean([x[2] for x in values])), "selection_isrj_bce": selection_bce})
        if selection_bce < best: best, best_epoch = selection_bce, epoch; torch.save(branch.state_dict(), checkpoint)
    branch.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); branch.eval()
    return branch, {"best_selection_isrj_bce": best, "best_epoch": best_epoch, "history": history, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "parameters": sum(p.numel() for p in branch.parameters())}


@torch.inference_mode()
def collect_record(base, branch, frontend, mode, loader, device, normalizer, metadata, thresholds):
    parts = {key: [] for key in ("logits", "labels", "prediction", "target", "mask")}
    for batch in loader:
        logits, prediction = _forward(base, branch, frontend, batch["iq"].to(device, non_blocking=True), mode)
        parts["logits"].append(logits.float().cpu()); parts["labels"].append(batch["label"].to(torch.uint8).cpu()); parts["prediction"].append(normalizer.inverse_transform(prediction.float().cpu()))
        parts["target"].append(reshape_selected_parameters(batch["parameter_values"].to(device)).float().cpu()); parts["mask"].append(reshape_selected_parameters(batch["parameter_mask"].to(device)).bool().cpu())
    labels = torch.cat(parts["labels"]).numpy().astype(np.uint8)
    if not np.array_equal(labels, metadata["labels"]): raise RuntimeError("P3o-U loader/metadata order mismatch")
    logits = torch.cat(parts["logits"]).numpy()
    return {"identity": metadata["identity"], "regime": metadata["regime"], "power": metadata["power"], "sample_index": metadata["sample_index"], "labels": labels, "logits": logits, "probabilities": 1.0 / (1.0 + np.exp(-logits)), "prediction": torch.cat(parts["prediction"]).numpy(), "target": torch.cat(parts["target"]).numpy(), "mask": torch.cat(parts["mask"]).numpy().astype(np.uint8), "thresholds": np.asarray(thresholds), "scale": normalizer.scale.numpy()}


def _verify(project, freeze_path):
    frozen = json.loads(freeze_path.read_text(encoding="utf-8")); files = {"preregister_sha256": project / "configs/p3ou_local_phase_dynamics_preregister_20260803.json", "implementation_sha256": project / "cj0724/p3ou_local_phase_dynamics.py", "runner_sha256": Path(__file__), "generator_sha256": project / "matlab/generate_p3ou_local_phase_dynamics_level.m"}
    mismatch = {key: {"expected": frozen.get(key), "actual": sha256_file(path)} for key, path in files.items() if frozen.get(key) != sha256_file(path)}
    if mismatch: raise RuntimeError(f"P3o-U method freeze mismatch: {mismatch}")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--rechunk-root", required=True); parser.add_argument("--rechunk-suffix", default=""); parser.add_argument("--method-freeze", required=True); parser.add_argument("--output-root"); parser.add_argument("--preflight-output"); parser.add_argument("--prior-identity-root", action="append", default=[]); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--warmup-epochs", type=int, default=2); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--conditional-weight", type=float, default=.25); parser.add_argument("--gradient-clip-norm", type=float, default=5.0); parser.add_argument("--bootstrap-resamples", type=int, default=2000); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    if bool(args.output_root) == bool(args.preflight_output): raise ValueError("Specify exactly one of --output-root or --preflight-output")
    _verify(project, project / args.method_freeze); audit = audit_roles(project, Path(args.rechunk_root), args.rechunk_suffix, [Path(x) for x in args.prior_identity_root])
    if args.preflight_output:
        destination = project / args.preflight_output; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): raise FileExistsError(destination)
        destination.write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8"); print(json.dumps({"preflight_passed": True, "identity_overlap_zero": True, "prior_files_checked": audit["prior_files_checked"]}, indent=2)); return
    output = project / args.output_root; output.mkdir(parents=True, exist_ok=False); selected, threshold_freezes, protocol = _base_receipts(project); normalizer = load_normalizer(protocol["normalizer"]); device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    paths = {role: role_paths(Path(args.rechunk_root), role, seed, args.rechunk_suffix) for role, seed in ROLES.items()}; datasets = {role: _dataset(paths[role]) for role in paths}; metadata = {role: _metadata(paths[role]) for role in paths}; loaders = {"fit": DataLoader(datasets["train"], args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(20260803), num_workers=0, pin_memory=device.type == "cuda"), "selection": DataLoader(datasets["selection"], args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda"), "development": DataLoader(datasets["development"], args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")}; frontend = LocalPhaseDynamics().to(device).eval(); runs = []; saved = {}
    try:
        for base_seed, residual_seed in zip(BASE_MODEL_SEEDS, RESIDUAL_SEEDS):
            base, checkpoint = _load_main(selected[base_seed], protocol, project, device); _freeze(base); thresholds = np.asarray(threshold_freezes[base_seed]["safe_thresholds"], dtype=np.float64); dev = {"U0_base": collect_record(base, None, frontend, "zero", loaders["development"], device, normalizer, metadata["development"], thresholds)}; sel = {"U0_base": collect_record(base, None, frontend, "zero", loaders["selection"], device, normalizer, metadata["selection"], thresholds)}; receipts = {}
            for name, mode in CONDITIONS:
                directory = output / f"base{base_seed}_residual{residual_seed}" / name; directory.mkdir(parents=True); branch, receipt = train_branch(base, frontend, mode, loaders["fit"], loaders["selection"], residual_seed, directory, args, device); receipts[name] = receipt; dev[name] = collect_record(base, branch, frontend, mode, loaders["development"], device, normalizer, metadata["development"], thresholds); sel[name] = collect_record(base, branch, frontend, mode, loaders["selection"], device, normalizer, metadata["selection"], thresholds); del branch
            selection = {name: choose_selection_isrj_threshold(value) for name, value in sel.items()}; primary = {name: classification_summary(value) for name, value in dev.items()}; calibrated = {name: classification_summary(_apply_selection_threshold(value, selection[name]["threshold"])) for name, value in dev.items()}; comparisons = {}
            for candidate, baseline, tag in (("U2_local_phase_dynamics", "U0_base", "U2_minus_U0"), ("U2_local_phase_dynamics", "U1_posterior_only_capacity_control", "U2_minus_U1"), ("U2_local_phase_dynamics", "U3_timebin_shuffled_phase_control", "U2_minus_U3")):
                boot = paired_bootstrap_source_fpr_reduction(dev[baseline]["probabilities"][:, 5], dev[candidate]["probabilities"][:, 5], dev[candidate]["labels"], dev[candidate]["identity"], thresholds[5], 920000 + base_seed + len(comparisons), args.bootstrap_resamples); comparisons[tag] = {"difference": _difference(primary[candidate], primary[baseline]), "bootstrap_source_fpr_reduction": boot, "regime_power_source_fpr_reduction": _regime_direction(primary[candidate], primary[baseline])}
            parameter = {name: gated_metrics(value) | {"worst_combo_predicted_gate_nmae": worst_combo_predicted_gate_nmae(value)} for name, value in dev.items()}; run = {"base_model_seed": base_seed, "residual_seed": residual_seed, "base_checkpoint": str(checkpoint), "base_safe_thresholds": thresholds, "branch_receipts": receipts, "primary_frozen_threshold_metrics": primary, "secondary_selection_calibration": selection, "secondary_calibrated_metrics": calibrated, "comparisons": comparisons, "predicted_gate_parameter_metrics": parameter}; (output / f"base{base_seed}_result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8"); runs.append(run)
            for name, value in dev.items():
                for key in ("identity", "labels", "regime", "power", "sample_index", "probabilities"): saved[f"base{base_seed}_{name}_{key}"] = value[key]
            del base
            if device.type == "cuda": torch.cuda.empty_cache()
    finally:
        for dataset in datasets.values(): _close(dataset)
    def every(tag, field, bound): return all(row["comparisons"][tag]["difference"][field] >= bound for row in runs)
    directions = [row["comparisons"]["U2_minus_U0"]["regime_power_source_fpr_reduction"] for row in runs]
    gates = {"identity_overlap_zero": audit["maximum_prior_overlap"] == 0 and not any(audit["cross_role_overlap"].values()), "U2_vs_U0_source_fpr_reduction_at_least_0p02_each_seed": every("U2_minus_U0", "source_fpr_reduction", .02), "U2_vs_U1_source_fpr_reduction_at_least_0p01_each_seed": every("U2_minus_U1", "source_fpr_reduction", .01), "U2_vs_U3_source_fpr_reduction_at_least_0p01_each_seed": every("U2_minus_U3", "source_fpr_reduction", .01), "all_paired_bootstrap_ci_lows_positive": all(row["comparisons"][tag]["bootstrap_source_fpr_reduction"]["ci95_low"] > 0 for row in runs for tag in ("U2_minus_U0", "U2_minus_U1", "U2_minus_U3")), "U2_recall_drop_at_most_0p02": every("U2_minus_U0", "isrj_recall_delta", -.02), "U2_worst_source_cell_nonworse": every("U2_minus_U0", "worst_source_fpr_reduction", 0), "U2_mft_ap_bitwise_equal": all(abs(row["comparisons"]["U2_minus_U0"]["difference"]["mft_ap_delta"]) <= 1e-12 for row in runs), "U2_macro_aupr_safe": every("U2_minus_U0", "macro_aupr_delta", -.005), "U2_macro_f1_safe": every("U2_minus_U0", "macro_f1_delta", -.005), "U2_predicted_gate_nmae_safe": all(row["predicted_gate_parameter_metrics"]["U2_local_phase_dynamics"]["predicted_gate_parameter_nmae"] <= 1.01 * row["predicted_gate_parameter_metrics"]["U0_base"]["predicted_gate_parameter_nmae"] for row in runs), "U2_miss_penalty_nonworse": all(row["predicted_gate_parameter_metrics"]["U2_local_phase_dynamics"]["miss_penalty_parameter_nmae"] <= row["predicted_gate_parameter_metrics"]["U0_base"]["miss_penalty_parameter_nmae"] for row in runs), "U2_worst_combo_nonworse": all(row["predicted_gate_parameter_metrics"]["U2_local_phase_dynamics"]["worst_combo_predicted_gate_nmae"] <= row["predicted_gate_parameter_metrics"]["U0_base"]["worst_combo_predicted_gate_nmae"] for row in runs), "U2_at_least_four_regimes_nonnegative": all(sum(all(value >= 0 for key, value in direction.items() if key.startswith(regime + "_")) for regime in REGIMES) >= 4 for direction in directions), "U2_power25_nonnegative": all(all(value >= 0 for key, value in direction.items() if key.endswith("_25")) for direction in directions), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    summary = {"experiment": "P3o-U local phase dynamics conditional ISRJ calibration", "development_passed": bool(all(gates.values())), "verdict": "promote_to_fresh_confirmation_without_retuning" if all(gates.values()) else "stop_no_confirmation_data", "promotion_gates": gates, "data_audit": audit, "runs": runs, "test_generated": False, "test_opened": False, "locked_test_used": False}; (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8"); np.savez_compressed(output / "development_predictions.npz", **saved); print(json.dumps({"development_passed": summary["development_passed"], "promotion_gates": gates}, indent=2))


if __name__ == "__main__": main()

"""Train a compact raw-IQ student only from beneficial frozen P1n corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cj0724.data import SIGNAL_LENGTH
from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3f_shared_isrj_router import choose_threshold, classification_metrics
from cj0724.p3g_cascaded_isrj_prefilter import collect_frozen_route, collect_shared
from cj0724.p3i_distilled_isrj_student import logit, mean_recall
from cj0724.paper_core_evidence import load_normalizer
from cj0724.train import set_seed


MODEL_SEEDS = (2041, 2042, 2043)
ISRJ_INDEX = 5


class DepthwiseStage(nn.Module):
    def __init__(self, input_width: int, output_width: int, stride: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_width, input_width,  nine := 9, stride=stride, padding=nine // 2, groups=input_width, bias=False),
            nn.GroupNorm(1, input_width), nn.GELU(),
            nn.Conv1d(input_width, output_width, 1, bias=False),
            nn.GroupNorm(1, output_width), nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class RawIQAdvantageStudent(nn.Module):
    def __init__(self, context_width: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, 12, 15, stride=5, padding=7, bias=False), nn.GroupNorm(1, 12), nn.GELU())
        self.encoder = nn.Sequential(DepthwiseStage(12, 20, 2), DepthwiseStage(20, 28, 2), DepthwiseStage(28, 40, 2), DepthwiseStage(40, 48, 2))
        self.head = nn.Sequential(nn.LayerNorm(96 + context_width), nn.Linear(96 + context_width, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 2))

    def forward(self, iq: torch.Tensor, context: torch.Tensor, main_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(self.stem(iq.float()))
        pooled = torch.cat((encoded.mean(-1), encoded.amax(-1)), dim=1)
        output = self.head(torch.cat((pooled, context.float()), dim=1))
        residual = 2.5 * torch.tanh(output[:, 0])
        return main_logit.float() + residual, output[:, 1]


def advantage_loss(final_logit, helpful_logit, truth, teacher, main_probability):
    main_error = (main_probability - truth).abs()
    teacher_advantage = (main_error - (teacher - truth).abs()).clamp_min(0.0)
    truth_raw = F.binary_cross_entropy_with_logits(final_logit, truth, reduction="none")
    truth_loss = (truth_raw * (1.0 + 3.0 * main_error)).mean()
    soft_raw = F.binary_cross_entropy_with_logits(final_logit, teacher, reduction="none")
    soft_loss = (soft_raw * teacher_advantage).sum() / teacher_advantage.sum().clamp_min(1.0)
    main_decision = main_probability >= 0.5
    teacher_decision = teacher >= 0.5
    helpful = ((teacher_decision == truth.bool()) & (main_decision != truth.bool())).float()
    helpful_loss = F.binary_cross_entropy_with_logits(helpful_logit, helpful)
    return truth_loss + 0.5 * soft_loss + 0.25 * helpful_loss


def load_iq_stack(files_by_regime) -> np.ndarray:
    parts = []
    for regime in REGIMES:
        for power in POWERS:
            with h5py.File(files_by_regime[regime][power], "r") as handle:
                raw = np.asarray(handle["data_legacy"][:, :], dtype=np.float32).T.reshape(-1, 2, SIGNAL_LENGTH)
            rms = np.sqrt(np.mean(np.square(raw[:, 0]) + np.square(raw[:, 1]), axis=1, keepdims=True)).clip(1e-12)
            parts.append((raw / rms[:, None]).astype(np.float16))
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3k_raw_iq_student_train_seed620000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3k_raw_iq_student_dev_seed630000000_{regime}_rechunked")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--preregister", default="configs/p3k_raw_iq_advantage_student_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3k_raw_iq_advantage_student_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=16); parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]; output = project / args.output_root; output.mkdir(parents=True, exist_ok=False)
    run_root, p2c_root = project / args.run_root, project / args.p2c_root
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8")); rows = json.loads((run_root / "summary.json").read_text(encoding="utf-8")); p2c = json.loads((p2c_root / "method_freeze.json").read_text(encoding="utf-8")); normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {int(row["seed"]): row for row in rows if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}; freezes = {int(row["seed"]): row for row in p2c["seed_freezes"]}
    excluded = [Path(protocol["data_file"])] + [Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files]
    roots = {role: {regime: transition_files(project / template.format(regime=regime), "train", POWERS) for regime in REGIMES} for role, template in (("train", args.train_root_template), ("development", args.development_root_template))}
    metadata, references = {"train": {}, "development": {}}, {}
    for role in ("train", "development"):
        reference = None
        for regime in REGIMES:
            audit_power_alignment(roots[role][regime]); identities, labels = read_identity_metadata(roots[role][regime][POWERS[0]])
            if reference is None: reference = identities
            elif not np.array_equal(reference, identities): raise RuntimeError("P3k regimes not paired")
            for path in excluded:
                previous, _ = read_identity_metadata(path)
                if np.intersect1d(identities, previous).size: raise RuntimeError("P3k identity overlap")
            metadata[role][regime] = {"identities": identities, "labels": labels}
        references[role] = reference
    if np.intersect1d(references["train"], references["development"]).size: raise RuntimeError("P3k cross-role overlap")
    roles = stratified_roles_10(metadata["train"]["ID_control"]["labels"], 20260801)
    shared_train, shared_dev = collect_shared(roots["train"], metadata["train"]), collect_shared(roots["development"], metadata["development"])
    iq_train, iq_dev = load_iq_stack(roots["train"]), load_iq_stack(roots["development"])
    fit_mask = np.isin(shared_train["sample_index"], roles["fit"]); selection_mask = np.isin(shared_train["sample_index"], roles["selection"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); results = []
    for seed in MODEL_SEEDS:
        route = collect_frozen_route(project, roots["train"], metadata["train"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root)
        dev_route = collect_frozen_route(project, roots["development"], metadata["development"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root)
        thresholds = np.asarray(freezes[seed]["routed_thresholds"], dtype=np.float64)
        context_train = np.concatenate((shared_train["features"], route["main"]), axis=1).astype(np.float32); context_dev = np.concatenate((shared_dev["features"], dev_route["main"]), axis=1).astype(np.float32)
        mean, scale = context_train[fit_mask].mean(0, keepdims=True), np.maximum(context_train[fit_mask].std(0, keepdims=True), 1e-4); context_train, context_dev = (context_train - mean) / scale, (context_dev - mean) / scale
        main_prob_train = route["main"][:, ISRJ_INDEX].astype(np.float32); main_logit_train = logit(main_prob_train).astype(np.float32); truth_train = route["labels"][:, ISRJ_INDEX].astype(np.float32); teacher_train = route["full"][:, ISRJ_INDEX].astype(np.float32)
        def loader(mask, shuffle):
            idx = np.flatnonzero(mask); ds = TensorDataset(torch.from_numpy(iq_train[idx]), torch.from_numpy(context_train[idx]), torch.from_numpy(main_logit_train[idx]), torch.from_numpy(truth_train[idx]), torch.from_numpy(teacher_train[idx]), torch.from_numpy(main_prob_train[idx])); return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed) if shuffle else None, pin_memory=device.type == "cuda")
        set_seed(seed); model = RawIQAdvantageStudent(context_train.shape[1]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); fit_loader, selection_loader = loader(fit_mask, True), loader(selection_mask, False); run_dir = output / f"seed_{seed}"; run_dir.mkdir(); np.savez_compressed(run_dir / "normalization.npz", mean=mean, scale=scale); best, stale, history = float("inf"), 0, []
        for epoch in range(1, args.epochs + 1):
            model.train(); train_losses = []
            for iq, context, ml, truth, teacher, mp in fit_loader:
                optimizer.zero_grad(set_to_none=True); final, helpful = model(iq.to(device), context.to(device), ml.to(device)); loss = advantage_loss(final, helpful, truth.to(device), teacher.to(device), mp.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); train_losses.append(float(loss.detach()))
            model.eval(); selection_losses = []
            with torch.inference_mode():
                for iq, context, ml, truth, teacher, mp in selection_loader:
                    final, helpful = model(iq.to(device), context.to(device), ml.to(device)); selection_losses.append(float(advantage_loss(final, helpful, truth.to(device), teacher.to(device), mp.to(device))))
            row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "selection_loss": float(np.mean(selection_losses))}; history.append(row)
            if row["selection_loss"] < best: best, stale = row["selection_loss"], 0; torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience: break
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)); model.eval()
        def predict(iq, context, main_probability):
            ml = logit(main_probability).astype(np.float32); ds = TensorDataset(torch.from_numpy(iq), torch.from_numpy(context), torch.from_numpy(ml)); loader_ = DataLoader(ds, batch_size=args.batch_size, pin_memory=device.type == "cuda"); values = []
            with torch.inference_mode():
                for ib, cb, mb in loader_: values.append(torch.sigmoid(model(ib.to(device), cb.to(device), mb.to(device))[0]).cpu().numpy())
            return np.concatenate(values)
        selection_probability = predict(iq_train[selection_mask], context_train[selection_mask], main_prob_train[selection_mask]); selected = choose_threshold(selection_probability, route["labels"][selection_mask, ISRJ_INDEX], shared_train["regime"][selection_mask], shared_train["power"][selection_mask])
        dev_probability = predict(iq_dev, context_dev, dev_route["main"][:, ISRJ_INDEX].astype(np.float32)); candidate_probability = dev_route["main"].copy(); candidate_probability[:, ISRJ_INDEX] = dev_probability; candidate_thresholds = thresholds.copy(); candidate_thresholds[ISRJ_INDEX] = selected["threshold"]
        safe = classification_metrics(dev_route["main"], dev_route["labels"], thresholds, shared_dev["regime"], shared_dev["power"]); teacher = classification_metrics(dev_route["full"], dev_route["labels"], thresholds, shared_dev["regime"], shared_dev["power"]); student = classification_metrics(candidate_probability, dev_route["labels"], candidate_thresholds, shared_dev["regime"], shared_dev["power"])
        safe_recall, teacher_recall, student_recall = mean_recall(safe), mean_recall(teacher), mean_recall(student); teacher_gain, student_gain = teacher_recall - safe_recall, student_recall - safe_recall; retention = student_gain / max(teacher_gain, 1e-12)
        run = {"seed": seed, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "epochs_ran": len(history), "best_selection_loss": best, "threshold_selection": selected, "teacher_recall_gain": teacher_gain, "student_recall_gain": student_gain, "recall_gain_retention": retention, "maximum_cell_fpr": float(max(row["fpr"] for row in student["cells"])), "macro_aupr_delta_vs_teacher": float(student["macro_aupr"] - teacher["macro_aupr"]), "safe": safe, "teacher": teacher, "student": student}
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8"); (run_dir / "result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8"); np.savez_compressed(run_dir / "development_predictions.npz", identity=shared_dev["identity"], regime=shared_dev["regime"], power=shared_dev["power"], labels=dev_route["labels"], safe_probability=dev_route["main"], teacher_probability=dev_route["full"], student_probability=candidate_probability, student_thresholds=candidate_thresholds); results.append(run); print(json.dumps({key: run[key] for key in ("seed", "teacher_recall_gain", "student_recall_gain", "recall_gain_retention", "maximum_cell_fpr", "macro_aupr_delta_vs_teacher")}), flush=True)
        del route, dev_route, model
        if device.type == "cuda": torch.cuda.empty_cache()
    gates = {"identity_overlap_zero": True, "each_seed_retain_90pct_own_teacher_gain": bool(all(run["recall_gain_retention"] >= 0.90 for run in results)), "maximum_cell_fpr_at_most_0p10": bool(all(run["maximum_cell_fpr"] <= 0.10 for run in results)), "each_seed_macro_aupr_delta_vs_teacher_at_least_minus_0p005": bool(all(run["macro_aupr_delta_vs_teacher"] >= -0.005 for run in results)), "three_of_three_positive_gain": bool(all(run["student_recall_gain"] > 0 for run in results))}; passed = bool(all(gates.values()))
    summary = {"experiment": "P3k raw-IQ advantage-weighted ISRJ student", "preregister_sha256": sha256_file(project / args.preregister), "identity_audit": {"train_sha256": _hash_int64(references["train"]), "development_sha256": _hash_int64(references["development"]), "cross_role_overlap": 0}, "promotion_gates": gates, "development_passed": passed, "runs": results, "interpretation": "Freeze for untouched efficiency confirmation." if passed else "Raw-IQ advantage student failed frozen gates; freeze development identity and move to shared separation representation.", "test_generated": False, "test_opened": False, "locked_test_used": False}; (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8"); print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__": main()

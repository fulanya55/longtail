"""Distill the frozen P1n ISRJ route into a compact shared-feature residual student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p2z_isrj_reset_consistent_separator import stratified_roles_10
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3f_shared_isrj_router import choose_threshold, classification_metrics
from cj0724.p3g_cascaded_isrj_prefilter import collect_shared
from cj0724.p3h_latent_physics_prefilter import collect_route_and_latent
from cj0724.paper_core_evidence import load_normalizer
from cj0724.train import set_seed


MODEL_SEEDS = (2041, 2042, 2043)
ISRJ_INDEX = 5


class DistilledISRJStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, residual_bound: float = 3.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.residual = nn.Linear(hidden_dim, 1); self.changed = nn.Linear(hidden_dim, 1); self.residual_bound = float(residual_bound)

    def forward(self, features: torch.Tensor, main_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features); final_logit = main_logit + self.residual_bound * torch.tanh(self.residual(hidden).squeeze(1)); return final_logit, self.changed(hidden).squeeze(1)


def distillation_loss(final_logit: torch.Tensor, changed_logit: torch.Tensor, truth: torch.Tensor, teacher_probability: torch.Tensor, changed: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(final_logit, truth) + 0.5 * F.binary_cross_entropy_with_logits(final_logit, teacher_probability) + 0.25 * F.binary_cross_entropy_with_logits(changed_logit, changed)


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability), 1.0e-5, 1.0 - 1.0e-5); return np.log(value) - np.log1p(-value)


def mean_recall(metric: dict[str, object]) -> float:
    return float(np.mean([row["recall"] for row in metric["cells"]]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3i_distilled_student_train_seed570000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3i_distilled_student_dev_seed580000000_{regime}_rechunked")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--preregister", default="configs/p3i_distilled_isrj_student_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3i_distilled_isrj_student_development_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--patience", type=int, default=5); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
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
            elif not np.array_equal(reference, identities): raise RuntimeError("P3i regimes not paired")
            for path in excluded:
                previous, _ = read_identity_metadata(path)
                if np.intersect1d(identities, previous).size: raise RuntimeError("P3i identity overlap")
            metadata[role][regime] = {"identities": identities, "labels": labels}
        references[role] = reference
    if np.intersect1d(references["train"], references["development"]).size: raise RuntimeError("P3i cross-role overlap")
    roles = stratified_roles_10(metadata["train"]["ID_control"]["labels"], 20260801); shared_train = collect_shared(roots["train"], metadata["train"]); shared_dev = collect_shared(roots["development"], metadata["development"]); fit_mask = np.isin(shared_train["sample_index"], roles["fit"]); selection_mask = np.isin(shared_train["sample_index"], roles["selection"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); results = []
    for seed in MODEL_SEEDS:
        train_route = collect_route_and_latent(project, roots["train"], metadata["train"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root, protocol); dev_route = collect_route_and_latent(project, roots["development"], metadata["development"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root, protocol)
        threshold = np.asarray(freezes[seed]["routed_thresholds"], dtype=np.float64); changed = ((train_route["full"][:, ISRJ_INDEX] >= threshold[ISRJ_INDEX]) != (train_route["main"][:, ISRJ_INDEX] >= threshold[ISRJ_INDEX])).astype(np.float32)
        x_train = np.concatenate((train_route["latent"], shared_train["features"], train_route["main"]), axis=1).astype(np.float32); x_dev = np.concatenate((dev_route["latent"], shared_dev["features"], dev_route["main"]), axis=1).astype(np.float32)
        mean, scale = x_train[fit_mask].mean(0, keepdims=True), np.maximum(x_train[fit_mask].std(0, keepdims=True), 1e-4); x_train = (x_train - mean) / scale; x_dev = (x_dev - mean) / scale
        main_logit_train = logit(train_route["main"][:, ISRJ_INDEX]).astype(np.float32); main_logit_dev = logit(dev_route["main"][:, ISRJ_INDEX]).astype(np.float32); truth_train = train_route["labels"][:, ISRJ_INDEX].astype(np.float32); teacher_train = train_route["full"][:, ISRJ_INDEX].astype(np.float32)
        set_seed(seed); model = DistilledISRJStudent(x_train.shape[1]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        def loader(mask, shuffle):
            indices = np.flatnonzero(mask); dataset = TensorDataset(torch.from_numpy(x_train[indices]), torch.from_numpy(main_logit_train[indices]), torch.from_numpy(truth_train[indices]), torch.from_numpy(teacher_train[indices]), torch.from_numpy(changed[indices])); return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed) if shuffle else None)
        fit_loader, selection_loader = loader(fit_mask, True), loader(selection_mask, False); run_dir = output / f"seed_{seed}"; run_dir.mkdir(); np.savez_compressed(run_dir / "normalization.npz", mean=mean, scale=scale)
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []
            for x, main_logit, truth, teacher, changed_target in fit_loader:
                optimizer.zero_grad(set_to_none=True); final, changed_logit = model(x.to(device), main_logit.to(device)); loss = distillation_loss(final, changed_logit, truth.to(device), teacher.to(device), changed_target.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach()))
            model.eval(); validation = []
            with torch.inference_mode():
                for x, main_logit, truth, teacher, changed_target in selection_loader:
                    final, changed_logit = model(x.to(device), main_logit.to(device)); validation.append(float(distillation_loss(final, changed_logit, truth.to(device), teacher.to(device), changed_target.to(device))))
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_loss": float(np.mean(validation))}; history.append(row)
            if row["selection_loss"] < best: best, stale = row["selection_loss"], 0; torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience: break
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)); model.eval()
        def predict(x, main_logit):
            batches = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(main_logit)), batch_size=args.batch_size, shuffle=False); values = []
            with torch.inference_mode():
                for xb, mb in batches: values.append(torch.sigmoid(model(xb.to(device), mb.to(device))[0]).cpu().numpy())
            return np.concatenate(values)
        selection_probability = predict(x_train[selection_mask], main_logit_train[selection_mask]); threshold_selection = choose_threshold(selection_probability, train_route["labels"][selection_mask, ISRJ_INDEX], shared_train["regime"][selection_mask], shared_train["power"][selection_mask]); dev_probability = predict(x_dev, main_logit_dev)
        candidate_probability = dev_route["main"].copy(); candidate_probability[:, ISRJ_INDEX] = dev_probability; candidate_thresholds = threshold.copy(); candidate_thresholds[ISRJ_INDEX] = threshold_selection["threshold"]
        safe = classification_metrics(dev_route["main"], dev_route["labels"], threshold, shared_dev["regime"], shared_dev["power"]); teacher_metric = classification_metrics(dev_route["full"], dev_route["labels"], threshold, shared_dev["regime"], shared_dev["power"]); student_metric = classification_metrics(candidate_probability, dev_route["labels"], candidate_thresholds, shared_dev["regime"], shared_dev["power"])
        safe_recall, teacher_recall, student_recall = mean_recall(safe), mean_recall(teacher_metric), mean_recall(student_metric); teacher_gain, student_gain = teacher_recall - safe_recall, student_recall - safe_recall; retention = student_gain / max(teacher_gain, 1e-12)
        run = {"seed": seed, "epochs_ran": len(history), "best_selection_loss": best, "threshold_selection": threshold_selection, "safe_recall": safe_recall, "teacher_recall": teacher_recall, "student_recall": student_recall, "teacher_recall_gain": teacher_gain, "student_recall_gain": student_gain, "recall_gain_retention": retention, "maximum_cell_fpr": float(max(row["fpr"] for row in student_metric["cells"])), "macro_aupr_delta_vs_teacher": float(student_metric["macro_aupr"] - teacher_metric["macro_aupr"]), "safe": safe, "teacher": teacher_metric, "student": student_metric}
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8"); (run_dir / "result.json").write_text(json.dumps(_jsonable(run), indent=2), encoding="utf-8"); np.savez_compressed(run_dir / "development_predictions.npz", identity=shared_dev["identity"], regime=shared_dev["regime"], power=shared_dev["power"], labels=dev_route["labels"], safe_probability=dev_route["main"], teacher_probability=dev_route["full"], student_probability=candidate_probability, safe_thresholds=threshold, student_thresholds=candidate_thresholds); results.append(run); print(json.dumps({key: run[key] for key in ("seed", "teacher_recall_gain", "student_recall_gain", "recall_gain_retention", "maximum_cell_fpr", "macro_aupr_delta_vs_teacher")}), flush=True); del model
        if device.type == "cuda": torch.cuda.empty_cache()
    gates = {"identity_overlap_zero": True, "each_seed_retain_90pct_teacher_gain": bool(all(run["recall_gain_retention"] >= 0.90 for run in results)), "maximum_cell_fpr_at_most_0p10": bool(all(run["maximum_cell_fpr"] <= 0.10 for run in results)), "each_seed_macro_aupr_delta_vs_teacher_at_least_minus_0p005": bool(all(run["macro_aupr_delta_vs_teacher"] >= -0.005 for run in results)), "three_of_three_positive_gain": bool(all(run["student_recall_gain"] > 0 for run in results))}; passed = bool(all(gates.values()))
    summary = {"experiment": "P3i distilled shared-representation ISRJ student", "preregister_sha256": sha256_file(project / args.preregister), "identity_audit": {"train_sha256": _hash_int64(references["train"]), "development_sha256": _hash_int64(references["development"]), "cross_role_overlap": 0}, "promotion_gates": gates, "development_passed": passed, "runs": results, "interpretation": "Freeze for end-to-end parameter and efficiency confirmation." if passed else "Compact distillation failed frozen gates; do not tune on seed580000000.", "test_generated": False, "test_opened": False, "locked_test_used": False}; (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8"); print(json.dumps({"development_passed": passed, "promotion_gates": gates, "interpretation": summary["interpretation"]}, indent=2))


if __name__ == "__main__": main()

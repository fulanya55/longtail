"""P3o-G: bounded, state-preserving residual trained for compound-group risk."""
from __future__ import annotations

import argparse
import copy
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.multitask import masked_parameter_smooth_l1, reshape_selected_parameters
from cj0724.multitask_benchmark import _combination_ids
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3n_cross_interference_backbone import ISRJ_INDEX, output_parameter_metrics, sha256_file
from cj0724.p3o_low_power_attribution import audit_identities, build_concat_dataset, close_concat_dataset, evaluate_outputs, fit_multi_file_normalizer, make_identity_roles, ordered_paths, role_metadata, save_prediction_payload, select_checkpoint_epochs, select_cp_safe_isrj_threshold
from cj0724.p3ob_dualview_representation import TemporalEncoder, collect_outputs, delay_peak_sequence, train_epoch
from cj0724.p3oc_parameter_matched_multiview import ParameterMatchedMultiView
from cj0724.train import get_device, set_seed
from cj0724.tuned_multitask_sweep import _set_warmup_cosine_lr, optimize_class_thresholds

MODEL_SEEDS = (2061, 2062, 2063)
METHODS = (("G0_bounded_mean", 0.0), ("G1_bounded_combo_cvar", 1.0))


class BoundedStatePreservingResidual(nn.Module):
    """A frozen F model plus a bounded delay residual; the F branch cannot train."""
    def __init__(self, base: ParameterMatchedMultiView, residual_bound: float = 0.25) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.residual_bound = float(residual_bound)
        self.delay = TemporalEncoder(3, 10)
        self.residual = nn.Sequential(nn.LayerNorm(104), nn.Linear(104, 12))

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward_with_embedding(self, iq: torch.Tensor):
        with torch.no_grad():
            logits, base_parameter, embedding = self.base.forward_with_embedding(iq)
        delay = self.delay(delay_peak_sequence(iq))
        raw = self.residual(torch.cat((embedding, delay), dim=1)).reshape(-1, 6, 2)
        residual = self.residual_bound * torch.tanh(raw)
        return logits, base_parameter + torch.sigmoid(logits).detach().unsqueeze(-1) * residual, embedding

    def forward(self, iq: torch.Tensor):
        logits, parameters, _ = self.forward_with_embedding(iq)
        return logits, parameters


def _paths(project: Path, template: str):
    return {regime: transition_files(project / template.format(regime=regime), "train", POWERS) for regime in REGIMES}


def _base_state_equal(before: dict[str, torch.Tensor], base: nn.Module) -> bool:
    state = base.state_dict()
    return bool(all(torch.equal(value, state[key]) for key, value in before.items()))


def _per_sample_parameter_loss(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & (labels[:, :, None] > 0.5)
    if bool((valid & ~torch.isfinite(target)).any()):
        raise ValueError("valid parameter targets must be finite")
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    counts = valid.sum(dim=(1, 2)).clamp_min(1)
    return (error * valid).sum(dim=(1, 2)) / counts


def compound_cvar_loss(per_sample: torch.Tensor, labels: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    """Equal-weight upper tail of observed six-label compound groups within a batch."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    powers = (2 ** torch.arange(labels.shape[1], device=labels.device, dtype=torch.int64))
    groups = (labels.detach().round().to(torch.int64) * powers).sum(dim=1)
    means = torch.stack([per_sample[groups == group].mean() for group in torch.unique(groups)])
    count = max(1, int(math.ceil(float(fraction) * means.numel())))
    return torch.topk(means, count).values.mean()


def train_side_epoch(model, loader, device, normalizer, optimizer, args, cvar_weight: float) -> dict[str, float]:
    model.train(); totals, means, risks = [], [], []
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True); labels = batch["label"].to(device, non_blocking=True)
        target = reshape_selected_parameters(batch["parameter_values"]).to(device, non_blocking=True)
        mask = reshape_selected_parameters(batch["parameter_mask"]).bool().to(device, non_blocking=True)
        autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
        with autocast:
            _, prediction = model(iq)
            per_sample = _per_sample_parameter_loss(prediction, normalizer.transform(target), labels, mask)
            mean = per_sample.mean(); risk = compound_cvar_loss(per_sample, labels, args.cvar_fraction)
            total = mean + float(cvar_weight) * risk
        optimizer.zero_grad(set_to_none=True); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm); optimizer.step()
        totals.append(float(total.detach())); means.append(float(mean.detach())); risks.append(float(risk.detach()))
    return {"total_loss": float(np.mean(totals)), "mean_parameter_loss": float(np.mean(means)), "compound_cvar_loss": float(np.mean(risks))}


def oracle_worst_combo_nmae(outputs: dict[str, object], normalizer) -> float:
    labels = outputs["labels"].numpy().astype(np.uint8); combo = _combination_ids(labels)
    valid = outputs["mask"] & (labels[:, :, None] > 0)
    error = np.abs(outputs["prediction"] - outputs["target"]) / normalizer.scale.numpy()[None, :, :]
    values = [float(error[(combo[:, None, None] == value) & valid].mean()) for value in np.unique(combo)]
    return float(max(values))


def _build_side(base, initial_state, bound: float, device):
    # Both controls must wrap the exact selected F instance, not merely a
    # value-identical copy, so the post-training audit covers the F state that
    # supplied the reported base prediction and threshold calibration.
    side = BoundedStatePreservingResidual(base, bound).to(device)
    side.load_state_dict(initial_state)
    return side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3og_bounded_combo_cvar_train_seed1230000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3og_bounded_combo_cvar_dev_seed1240000000_{regime}_rechunked")
    parser.add_argument("--preregister", default="configs/p3og_bounded_combo_cvar_preregister_20260802.json")
    parser.add_argument("--method-freeze", default="configs/p3og_bounded_combo_cvar_method_freeze_20260802.json")
    parser.add_argument("--output-root", default="outputs/p3og_bounded_combo_cvar_development_20260802")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--warmup-epochs", type=int, default=2); parser.add_argument("--batch-size", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--parameter-weight", type=float, default=.1); parser.add_argument("--contrast-weight", type=float, default=.0); parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--residual-bound", type=float, default=.25); parser.add_argument("--cvar-fraction", type=float, default=.25); parser.add_argument("--split-seed", type=int, default=20260802)
    parser.add_argument("--cp-cap", type=float, default=.1); parser.add_argument("--cp-confidence", type=float, default=.95); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    freeze = json.loads((project / args.method_freeze).read_text(encoding="utf-8"))
    if freeze["preregister_sha256"] != sha256_file(project / args.preregister) or freeze["implementation_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("P3o-G frozen hash mismatch")
    if freeze["data_generator_sha256"] != sha256_file(project / "matlab/generate_p3o_low_power_attribution_data.m"):
        raise RuntimeError("P3o-G generator hash mismatch")
    output = project / args.output_root; output.mkdir(parents=True, exist_ok=False); device = get_device(args.device)
    roots = {"train": _paths(project, args.train_root_template), "development": _paths(project, args.development_root_template)}; metadata, references = {"train": {}, "development": {}}, {}
    for role, role_roots in roots.items():
        paired = None
        for regime, paths in role_roots.items():
            audit_power_alignment(paths); identities, labels = read_identity_metadata(paths[25]); paired = identities if paired is None else paired
            if not np.array_equal(paired, identities): raise RuntimeError(f"unpaired {role} regimes")
            metadata[role][regime] = {"identities": identities, "labels": labels}
        references[role] = paired
    labels, development_labels = metadata["train"]["ID_control"]["labels"], metadata["development"]["ID_control"]["labels"]
    if not np.array_equal(labels, development_labels): raise RuntimeError("label contract mismatch")
    roles = make_identity_roles(labels, args.split_seed); old = json.loads((project / args.run_root / "protocol.json").read_text(encoding="utf-8"))
    audit = audit_identities(project, references["train"], references["development"], {1230000000, 1240000000}, Path(old["data_file"]))
    if audit["cross_role_overlap"] or audit["maximum_prior_overlap"]: raise RuntimeError("identity collision")
    (output / "identity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); np.savez_compressed(output / "identity_roles.npz", **roles)
    protocol = {"args": vars(args), "model_seeds": list(MODEL_SEEDS), "methods": [name for name, _ in METHODS], "preregister_sha256": freeze["preregister_sha256"], "implementation_sha256": freeze["implementation_sha256"], "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    normalizer = fit_multi_file_normalizer(ordered_paths(roots["train"]), roles["full_fit"], labels)
    datasets = {role: build_concat_dataset(ordered_paths(roots["train"]), roles[role]) for role in ("full_fit", "early", "calibration")}; development = build_concat_dataset(ordered_paths(roots["development"]), np.arange(len(development_labels)))
    meta = {role: role_metadata(metadata["train"], roles[role]) for role in ("calibration",)}; development_meta = role_metadata(metadata["development"], np.arange(len(development_labels))); results = []
    try:
        for seed in MODEL_SEEDS:
            set_seed(seed)
            base_fit = DataLoader(datasets["full_fit"], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
            early_loader = DataLoader(datasets["early"], batch_size=args.batch_size); calibration_loader = DataLoader(datasets["calibration"], batch_size=args.batch_size); development_loader = DataLoader(development, batch_size=args.batch_size)
            base = ParameterMatchedMultiView("F_time_frequency_matched").to(device); optimizer = torch.optim.AdamW(base.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); base_history = []
            for epoch in range(1, args.epochs + 1):
                lr = _set_warmup_cosine_lr(optimizer, args.learning_rate, epoch, args.epochs, args.warmup_epochs); train = train_epoch(base, base_fit, device, normalizer, optimizer, None, args); early = collect_outputs(base, early_loader, device, normalizer, args.parameter_weight)
                row = {"epoch": epoch, "lr": lr, "train": train, "early_parameter_macro_nmae": float(output_parameter_metrics(early, normalizer)["oracle_macro_target_nmae"]), "early_macro_aupr": float(average_precision_score(early["labels"].numpy(), torch.sigmoid(early["logits"]).numpy(), average="macro"))}; base_history.append(row); torch.save(base.state_dict(), output / f"base_{seed}_{epoch}.pt")
            base_epoch = int(select_checkpoint_epochs(base_history)["C_joint_epoch"]); base.load_state_dict(torch.load(output / f"base_{seed}_{base_epoch}.pt", map_location=device, weights_only=True)); base.eval()
            calibration = collect_outputs(base, calibration_loader, device, normalizer, args.parameter_weight); thresholds = optimize_class_thresholds(torch.sigmoid(calibration["logits"]).numpy(), calibration["labels"].numpy()); safe = select_cp_safe_isrj_threshold(torch.sigmoid(calibration["logits"]).numpy(), calibration["labels"].numpy(), meta["calibration"]["regime"], meta["calibration"]["power"], args.cp_cap, args.cp_confidence); thresholds[ISRJ_INDEX] = safe["threshold"]
            base_development = collect_outputs(base, development_loader, device, normalizer, args.parameter_weight); base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
            template = BoundedStatePreservingResidual(copy.deepcopy(base), args.residual_bound).to(device); initial_state = copy.deepcopy(template.state_dict()); method_rows = {}
            for method, cvar_weight in METHODS:
                side = _build_side(base, initial_state, args.residual_bound, device); optimizer = torch.optim.AdamW((p for p in side.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay); history = []
                for epoch in range(1, args.epochs + 1):
                    lr = _set_warmup_cosine_lr(optimizer, args.learning_rate, epoch, args.epochs, args.warmup_epochs)
                    fit = DataLoader(datasets["full_fit"], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 7000))
                    train = train_side_epoch(side, fit, device, normalizer, optimizer, args, cvar_weight); early = collect_outputs(side, early_loader, device, normalizer, args.parameter_weight)
                    row = {"epoch": epoch, "lr": lr, "train": train, "early_oracle_worst_combo_nmae": oracle_worst_combo_nmae(early, normalizer), "early_parameter_macro_nmae": float(output_parameter_metrics(early, normalizer)["oracle_macro_target_nmae"])}; history.append(row); torch.save(side.state_dict(), output / f"{method}_{seed}_{epoch}.pt")
                chosen = min(history, key=lambda row: (row["early_oracle_worst_combo_nmae"], row["early_parameter_macro_nmae"], row["epoch"]))["epoch"]; side.load_state_dict(torch.load(output / f"{method}_{seed}_{chosen}.pt", map_location=device, weights_only=True)); side_development = collect_outputs(side, development_loader, device, normalizer, args.parameter_weight)
                logits_equal = bool(torch.equal(base_development["logits"], side_development["logits"])); state_equal = _base_state_equal(base_state, side.base)
                method_rows[method] = {"evaluation": evaluate_outputs(side_development, thresholds, development_meta, normalizer), "history": history, "epoch": int(chosen), "logits_bitwise_equal": logits_equal, "base_state_bitwise_equal": state_equal}
                save_prediction_payload(output / f"seed_{seed}_{method}.npz", side_development, development_meta)
            save_prediction_payload(output / f"seed_{seed}_F_base.npz", base_development, development_meta)
            results.append({"seed": seed, "base": evaluate_outputs(base_development, thresholds, development_meta, normalizer), "base_history": base_history, "base_epoch": base_epoch, "safe_audit": safe, "methods": method_rows})
    finally:
        for dataset in datasets.values(): close_concat_dataset(dataset)
        close_concat_dataset(development)
    comparisons = []
    for row in results:
        base, g0, g1 = row["base"], row["methods"]["G0_bounded_mean"]["evaluation"], row["methods"]["G1_bounded_combo_cvar"]["evaluation"]
        comparisons.append({"seed": row["seed"], "g1_oracle_ratio_vs_base": g1["parameter"]["oracle_macro_target_nmae"] / base["parameter"]["oracle_macro_target_nmae"], "g1_predicted_ratio_vs_base": g1["predicted_gate"]["predicted_gate_parameter_nmae"] / base["predicted_gate"]["predicted_gate_parameter_nmae"], "g1_miss_delta_vs_base": g1["predicted_gate"]["miss_penalty_parameter_nmae"] - base["predicted_gate"]["miss_penalty_parameter_nmae"], "g1_worst_ratio_vs_base": g1["predicted_gate"]["worst_combo_predicted_gate_nmae"] / base["predicted_gate"]["worst_combo_predicted_gate_nmae"], "g1_worst_ratio_vs_g0": g1["predicted_gate"]["worst_combo_predicted_gate_nmae"] / g0["predicted_gate"]["worst_combo_predicted_gate_nmae"]})
    gates = {"identity_overlap_zero": not audit["cross_role_overlap"] and not audit["maximum_prior_overlap"], "all_side_runs_preserve_F_state_and_logits_bitwise": all(item["methods"][method]["logits_bitwise_equal"] and item["methods"][method]["base_state_bitwise_equal"] for item in results for method, _ in METHODS), "all_seeds_base_safe_calibration_feasible_and_development_maximum_cell_fpr_at_most_0p10": all(item["safe_audit"]["feasible"] and item["base"]["maximum_cell_fpr"] <= .1 for item in results), "all_seeds_base_overall_isrj_recall_at_least_0p85_and_25db_recall_at_least_0p78": all(item["base"]["isrj_recall"] >= .85 and item["base"]["per_power"]["25"]["isrj_recall"] >= .78 for item in results), "all_seeds_G1_oracle_and_predicted_gate_nmae_ratio_vs_F_base_at_most_0p98": all(item["g1_oracle_ratio_vs_base"] <= .98 and item["g1_predicted_ratio_vs_base"] <= .98 for item in comparisons), "all_seeds_G1_miss_penalty_nonworse_vs_F_base": all(item["g1_miss_delta_vs_base"] <= 0 for item in comparisons), "all_seeds_G1_worst_combo_predicted_gate_nmae_ratio_vs_F_base_at_most_0p98": all(item["g1_worst_ratio_vs_base"] <= .98 for item in comparisons), "all_seeds_G1_worst_combo_predicted_gate_nmae_ratio_vs_G0_at_most_0p99": all(item["g1_worst_ratio_vs_g0"] <= .99 for item in comparisons), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    summary = {"experiment": "P3o-G bounded compound-CVaR residual", "development_passed": bool(all(gates.values())), "promotion_gates": gates, "comparisons": comparisons, "results": results, "identity_audit": audit, "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(summary, default=lambda value: value.item() if hasattr(value, "item") else str(value), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

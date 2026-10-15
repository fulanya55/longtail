"""P3o-E: state-preserving F recognition plus delay-only parameter residual."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3n_cross_interference_backbone import ISRJ_INDEX, output_parameter_metrics, sha256_file
from cj0724.p3o_low_power_attribution import MODEL_SEEDS, audit_identities, build_concat_dataset, close_concat_dataset, evaluate_outputs, fit_multi_file_normalizer, make_identity_roles, ordered_paths, role_metadata, save_prediction_payload, select_checkpoint_epochs, select_cp_safe_isrj_threshold
from cj0724.p3ob_dualview_representation import TemporalEncoder, collect_outputs, delay_peak_sequence, train_epoch
from cj0724.p3oc_parameter_matched_multiview import ParameterMatchedMultiView
from cj0724.train import get_device, set_seed
from cj0724.tuned_multitask_sweep import _set_warmup_cosine_lr, optimize_class_thresholds


class StatePreservingFrequencyDelayResidual(nn.Module):
    """Train only the residual while the complete F base remains immutable and eval-mode."""
    def __init__(self, base: ParameterMatchedMultiView) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
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
        residual = self.residual(torch.cat((embedding, delay), dim=1)).reshape(-1, 6, 2)
        return logits, base_parameter + torch.sigmoid(logits).detach().unsqueeze(-1) * residual, embedding

    def forward(self, iq: torch.Tensor):
        logits, parameters, _ = self.forward_with_embedding(iq)
        return logits, parameters

    def exclusive_contrast_loss(self, embedding: torch.Tensor, labels: torch.Tensor):
        return embedding.new_zeros(())


def _paths(project: Path, template: str):
    return {regime: transition_files(project / template.format(regime=regime), "train", POWERS) for regime in REGIMES}


def _class_nmae(record: dict, name: str) -> float:
    return float(np.mean([item["normalized_mae"] for item in record["parameter"]["per_target"] if item["class_name"] == name]))


def _f_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def _immutable_f_audit(base: ParameterMatchedMultiView, side: StatePreservingFrequencyDelayResidual, iq: torch.Tensor) -> bool:
    base.eval(); side.eval()
    before_logits = base(iq)[0].detach().clone()
    before_buffers = _f_buffers(base)
    side.train()
    if base.training or any(module.training for module in base.modules()):
        return False
    side_logits = side(iq)[0]
    return bool(torch.equal(before_logits, side_logits) and all(torch.equal(before_buffers[name], value) for name, value in base.named_buffers()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3oe_state_preserving_train_seed1180000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3oe_state_preserving_dev_seed1190000000_{regime}_rechunked")
    parser.add_argument("--preregister", default="configs/p3oe_state_preserving_frequency_delay_residual_preregister_20260802.json")
    parser.add_argument("--method-freeze", default="configs/p3oe_state_preserving_frequency_delay_residual_method_freeze_20260802.json")
    parser.add_argument("--output-root", default="outputs/p3oe_state_preserving_frequency_delay_residual_development_20260802")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--warmup-epochs", type=int, default=2); parser.add_argument("--batch-size", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--parameter-weight", type=float, default=.1); parser.add_argument("--contrast-weight", type=float, default=.0); parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--cp-cap", type=float, default=.1); parser.add_argument("--cp-confidence", type=float, default=.95); parser.add_argument("--split-seed", type=int, default=20260802); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    freeze = json.loads((project / args.method_freeze).read_text(encoding="utf-8"))
    if freeze["preregister_sha256"] != sha256_file(project / args.preregister) or freeze["implementation_sha256"] != sha256_file(Path(__file__).resolve()): raise RuntimeError("P3o-E frozen hash mismatch")
    if freeze["data_generator_sha256"] != sha256_file(project / "matlab/generate_p3o_low_power_attribution_data.m"): raise RuntimeError("P3o-E generator hash mismatch")
    output = project / args.output_root; output.mkdir(parents=True, exist_ok=False); device = get_device(args.device)
    roots = {"train": _paths(project, args.train_root_template), "development": _paths(project, args.development_root_template)}; metadata = {"train": {}, "development": {}}; references = {}
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
    audit = audit_identities(project, references["train"], references["development"], {1180000000, 1190000000}, Path(old["data_file"]))
    if audit["cross_role_overlap"] or audit["maximum_prior_overlap"]: raise RuntimeError("identity collision")
    (output / "identity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); np.savez_compressed(output / "identity_roles.npz", **roles)
    (output / "protocol.json").write_text(json.dumps({"args": vars(args), "preregister_sha256": freeze["preregister_sha256"], "implementation_sha256": freeze["implementation_sha256"], "test_generated": False, "test_opened": False, "locked_test_used": False}, indent=2), encoding="utf-8")
    train_paths, development_paths = ordered_paths(roots["train"]), ordered_paths(roots["development"]); normalizer = fit_multi_file_normalizer(train_paths, roles["full_fit"], labels)
    datasets = {key: build_concat_dataset(train_paths, roles[key]) for key in ("full_fit", "early", "calibration", "selection")}; development = build_concat_dataset(development_paths, np.arange(len(development_labels)))
    role_meta = {key: role_metadata(metadata["train"], roles[key]) for key in ("early", "calibration", "selection")}; development_meta = role_metadata(metadata["development"], np.arange(len(development_labels))); results = []
    try:
        for seed in MODEL_SEEDS:
            set_seed(seed); fit = DataLoader(datasets["full_fit"], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)); loaders = {key: DataLoader(datasets[key], batch_size=args.batch_size) for key in ("early", "calibration", "selection")}; development_loader = DataLoader(development, batch_size=args.batch_size)
            base = ParameterMatchedMultiView("F_time_frequency_matched").to(device); optimizer = torch.optim.AdamW(base.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); base_history = []
            for epoch in range(1, args.epochs + 1):
                _set_warmup_cosine_lr(optimizer, args.learning_rate, epoch, args.epochs, args.warmup_epochs); train = train_epoch(base, fit, device, normalizer, optimizer, None, args); early = collect_outputs(base, loaders["early"], device, normalizer, args.parameter_weight); base_history.append({"epoch": epoch, "train": train, "early_parameter_macro_nmae": float(output_parameter_metrics(early, normalizer)["oracle_macro_target_nmae"]), "early_macro_aupr": float(average_precision_score(early["labels"].numpy(), torch.sigmoid(early["logits"]).numpy(), average="macro"))}); torch.save(base.state_dict(), output / f"base_{seed}_{epoch}.pt")
            base_epoch = int(select_checkpoint_epochs(base_history)["C_joint_epoch"]); base.load_state_dict(torch.load(output / f"base_{seed}_{base_epoch}.pt", map_location=device, weights_only=True)); calibration = collect_outputs(base, loaders["calibration"], device, normalizer, args.parameter_weight); thresholds = optimize_class_thresholds(torch.sigmoid(calibration["logits"]).numpy(), calibration["labels"].numpy()); safe = select_cp_safe_isrj_threshold(torch.sigmoid(calibration["logits"]).numpy(), calibration["labels"].numpy(), role_meta["calibration"]["regime"], role_meta["calibration"]["power"], args.cp_cap, args.cp_confidence); thresholds[ISRJ_INDEX] = safe["threshold"]
            base_development = collect_outputs(base, development_loader, device, normalizer, args.parameter_weight); side = StatePreservingFrequencyDelayResidual(base).to(device); immutable_probe = _immutable_f_audit(base, side, next(iter(fit))["iq"][:4].to(device)); optimizer = torch.optim.AdamW((parameter for parameter in side.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay); side_history = []
            for epoch in range(1, args.epochs + 1):
                _set_warmup_cosine_lr(optimizer, args.learning_rate, epoch, args.epochs, args.warmup_epochs); train = train_epoch(side, fit, device, normalizer, optimizer, None, args); early = collect_outputs(side, loaders["early"], device, normalizer, args.parameter_weight); side_history.append({"epoch": epoch, "train": train, "early_parameter_macro_nmae": float(output_parameter_metrics(early, normalizer)["oracle_macro_target_nmae"])}); torch.save(side.state_dict(), output / f"side_{seed}_{epoch}.pt")
            side_epoch = min(side_history, key=lambda item: item["early_parameter_macro_nmae"])["epoch"]; side.load_state_dict(torch.load(output / f"side_{seed}_{side_epoch}.pt", map_location=device, weights_only=True)); side_development = collect_outputs(side, development_loader, device, normalizer, args.parameter_weight); logits_equal = bool(torch.equal(base_development["logits"], side_development["logits"])); results.append({"seed": seed, "base": evaluate_outputs(base_development, thresholds, development_meta, normalizer), "side": evaluate_outputs(side_development, thresholds, development_meta, normalizer), "logits_bitwise_equal": logits_equal, "immutable_f_probe_passed": immutable_probe, "base_history": base_history, "side_history": side_history, "base_epoch": base_epoch, "side_epoch": side_epoch, "safe_audit": safe}); save_prediction_payload(output / f"seed_{seed}_base.npz", base_development, development_meta); save_prediction_payload(output / f"seed_{seed}_side.npz", side_development, development_meta)
    finally:
        for dataset in datasets.values(): close_concat_dataset(dataset)
        close_concat_dataset(development)
    gates = {"identity_overlap_zero": not audit["cross_role_overlap"] and not audit["maximum_prior_overlap"], "post_optimizer_step_F_logits_bitwise_equal_to_base_and_F_buffers_bitwise_unchanged": all(item["logits_bitwise_equal"] and item["immutable_f_probe_passed"] for item in results), "all_seeds_base_safe_calibration_feasible_and_development_maximum_cell_fpr_at_most_0p10": all(item["safe_audit"]["feasible"] and item["base"]["maximum_cell_fpr"] <= .1 for item in results), "all_seeds_base_overall_isrj_recall_at_least_0p85_and_25db_recall_at_least_0p78": all(item["base"]["isrj_recall"] >= .85 and item["base"]["per_power"]["25"]["isrj_recall"] >= .78 for item in results), "all_seeds_residual_oracle_and_predicted_gate_parameter_nmae_ratio_vs_base_at_most_0p98": all(item["side"]["parameter"]["oracle_macro_target_nmae"] / item["base"]["parameter"]["oracle_macro_target_nmae"] <= .98 and item["side"]["predicted_gate"]["predicted_gate_parameter_nmae"] / item["base"]["predicted_gate"]["predicted_gate_parameter_nmae"] <= .98 for item in results), "all_seeds_residual_miss_penalty_and_worst_combination_nonworse_vs_base": all(item["side"]["predicted_gate"]["miss_penalty_parameter_nmae"] <= item["base"]["predicted_gate"]["miss_penalty_parameter_nmae"] and item["side"]["predicted_gate"]["worst_combo_predicted_gate_nmae"] <= item["base"]["predicted_gate"]["worst_combo_predicted_gate_nmae"] for item in results), "all_seeds_residual_mft_and_isrj_oracle_nmae_ratio_vs_base_at_most_0p99": all(_class_nmae(item["side"], "MFT") / _class_nmae(item["base"], "MFT") <= .99 and _class_nmae(item["side"], "ISRJ") / _class_nmae(item["base"], "ISRJ") <= .99 for item in results), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    (output / "summary.json").write_text(json.dumps({"development_passed": all(gates.values()), "promotion_gates": gates, "results": results, "identity_audit": audit, "test_generated": False, "test_opened": False, "locked_test_used": False}, default=lambda value: value.item() if hasattr(value, "item") else str(value), indent=2), encoding="utf-8")


if __name__ == "__main__": main()

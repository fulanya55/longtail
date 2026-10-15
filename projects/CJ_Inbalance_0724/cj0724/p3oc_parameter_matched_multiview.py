"""P3o-C: fair parameter-matched test of delay-view complementarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.multitask import PARAMETER_COLUMNS
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p2w_isrj_width_oracle_ood import POWERS
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3n_cross_interference_backbone import ISRJ_INDEX, output_parameter_metrics, sha256_file
from cj0724.p3o_low_power_attribution import (
    MODEL_SEEDS, audit_identities, build_concat_dataset, bootstrap_isrj_recall_difference,
    close_concat_dataset, evaluate_outputs, fit_multi_file_normalizer, make_identity_roles,
    ordered_paths, role_metadata, save_prediction_payload, select_checkpoint_epochs,
    select_cp_safe_isrj_threshold,
)
from cj0724.p3ob_dualview_representation import (
    TemporalEncoder, TimeFrequencyEncoder, collect_outputs, delay_peak_sequence,
    local_time_frequency, train_epoch,
)
from cj0724.train import get_device, set_seed
from cj0724.tuned_multitask_sweep import _set_warmup_cosine_lr, optimize_class_thresholds


METHODS = ("T_time_matched", "D_delay_matched", "F_time_frequency_matched", "DF_dual_matched")
EXPECTED_PARAMETERS = {"T_time_matched": 26370, "D_delay_matched": 26640, "F_time_frequency_matched": 25490, "DF_dual_matched": 24400}


class ParameterMatchedMultiView(nn.Module):
    def __init__(self, method: str) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        if method == "T_time_matched":
            self.time = TemporalEncoder(2, 18); dimension = self.time.output_dim
        elif method == "D_delay_matched":
            self.delay = TemporalEncoder(3, 18); dimension = self.delay.output_dim
        elif method == "F_time_frequency_matched":
            self.frequency = TimeFrequencyEncoder(16); dimension = self.frequency.output_dim
        else:
            self.delay = TemporalEncoder(3, 10); self.frequency = TimeFrequencyEncoder(10)
            dimension = 64
            self.fusion = nn.Sequential(nn.LayerNorm(80), nn.Linear(80, dimension), nn.GELU())
        self.classifier = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 6))
        self.parameter_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 12))

    def embedding(self, iq: torch.Tensor) -> torch.Tensor:
        if self.method == "T_time_matched": return self.time(iq.float())
        if self.method == "D_delay_matched": return self.delay(delay_peak_sequence(iq))
        if self.method == "F_time_frequency_matched": return self.frequency(local_time_frequency(iq))
        return self.fusion(torch.cat((self.delay(delay_peak_sequence(iq)), self.frequency(local_time_frequency(iq))), dim=1))

    def forward_with_embedding(self, iq: torch.Tensor):
        embedding = self.embedding(iq)
        return self.classifier(embedding), self.parameter_head(embedding).reshape(-1, 6, 2), embedding

    def forward(self, iq: torch.Tensor):
        logits, parameters, _ = self.forward_with_embedding(iq)
        return logits, parameters

    def exclusive_contrast_loss(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return embedding.new_zeros(())


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def power_aupr(row: dict[str, object], power: int) -> float:
    return float(np.mean([float(cell["aupr"]) for cell in row["cells"] if int(cell["power"]) == int(power)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root-template", default="generated_data/p3oc_parameter_matched_train_seed1030000000_{regime}_rechunked")
    parser.add_argument("--development-root-template", default="generated_data/p3oc_parameter_matched_dev_seed1040000000_{regime}_rechunked")
    parser.add_argument("--preregister", default="configs/p3oc_parameter_matched_multiview_preregister_20260801.json")
    parser.add_argument("--method-freeze", default="configs/p3oc_parameter_matched_multiview_method_freeze_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3oc_parameter_matched_multiview_development_20260801")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--warmup-epochs", type=int, default=2); parser.add_argument("--batch-size", type=int, default=63); parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--parameter-weight", type=float, default=.1); parser.add_argument("--contrast-weight", type=float, default=.0); parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--checkpoint-tolerance", type=float, default=1.01); parser.add_argument("--cp-cap", type=float, default=.10); parser.add_argument("--cp-confidence", type=float, default=.95); parser.add_argument("--bootstrap-draws", type=int, default=2000); parser.add_argument("--split-seed", type=int, default=20260801); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; preregister = project / args.preregister; freeze = json.loads((project / args.method_freeze).read_text(encoding="utf-8"))
    if freeze["preregister_sha256"] != sha256_file(preregister) or freeze["implementation_sha256"] != sha256_file(Path(__file__).resolve()): raise RuntimeError("P3o-C frozen hash mismatch")
    counts = {method: parameter_count(ParameterMatchedMultiView(method)) for method in METHODS}
    reference = counts["F_time_frequency_matched"]
    if counts != EXPECTED_PARAMETERS or any(abs(value / reference - 1.0) > .05 for value in counts.values()): raise RuntimeError(f"parameter match failed: {counts}")
    output = project / args.output_root; output.mkdir(parents=True, exist_ok=False); device = get_device(args.device)
    roots = {role: {regime: transition_files(project / template.format(regime=regime), "train", POWERS) for regime in REGIMES} for role, template in (("train", args.train_root_template), ("development", args.development_root_template))}
    metadata, references = {"train": {}, "development": {}}, {}
    for role in roots:
        paired = None
        for regime in REGIMES:
            audit_power_alignment(roots[role][regime]); identities, labels = read_identity_metadata(roots[role][regime][POWERS[0]])
            if paired is None: paired = identities
            elif not np.array_equal(paired, identities): raise RuntimeError(f"unpaired {role} regimes")
            metadata[role][regime] = {"identities": identities, "labels": labels}
        references[role] = paired
    labels, development_labels = metadata["train"]["ID_control"]["labels"], metadata["development"]["ID_control"]["labels"]
    if not np.array_equal(labels, development_labels): raise RuntimeError("label contract mismatch")
    roles = make_identity_roles(labels, args.split_seed); old = json.loads((project / args.run_root / "protocol.json").read_text(encoding="utf-8")); audit = audit_identities(project, references["train"], references["development"], {1030000000,1040000000}, Path(old["data_file"]))
    if audit["cross_role_overlap"] or audit["maximum_prior_overlap"]: raise RuntimeError("identity collision")
    (output / "identity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); np.savez_compressed(output / "identity_roles.npz", **roles)
    train_paths, development_paths = ordered_paths(roots["train"]), ordered_paths(roots["development"]); normalizer = fit_multi_file_normalizer(train_paths, roles["full_fit"], labels)
    datasets = {role: build_concat_dataset(train_paths, roles[role]) for role in ("full_fit", "early", "calibration", "selection")}; dev_indices = np.arange(len(development_labels), dtype=np.int64); development_dataset = build_concat_dataset(development_paths, dev_indices)
    meta = {role: role_metadata(metadata["train"], roles[role]) for role in ("early", "calibration", "selection")}; development_meta = role_metadata(metadata["development"], dev_indices)
    protocol = vars(args) | {"methods": list(METHODS), "model_seeds": list(MODEL_SEEDS), "parameter_counts": counts, "parameter_match_reference": reference, "role_identity_counts": {key: int(len(value)) for key,value in roles.items()}, "preregister_sha256": freeze["preregister_sha256"], "implementation_sha256": freeze["implementation_sha256"], "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "protocol.json").write_text(json.dumps(_jsonable(protocol), indent=2), encoding="utf-8")
    results = []
    try:
        for method in METHODS:
            for model_seed in MODEL_SEEDS:
                print(f"START method={method} seed={model_seed}", flush=True); set_seed(model_seed); generator = torch.Generator().manual_seed(model_seed)
                loaders = {"fit": DataLoader(datasets["full_fit"], batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.workers, pin_memory=device.type == "cuda")}
                loaders.update({role: DataLoader(datasets[role], batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda") for role in ("early","calibration","selection")}); development_loader = DataLoader(development_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
                model = ParameterMatchedMultiView(method).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None; run_dir = output / method / f"seed_{model_seed}"; run_dir.mkdir(parents=True); history=[]
                for epoch in range(1,args.epochs+1):
                    lr=_set_warmup_cosine_lr(optimizer,args.learning_rate,epoch,args.epochs,args.warmup_epochs); train=train_epoch(model,loaders["fit"],device,normalizer,optimizer,scaler,args); early=collect_outputs(model,loaders["early"],device,normalizer,args.parameter_weight); row={"epoch":epoch,"lr":lr,"train":train,"early_parameter_macro_nmae":float(output_parameter_metrics(early,normalizer)["oracle_macro_target_nmae"]),"early_macro_aupr":float(average_precision_score(early["labels"].numpy(),torch.sigmoid(early["logits"]).numpy(),average="macro") )}; history.append(row); torch.save(model.state_dict(),run_dir/f"epoch_{epoch:03d}.pt"); print(json.dumps({"method":method,"seed":model_seed,**row}),flush=True)
                choice=select_checkpoint_epochs(history,args.checkpoint_tolerance); epoch=int(choice["C_joint_epoch"]); model.load_state_dict(torch.load(run_dir/f"epoch_{epoch:03d}.pt",map_location=device,weights_only=True)); torch.save(model.state_dict(),run_dir/"C_joint.pt")
                calibration=collect_outputs(model,loaders["calibration"],device,normalizer,args.parameter_weight); f1=optimize_class_thresholds(torch.sigmoid(calibration["logits"]).numpy(),calibration["labels"].numpy()); safe_audit=select_cp_safe_isrj_threshold(torch.sigmoid(calibration["logits"]).numpy(),calibration["labels"].numpy(),meta["calibration"]["regime"],meta["calibration"]["power"],args.cp_cap,args.cp_confidence)
                if not safe_audit["feasible"]: raise RuntimeError(f"CP-safe infeasible: {method}/{model_seed}")
                safe=np.asarray(f1,dtype=np.float64).copy(); safe[ISRJ_INDEX]=float(safe_audit["threshold"]); selection=collect_outputs(model,loaders["selection"],device,normalizer,args.parameter_weight); development=collect_outputs(model,development_loader,device,normalizer,args.parameter_weight); prediction_file=run_dir/"C_joint_development_predictions.npz"; save_prediction_payload(prediction_file,development,development_meta)
                result={"method":method,"seed":model_seed,"parameters":counts[method],"history":history,"checkpoint_selection":choice,"C_joint_epoch":epoch,"safe_audit":safe_audit,"thresholds":{"T_F1":np.asarray(f1),"T_safe":safe},"selection":{"T_F1":evaluate_outputs(selection,f1,meta["selection"],normalizer),"T_safe":evaluate_outputs(selection,safe,meta["selection"],normalizer)},"development":{"T_F1":evaluate_outputs(development,f1,development_meta,normalizer),"T_safe":evaluate_outputs(development,safe,development_meta,normalizer)},"prediction_file":str(prediction_file.relative_to(project))}
                (run_dir/"result.json").write_text(json.dumps(_jsonable(result),indent=2),encoding="utf-8");results.append(result); candidate=result["development"]["T_safe"];print(json.dumps({"method":method,"seed":model_seed,"candidate_isrj_recall":candidate["isrj_recall"],"candidate_maximum_cell_fpr":candidate["maximum_cell_fpr"],"candidate_macro_aupr":candidate["macro_aupr"]}),flush=True);del model
                if device.type=="cuda":torch.cuda.empty_cache()
    finally:
        for dataset in datasets.values():close_concat_dataset(dataset)
        close_concat_dataset(development_dataset)
    indexed={(row["method"],int(row["seed"])):row for row in results}; comparisons=[];bootstrap=[]
    for seed in MODEL_SEEDS:
        candidate=indexed[("DF_dual_matched",seed)]; baseline=indexed[("F_time_frequency_matched",seed)]; c=candidate["development"]["T_safe"];b=baseline["development"]["T_safe"]
        comparisons.append({"seed":seed,"recall_delta_vs_frequency":c["isrj_recall"]-b["isrj_recall"],"aupr25_delta_vs_frequency":power_aupr(c,25)-power_aupr(b,25),"oracle_nmae_ratio":c["parameter"]["oracle_macro_target_nmae"]/b["parameter"]["oracle_macro_target_nmae"],"predicted_nmae_ratio":c["predicted_gate"]["predicted_gate_parameter_nmae"]/b["predicted_gate"]["predicted_gate_parameter_nmae"],"miss_delta":c["predicted_gate"]["miss_penalty_parameter_nmae"]-b["predicted_gate"]["miss_penalty_parameter_nmae"],"worst_delta":c["predicted_gate"]["worst_combo_predicted_gate_nmae"]-b["predicted_gate"]["worst_combo_predicted_gate_nmae"]})
        bootstrap.append({"seed":seed,"dual_vs_frequency_recall":bootstrap_isrj_recall_difference(project/candidate["prediction_file"],float(candidate["thresholds"]["T_safe"][ISRJ_INDEX]),project/baseline["prediction_file"],float(baseline["thresholds"]["T_safe"][ISRJ_INDEX]),args.bootstrap_draws,args.split_seed+seed)})
    candidate_rows=[indexed[("DF_dual_matched",seed)]["development"]["T_safe"] for seed in MODEL_SEEDS]
    gates={"identity_overlap_zero":audit["cross_role_overlap"]==0 and audit["maximum_prior_overlap"]==0,"parameter_match_within_5_percent":all(abs(value/reference-1)<=.05 for value in counts.values()),"all_seeds_maximum_cell_fpr_at_most_0p10":all(row["maximum_cell_fpr"]<=.10 for row in candidate_rows),"all_seeds_overall_isrj_recall_at_least_0p85":all(row["isrj_recall"]>=.85 for row in candidate_rows),"all_seeds_25db_isrj_recall_at_least_0p78":all(row["per_power"]["25"]["isrj_recall"]>=.78 for row in candidate_rows),"all_seeds_25db_aupr_delta_vs_F_time_frequency_matched_at_least_0p01":all(row["aupr25_delta_vs_frequency"]>=.01 for row in comparisons),"all_seeds_recall_delta_vs_F_time_frequency_matched_nonnegative":all(row["recall_delta_vs_frequency"]>=0 for row in comparisons),"all_seeds_oracle_and_predicted_gate_parameter_nmae_ratio_vs_F_time_frequency_matched_at_most_1p01":all(row["oracle_nmae_ratio"]<=1.01 and row["predicted_nmae_ratio"]<=1.01 for row in comparisons),"all_seeds_miss_penalty_and_worst_combination_nonworse_vs_F_time_frequency_matched":all(row["miss_delta"]<=0 and row["worst_delta"]<=0 for row in comparisons),"all_seeds_safe_calibration_feasible":all(row["safe_audit"]["feasible"] for row in results),"test_generated_false":True,"test_opened_false":True,"locked_test_used_false":True}
    summary={"experiment":"P3o-C parameter-matched multiview","development_passed":bool(all(gates.values())),"promotion_gates":gates,"parameter_counts":counts,"comparisons":comparisons,"clustered_paired_bootstrap":bootstrap,"results":results,"identity_audit":{key:audit[key] for key in audit if key!="rows"},"test_generated":False,"test_opened":False,"locked_test_used":False};(output/"summary.json").write_text(json.dumps(_jsonable(summary),indent=2),encoding="utf-8");print(json.dumps({"development_passed":summary["development_passed"],"promotion_gates":gates},indent=2),flush=True)


if __name__ == "__main__": main()

"""P5b3: attach a posterior-conditioned JNR estimator without changing classification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import (
    FeatureRole, PairedPowerOracleDataset, _power_files, bootstrap_gain,
    metrics, target_statistics, train_feature_probe, sha256_file,
)
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role, collect_frozen_posterior, with_context
from cj0724.multitask import build_multitask_model
from cj0724.paper_core_evidence import load_normalizer

SOURCE = "mlcnn_iq_grouped_task_routed"
TARGET_CLASSES = (3, 4, 5)
TARGET_NAMES = ("MFT.JNR", "VFT.JNR", "ISRJ.JNR")


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_base(project: Path, seed: int, base_root: str | Path | None = None):
    root = project / base_root if base_root is not None else project / "outputs/p4c_heterogeneous_route_variable_jnr_20260801/tuned_multitask_20260801_063924"
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    row = next(x for x in json.loads((root / "summary.json").read_text(encoding="utf-8"))
               if x["model"] == SOURCE and int(x["seed"]) == seed and x.get("checkpoint_objective") == "parameter")
    model = build_multitask_model(SOURCE, protocol["width"], protocol["dropout"], protocol["head_dim"],
                                  protocol["parameter_hidden_dim"], protocol["legacy_attention"])
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = project / checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"])
    return model, load_normalizer(protocol["normalizer"]), np.asarray(json.loads(row["thresholds"]), np.float32), {
        "source_seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "thresholds": json.loads(row["thresholds"]),
    }


def collect_base_parameter(dataset, model, normalizer, device):
    """Return the full base output plus classification values; no candidate is run here."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=126, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    logits, parameter = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            raw_logits, normalized = model(batch["iq"].to(device))
            logits.append(raw_logits.float().cpu().numpy())
            parameter.append(normalizer.inverse_transform(normalized.float().cpu()).numpy())
    return np.concatenate(logits), np.concatenate(parameter)


def predicted_gated_role(role: FeatureRole, probabilities: np.ndarray, thresholds: np.ndarray) -> FeatureRole:
    valid = role.valid.copy()
    for local, class_index in enumerate(TARGET_CLASSES):
        valid[:, local] &= probabilities[:, class_index] >= thresholds[class_index]
    return FeatureRole(role.feature, role.composition, role.target, valid, role.identities, role.power, role.combination)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("preregister", "data-root", "roles", "role-receipt", "data-audit", "output-root"):
        p.add_argument(f"--{name}", required=True)
    args = p.parse_args(); project = Path.cwd()
    prereg_path = project / args.preregister; protocol = json.loads(prereg_path.read_text(encoding="utf-8"))
    audit = json.loads((project / args.data_audit).read_text(encoding="utf-8")); receipt = json.loads((project / args.role_receipt).read_text(encoding="utf-8"))
    if not audit["valid_for_frozen_inference"] or not receipt["all_roles_disjoint"] or not receipt["same_roles_required_at_all_powers"]:
        raise RuntimeError("P5b3 audit/roles did not authorize the run")
    indices = np.load(project / args.roles); files = _power_files(project / args.data_root)
    out = project / args.output_root; out.mkdir(parents=True, exist_ok=True)
    datasets = {k: PairedPowerOracleDataset(files, indices[k]) for k in ("train", "early", "confirm")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analytic = {k: collect_analytic_role(v, device) for k, v in datasets.items()}
    _median, target_mean, target_scale = target_statistics(analytic["train"])
    rows = []
    for source_seed, head_seed in zip(protocol["frozen_base"]["model_seeds"], (14041, 14042, 14043)):
        model, normalizer, thresholds, source = load_base(project, int(source_seed)); model = model.to(device).eval()
        posterior = {k: collect_frozen_posterior(v, model, device) for k, v in datasets.items()}
        conditioned = {k: with_context(analytic[k], posterior[k]) for k in analytic}
        checkpoint_path = out / "frozen_candidate_heads" / f"source{source_seed}_head{head_seed}.pt"
        _early, estimate, train_receipt = train_feature_probe(
            conditioned["train"], conditioned["early"], conditioned["confirm"], int(head_seed), target_mean, target_scale, 30, 5,
            checkpoint_path=checkpoint_path,
        )
        logits, base_full = collect_base_parameter(datasets["confirm"], model, normalizer, device)
        probability = 1.0 / (1.0 + np.exp(-logits)); decision = probability >= thresholds[None, :]
        base = base_full[:, TARGET_CLASSES, 0].copy(); attached_full = base_full.copy()
        for local, class_index in enumerate(TARGET_CLASSES):
            attached_full[decision[:, class_index], class_index, 0] = estimate[decision[:, class_index], local]
        attached = attached_full[:, TARGET_CLASSES, 0]
        # All locations except the three explicitly predicted-positive JNR cells are invariant.
        allowed = np.zeros_like(base_full, bool)
        for class_index in TARGET_CLASSES: allowed[:, class_index, 0] = decision[:, class_index]
        invariant = bool(np.array_equal(base_full[~allowed], attached_full[~allowed]))
        oracle_base, oracle_candidate = metrics(base, conditioned["confirm"], target_scale), metrics(attached, conditioned["confirm"], target_scale)
        gated_role = predicted_gated_role(conditioned["confirm"], probability, thresholds)
        gated_base, gated_candidate = metrics(base, gated_role, target_scale), metrics(attached, gated_role, target_scale)
        rows.append({"source": source, "head_seed": int(head_seed), **train_receipt, "base": oracle_base, "attached": oracle_candidate,
                     "predicted_gated_base": gated_base, "predicted_gated_attached": gated_candidate,
                     "classification_hash": {"logits": array_hash(logits), "probabilities": array_hash(probability), "decisions": array_hash(decision)},
                     "other_parameter_invariant": invariant, "base": oracle_base, "base_prediction": base, "attached_prediction": attached,
                     "role": conditioned["confirm"], "powers": conditioned["confirm"].power})
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {}; bootstrap = {}; power_fail = []
    for local, name in enumerate(TARGET_NAMES):
        base_n = [r["base"]["by_target"][name]["nmae"] for r in rows]; new_n = [r["attached"]["by_target"][name]["nmae"] for r in rows]
        gain = [1.0 - new / old for old, new in zip(base_n, new_n)]
        gated_old = [r["predicted_gated_base"]["by_target"][name]["nmae"] for r in rows]
        gated_new = [r["predicted_gated_attached"]["by_target"][name]["nmae"] for r in rows]
        for r in rows:
            for power in (10, 30):
                delta = r["attached"]["by_target"][name]["per_power_nmae"][str(power)] - r["base"]["by_target"][name]["per_power_nmae"][str(power)]
                if delta > 0.03: power_fail.append({"target": name, "power": power, "delta": float(delta)})
        bootstrap[name] = [bootstrap_gain(r["base_prediction"], r["attached_prediction"], r["role"], local, target_scale, r["head_seed"]) for r in rows]
        summary[name] = {"base_nmae": base_n, "attached_nmae": new_n, "mean_reduction": float(np.mean(gain)), "reduction_by_seed": gain,
                         "predicted_gated_base_nmae": gated_old, "predicted_gated_attached_nmae": gated_new,
                         "mean_predicted_gated_delta": float(np.mean(np.asarray(gated_new)-np.asarray(gated_old)))}
    mft, vft, isrj = (summary[x] for x in TARGET_NAMES); gates = protocol["promotion_gates"]
    decision = {"classification_and_other_parameter_invariance": bool(all(r["other_parameter_invariant"] for r in rows)),
                "mft_gain": mft["mean_reduction"] >= gates["each_of_MFT_and_ISRJ_mean_oracle_gated_nmae_reduction_vs_base_at_least"],
                "isrj_gain": isrj["mean_reduction"] >= gates["each_of_MFT_and_ISRJ_mean_oracle_gated_nmae_reduction_vs_base_at_least"],
                "mft_three_seed": all(x > 0 for x in mft["reduction_by_seed"]), "isrj_three_seed": all(x > 0 for x in isrj["reduction_by_seed"]),
                "mft_predicted_gate": mft["mean_predicted_gated_delta"] <= 0, "isrj_predicted_gate": isrj["mean_predicted_gated_delta"] <= 0,
                "vft_control": vft["mean_reduction"] >= 0, "bootstrap": all(all(x["ci95_low"] > 0 for x in bootstrap[n]) for n in ("MFT.JNR","ISRJ.JNR")), "power": not power_fail}
    clean = [{k:v for k,v in r.items() if k not in ("base_prediction","attached_prediction","role","powers")} for r in rows]
    result = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(prereg_path), "data_audit": audit, "role_receipt": receipt,
              "rows": clean, "summary": summary, "bootstrap": bootstrap, "power_failures": power_fail, "promotion_gates": decision,
              "attachment_passed": bool(all(decision.values())), "test_generated": False, "test_opened": False, "locked_test_used": False}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary":summary,"promotion_gates":decision,"attachment_passed":result["attachment_passed"]},indent=2))
    for d in datasets.values(): d.close()

if __name__ == "__main__": main()

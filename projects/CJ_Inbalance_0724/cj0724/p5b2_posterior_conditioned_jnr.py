"""P5b2 deployability screen: frozen label posterior versus oracle composition.

This study is intentionally narrower than P5b1.  It never updates the
classification branch and never attaches an estimator to a classifier.  Its
only question is whether the frozen six-label posterior retains enough context
for a small analytic JNR estimator on wholly new paired-power identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cj0724.p5b1_jnr_physical_oracle import (
    POWERS,
    ROUTED,
    FeatureRole,
    PairedPowerOracleDataset,
    _power_files,
    analytic_features,
    bootstrap_gain,
    frozen_source_model,
    metrics,
    target_statistics,
    train_feature_probe,
    sha256_file,
)


TARGET_NAMES = ("MFT.JNR", "VFT.JNR", "ISRJ.JNR")


def collect_analytic_role(dataset: PairedPowerOracleDataset, device: torch.device) -> FeatureRole:
    """Collect P5b1's fixed physical feature bank and metadata once."""
    loader = DataLoader(dataset, batch_size=126, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    feature, composition, target, valid, identity, power, combination = [], [], [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            feature.append(analytic_features(batch["iq"].to(device)).float().cpu().numpy())
            composition.append(batch["composition"].numpy())
            target.append(batch["target"].numpy())
            valid.append(batch["valid"].numpy().astype(bool))
            identity.append(batch["identity"].numpy())
            power.append(batch["power"].numpy())
            combination.append(batch["combination"].numpy())
    return FeatureRole(
        feature=np.concatenate(feature), composition=np.concatenate(composition),
        target=np.concatenate(target), valid=np.concatenate(valid),
        identities=np.concatenate(identity), power=np.concatenate(power),
        combination=np.concatenate(combination),
    )


def collect_frozen_posterior(
    dataset: PairedPowerOracleDataset, model: torch.nn.Module, device: torch.device,
) -> np.ndarray:
    """Return six independent sigmoid probabilities; labels/power are never model inputs."""
    loader = DataLoader(dataset, batch_size=126, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    model.eval()
    pieces = []
    with torch.inference_mode():
        for batch in loader:
            iq = batch["iq"].to(device)
            feature = model.classification_adapter(model.classification_backbone(iq))
            pieces.append(torch.sigmoid(model.classifier(feature)).float().cpu().numpy())
    return np.concatenate(pieces).astype(np.float32)


def with_context(role: FeatureRole, context: np.ndarray) -> FeatureRole:
    context = np.asarray(context, dtype=np.float32)
    if context.shape != role.composition.shape:
        raise ValueError(f"Expected six-dimensional context {role.composition.shape}, got {context.shape}")
    return FeatureRole(
        feature=role.feature, composition=context, target=role.target, valid=role.valid,
        identities=role.identities, power=role.power, combination=role.combination,
    )


def summarize_rows(rows: list[dict[str, object]], baseline: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for target in TARGET_NAMES:
        baseline_target = baseline[target]
        baseline_nmae = float(baseline_target["nmae"])
        nmae = [float(row["confirm"]["by_target"][target]["nmae"]) for row in rows]
        correlation = [float(row["confirm"]["by_target"][target]["pearson"]) for row in rows]
        reduction = [1.0 - value / baseline_nmae for value in nmae]
        low_power = [
            1.0 - float(row["confirm"]["by_target"][target]["per_power_nmae"]["10"])
            / float(baseline_target["per_power_nmae"]["10"])
            for row in rows
        ]
        summary[target] = {
            "constant_confirm_nmae": baseline_nmae,
            "confirm_nmae_by_seed": nmae,
            "mean_nmae_reduction_vs_constant": float(np.mean(reduction)),
            "nmae_reduction_by_seed": reduction,
            "mean_confirm_pearson": float(np.mean(correlation)),
            "confirm_pearson_by_seed": correlation,
            "mean_10db_nmae_reduction_vs_constant": float(np.mean(low_power)),
            "10db_nmae_reduction_by_seed": low_power,
        }
    return summary


def gate_d1(
    summary: dict[str, object], rows: list[dict[str, object]], baseline: dict[str, object],
    target_scale: np.ndarray, gates: dict[str, object],
) -> tuple[dict[str, bool], dict[str, object], list[dict[str, object]]]:
    mft, vft, isrj = (summary[target] for target in TARGET_NAMES)
    worst_failures: list[dict[str, object]] = []
    bootstrap: dict[str, object] = {}
    for target_index, target in enumerate(TARGET_NAMES):
        baseline_target = baseline[target]
        for row in rows:
            candidate = row["confirm"]["by_target"][target]
            for combination, nmae in candidate["per_combination_nmae"].items():
                delta = float(nmae) - float(baseline_target["per_combination_nmae"][combination])
                if delta > float(gates["no_target_has_confirm_worst_combination_nmae_worse_than_D0_by_more_than"]):
                    worst_failures.append({"target": target, "combination": combination, "delta": delta})
        bootstrap[target] = [
            bootstrap_gain(
                row["constant_prediction"], row["prediction"], row["role"], target_index,
                target_scale, int(row["probe_seed"]),
            )
            for row in rows
        ]
    result = {
        "mft_mean_nmae_reduction": mft["mean_nmae_reduction_vs_constant"] >= gates["each_of_MFT_and_ISRJ_mean_confirm_nmae_reduction_vs_D0_at_least"],
        "isrj_mean_nmae_reduction": isrj["mean_nmae_reduction_vs_constant"] >= gates["each_of_MFT_and_ISRJ_mean_confirm_nmae_reduction_vs_D0_at_least"],
        "mft_three_seed_positive": all(value > 0.0 for value in mft["nmae_reduction_by_seed"]),
        "isrj_three_seed_positive": all(value > 0.0 for value in isrj["nmae_reduction_by_seed"]),
        "mft_pearson": mft["mean_confirm_pearson"] >= gates["each_of_MFT_and_ISRJ_mean_confirm_pearson_at_least"],
        "isrj_pearson": isrj["mean_confirm_pearson"] >= gates["each_of_MFT_and_ISRJ_mean_confirm_pearson_at_least"],
        "mft_10db": mft["mean_10db_nmae_reduction_vs_constant"] >= gates["each_of_MFT_and_ISRJ_10db_confirm_nmae_reduction_vs_D0_at_least"],
        "isrj_10db": isrj["mean_10db_nmae_reduction_vs_constant"] >= gates["each_of_MFT_and_ISRJ_10db_confirm_nmae_reduction_vs_D0_at_least"],
        "vft_control": vft["mean_nmae_reduction_vs_constant"] >= 0.0,
        "worst_combination": not worst_failures,
        "bootstrap_ci": all(
            all(float(item["ci95_low"]) > float(gates["each_of_MFT_and_ISRJ_identity_clustered_bootstrap_gain_ci95_low_greater_than"])
            for item in bootstrap[target]
        ) for target in ("MFT.JNR", "ISRJ.JNR")
        ),
    }
    return result, bootstrap, worst_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--role-receipt", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    project = Path.cwd()
    preregister_path = project / args.preregister
    protocol = json.loads(preregister_path.read_text(encoding="utf-8"))
    role_receipt = json.loads((project / args.role_receipt).read_text(encoding="utf-8"))
    data_audit = json.loads((project / args.data_audit).read_text(encoding="utf-8"))
    if not data_audit["valid_for_frozen_inference"] or not role_receipt["all_roles_disjoint"]:
        raise RuntimeError("P5b2 data or role audit did not authorize inference")
    if not role_receipt["same_roles_required_at_all_powers"]:
        raise RuntimeError("P5b2 roles are not paired across power views")

    role_indices = np.load(project / args.roles)
    files = _power_files(project / args.data_root)
    datasets = {name: PairedPowerOracleDataset(files, role_indices[name]) for name in ("train", "early", "confirm")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analytic = {name: collect_analytic_role(dataset, device) for name, dataset in datasets.items()}
    constant, target_mean, target_scale = target_statistics(analytic["train"])
    constant_prediction = np.broadcast_to(constant, analytic["confirm"].target.shape).copy()
    constant_metrics = metrics(constant_prediction, analytic["confirm"], target_scale)
    probe_seeds = [int(value) for value in protocol["training"]["probe_seeds"]]
    source_seeds = [int(value) for value in protocol["frozen_sources"]["model_seeds"]]
    if len(source_seeds) != len(probe_seeds):
        raise RuntimeError("P5b2 preregistration requires one frozen source per probe seed")

    d1_rows, d2_rows = [], []
    for source_seed, probe_seed in zip(source_seeds, probe_seeds):
        source_model, source_receipt = frozen_source_model(project, source_seed)
        source_model = source_model.to(device).eval()
        posterior = {
            name: collect_frozen_posterior(dataset, source_model, device)
            for name, dataset in datasets.items()
        }
        d1_roles = {name: with_context(analytic[name], posterior[name]) for name in analytic}
        early_prediction, confirm_prediction, receipt = train_feature_probe(
            d1_roles["train"], d1_roles["early"], d1_roles["confirm"], probe_seed,
            target_mean, target_scale, int(protocol["training"]["maximum_epochs"]),
            int(protocol["training"]["early_stopping_patience"]),
        )
        d1_rows.append({
            "probe_seed": probe_seed, "frozen_source": source_receipt, **receipt,
            "early": metrics(early_prediction, d1_roles["early"], target_scale),
            "confirm": metrics(confirm_prediction, d1_roles["confirm"], target_scale),
            "prediction": confirm_prediction, "role": d1_roles["confirm"],
            "constant_prediction": constant_prediction,
        })

        # D2 is an oracle upper bound with exactly the same physical features,
        # optimiser and model seed; it may never be a deployable candidate.
        early_prediction, confirm_prediction, receipt = train_feature_probe(
            analytic["train"], analytic["early"], analytic["confirm"], probe_seed,
            target_mean, target_scale, int(protocol["training"]["maximum_epochs"]),
            int(protocol["training"]["early_stopping_patience"]),
        )
        d2_rows.append({
            "probe_seed": probe_seed, **receipt,
            "early": metrics(early_prediction, analytic["early"], target_scale),
            "confirm": metrics(confirm_prediction, analytic["confirm"], target_scale),
        })
        del source_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline = constant_metrics["by_target"]
    d1_summary, d2_summary = summarize_rows(d1_rows, baseline), summarize_rows(d2_rows, baseline)
    d1_gates, bootstrap, worst_failures = gate_d1(
        d1_summary, d1_rows, baseline, target_scale, protocol["promotion_gates_for_D1"],
    )
    ratio_gate = float(protocol["promotion_gates_for_D1"]["D1_mft_and_isrj_mean_nmae_no_more_than_oracle_D2_times"])
    d1_gates["posterior_oracle_gap"] = all(
        float(d1_summary[target]["constant_confirm_nmae"]) > 0.0
        and float(np.mean(d1_summary[target]["confirm_nmae_by_seed"]))
        <= ratio_gate * float(np.mean(d2_summary[target]["confirm_nmae_by_seed"]))
        for target in ("MFT.JNR", "ISRJ.JNR")
    )
    output_root = project / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / "d1_confirm_predictions.npz"
    np.savez_compressed(
        prediction_path,
        predictions=np.stack([row["prediction"] for row in d1_rows]),
        target=d1_rows[0]["role"].target, valid=d1_rows[0]["role"].valid.astype(np.uint8),
        identities=d1_rows[0]["role"].identities, powers=d1_rows[0]["role"].power,
        combinations=d1_rows[0]["role"].combination,
    )
    compact_d1 = [{key: value for key, value in row.items() if key not in ("prediction", "role", "constant_prediction")} for row in d1_rows]
    payload = {
        "experiment": protocol["experiment"],
        "preregister": str(preregister_path), "preregister_sha256": sha256_file(preregister_path),
        "data_audit": data_audit, "role_receipt": role_receipt,
        "input_contract": {
            "D1_true_labels_input": False, "D1_nominal_power_input": False,
            "D1_classifier_updated": False, "D1_raw_iq_deep_expert": False,
            "D2_oracle_only": True, "powers_used": list(POWERS),
        },
        "target_normalization_train_only": {"mean": target_mean.tolist(), "scale": target_scale.tolist()},
        "results": {"D0_constant": {"prediction": constant.tolist(), "confirm": constant_metrics},
                    "D1_analytic_posterior": {"rows": compact_d1},
                    "D2_analytic_oracle": {"rows": d2_rows}},
        "D1_target_summary": d1_summary, "D2_oracle_target_summary": d2_summary,
        "D1_identity_clustered_bootstrap": bootstrap, "D1_worst_combination_failures": worst_failures,
        "promotion_gates": d1_gates, "posterior_conditioning_sufficient": bool(all(d1_gates.values())),
        "D1_confirm_predictions": str(prediction_path),
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    output_path = output_root / "result.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"D1_target_summary": d1_summary, "D2_oracle_target_summary": d2_summary,
                      "promotion_gates": d1_gates, "posterior_conditioning_sufficient": payload["posterior_conditioning_sufficient"]}, indent=2))
    for dataset in datasets.values():
        dataset.close()


if __name__ == "__main__":
    main()

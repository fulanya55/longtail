"""Train the two pre-registered P5e source/head seeds with no mechanism change."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import (
    PairedPowerOracleDataset, _power_files, bootstrap_gain, metrics, sha256_file,
    target_statistics, train_feature_probe,
)
from cj0724.p5b2_posterior_conditioned_jnr import (
    collect_analytic_role, collect_frozen_posterior, with_context,
)
from cj0724.p5b3_bounded_jnr_attachment import (
    TARGET_CLASSES, TARGET_NAMES, array_hash, collect_base_parameter, load_base,
    predicted_gated_role,
)


def _clean(rows):
    return [{key: value for key, value in row.items() if key not in {
        "base_prediction", "attached_prediction", "role", "target_scale",
    }} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preregister", "base-root", "data-root", "roles", "role-receipt", "data-audit", "output-root"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    project = Path.cwd()
    protocol_path = project / args.preregister
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = json.loads((project / args.data_audit).read_text(encoding="utf-8"))
    receipt = json.loads((project / args.role_receipt).read_text(encoding="utf-8"))
    if not audit["valid_for_frozen_inference"] or not receipt["all_roles_disjoint"]:
        raise RuntimeError("P5e requires the P5b3 development data audit and disjoint roles")
    p5b3 = project / protocol["parents"]["P5b3_result"]
    if sha256_file(p5b3) != protocol["parents"]["P5b3_result_sha256"]:
        raise RuntimeError("P5b3 result hash does not match the P5e preregistration")

    source_seeds = tuple(protocol["source_model_replication"]["new_source_model_seeds"])
    head_seeds = tuple(protocol["source_model_replication"]["new_head_model_seeds"])
    if len(source_seeds) != 2 or len(head_seeds) != 2:
        raise RuntimeError("P5e requires exactly two extension source/head seeds")
    files = _power_files(project / args.data_root)
    indices = np.load(project / args.roles)
    datasets = {name: PairedPowerOracleDataset(files, indices[name]) for name in ("train", "early", "confirm")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analytic = {name: collect_analytic_role(dataset, device) for name, dataset in datasets.items()}
    _, target_mean, target_scale = target_statistics(analytic["train"])
    output = project / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_seed, head_seed in zip(source_seeds, head_seeds):
        base_model, normalizer, thresholds, source = load_base(project, int(source_seed), args.base_root)
        base_model = base_model.to(device).eval()
        posterior = {name: collect_frozen_posterior(dataset, base_model, device) for name, dataset in datasets.items()}
        conditioned = {name: with_context(analytic[name], posterior[name]) for name in analytic}
        checkpoint = output / "frozen_candidate_heads" / f"source{source_seed}_head{head_seed}.pt"
        _, estimate, train_receipt = train_feature_probe(
            conditioned["train"], conditioned["early"], conditioned["confirm"], int(head_seed),
            target_mean, target_scale, 30, 5, checkpoint_path=checkpoint,
        )
        logits, base_full = collect_base_parameter(datasets["confirm"], base_model, normalizer, device)
        probability = 1.0 / (1.0 + np.exp(-logits))
        decision = probability >= thresholds[None, :]
        base = base_full[:, TARGET_CLASSES, 0].copy()
        attached_full = base_full.copy()
        for local, class_index in enumerate(TARGET_CLASSES):
            attached_full[decision[:, class_index], class_index, 0] = estimate[decision[:, class_index], local]
        attached = attached_full[:, TARGET_CLASSES, 0]
        allowed = np.zeros_like(base_full, dtype=bool)
        for class_index in TARGET_CLASSES:
            allowed[:, class_index, 0] = decision[:, class_index]
        gated = predicted_gated_role(conditioned["confirm"], probability, thresholds)
        rows.append({
            "source": source,
            "head_seed": int(head_seed),
            **train_receipt,
            "base": metrics(base, conditioned["confirm"], target_scale),
            "attached": metrics(attached, conditioned["confirm"], target_scale),
            "predicted_gated_base": metrics(base, gated, target_scale),
            "predicted_gated_attached": metrics(attached, gated, target_scale),
            "classification_hash": {
                "logits": array_hash(logits), "probabilities": array_hash(probability),
                "decisions": array_hash(decision),
            },
            "other_parameter_invariant": bool(np.array_equal(base_full[~allowed], attached_full[~allowed])),
            "base_prediction": base,
            "attached_prediction": attached,
            "role": conditioned["confirm"],
            "target_scale": target_scale,
        })
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary, bootstrap, power_failures = {}, {}, []
    for local, target in enumerate(TARGET_NAMES):
        oracle_gain, predicted_delta = [], []
        for row in rows:
            old = row["base"]["by_target"][target]["nmae"]
            new = row["attached"]["by_target"][target]["nmae"]
            oracle_gain.append(float(1.0 - new / old))
            predicted_delta.append(float(
                row["predicted_gated_attached"]["by_target"][target]["nmae"]
                - row["predicted_gated_base"]["by_target"][target]["nmae"]
            ))
            for power in (10, 30):
                delta = float(
                    row["attached"]["by_target"][target]["per_power_nmae"][str(power)]
                    - row["base"]["by_target"][target]["per_power_nmae"][str(power)]
                )
                if delta > 0.03:
                    power_failures.append({"source_seed": row["source"]["source_seed"], "target": target, "power": power, "delta": delta})
        bootstrap[target] = [
            bootstrap_gain(row["base_prediction"], row["attached_prediction"], row["role"], local, row["target_scale"], row["head_seed"])
            for row in rows
        ]
        summary[target] = {"oracle_gain_by_model": oracle_gain, "predicted_gated_delta_by_model": predicted_delta}
    gates = {
        "classification_and_other_parameter_invariance": all(row["other_parameter_invariant"] for row in rows),
        "mft_positive": all(value > 0.0 for value in summary["MFT.JNR"]["oracle_gain_by_model"]),
        "isrj_positive": all(value > 0.0 for value in summary["ISRJ.JNR"]["oracle_gain_by_model"]),
        "mft_predicted_not_worse": all(value <= 0.0 for value in summary["MFT.JNR"]["predicted_gated_delta_by_model"]),
        "isrj_predicted_not_worse": all(value <= 0.0 for value in summary["ISRJ.JNR"]["predicted_gated_delta_by_model"]),
        "mft_isrj_bootstrap_ci_low_positive": all(
            item["ci95_low"] > 0.0 for target in ("MFT.JNR", "ISRJ.JNR") for item in bootstrap[target]
        ),
        "no_10_or_30_power_degradation_over_0.03": not power_failures,
    }
    result = {
        "experiment": protocol["experiment"],
        "preregister_sha256": sha256_file(protocol_path),
        "base_root": str((project / args.base_root).resolve()),
        "data_audit": audit,
        "role_receipt": receipt,
        "rows": _clean(rows),
        "summary": summary,
        "bootstrap": bootstrap,
        "power_failures": power_failures,
        "development_gates": gates,
        "extension_passed": bool(all(gates.values())),
        "test_generated": False,
        "test_opened": False,
        "locked_test_used": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"extension_passed": result["extension_passed"], "development_gates": gates}, indent=2))
    for dataset in datasets.values():
        dataset.close()


if __name__ == "__main__":
    main()

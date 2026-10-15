"""Read-only two-new-identity confirmation for the frozen P5e five-model set."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cj0724.p5b1_jnr_physical_oracle import PairedPowerOracleDataset, _power_files, bootstrap_gain, metrics, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role, collect_frozen_posterior, with_context
from cj0724.p5b3_bounded_jnr_attachment import TARGET_CLASSES, TARGET_NAMES, collect_base_parameter, load_base, predicted_gated_role
from cj0724.p5b4_bounded_jnr_confirmation import estimate, load_head


def _clean(rows):
    return [{key: value for key, value in row.items() if key not in {
        "base_prediction", "attached_prediction", "role", "target_scale",
    }} for row in rows]


def evaluate_root(project: Path, root: Path, protocol: dict) -> tuple[list[dict], dict]:
    files = _power_files(root)
    dataset = PairedPowerOracleDataset(files, np.arange(3780, dtype=np.int64))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_started = time.perf_counter()
    analytic = collect_analytic_role(dataset, device)
    elapsed = {"analytic_feature_seconds": float(time.perf_counter() - feature_started), "per_model": []}
    rows = []
    for source_seed in protocol["frozen_candidate"]["source_model_seeds"]:
        seed_key = str(source_seed)
        started = time.perf_counter()
        model, normalizer, thresholds, source = load_base(project, int(source_seed), protocol["frozen_candidate"]["base_roots"][seed_key])
        model = model.to(device).eval()
        posterior = collect_frozen_posterior(dataset, model, device)
        role = with_context(analytic, posterior)
        head_path = project / protocol["frozen_candidate"]["head_checkpoints"][seed_key]
        head, state = load_head(head_path, device)
        value = estimate(role, head, state, device)
        logits, base_full = collect_base_parameter(dataset, model, normalizer, device)
        probability = 1.0 / (1.0 + np.exp(-logits))
        decision = probability >= thresholds[None, :]
        base = base_full[:, TARGET_CLASSES, 0].copy()
        attached_full = base_full.copy()
        for local, class_index in enumerate(TARGET_CLASSES):
            attached_full[decision[:, class_index], class_index, 0] = value[decision[:, class_index], local]
        attached = attached_full[:, TARGET_CLASSES, 0]
        allowed = np.zeros_like(base_full, dtype=bool)
        for class_index in TARGET_CLASSES:
            allowed[:, class_index, 0] = decision[:, class_index]
        gated = predicted_gated_role(role, probability, thresholds)
        elapsed["per_model"].append({
            "source_seed": int(source_seed),
            "seconds": float(time.perf_counter() - started),
            "attachment_active_cells": int(decision[:, TARGET_CLASSES].sum()),
            "candidate_target_cells": int(decision.shape[0] * len(TARGET_CLASSES)),
            "attachment_active_fraction": float(decision[:, TARGET_CLASSES].mean()),
        })
        rows.append({
            "source": source,
            "head": str(head_path),
            "head_sha256": sha256_file(head_path),
            "base": metrics(base, role, np.asarray(state["target_scale"])),
            "attached": metrics(attached, role, np.asarray(state["target_scale"])),
            "predicted_base": metrics(base, gated, np.asarray(state["target_scale"])),
            "predicted_attached": metrics(attached, gated, np.asarray(state["target_scale"])),
            "invariant": bool(np.array_equal(base_full[~allowed], attached_full[~allowed])),
            "base_prediction": base,
            "attached_prediction": attached,
            "role": role,
            "target_scale": np.asarray(state["target_scale"]),
        })
        del model, head
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dataset.close()
    return rows, elapsed


def summarize(rows: list[dict]) -> tuple[dict, dict, list[dict]]:
    summary, bootstrap, failures = {}, {}, []
    for local, target in enumerate(TARGET_NAMES):
        oracle_gain, predicted_delta = [], []
        for row in rows:
            old = row["base"]["by_target"][target]["nmae"]
            new = row["attached"]["by_target"][target]["nmae"]
            oracle_gain.append(float(1.0 - new / old))
            predicted_delta.append(float(row["predicted_attached"]["by_target"][target]["nmae"] - row["predicted_base"]["by_target"][target]["nmae"]))
            for power in (10, 30):
                delta = float(row["attached"]["by_target"][target]["per_power_nmae"][str(power)] - row["base"]["by_target"][target]["per_power_nmae"][str(power)])
                if delta > 0.03:
                    failures.append({"source_seed": row["source"]["source_seed"], "target": target, "power": power, "delta": delta})
        bootstrap[target] = [
            bootstrap_gain(row["base_prediction"], row["attached_prediction"], row["role"], local, row["target_scale"], 15000 + index)
            for index, row in enumerate(rows)
        ]
        summary[target] = {"oracle_gain_by_model": oracle_gain, "predicted_gated_delta_by_model": predicted_delta}
    return summary, bootstrap, failures


def data_gates(rows: list[dict], summary: dict, bootstrap: dict, failures: list[dict]) -> dict:
    return {
        "invariance": all(row["invariant"] for row in rows),
        "mft_positive": all(value > 0.0 for value in summary["MFT.JNR"]["oracle_gain_by_model"]),
        "isrj_positive": all(value > 0.0 for value in summary["ISRJ.JNR"]["oracle_gain_by_model"]),
        "mft_predicted_not_worse": all(value <= 0.0 for value in summary["MFT.JNR"]["predicted_gated_delta_by_model"]),
        "isrj_predicted_not_worse": all(value <= 0.0 for value in summary["ISRJ.JNR"]["predicted_gated_delta_by_model"]),
        "bootstrap": all(item["ci95_low"] > 0.0 for target in ("MFT.JNR", "ISRJ.JNR") for item in bootstrap[target]),
        "power": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    project = Path.cwd()
    protocol_path = project / args.preregister
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256_file(project / protocol["extension_result"]) != protocol["extension_result_sha256"]:
        raise RuntimeError("P5e extension result hash does not match confirmation freeze")
    if len(args.data_roots) != 2 or len(args.audits) != 2:
        raise RuntimeError("P5e confirmation requires exactly two fresh data audits")
    output = {"experiment": protocol["experiment"], "preregister_sha256": sha256_file(protocol_path), "data": {}, "test_generated": False, "test_opened": False, "locked_test_used": False}
    passed = True
    for data_root, audit_path in zip(args.data_roots, args.audits):
        audit = json.loads((project / audit_path).read_text(encoding="utf-8"))
        if not audit["valid_for_frozen_inference"]:
            raise RuntimeError(f"invalid P5e confirmation audit: {audit_path}")
        rows, runtime = evaluate_root(project, project / data_root, protocol)
        summary, bootstrap, failures = summarize(rows)
        gates = data_gates(rows, summary, bootstrap, failures)
        passed &= all(gates.values())
        output["data"][data_root] = {"audit": audit, "rows": _clean(rows), "summary": summary, "bootstrap": bootstrap, "power_failures": failures, "gates": gates, "runtime_and_lazy_call_cost": runtime}
    output["confirmation_passed"] = bool(passed)
    output["runtime_and_lazy_call_cost_reported"] = True
    root = project / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"confirmation_passed": output["confirmation_passed"], "gates": {key: value["gates"] for key, value in output["data"].items()}}, indent=2))


if __name__ == "__main__":
    main()

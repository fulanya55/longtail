"""Two fresh-identity confirmation of the unchanged frozen P1n ISRJ route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cj0724.dual_axis import audit_power_alignment, read_identity_metadata
from cj0724.integrate_swept_experts_with_six_class import _jsonable
from cj0724.p1f_isrj_confidence_bound_router import transition_files
from cj0724.p1k_joint_level3_system import _hash_int64
from cj0724.p2w_isrj_width_oracle_ood import POWERS, sha256_file
from cj0724.p3c_mft_sparse_deconvolution import REGIMES
from cj0724.p3f_shared_isrj_router import classification_metrics
from cj0724.p3g_cascaded_isrj_prefilter import collect_frozen_route
from cj0724.p3i_distilled_isrj_student import mean_recall
from cj0724.paper_core_evidence import load_normalizer


MODEL_SEEDS = (2041, 2042, 2043)
DATA_SEEDS = (640000000, 650000000)
ISRJ_INDEX = 5


def clustered_recall_contributions(labels, safe_probability, routed_probability, thresholds, identities):
    truth = labels[:, ISRJ_INDEX].astype(bool)
    safe = safe_probability[:, ISRJ_INDEX] >= thresholds[ISRJ_INDEX]
    routed = routed_probability[:, ISRJ_INDEX] >= thresholds[ISRJ_INDEX]
    positive_identities = np.unique(identities[truth])
    values = []
    for identity in positive_identities:
        selected = truth & (identities == identity)
        values.append(float(np.mean(routed[selected].astype(np.float64) - safe[selected].astype(np.float64))))
    return np.asarray(values, dtype=np.float64)


def paired_bootstrap(values: np.ndarray, seed: int, replicates: int = 10000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[sampled].mean(axis=1)
    return {"estimate": float(values.mean()), "ci_lower": float(np.quantile(draws, 0.025)), "ci_upper": float(np.quantile(draws, 0.975)), "replicates": int(replicates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-template", default="generated_data/p3l_p1n_confirmation_seed{data_seed}_{regime}_rechunked")
    parser.add_argument("--run-root", default="outputs/paper_mechanism_fullscale/tuned_multitask_20260725_165156")
    parser.add_argument("--p2c-root", default="outputs/p2c_lazy_joint_confirmation_20260801")
    parser.add_argument("--preregister", default="configs/p3l_p1n_two_identity_confirmation_preregister_20260801.json")
    parser.add_argument("--output-root", default="outputs/p3l_p1n_two_identity_confirmation_20260801")
    parser.add_argument("--excluded-identity-files", nargs="*", default=[])
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; output = project / args.output_root; output.mkdir(parents=True, exist_ok=False)
    run_root, p2c_root = project / args.run_root, project / args.p2c_root
    protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8")); rows = json.loads((run_root / "summary.json").read_text(encoding="utf-8")); p2c = json.loads((p2c_root / "method_freeze.json").read_text(encoding="utf-8")); normalizer = load_normalizer(protocol["normalizer"])
    selected_rows = {int(row["seed"]): row for row in rows if row["model"] == "mlcnn_iq_grouped_task_routed" and row.get("checkpoint_objective") == "parameter"}; freezes = {int(row["seed"]): row for row in p2c["seed_freezes"]}
    excluded = [Path(protocol["data_file"])] + [Path(value) if Path(value).is_absolute() else project / value for value in args.excluded_identity_files]
    datasets, all_references = {}, []
    for data_seed in DATA_SEEDS:
        roots = {regime: transition_files(project / args.data_root_template.format(data_seed=data_seed, regime=regime), "train", POWERS) for regime in REGIMES}; metadata, reference = {}, None
        for regime in REGIMES:
            audit_power_alignment(roots[regime]); identities, labels = read_identity_metadata(roots[regime][POWERS[0]])
            if reference is None: reference = identities
            elif not np.array_equal(reference, identities): raise RuntimeError("P3l regimes not paired")
            for path in excluded:
                previous, _ = read_identity_metadata(path)
                if np.intersect1d(identities, previous).size: raise RuntimeError("P3l identity overlap")
            for prior in all_references:
                if np.intersect1d(identities, prior).size: raise RuntimeError("P3l confirmation seed overlap")
            metadata[regime] = {"identities": identities, "labels": labels}
        all_references.append(reference); datasets[data_seed] = {"roots": roots, "metadata": metadata, "reference": reference}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); runs, contributions = [], {seed: [] for seed in MODEL_SEEDS}
    for data_seed in DATA_SEEDS:
        entry = datasets[data_seed]
        for seed in MODEL_SEEDS:
            route = collect_frozen_route(project, entry["roots"], entry["metadata"], seed, selected_rows[seed], freezes[seed], normalizer, device, p2c_root)
            thresholds = np.asarray(freezes[seed]["routed_thresholds"], dtype=np.float64)
            safe = classification_metrics(route["main"], route["labels"], thresholds, route["regime"], route["power"]); routed = classification_metrics(route["full"], route["labels"], thresholds, route["regime"], route["power"])
            recall_gain = mean_recall(routed) - mean_recall(safe); values = clustered_recall_contributions(route["labels"], route["main"], route["full"], thresholds, route["identity"]); contributions[seed].append(values)
            run = {"data_seed": data_seed, "model_seed": seed, "recall_gain": recall_gain, "paired_bootstrap": paired_bootstrap(values, 20260801 + data_seed + seed), "maximum_cell_fpr": float(max(row["fpr"] for row in routed["cells"])), "macro_aupr_delta_vs_main": float(routed["macro_aupr"] - safe["macro_aupr"]), "main": safe, "p1n": routed}
            runs.append(run); print(json.dumps({key: run[key] for key in ("data_seed", "model_seed", "recall_gain", "maximum_cell_fpr", "macro_aupr_delta_vs_main")}), flush=True)
    aggregate_by_model = {str(seed): paired_bootstrap(np.concatenate(contributions[seed]), 20260801 + seed) for seed in MODEL_SEEDS}
    overall_parts = []
    for data_index in range(len(DATA_SEEDS)):
        matrix = np.column_stack([contributions[seed][data_index] for seed in MODEL_SEEDS]); overall_parts.append(matrix.mean(axis=1))
    overall = paired_bootstrap(np.concatenate(overall_parts), 20260801)
    gates = {"identity_overlap_zero": True, "all_six_recall_gains_positive": bool(all(run["recall_gain"] > 0 for run in runs)), "each_model_aggregate_ci_lower_above_zero": bool(all(row["ci_lower"] > 0 for row in aggregate_by_model.values())), "overall_ci_lower_above_zero": bool(overall["ci_lower"] > 0), "maximum_cell_fpr_at_most_0p10": bool(all(run["maximum_cell_fpr"] <= 0.10 for run in runs)), "each_run_macro_aupr_delta_at_least_minus_0p005": bool(all(run["macro_aupr_delta_vs_main"] >= -0.005 for run in runs))}; passed = bool(all(gates.values()))
    summary = {"experiment": "P3l frozen P1n two-identity independent confirmation", "preregister_sha256": sha256_file(project / args.preregister), "identity_audit": {"data_seeds": {str(seed): _hash_int64(datasets[seed]["reference"]) for seed in DATA_SEEDS}, "cross_seed_overlap": 0}, "runs": runs, "aggregate_by_model_seed": aggregate_by_model, "overall_paired_bootstrap": overall, "promotion_gates": gates, "confirmation_passed": passed, "evidence_level": "Level 4 independent confirmation" if passed else "Level 3 development evidence only", "locked_test_used": False}; (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8"); print(json.dumps({"confirmation_passed": passed, "overall_paired_bootstrap": overall, "promotion_gates": gates, "evidence_level": summary["evidence_level"]}, indent=2))


if __name__ == "__main__": main()

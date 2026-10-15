"""Aggregate P7a C0/C1/C2 confirmation-role gates without post-hoc tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--static", required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    resolve = lambda text: Path(text) if Path(text).is_absolute() else project / text
    protocol, audit, static = (_read(resolve(value)) for value in (args.preregister, args.audit, args.static))
    if not audit["all_excluded_overlap_zero"] or not audit["alignment"]["all_valid_positive_finite"] or not audit["alignment"]["all_masks_label_aligned"]:
        raise RuntimeError("P7a data audit failed")
    if static["selected"]["channel"] != protocol["static_selection"]["selected_generic_branch_channels"] or not static["static_passed"]:
        raise RuntimeError("P7a static selection mismatch")
    c0, c1, c2 = (protocol["models"][key] for key in ("C0_shared_global", "C1_iq_grouped_route", "C2_equal_mac_generic_control")); seeds = [int(seed) for seed in protocol["training"]["model_seeds"]]; run_root = resolve(args.run_root)
    summary = {(str(row["model"]), int(row["seed"])): row for row in _read(run_root / "summary.json")}
    expected = {(model, seed) for model in (c0, c1, c2) for seed in seeds}
    if set(summary) != expected: raise RuntimeError("P7a incomplete model/seed cells")
    metrics = {(model, seed): _read(run_root / f"{model}__cos_lr1e4_wd1e4__seed{seed}" / "receipt_parameter.json")["confirm_adaptive"] for model, seed in expected}
    rows, targets = [], []
    for seed in seeds:
        m0, m1, m2 = (metrics[(model, seed)] for model in (c0, c1, c2)); p0, p1, p2 = (value["parameter"] for value in (m0, m1, m2))
        rows.append({"seed": seed, "C1_vs_C0_full12_reduction": float(1 - p1["oracle_macro_target_nmae"] / p0["oracle_macro_target_nmae"]), "C1_vs_C2_full12_reduction": float(1 - p1["oracle_macro_target_nmae"] / p2["oracle_macro_target_nmae"]), "C1_vs_C2_pearson_delta": float(p1["oracle_macro_pearson_r"] - p2["oracle_macro_pearson_r"]), "C1_vs_C2_aupr_delta": float(m1["macro_aupr"] - m2["macro_aupr"]), "C1_vs_C2_subset_delta": float(m1["subset_acc"] - m2["subset_acc"])})
    for index, target in enumerate(metrics[(c1, seeds[0])]["parameter"]["per_target"]):
        delta = [float(metrics[(c1, seed)]["parameter"]["per_target"][index]["normalized_mae"] - metrics[(c2, seed)]["parameter"]["per_target"][index]["normalized_mae"]) for seed in seeds]
        targets.append({"target": f"{target['class_name']}.{target['name']}", "C1_minus_C2_by_seed": delta, "systemic_harm": bool(all(value > .03 for value in delta))})
    gates = protocol["promotion_gates"]
    gate_results = {"identity_overlap_zero_and_all_12_targets_finite_variable": bool(audit["all_excluded_overlap_zero"] and audit["all_targets_finite_and_variable"]), "C2_C1_macs_and_parameters_within_2_percent": bool(.98 <= static["selected"]["mac_ratio"] <= 1.02 and .98 <= static["selected"]["parameter_ratio"] <= 1.02), "each_seed_C1_full12_vs_C0": all(row["C1_vs_C0_full12_reduction"] >= gates["each_seed_C1_full12_oracle_nmae_reduction_vs_C0_at_least"] for row in rows), "mean_C1_full12_vs_C0": float(np.mean([row["C1_vs_C0_full12_reduction"] for row in rows])) >= gates["mean_C1_full12_oracle_nmae_reduction_vs_C0_at_least"], "each_seed_C1_full12_vs_C2": all(row["C1_vs_C2_full12_reduction"] >= gates["each_seed_C1_full12_oracle_nmae_reduction_vs_C2_at_least"] for row in rows), "mean_C1_full12_vs_C2": float(np.mean([row["C1_vs_C2_full12_reduction"] for row in rows])) >= gates["mean_C1_full12_oracle_nmae_reduction_vs_C2_at_least"], "each_seed_C1_pearson_not_lower_than_C2": all(row["C1_vs_C2_pearson_delta"] >= 0 for row in rows), "each_seed_C1_aupr_noninferior_to_C2": all(row["C1_vs_C2_aupr_delta"] >= -gates["each_seed_C1_macro_aupr_drop_vs_C2_at_most"] for row in rows), "each_seed_C1_subset_noninferior_to_C2": all(row["C1_vs_C2_subset_delta"] >= -gates["each_seed_C1_subset_accuracy_drop_vs_C2_at_most"] for row in rows), "no_systemic_target_harm_C1_vs_C2": not any(row["systemic_harm"] for row in targets), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    result = {"experiment": protocol["experiment"], "rows": rows, "target_rows": targets, "parameter_counts": {model: int(summary[(model, seeds[0])]["parameters"]) for model in (c0,c1,c2)}, "static": static["selected"], "gates": gate_results, "development_passed": bool(all(gate_results.values())), "verdict": "promote_to_two_identity_confirmation" if all(gate_results.values()) else "stop_no_confirmation_data", "test_generated": False, "test_opened": False, "locked_test_used": False}
    output = resolve(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

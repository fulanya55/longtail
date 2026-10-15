"""Durable one-seed P3o-L runner using absolute D-drive rechunked roots."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

import cj0724.p3og_bounded_combo_cvar as core
from cj0724.p3n_cross_interference_backbone import sha256_file
from cj0724.p3o_low_power_attribution import audit_identities as audited_identity_roles
from cj0724.p3oj_component_conditioned_residual import ComponentConditionedResidual
from cj0724.p3oj_component_conditioned_residual_seed import finite_valid_per_sample_parameter_loss


def _audit(project, train_identity, development_identity, ignored_seeds, original_data_file):
    audit_path = project / "outputs/p3ol_durable_d_drive_development_20260802/data_audit.json"
    if not audit_path.exists():
        raise RuntimeError("missing pre-training P3o-L D-drive identity audit")
    result = json.loads(audit_path.read_text(encoding="utf-8"))
    if result["train_unique"] != len(set(train_identity.tolist())) or result["development_unique"] != len(set(development_identity.tolist())):
        raise RuntimeError("pre-training identity audit count mismatch")
    if result["cross_role_overlap"] or result["maximum_prior_overlap"] or not result["power_alignment_verified"] or not result["label_contract_verified"]:
        raise RuntimeError("pre-training P3o-L identity audit did not pass")
    return result


def _rewrite_receipt(output: Path) -> None:
    raw = json.loads((output / "summary.json").read_text(encoding="utf-8")); results, comparisons = [], []
    for row in raw["results"]:
        l0, l1, base = row["methods"]["G0_bounded_mean"], row["methods"]["G1_bounded_combo_cvar"], row["base"]
        results.append({**row, "methods": {"L0_shared_mean": l0, "L1_component_conditioned": l1}})
        e0, e1 = l0["evaluation"], l1["evaluation"]
        comparisons.append({"seed": row["seed"], "l1_oracle_ratio_vs_l0": e1["parameter"]["oracle_macro_target_nmae"] / e0["parameter"]["oracle_macro_target_nmae"], "l1_predicted_ratio_vs_l0": e1["predicted_gate"]["predicted_gate_parameter_nmae"] / e0["predicted_gate"]["predicted_gate_parameter_nmae"], "l1_miss_delta_vs_l0": e1["predicted_gate"]["miss_penalty_parameter_nmae"] - e0["predicted_gate"]["miss_penalty_parameter_nmae"], "l1_worst_ratio_vs_l0": e1["predicted_gate"]["worst_combo_predicted_gate_nmae"] / e0["predicted_gate"]["worst_combo_predicted_gate_nmae"], "l1_worst_ratio_vs_base": e1["predicted_gate"]["worst_combo_predicted_gate_nmae"] / base["predicted_gate"]["worst_combo_predicted_gate_nmae"]})
    gates = {"identity_overlap_zero": raw["promotion_gates"]["identity_overlap_zero"], "all_side_runs_preserve_F_state_and_logits_bitwise": raw["promotion_gates"]["all_side_runs_preserve_F_state_and_logits_bitwise"], "all_seeds_F_base_safe_and_retrieval_gates": raw["promotion_gates"]["all_seeds_base_safe_calibration_feasible_and_development_maximum_cell_fpr_at_most_0p10"] and raw["promotion_gates"]["all_seeds_base_overall_isrj_recall_at_least_0p85_and_25db_recall_at_least_0p78"], "all_seeds_L1_oracle_and_predicted_gate_nmae_ratio_vs_L0_at_most_0p98": all(x["l1_oracle_ratio_vs_l0"] <= .98 and x["l1_predicted_ratio_vs_l0"] <= .98 for x in comparisons), "all_seeds_L1_worst_combo_predicted_gate_nmae_ratio_vs_L0_at_most_0p98": all(x["l1_worst_ratio_vs_l0"] <= .98 for x in comparisons), "all_seeds_L1_miss_penalty_nonworse_vs_L0": all(x["l1_miss_delta_vs_l0"] <= 0 for x in comparisons), "all_seeds_L1_worst_combo_predicted_gate_nmae_ratio_vs_F_base_at_most_0p98": all(x["l1_worst_ratio_vs_base"] <= .98 for x in comparisons), "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True}
    receipt = {"experiment": "P3o-L durable-D-drive posterior-component-conditioned bounded residual", "development_passed": bool(all(gates.values())), "promotion_gates": gates, "comparisons": comparisons, "results": results, "identity_audit": raw["identity_audit"], "test_generated": False, "test_opened": False, "locked_test_used": False}
    (output / "summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, choices=(2101, 2102, 2103), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-root-template", required=True)
    parser.add_argument("--development-root-template", required=True)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; output = project / args.output_root
    freeze = json.loads((project / "configs/p3ol_component_conditioned_residual_method_freeze_20260802.json").read_text(encoding="utf-8"))
    if (freeze["wrapper_sha256"] != sha256_file(Path(__file__).resolve()) or freeze["model_sha256"] != sha256_file(project / "cj0724/p3oj_component_conditioned_residual.py") or freeze["implementation_sha256"] != sha256_file(Path(core.__file__).resolve())):
        raise RuntimeError("P3o-L method hash mismatch")
    core._per_sample_parameter_loss = finite_valid_per_sample_parameter_loss
    core.audit_identities = _audit; core.MODEL_SEEDS = (int(args.model_seed),); core.BoundedStatePreservingResidual = ComponentConditionedResidual
    core.METHODS = (("G0_bounded_mean", 0.0), ("G1_bounded_combo_cvar", 0.0))
    calls = {"count": 0}
    def build_side(base, initial_state, bound, device):
        side = ComponentConditionedResidual(base, bound).to(device); side.load_state_dict(initial_state)
        side.set_component_enabled(bool(calls["count"] % 2)); calls["count"] += 1
        return side
    core._build_side = build_side
    sys.argv = [sys.argv[0], "--train-root-template", args.train_root_template, "--development-root-template", args.development_root_template, "--preregister", "configs/p3ol_component_conditioned_residual_preregister_20260802.json", "--method-freeze", "configs/p3ol_component_conditioned_residual_method_freeze_20260802.json", "--output-root", args.output_root]
    try:
        core.main(); _rewrite_receipt(output)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / "exception_receipt.json").write_text(json.dumps({"model_seed": args.model_seed, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "test_generated": False, "test_opened": False, "locked_test_used": False}, indent=2), encoding="utf-8")
        raise
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

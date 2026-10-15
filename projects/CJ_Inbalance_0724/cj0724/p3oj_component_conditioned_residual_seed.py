"""One durable P3o-J seed: posterior-component residual versus shared mean residual."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

import cj0724.p3og_bounded_combo_cvar as core
from cj0724.p3n_cross_interference_backbone import sha256_file
from cj0724.p3o_low_power_attribution import audit_identities as audited_identity_roles
from cj0724.p3oj_component_conditioned_residual import ComponentConditionedResidual


def finite_valid_per_sample_parameter_loss(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & (labels[:, :, None] > 0.5)
    if bool((valid & ~torch.isfinite(target)).any()):
        raise ValueError("valid parameter targets must be finite")
    rows = []
    for index in range(prediction.shape[0]):
        selected = valid[index]
        rows.append(F.smooth_l1_loss(prediction[index][selected], target[index][selected], reduction="mean") if bool(selected.any()) else prediction[index].sum() * 0.0)
    return torch.stack(rows)


def _audit(project, train_identity, development_identity, ignored_seeds, original_data_file):
    return audited_identity_roles(project, train_identity, development_identity, {1450000000, 1460000000}, original_data_file)


def _rewrite_receipt(output: Path) -> None:
    raw = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    results, comparisons = [], []
    for row in raw["results"]:
        methods = row["methods"]
        j0, j1, base = methods["G0_bounded_mean"], methods["G1_bounded_combo_cvar"], row["base"]
        results.append({**row, "methods": {"J0_shared_mean": j0, "J1_component_conditioned": j1}})
        j0_eval, j1_eval = j0["evaluation"], j1["evaluation"]
        comparisons.append({
            "seed": row["seed"],
            "j1_oracle_ratio_vs_j0": j1_eval["parameter"]["oracle_macro_target_nmae"] / j0_eval["parameter"]["oracle_macro_target_nmae"],
            "j1_predicted_ratio_vs_j0": j1_eval["predicted_gate"]["predicted_gate_parameter_nmae"] / j0_eval["predicted_gate"]["predicted_gate_parameter_nmae"],
            "j1_miss_delta_vs_j0": j1_eval["predicted_gate"]["miss_penalty_parameter_nmae"] - j0_eval["predicted_gate"]["miss_penalty_parameter_nmae"],
            "j1_worst_ratio_vs_j0": j1_eval["predicted_gate"]["worst_combo_predicted_gate_nmae"] / j0_eval["predicted_gate"]["worst_combo_predicted_gate_nmae"],
            "j1_worst_ratio_vs_base": j1_eval["predicted_gate"]["worst_combo_predicted_gate_nmae"] / base["predicted_gate"]["worst_combo_predicted_gate_nmae"],
        })
    gates = {
        "identity_overlap_zero": raw["promotion_gates"]["identity_overlap_zero"],
        "all_side_runs_preserve_F_state_and_logits_bitwise": raw["promotion_gates"]["all_side_runs_preserve_F_state_and_logits_bitwise"],
        "all_seeds_F_base_safe_and_retrieval_gates": raw["promotion_gates"]["all_seeds_base_safe_calibration_feasible_and_development_maximum_cell_fpr_at_most_0p10"] and raw["promotion_gates"]["all_seeds_base_overall_isrj_recall_at_least_0p85_and_25db_recall_at_least_0p78"],
        "all_seeds_J1_oracle_and_predicted_gate_nmae_ratio_vs_J0_at_most_0p98": all(item["j1_oracle_ratio_vs_j0"] <= .98 and item["j1_predicted_ratio_vs_j0"] <= .98 for item in comparisons),
        "all_seeds_J1_worst_combo_predicted_gate_nmae_ratio_vs_J0_at_most_0p98": all(item["j1_worst_ratio_vs_j0"] <= .98 for item in comparisons),
        "all_seeds_J1_miss_penalty_nonworse_vs_J0": all(item["j1_miss_delta_vs_j0"] <= 0 for item in comparisons),
        "all_seeds_J1_worst_combo_predicted_gate_nmae_ratio_vs_F_base_at_most_0p98": all(item["j1_worst_ratio_vs_base"] <= .98 for item in comparisons),
        "test_generated_false": True, "test_opened_false": True, "locked_test_used_false": True,
    }
    receipt = {
        "experiment": "P3o-J posterior-component-conditioned bounded parameter residual",
        "development_passed": bool(all(gates.values())), "promotion_gates": gates,
        "comparisons": comparisons, "results": results, "identity_audit": raw["identity_audit"],
        "test_generated": False, "test_opened": False, "locked_test_used": False,
    }
    (output / "summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, required=True, choices=(2091, 2092, 2093))
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; output = project / args.output_root
    freeze = json.loads((project / "configs/p3oj_component_conditioned_residual_method_freeze_20260802.json").read_text(encoding="utf-8"))
    if freeze["wrapper_sha256"] != sha256_file(Path(__file__).resolve()) or freeze["model_sha256"] != sha256_file(project / "cj0724/p3oj_component_conditioned_residual.py") or freeze["implementation_sha256"] != sha256_file(Path(core.__file__).resolve()):
        raise RuntimeError("P3o-J method hash mismatch")
    core._per_sample_parameter_loss = finite_valid_per_sample_parameter_loss
    core.audit_identities = _audit; core.MODEL_SEEDS = (int(args.model_seed),)
    core.BoundedStatePreservingResidual = ComponentConditionedResidual
    core.METHODS = (("G0_bounded_mean", 0.0), ("G1_bounded_combo_cvar", 0.0))
    calls = {"count": 0}
    def build_side(base, initial_state, bound, device):
        side = ComponentConditionedResidual(base, bound).to(device); side.load_state_dict(initial_state)
        side.set_component_enabled(bool(calls["count"] % 2)); calls["count"] += 1
        return side
    core._build_side = build_side
    sys.argv = [sys.argv[0], "--train-root-template", "generated_data/p3oj_component_conditioned_train_seed1450000000_{regime}_rechunked", "--development-root-template", "generated_data/p3oj_component_conditioned_dev_seed1460000000_{regime}_rechunked", "--preregister", "configs/p3oj_component_conditioned_residual_preregister_20260802.json", "--method-freeze", "configs/p3oj_component_conditioned_residual_method_freeze_20260802.json", "--output-root", args.output_root]
    try:
        core.main(); _rewrite_receipt(output)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / "exception_receipt.json").write_text(json.dumps({"model_seed": args.model_seed, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "test_generated": False, "test_opened": False, "locked_test_used": False}, indent=2), encoding="utf-8")
        raise
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__": main()

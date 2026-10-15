"""P3o-I one-seed durable runner for the finite-valid compound-CVaR comparison."""
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
    return audited_identity_roles(project, train_identity, development_identity, {1330000000, 1340000000}, original_data_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, required=True, choices=(2081, 2082, 2083))
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]; output = project / args.output_root
    freeze = json.loads((project / "configs/p3oi_durable_finite_combo_cvar_method_freeze_20260802.json").read_text(encoding="utf-8"))
    if freeze["wrapper_sha256"] != sha256_file(Path(__file__).resolve()) or freeze["implementation_sha256"] != sha256_file(Path(core.__file__).resolve()):
        raise RuntimeError("P3o-I method hash mismatch")
    core._per_sample_parameter_loss = finite_valid_per_sample_parameter_loss
    core.audit_identities = _audit; core.MODEL_SEEDS = (int(args.model_seed),)
    sys.argv = [sys.argv[0], "--train-root-template", "generated_data/p3oi_durable_finite_combo_cvar_train_seed1330000000_{regime}_rechunked", "--development-root-template", "generated_data/p3oi_durable_finite_combo_cvar_dev_seed1340000000_{regime}_rechunked", "--preregister", "configs/p3oi_durable_finite_combo_cvar_preregister_20260802.json", "--method-freeze", "configs/p3oi_durable_finite_combo_cvar_method_freeze_20260802.json", "--output-root", args.output_root]
    try:
        core.main()
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / "exception_receipt.json").write_text(json.dumps({"model_seed": args.model_seed, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "test_generated": False, "test_opened": False, "locked_test_used": False}, indent=2), encoding="utf-8")
        raise
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__": main()
